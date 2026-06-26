"""Offline RL on the pooled D_logs (Spec Section 20).

Three methods on a *static* dataset (no env interaction during training):
  - bc    : behavioural cloning (cross-entropy on logged actions)
  - naive : vanilla offline DQN  -> meant to FAIL (Q overestimation / OOD blow-up)
  - cql   : Conservative Q-Learning (DQN + log-sum-exp penalty) -> the fix

Eval uses the env via evaluate() with a _flatten_obs+standardize wrapper.
Run:  python code/offline_rl.py --data offline_pool.npz --steps 40000 --cql-alpha 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from dqn_agent import QNetwork  # reuse the exact MLP arch from Role A

from drone_dispatch_env.config import Config
from drone_dispatch_env.offline import load_offline_dataset, _flatten_obs
from drone_dispatch_env.evaluate import evaluate

OBS_DIM = 181
N_ACTIONS = 169
HIDDEN = [256, 256]
GAMMA = 0.99
EVAL_SEEDS = [0, 1, 2]  # locked up front, same seeds as Role A's reported eval


def load_pool(path, device):
    d = load_offline_dataset(path)
    obs = d["observations"].astype(np.float32)
    nobs = d["next_observations"].astype(np.float32)
    # Standardize once from the pool; reuse identical stats at eval (advisor #1).
    mean = obs.mean(0)
    std = obs.std(0) + 1e-6
    obs = (obs - mean) / std
    nobs = (nobs - mean) / std
    data = dict(
        obs=torch.as_tensor(obs, device=device),
        act=torch.as_tensor(d["actions"].astype(np.int64), device=device),
        rew=torch.as_tensor(d["rewards"].astype(np.float32), device=device),
        nobs=torch.as_tensor(nobs, device=device),
        # d4rl / offline-RL convention: a truncation (timeout) is NOT a terminal, so we
        # bootstrap THROUGH it and exclude `timeouts` from the Bellman backup; only a
        # true terminal (all drones lost) zeros the bootstrap. The dataset ships a
        # separate `timeouts` array precisely so it can be excluded here.
        done=torch.as_tensor(d["terminals"].astype(np.float32), device=device),
    )
    return data, mean.astype(np.float32), std.astype(np.float32)


def _qnet(device):
    return QNetwork(OBS_DIM, N_ACTIONS, HIDDEN).to(device)


def _q_diag(net, obs):
    """Mean/max of max_a Q(s,a) on a fixed batch — the overestimation probe."""
    with torch.no_grad():
        q = net(obs).max(1).values
    return float(q.mean()), float(q.max())


def train(method, data, steps, batch, lr, cql_alpha, device, log, seed):
    net, tgt = _qnet(device), _qnet(device)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = data["obs"].shape[0]
    probe = data["obs"][:4096]  # fixed states for the Q-magnitude probe
    rng = np.random.default_rng(seed)

    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, n, batch), device=device)
        o, a, r, no, dn = (data["obs"][idx], data["act"][idx], data["rew"][idx],
                           data["nobs"][idx], data["done"][idx])

        if method == "bc":
            loss = F.cross_entropy(net(o), a)
        else:
            q = net(o).gather(1, a[:, None]).squeeze(1)
            with torch.no_grad():
                # ponytail: no masks stored -> Bellman max over all 169 actions.
                # For naive DQN this is exactly the OOD-overestimation we want to show.
                target = r + GAMMA * (1 - dn) * tgt(no).max(1).values
            loss = F.mse_loss(q, target)
            if method == "cql":
                # conservative penalty: push down all actions, pull up data action
                cql = (torch.logsumexp(net(o), 1) - q).mean()
                loss = loss + cql_alpha * cql

        opt.zero_grad(); loss.backward(); opt.step()
        if method != "bc" and step % 1000 == 0:
            tgt.load_state_dict(net.state_dict())
        if step % max(1, steps // 20) == 0:
            mq, xq = _q_diag(net, probe)
            log.append(dict(seed=seed, method=method, step=step,
                            loss=float(loss.detach()),
                            mean_q=mq, max_q=xq))
            print(f"  [{method}] step {step:6d} loss={float(loss):8.3f} "
                  f"mean_q={mq:8.2f} max_q={xq:8.2f}")
    return net


class _Wrapped:
    """Adapts a trained net to the evaluate() policy interface (.act)."""
    def __init__(self, net, mean, std, device, greedy_logits=False):
        self.net, self.device = net, device
        self.mean = torch.as_tensor(mean, device=device)
        self.std = torch.as_tensor(std, device=device)
        self.net.eval()

    def act(self, obs):
        x = (torch.as_tensor(_flatten_obs(obs), device=self.device) - self.mean) / self.std
        with torch.no_grad():
            q = self.net(x[None]).squeeze(0).cpu().numpy()
        mask = np.asarray(obs["action_mask"], dtype=bool)
        return int(np.argmax(np.where(mask, q, -np.inf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/offline_cql.yaml",
                    help="YAML with the hyperparameters (numbers live here, not in code)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--bc-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--cql-alpha", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--qstats", default=None)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # config file holds the defaults; explicit CLI flags override it
    import yaml
    cfgd = {}
    if args.config and Path(args.config).exists():
        cfgd = yaml.safe_load(open(args.config)) or {}
    def pick(name, fallback):
        v = getattr(args, name.replace("-", "_"))
        return v if v is not None else cfgd.get(name, fallback)
    args.data = pick("data", "offline_pool.npz")
    args.steps = int(pick("steps", 40000))
    args.bc_steps = int(pick("bc-steps", 15000))
    args.batch = int(pick("batch", 256))
    args.lr = float(pick("lr", 1e-3))
    args.cql_alpha = float(pick("cql-alpha", 1.0))
    args.out = pick("out", "logs/offline_results.json")
    global OBS_DIM, N_ACTIONS, HIDDEN, GAMMA, EVAL_SEEDS
    OBS_DIM = int(cfgd.get("obs-dim", 181))
    N_ACTIONS = int(cfgd.get("n-actions", 169))
    HIDDEN = [int(x) for x in cfgd.get("hidden", [256, 256])]
    GAMMA = float(cfgd.get("gamma", 0.99))
    EVAL_SEEDS = [int(s) for s in cfgd.get("eval-seeds", [0, 1, 2])]
    training_seeds = [int(s) for s in cfgd.get("seeds", [0, 1, 2])]
    seed = int(args.seed if args.seed is not None else training_seeds[0])
    args.qstats = args.qstats or f"logs/offline_qstats_seed{seed}.csv"
    args.weight = args.weight or f"weights/offline_cql_seed{seed}.pt"

    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  data={args.data}")
    data, mean, std = load_pool(args.data, device)
    print(f"pool: {data['obs'].shape[0]} transitions")

    cfg = Config()
    log, results = [], {}
    refs = {"random": 18.78, "greedy_nearest": 4.57, "online_dqn_1M": 6.76}

    plans = [("bc", args.bc_steps), ("naive", args.steps), ("cql", args.steps)]
    nets = {}
    for method, steps in plans:
        print(f"== train {method} ({steps} steps) ==")
        nets[method] = train(method, data, steps, args.batch, args.lr,
                             args.cql_alpha, device, log, seed)
        pol = _Wrapped(nets[method], mean, std, device)
        m = evaluate(pol, cfg, seeds=EVAL_SEEDS)["mean"]
        results[method] = m
        print(f"   -> cost_per_order={m['cost_per_order']:.2f} "
              f"success={m['success_rate']:.2f} return={m['episode_return']:.1f}")

    Path("logs").mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"refs": refs, "results": results, "training_seed": seed,
                   "eval_seeds": EVAL_SEEDS}, f, indent=2)
    import csv
    with open(args.qstats, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["seed", "method", "step", "loss", "mean_q", "max_q"])
        w.writeheader(); w.writerows(log)

    # save the submitted offline policy (CQL) + its normalization stats, so
    # run_all.py can load and evaluate it without retraining (spec: one weight/method)
    Path("weights").mkdir(exist_ok=True)
    torch.save({"model_state": nets["cql"].state_dict(), "mean": mean, "std": std,
                "obs_dim": OBS_DIM, "n_actions": N_ACTIONS, "hidden": HIDDEN},
               args.weight)

    print("\n=== OFFLINE RL — cost_per_order (lower better), eval seeds", EVAL_SEEDS, "===")
    for k, v in refs.items():
        print(f"  {k:16s} {v:6.2f}  (reference)")
    for k in ("bc", "naive", "cql"):
        print(f"  {k:16s} {results[k]['cost_per_order']:6.2f}")

    # one runnable check: naive must overestimate more than cql by the end
    qend = {m: [r['max_q'] for r in log if r['method'] == m][-1] for m in ("naive", "cql")}
    print(f"\nfinal max_q  naive={qend['naive']:.1f}  cql={qend['cql']:.1f}")
    assert qend["naive"] > qend["cql"], "expected naive DQN to overestimate vs CQL"
    print("OK: naive overestimates vs CQL (conservatism confirmed)")


if __name__ == "__main__":
    main()
