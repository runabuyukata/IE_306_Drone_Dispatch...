"""Multi-agent IDQN with parameter sharing on DroneDispatchMA-v0 (Spec Section 21).

Decentralized: each of the 8 drones is an independent DQN agent, but all agents
*share one Q-network* (parameter sharing) and pool their transitions into one
replay buffer. Per-agent obs = 59-dim local view, 4 actions (accept/move/charge/
idle). This is the canonical IDQN setup; the catch it exposes is **non-stationarity**
— from any one agent's view the other 7 are part of the environment and keep
changing as they learn, so the transition distribution is non-stationary.

Eval reconstructs the same cost_per_order used by the centralized env from the
reward stream: cost = 10*delivered + 5*ontime - return  (deliveries counted
exactly via a TO_DROPOFF->IDLE status transition on a live drone).

Run:  python code/train_ma_idqn.py --steps 150000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from dqn_agent import QNetwork

from drone_dispatch_env.config import Config, IDLE, TO_DROPOFF
from drone_dispatch_env.env_ma import DroneDispatchMAEnv

def eval_ma(net, cfg, device, policy="greedy", seeds=(0, 1, 2), rng=None):
    """Returns (mean_return, mean_cost_per_order, mean_delivered)."""
    env = DroneDispatchMAEnv(cfg)
    rng = rng or np.random.default_rng(0)
    rets, costs, delivs = [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        prev = {i: IDLE for i in range(cfg.n_drones)}
        R = D = OT = 0
        done = False
        while not done:
            actions = {}
            for i, a in enumerate(env.agents):
                if policy == "random":
                    actions[a] = int(rng.integers(4))
                else:
                    x = torch.as_tensor(obs[a], device=device)[None]
                    with torch.no_grad():
                        actions[a] = int(net(x).argmax(1).item())
            delivery_deadlines = {
                i: next((o.deadline for o in env.pending
                         if o.oid == env.drones[i].order_id), None)
                for i in range(cfg.n_drones)
                if env.drones[i].status == TO_DROPOFF
            }
            delivery_t = env.t
            obs, rew, terms, truncs, _ = env.step(actions)
            R += sum(rew.values())
            for i, a in enumerate(env.agents):
                cs = env.drones[i].status
                if prev[i] == TO_DROPOFF and cs == IDLE and not env.drones[i].lost:
                    D += 1
                    deadline = delivery_deadlines.get(i)
                    if deadline is not None and delivery_t <= deadline:
                        OT += 1
                prev[i] = cs
            done = all(terms.values())
        rets.append(R)
        costs.append((10 * D + 5 * OT - R) / max(D, 1))
        delivs.append(D)
    return float(np.mean(rets)), float(np.mean(costs)), float(np.mean(delivs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ma_idqn.yaml",
                    help="YAML with the hyperparameters (numbers live here, not in code)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--buffer", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--eps-frac", type=float, default=None)
    ap.add_argument("--target-every", type=int, default=None)
    ap.add_argument("--train-every", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None,
                    help="periodic eval interval (0 = steps//6)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import yaml
    cfgd = {}
    if args.config and Path(args.config).exists():
        cfgd = yaml.safe_load(open(args.config)) or {}
    def pick(name, fallback):
        v = getattr(args, name.replace("-", "_"))
        return v if v is not None else cfgd.get(name, fallback)
    args.steps = int(pick("steps", 60000))
    args.batch = int(pick("batch", 128))
    args.lr = float(pick("lr", 1e-3))
    args.buffer = int(pick("buffer", 100000))
    args.warmup = int(pick("warmup", 2000))
    args.eps_frac = float(pick("eps-frac", 0.5))
    args.target_every = int(pick("target-every", 1000))
    args.train_every = int(pick("train-every", 4))
    args.eval_every = int(pick("eval-every", 0))
    args.out = pick("out", "logs/ma_results.json")
    seed = int(args.seed if args.seed is not None else cfgd.get("seed", 0))
    hidden = [int(x) for x in cfgd.get("hidden", [128, 128])]
    gamma = float(cfgd.get("gamma", 0.99))
    obs_dim = int(cfgd.get("obs-dim", 59))
    n_actions = int(cfgd.get("n-actions", 4))
    eval_seeds = [int(s) for s in cfgd.get("eval-seeds", [0, 1, 2])]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    torch.set_num_threads(1)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    net = QNetwork(obs_dim, n_actions, hidden).to(device)
    tgt = QNetwork(obs_dim, n_actions, hidden).to(device)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buf = deque(maxlen=args.buffer)
    rng = np.random.default_rng(seed)

    env = DroneDispatchMAEnv(cfg)
    obs, _ = env.reset(seed=0)
    eps_steps = int(args.steps * args.eps_frac)
    log = []
    print(f"device={device} agents={cfg.n_drones} steps={args.steps}")

    for step in range(1, args.steps + 1):
        eps = max(0.05, 1.0 - (1 - 0.05) * step / eps_steps)
        actions = {}
        for a in env.agents:
            if rng.random() < eps:
                actions[a] = int(rng.integers(n_actions))
            else:
                with torch.no_grad():
                    x = torch.as_tensor(obs[a], device=device)[None]
                    actions[a] = int(net(x).argmax(1).item())
        nobs, rew, terms, truncs, _ = env.step(actions)
        for a in env.agents:                       # pooled, shared-parameter buffer
            buf.append((obs[a], actions[a], rew[a], nobs[a], float(terms[a])))
        obs = nobs
        if all(terms.values()):
            obs, _ = env.reset(seed=int(rng.integers(1_000_000)))

        if (len(buf) >= max(args.warmup, args.batch)
                and step % args.train_every == 0):
            batch = random.sample(buf, args.batch)
            o, a, r, no, d = zip(*batch)
            o = torch.as_tensor(np.stack(o), device=device)
            a = torch.as_tensor(np.asarray(a, np.int64), device=device)
            r = torch.as_tensor(np.asarray(r, np.float32), device=device)
            no = torch.as_tensor(np.stack(no), device=device)
            d = torch.as_tensor(np.asarray(d, np.float32), device=device)
            q = net(o).gather(1, a[:, None]).squeeze(1)
            with torch.no_grad():
                target = r + gamma * (1 - d) * tgt(no).max(1).values
            loss = F.smooth_l1_loss(q, target)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % args.target_every == 0:
                tgt.load_state_dict(net.state_dict())

        eval_every = args.eval_every or max(1, args.steps // 6)
        if step % eval_every == 0:
            ret, cost, deliv = eval_ma(net, cfg, device, seeds=eval_seeds)
            log.append(dict(step=step, eps=round(eps, 3), ret=ret, cost=cost, deliv=deliv))
            print(f"  step {step:6d} eps={eps:.2f} return={ret:8.1f} "
                  f"cost={cost:7.2f} delivered={deliv:.1f}")

    # baselines + final head-to-head
    rnd = eval_ma(net, cfg, device, policy="random", seeds=eval_seeds)
    idqn = eval_ma(net, cfg, device, policy="greedy", seeds=eval_seeds)
    results = {
        "idqn_ma": {"return": idqn[0], "cost_per_order": idqn[1], "delivered": idqn[2]},
        "random_ma": {"return": rnd[0], "cost_per_order": rnd[1], "delivered": rnd[2]},
        "ref_centralized_double_dqn": {"cost_per_order": 6.76,
            "note": "different env/action abstraction; paradigm reference, not identical metric"},
        "training_seed": seed,
        "eval_seeds": eval_seeds,
    }
    Path("logs").mkdir(exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    with open(f"logs/ma_idqn_seed{seed}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "eps", "ret", "cost", "deliv"])
        w.writeheader(); w.writerows(log)
    Path("weights").mkdir(exist_ok=True)
    torch.save({"model_state": net.state_dict(), "obs_dim": obs_dim,
                "n_actions": n_actions, "hidden": hidden,
                "training_seed": seed}, f"weights/ma_idqn_seed{seed}.pt")

    print("\n=== MULTI-AGENT IDQN (param sharing) — eval seeds", eval_seeds, "===")
    print(f"  {'random (MA)':22s} return={rnd[0]:8.1f} cost={rnd[1]:7.2f} delivered={rnd[2]:.1f}")
    print(f"  {'IDQN  (MA)':22s} return={idqn[0]:8.1f} cost={idqn[1]:7.2f} delivered={idqn[2]:.1f}")
    print(f"  {'ref centralized DQN':22s} cost=6.76 (own env, paradigm reference)")
    # check: a trained shared-param IDQN should beat the random baseline in-env
    if idqn[0] > rnd[0]:
        print("OK: IDQN beats random baseline in the same MA env")
    else:
        print("WARN: IDQN did not beat random — undertrained; raise --steps")


if __name__ == "__main__":
    main()
