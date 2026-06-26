"""Reproduce the complete team results table from saved policies.

Single command (config + seeds overridable so the grader can swap in held-out):
    python run_all.py --config configs/eval_standard.yaml --seeds 0,1,2

Prints cost_per_order (mean +/- std over seeds, the primary metric) and success
rate, and writes logs/run_all_table.md. Continuous-control DDPG and the
multi-agent policy are reported separately because they use different envs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "code"))

from drone_dispatch_env.baselines import make_baseline
from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate
from dqn_agent import load_policy
from train_factored_dqn import load_policy as load_factored_dqn
from role_b.adapters import load_dispatch_agent
from role_c.role_c_rollout import RoleCRolloutPlanner

# One file per learned method. Double DQN uses its best (validation-selected) 1M
# checkpoint; see logs/engineering_log.md for why (it diverges after ~2.5M).
METHODS = {
    "DQN n=3":         "weights/dqn_nstep_600k.pt",
    "Double DQN n=3":  "weights/double_dqn_nstep_3m_step_1000000.pt",
    "Dueling DQN n=3": "weights/dueling_dqn_nstep_600k.pt",
}
ROLE_B_METHODS = {
    "REINFORCE + GAE": "weights/reinforce.pt",
    "A2C": "weights/a2c.pt",
}
BASELINES = ["random", "greedy_nearest", "milp_rolling"]
# Joint methods (separate weights). CQL runs on the same centralized env so it
# joins the main table; MA runs on DroneDispatchMA-v0 and is reported separately.
CQL_WEIGHTS = "weights/offline_cql.pt"
MA_WEIGHTS = "weights/ma_idqn.pt"


def stats(res: dict):
    cps = [m["cost_per_order"] for m in res["per_seed"]]
    sr = [m["success_rate"] for m in res["per_seed"]]
    return float(np.mean(cps)), float(np.std(cps)), float(np.mean(sr))


def load_cql_policy(path, device="cpu"):
    """Rebuild the offline-CQL eval wrapper from its saved weights + norm stats."""
    import torch
    from offline_rl import _Wrapped
    from dqn_agent import QNetwork
    ck = torch.load(path, map_location=device, weights_only=False)  # our own file (has numpy stats)
    net = QNetwork(ck["obs_dim"], ck["n_actions"], ck["hidden"])
    net.load_state_dict(ck["model_state"])
    return _Wrapped(net, np.asarray(ck["mean"]), np.asarray(ck["std"]), device)


def eval_ma_policy(path, cfg, seeds, device="cpu"):
    """Eval the shared-param IDQN on the MA env; returns (return, cost, delivered)."""
    import torch
    from train_ma_idqn import eval_ma
    from dqn_agent import QNetwork
    ck = torch.load(path, map_location=device, weights_only=False)
    net = QNetwork(ck["obs_dim"], ck["n_actions"], ck["hidden"])
    net.load_state_dict(ck["model_state"])
    return eval_ma(net, cfg, device, policy="greedy", seeds=seeds)


def eval_ddpg_policy(path, cfg, seeds, device="cpu"):
    import torch
    from role_b.ddpg import eval_ddpg
    from role_b.networks import DDPGActor
    ck = torch.load(path, map_location=device, weights_only=False)
    actor = DDPGActor(ck["obs_dim"], ck["hidden"])
    actor.load_state_dict(ck["state_dict"])
    actor.eval()
    return eval_ddpg(actor, cfg, seeds, torch.device(device))


def eval_go_straight(cfg, seeds):
    """Go-straight baseline on DroneControl-v0 (the DDPG comparison bar)."""
    from drone_dispatch_env.env_control import DroneControlEnv
    from role_b.adapters import GoStraight
    from role_b.ddpg import _clip_action
    env = DroneControlEnv(cfg)
    pol = GoStraight(cfg)
    rets, succ, steps = [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done, ret, n, last_term = False, 0.0, 0, False
        while not done:
            obs, r, term, trunc, _ = env.step(_clip_action(pol.act(obs)))
            ret += r
            n += 1
            last_term = term
            done = term or trunc
        rets.append(ret)
        succ.append(1.0 if (last_term and obs[2] > 0.0) else 0.0)
        steps.append(n)
    return {"return": float(np.mean(rets)), "success_rate": float(np.mean(succ)),
            "mean_steps": float(np.mean(steps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_standard.yaml")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    seeds = [int(s) for s in args.seeds.split(",") if s]

    rows = []
    for name in BASELINES:
        rows.append((name, *stats(evaluate(make_baseline(name, cfg), cfg, seeds))))
    for name, wp in METHODS.items():
        if not Path(wp).exists():
            rows.append((f"{name} [weights missing]", float("nan"), 0.0, 0.0))
            continue
        try:
            rows.append((name, *stats(evaluate(load_policy(wp), cfg, seeds))))
        except (RuntimeError, ValueError) as exc:
            rows.append((f"{name} [config-incompatible]", float("nan"),
                         float("nan"), 0.0))
    factored_path = Path("weights/factored_double_dqn.pt")
    if factored_path.exists():
        rows.append(("Factored Double DQN (demo warm-start)",
                     *stats(evaluate(
                         load_factored_dqn(
                             factored_path, cfg_override=cfg), cfg, seeds))))
    for name, wp in ROLE_B_METHODS.items():
        if not Path(wp).exists():
            rows.append((f"{name} [weights missing]", float("nan"), 0.0, 0.0))
            continue
        rows.append((name, *stats(evaluate(load_dispatch_agent(wp, cfg), cfg, seeds))))
    role_c = RoleCRolloutPlanner.from_yaml(
        cfg, "configs/role_c_rollout.yaml", depth=1)
    rows.append(("Role C rollout depth=1",
                 *stats(evaluate(role_c, cfg, seeds))))
    if Path(CQL_WEIGHTS).exists():  # joint offline method, same centralized env
        try:
            rows.append(("Offline CQL (joint)",
                         *stats(evaluate(
                             load_cql_policy(CQL_WEIGHTS), cfg, seeds))))
        except (RuntimeError, ValueError):
            rows.append(("Offline CQL (joint) [config-incompatible]",
                         float("nan"), float("nan"), 0.0))

    header = f"Eval config = {args.config} | seeds = {seeds} | primary metric = cost_per_order (lower better)"
    lines = [header, "", f"{'policy':24} {'cost/order (mean+/-std)':26} {'success':>8}", "-" * 60]
    md = ["# run_all results table", "", header, "",
          "| policy | cost/order (mean±std) | success rate |", "|---|---|---|"]
    for name, m, s, sr in rows:
        lines.append(f"{name:24} {m:8.2f} +/- {s:5.2f}        {sr:8.3f}")
        md.append(f"| {name} | {m:.2f} ± {s:.2f} | {sr:.3f} |")

    ddpg_path = Path("weights/ddpg.pt")
    if ddpg_path.exists():
        gs = eval_go_straight(cfg, seeds)
        ddpg = eval_ddpg_policy(ddpg_path, cfg, seeds)
        gs_line = (f"go_straight (baseline):  return={gs['return']:.2f} "
                   f"success={gs['success_rate']:.3f} "
                   f"mean_steps={gs['mean_steps']:.1f}")
        ddpg_line = (f"DDPG (DroneControl-v0):  return={ddpg['return']:.2f} "
                     f"success={ddpg['success_rate']:.3f} "
                     f"mean_steps={ddpg['mean_steps']:.1f}")
        lines += ["", "-- Role B continuous control (separate env) --",
                  gs_line, ddpg_line]
        md += ["", "## Role B continuous control", "",
               f"go_straight baseline: return = {gs['return']:.2f}, "
               f"success = {gs['success_rate']:.3f}, "
               f"mean steps = {gs['mean_steps']:.1f}.",
               f"DDPG on DroneControl-v0: return = {ddpg['return']:.2f}, "
               f"success = {ddpg['success_rate']:.3f}, "
               f"mean steps = {ddpg['mean_steps']:.1f}."]
    # Multi-agent: different env (DroneDispatchMA-v0), reported separately. Its
    # cost_per_order is reconstructed from the reward stream (see train_ma_idqn).
    if Path(MA_WEIGHTS).exists():
        ret, cost, deliv = eval_ma_policy(MA_WEIGHTS, cfg, seeds)
        ma_line = (f"Multi-agent IDQN (DroneDispatchMA-v0): cost_per_order={cost:.2f} "
                   f"delivered/ep={deliv:.1f} return={ret:.1f}")
        lines += ["", "-- joint multi-agent (separate env) --", ma_line]
        md += ["", f"**Joint multi-agent** (DroneDispatchMA-v0, separate env): "
               f"cost_per_order = {cost:.2f}, delivered/ep = {deliv:.1f}, return = {ret:.1f}."]

    out = "\n".join(lines)
    print(out)
    Path("logs/run_all_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
