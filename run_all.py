"""Reproduce the Role-A results table: load saved DQN policies and compare them
to the shipped baselines (random, greedy_nearest, milp_rolling) on a config.

Single command (config + seeds overridable so the grader can swap in held-out):
    python run_all.py --config configs/eval_standard.yaml --seeds 0,1,2

Prints cost_per_order (mean +/- std over seeds, the primary metric) and success
rate, and writes logs/run_all_table.md. The objective bar is greedy_nearest.
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

# One file per learned method. Double DQN uses its best (validation-selected) 1M
# checkpoint; see logs/engineering_log.md for why (it diverges after ~2.5M).
METHODS = {
    "DQN n=3":         "weights/dqn_nstep_600k.pt",
    "Double DQN n=3":  "weights/double_dqn_nstep_3m_step_1000000.pt",
    "Dueling DQN n=3": "weights/dueling_dqn_nstep_600k.pt",
}
BASELINES = ["random", "greedy_nearest", "milp_rolling"]


def stats(res: dict):
    cps = [m["cost_per_order"] for m in res["per_seed"]]
    sr = [m["success_rate"] for m in res["per_seed"]]
    return float(np.mean(cps)), float(np.std(cps)), float(np.mean(sr))


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
        rows.append((name, *stats(evaluate(load_policy(wp), cfg, seeds))))

    header = f"Eval config = {args.config} | seeds = {seeds} | primary metric = cost_per_order (lower better)"
    lines = [header, "", f"{'policy':24} {'cost/order (mean+/-std)':26} {'success':>8}", "-" * 60]
    md = ["# run_all results table", "", header, "",
          "| policy | cost/order (mean±std) | success rate |", "|---|---|---|"]
    for name, m, s, sr in rows:
        lines.append(f"{name:24} {m:8.2f} +/- {s:5.2f}        {sr:8.3f}")
        md.append(f"| {name} | {m:.2f} ± {s:.2f} | {sr:.3f} |")
    out = "\n".join(lines)
    print(out)
    Path("logs/run_all_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
