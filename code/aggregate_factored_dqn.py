"""Aggregate three factored Double-DQN training seeds and select a checkpoint."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    seeds = [0, 1, 2]
    per_seed = {}
    for seed in seeds:
        rows = list(csv.DictReader(
            open(f"logs/factored_double_dqn_seed{seed}.csv", newline="")))
        best = min(rows, key=lambda r: float(r["cost_per_order"]))
        per_seed[str(seed)] = {
            "best_step": int(best["step"]),
            "cost_per_order": float(best["cost_per_order"]),
            "success_rate": float(best["success_rate"]),
            "episode_return": float(best["episode_return"]),
        }
    costs = [per_seed[str(s)]["cost_per_order"] for s in seeds]
    selected_seed = min(seeds, key=lambda s: per_seed[str(s)]["cost_per_order"])
    shutil.copy2(
        f"weights/factored_double_dqn_seed{selected_seed}.pt",
        "weights/factored_double_dqn.pt",
    )
    output = {
        "per_training_seed": per_seed,
        "cost_per_order_mean": float(np.mean(costs)),
        "cost_per_order_std": float(np.std(costs)),
        "selected_seed": selected_seed,
        "eval_seeds": [0, 1, 2],
    }
    Path("logs/factored_double_dqn_results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
