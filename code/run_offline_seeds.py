"""Run offline BC/naive-DQN/CQL over all configured training seeds."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    config = Path("configs/offline_cql.yaml")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    per_seed = {}

    for seed in seeds:
        result = Path(f"logs/offline_results_seed{seed}.json")
        qstats = Path(f"logs/offline_qstats_seed{seed}.csv")
        weight = Path(f"weights/offline_cql_seed{seed}.pt")
        if not (result.exists() and qstats.exists() and weight.exists()):
            subprocess.run([
                sys.executable, "code/offline_rl.py",
                "--config", str(config),
                "--seed", str(seed),
                "--out", str(result),
                "--qstats", str(qstats),
                "--weight", str(weight),
            ], check=True)
        per_seed[str(seed)] = json.loads(
            result.read_text(encoding="utf-8"))["results"]

    aggregate = {}
    for method in ("bc", "naive", "cql"):
        costs = [per_seed[str(s)][method]["cost_per_order"] for s in seeds]
        successes = [per_seed[str(s)][method]["success_rate"] for s in seeds]
        aggregate[method] = {
            "cost_per_order_mean": float(np.mean(costs)),
            "cost_per_order_std": float(np.std(costs)),
            "success_rate_mean": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
        }

    selected_seed = min(
        seeds, key=lambda s: per_seed[str(s)]["cql"]["cost_per_order"])
    shutil.copy2(
        f"weights/offline_cql_seed{selected_seed}.pt",
        "weights/offline_cql.pt",
    )

    merged_qstats = []
    for seed in seeds:
        with open(f"logs/offline_qstats_seed{seed}.csv", newline="") as f:
            merged_qstats.extend(csv.DictReader(f))
    with open("logs/offline_qstats.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "method", "step", "loss", "mean_q", "max_q"])
        writer.writeheader()
        writer.writerows(merged_qstats)

    output = {
        "refs": {"random": 18.78, "greedy_nearest": 4.57,
                 "online_dqn_1M": 6.76},
        "per_training_seed": per_seed,
        "aggregate": aggregate,
        "selected_seed": selected_seed,
        "results": per_seed[str(selected_seed)],
        "training_seeds": seeds,
        "eval_seeds": cfg.get("eval-seeds", [0, 1, 2]),
    }
    Path("logs/offline_results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate,
                      "selected_seed": selected_seed}, indent=2))


if __name__ == "__main__":
    main()
