from __future__ import annotations

import argparse
import csv
import json
import os

from drone_dispatch_env import Config, evaluate
from role_c_rollout import RoleCRolloutPlanner


def save_per_seed_csv(path, seeds, results):
    rows = results["per_seed"]
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = ["seed"] + list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed, row in zip(seeds, rows):
            out = {"seed": seed}
            out.update(row)
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--weights-dir", default="weights")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    policy = RoleCRolloutPlanner.from_yaml(
        cfg, "configs/role_c_rollout.yaml", depth=args.depth)
    results = evaluate(policy, cfg, seeds)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    csv_path = os.path.join(args.log_dir, f"role_c_rollout_depth{args.depth}.csv")
    save_per_seed_csv(csv_path, seeds, results)

    settings_path = os.path.join(args.weights_dir, f"role_c_rollout_depth{args.depth}.json")
    with open(settings_path, "w") as f:
        json.dump(
            {
                "method": "RoleCRolloutPlanner",
                "role": "C",
                "depth": args.depth,
                "config": args.config,
                "seeds": seeds,
                "note": "No neural weights; this file stores the exact planning settings.",
            },
            f,
            indent=2,
        )

    print(json.dumps(results["mean"], indent=2))
    print(f"\nSaved per-seed log to: {csv_path}")
    print(f"Saved planner settings to: {settings_path}")


if __name__ == "__main__":
    main()
