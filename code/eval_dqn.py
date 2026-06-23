from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from drone_dispatch_env.evaluate import evaluate

sys.path.append(str(Path(__file__).resolve().parent))
from dqn_agent import load_policy


def action_summary(policy, seeds):
    cfg = policy.cfg
    counts = {"assign": 0, "charge": 0, "noop": 0, "charge_open_steps": 0}
    for seed in seeds:
        env = DroneDispatchEnv(cfg)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            charge_indices = [cfg.charge_index(d) for d in range(cfg.n_drones)]
            if np.asarray(obs["action_mask"], dtype=bool)[charge_indices].any():
                counts["charge_open_steps"] += 1
            action = int(policy.act(obs))
            counts[cfg.decode(action)[0]] += 1
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/dqn_seed0.pt")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    policy = load_policy(args.weights)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    results = evaluate(policy, policy.cfg, seeds)
    out = dict(results["mean"])
    out["actions"] = action_summary(policy, seeds)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
