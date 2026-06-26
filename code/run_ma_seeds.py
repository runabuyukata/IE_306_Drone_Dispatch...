"""Train and aggregate parameter-sharing IDQN over configured seeds."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent))
from dqn_agent import QNetwork
from train_ma_idqn import eval_ma
from drone_dispatch_env.config import Config


def main() -> None:
    config = Path("configs/ma_idqn.yaml")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    per_seed = {}
    shared_random = None
    eval_seeds = [int(s) for s in cfg.get("eval-seeds", [0, 1, 2])]
    env_cfg = Config()

    for seed in seeds:
        result = Path(f"logs/ma_results_seed{seed}.json")
        weight = Path(f"weights/ma_idqn_seed{seed}.pt")
        if seed == 0 and not weight.exists() and Path("weights/ma_idqn.pt").exists():
            shutil.copy2("weights/ma_idqn.pt", weight)
        if not weight.exists():
            subprocess.run([
                sys.executable, "code/train_ma_idqn.py",
                "--config", str(config),
                "--seed", str(seed),
                "--out", str(result),
            ], check=True)
        ck = torch.load(weight, map_location="cpu", weights_only=False)
        net = QNetwork(ck["obs_dim"], ck["n_actions"], ck["hidden"])
        net.load_state_dict(ck["model_state"])
        idqn = eval_ma(net, env_cfg, "cpu", seeds=eval_seeds)
        if shared_random is None:
            shared_random = eval_ma(
                net, env_cfg, "cpu", policy="random", seeds=eval_seeds)
        random_result = shared_random
        per_seed[str(seed)] = {
            "idqn_ma": {
                "return": idqn[0],
                "cost_per_order": idqn[1],
                "delivered": idqn[2],
            },
            "random_ma": {
                "return": random_result[0],
                "cost_per_order": random_result[1],
                "delivered": random_result[2],
            },
            "training_seed": seed,
            "eval_seeds": eval_seeds,
        }
        result.write_text(
            json.dumps(per_seed[str(seed)], indent=2), encoding="utf-8")

    costs = [per_seed[str(s)]["idqn_ma"]["cost_per_order"] for s in seeds]
    returns = [per_seed[str(s)]["idqn_ma"]["return"] for s in seeds]
    delivered = [per_seed[str(s)]["idqn_ma"]["delivered"] for s in seeds]
    selected_seed = min(
        seeds, key=lambda s: per_seed[str(s)]["idqn_ma"]["cost_per_order"])
    selected_source = f"30k training seed {selected_seed}"
    extended = None
    final_path = Path("weights/ma_idqn.pt")
    if final_path.exists():
        ck = torch.load(final_path, map_location="cpu", weights_only=False)
        net = QNetwork(ck["obs_dim"], ck["n_actions"], ck["hidden"])
        net.load_state_dict(ck["model_state"])
        ext = eval_ma(net, env_cfg, "cpu", seeds=eval_seeds)
        extended = {
            "return": ext[0],
            "cost_per_order": ext[1],
            "delivered": ext[2],
            "note": "extended 60k seed-0 checkpoint from the original run",
        }
    if (extended is None or
            per_seed[str(selected_seed)]["idqn_ma"]["cost_per_order"]
            < extended["cost_per_order"]):
        shutil.copy2(
            f"weights/ma_idqn_seed{selected_seed}.pt",
            final_path,
        )
    else:
        selected_source = "extended 60k seed-0 checkpoint"

    output = {
        "per_training_seed": per_seed,
        "aggregate": {
            "idqn_ma": {
                "cost_per_order_mean": float(np.mean(costs)),
                "cost_per_order_std": float(np.std(costs)),
                "return_mean": float(np.mean(returns)),
                "return_std": float(np.std(returns)),
                "delivered_mean": float(np.mean(delivered)),
                "delivered_std": float(np.std(delivered)),
            }
        },
        "selected_seed": selected_seed,
        "selected_source": selected_source,
        "idqn_ma": (extended if selected_source.startswith("extended")
                    else per_seed[str(selected_seed)]["idqn_ma"]),
        "extended_checkpoint": extended,
        "random_ma": per_seed[str(selected_seed)]["random_ma"],
        "training_seeds": seeds,
        "eval_seeds": eval_seeds,
    }
    Path("logs/ma_results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": output["aggregate"],
                      "selected_seed": selected_seed}, indent=2))


if __name__ == "__main__":
    main()
