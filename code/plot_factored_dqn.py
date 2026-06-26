"""Plot the three-seed factored Double-DQN validation curve."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    series = []
    for seed in (0, 1, 2):
        rows = list(csv.DictReader(
            open(f"logs/factored_double_dqn_seed{seed}.csv", newline="")))
        series.append((
            np.asarray([int(r["step"]) for r in rows]),
            np.asarray([float(r["cost_per_order"]) for r in rows]),
        ))
    steps = series[0][0]
    values = np.stack([v for _, v in series])
    mean, std = values.mean(0), values.std(0)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(steps, mean, marker="o", label="Factored Double DQN")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25,
                    label="±1 std over 3 training seeds")
    ax.axhline(4.57, color="tab:red", linestyle="--",
               label="greedy_nearest (4.57)")
    ax.set_xlabel("environment steps (0 = demonstration warm-start)")
    ax.set_ylabel("validation cost per delivered order")
    ax.set_title("Role A: factored Double DQN, 3-seed mean ± std")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/factored_double_dqn_curve.png", dpi=160)


if __name__ == "__main__":
    main()
