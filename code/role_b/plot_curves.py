"""Plot Role B learning curves (mean +/- std over >=3 seeds) and the GAE-lambda
ablation. Reads logs/*.csv written by the trainers; writes PNGs to figures/.

    python code/role_b/plot_curves.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from drone_dispatch_env.config import Config                    # noqa: E402
from drone_dispatch_env.evaluate import evaluate                # noqa: E402
from drone_dispatch_env.baselines import make_baseline          # noqa: E402

FIG = "figures"
VAL_SEEDS = [200, 201, 202]


def _read(path):
    """CSV -> dict of float arrays. The csv module avoids numpy's structured-array
    name mangling of fields like 'return' (a Python keyword)."""
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return {k: np.array([float(r[k]) for r in rows], dtype=float) for k in rows[0]}
    except Exception:
        return None


def _stack(method, metric):
    """Return (steps, mean, std) over all seed CSVs for `method`."""
    files = sorted(glob.glob(f"logs/{method}_seed*.csv"))
    series, steps = [], None
    for f in files:
        d = _read(f)
        if d is None or metric not in d or len(d["step"]) == 0:
            continue
        series.append(d[metric])
        steps = d["step"]
    if not series:
        return None, None, None
    n = min(len(s) for s in series)
    arr = np.stack([s[:n] for s in series], axis=0)
    return steps[:n], arr.mean(0), arr.std(0)


def _greedy_ref(metric):
    cfg = Config()
    m = evaluate(make_baseline("greedy_nearest", cfg), cfg, VAL_SEEDS)["mean"]
    return m.get(metric)


def plot_dispatch():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in [(axes[0], "cost_per_order", "cost_per_order (lower better)"),
                              (axes[1], "n_delivered", "orders delivered / episode")]:
        for method, color in [("reinforce", "tab:blue"), ("a2c", "tab:orange")]:
            steps, mean, std = _stack(method, metric)
            if steps is None:
                continue
            ax.plot(steps, mean, color=color, label=method)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ref = _greedy_ref(metric)
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=1, label="greedy_nearest")
        ax.set_xlabel("env steps")
        ax.set_title(title)
        ax.legend()
    if axes[0].has_data():
        axes[0].set_ylim(bottom=0, top=min(20, axes[0].get_ylim()[1]))
    fig.suptitle("Role B dispatch learning curves (mean +/- std, 3 seeds)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/dispatch_curves.png", dpi=120)
    print(f"wrote {FIG}/dispatch_curves.png")


def plot_ddpg():
    steps, mean, std = _stack("ddpg", "return")
    if steps is None:
        print("no ddpg logs yet")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, mean, color="tab:green", label="DDPG")
    ax.fill_between(steps, mean - std, mean + std, color="tab:green", alpha=0.2)
    ax.set_xlabel("env steps")
    ax.set_ylabel("eval return")
    ax.set_title("DDPG on DroneControl-v0 (mean +/- std, 3 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG}/ddpg_curve.png", dpi=120)
    print(f"wrote {FIG}/ddpg_curve.png")


def plot_ablation():
    files = sorted(glob.glob("logs/ablation_gae_lam*_seed*.csv"))
    if not files:
        print("no ablation logs yet")
        return
    by_lambda: dict[float, list[float]] = {}
    for f in files:
        tag = os.path.basename(f).split("_seed")[0].replace("ablation_gae_lam", "")
        lam = float(tag.replace("p", "."))
        d = _read(f)
        if d is None or "cost_per_order" not in d or len(d["cost_per_order"]) == 0:
            continue
        by_lambda.setdefault(lam, []).append(float(np.min(d["cost_per_order"])))
    if not by_lambda:
        return
    lams = sorted(by_lambda)
    means = [np.mean(by_lambda[l]) for l in lams]
    stds = [np.std(by_lambda[l]) for l in lams]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(lams, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("GAE lambda")
    ax.set_ylabel("best val cost_per_order")
    ax.set_title("GAE-lambda ablation on A2C (mean +/- std, 3 seeds)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/ablation_gae.png", dpi=120)
    print(f"wrote {FIG}/ablation_gae.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    plot_dispatch()
    plot_ddpg()
    plot_ablation()


if __name__ == "__main__":
    main()
