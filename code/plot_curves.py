"""3-seed mean±std learning curves from the per-seed eval CSVs (Spec §4).

Reads `logs/*_eval.csv` (columns step, eval_cost_per_order, eval_episode_return),
aligns the three seeds on their common steps, and plots the mean with a ±std band
(not three overlaid lines). Writes PNGs to figures/.

    python code/plot_curves.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = "logs"
G600 = {
    "DQN n=3": [f"{L}/dqn_nstep_600k_eval.csv", f"{L}/dqn_nstep_600k_seed1_eval.csv", f"{L}/dqn_nstep_600k_seed2_eval.csv"],
    "Double DQN n=3": [f"{L}/double_dqn_nstep_600k_eval.csv", f"{L}/double_dqn_nstep_600k_seed1_eval.csv", f"{L}/double_dqn_nstep_600k_seed2_eval.csv"],
    "Dueling DQN n=3": [f"{L}/dueling_dqn_nstep_600k_eval.csv", f"{L}/dueling_dqn_nstep_600k_seed1_eval.csv", f"{L}/dueling_dqn_nstep_600k_seed2_eval.csv"],
}
DBL3M = [f"{L}/double_dqn_nstep_3m_eval.csv", f"{L}/double_dqn_nstep_3m_seed1_eval.csv", f"{L}/double_dqn_nstep_3m_seed2_eval.csv"]


def _read(path, col):
    out = {}
    for r in csv.DictReader(open(path)):
        out[int(float(r["step"]))] = float(r[col])
    return out


def band(paths, col):
    """Align seeds on common steps; return (steps, mean, std)."""
    series = [_read(p, col) for p in paths if Path(p).exists()]
    if len(series) < 2:
        return None
    common = sorted(set.intersection(*[set(s) for s in series]))
    M = np.array([[s[k] for k in common] for s in series])  # (n_seeds, n_steps)
    return np.array(common), M.mean(0), M.std(0)


def plot_methods(groups, col, ylabel, title, out, ymax=None):
    plt.figure(figsize=(7, 4.2))
    for name, paths in groups.items():
        b = band(paths, col)
        if b is None:
            continue
        x, m, s = b
        plt.plot(x / 1000, m, label=name, lw=1.8)
        plt.fill_between(x / 1000, m - s, m + s, alpha=0.18)
    plt.xlabel("training steps (k = 1000)"); plt.ylabel(ylabel); plt.title(title)
    if ymax:
        plt.ylim(top=ymax)
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close()
    print("wrote", out)


def main():
    n_seeds = sum(Path(p).exists() for p in G600["Double DQN n=3"])
    print(f"Double DQN 600k seeds found: {n_seeds}")
    plot_methods(G600, "eval_cost_per_order", "cost_per_order (lower better)",
                 "600k — 3-seed mean ± std (cost)", "figures/dqn_family_600k_cost.png", ymax=60)
    plot_methods(G600, "eval_episode_return", "episode_return",
                 "600k — 3-seed mean ± std (return)", "figures/dqn_family_600k_return.png")
    plot_methods({"Double DQN n=3 (3M)": DBL3M}, "eval_cost_per_order",
                 "cost_per_order (lower better)",
                 "Double DQN 3M — 3-seed mean ± std (good band then divergence)",
                 "figures/double_dqn_3m_cost.png", ymax=90)


if __name__ == "__main__":
    main()
