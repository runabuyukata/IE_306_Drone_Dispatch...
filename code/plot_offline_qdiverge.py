"""Plot the offline-RL Q-value divergence: naive offline DQN over-estimates
(max Q blows up on OOD actions) while CQL stays bounded. Reads logs/offline_qstats.csv
(columns: seed, method, step, loss, mean_q, max_q) and writes
figures/offline_q_divergence.png. Uses seed 0 (representative; all seeds diverge).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "logs" / "offline_qstats.csv"
OUT = ROOT / "figures" / "offline_q_divergence.png"
SEED = 0
LABELS = {"naive": "naive offline DQN", "cql": "CQL", "bc": "behavioral cloning"}
COLORS = {"naive": "tab:red", "cql": "tab:green", "bc": "tab:gray"}

series = defaultdict(lambda: ([], []))  # method -> (steps, max_q)
with SRC.open() as f:
    for row in csv.DictReader(f):
        if int(row["seed"]) != SEED:
            continue
        m = row["method"]
        series[m][0].append(int(row["step"]))
        series[m][1].append(float(row["max_q"]))

fig, ax = plt.subplots(figsize=(7, 4.2))
for m in ("naive", "cql", "bc"):
    if m not in series:
        continue
    steps, q = series[m]
    ax.plot(steps, q, marker=".", color=COLORS[m], label=LABELS[m])
ax.set_xlabel("training step")
ax.set_ylabel("max Q over batch")
ax.set_title("Offline RL: naive DQN over-estimates (OOD actions); CQL stays bounded")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")

# sanity: naive must end far above CQL, else the divergence story is wrong
naive_end = series["naive"][1][-1]
cql_end = series["cql"][1][-1]
assert naive_end > 5 * cql_end, f"expected naive >> cql, got {naive_end:.0f} vs {cql_end:.0f}"
print(f"naive max_q final={naive_end:.0f}, cql final={cql_end:.0f} -> divergence confirmed")
