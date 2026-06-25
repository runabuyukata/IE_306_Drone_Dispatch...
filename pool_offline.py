"""Üç üyenin offline .npz'lerini tek mixed-quality havuza birleştirir.

Kullanim:  python pool_offline.py <out.npz> <in1.npz> <in2.npz> ...
Her girdi check_offline_npz.py formatina uymali (ayni anahtarlar/dtype).
"""
import sys
import numpy as np

PER_STEP = ["observations", "actions", "rewards", "next_observations",
            "terminals", "timeouts"]


def pool(out_path, in_paths):
    parts = {k: [] for k in PER_STEP + ["episode_returns"]}
    total = 0
    for p in in_paths:
        d = np.load(p)
        n = len(d["actions"])
        total += n
        for k in parts:
            parts[k].append(d[k])
        print(f"  + {p}: {n} transition, {len(d['episode_returns'])} episode")

    merged = {k: np.concatenate(v) for k, v in parts.items()}
    assert len(merged["actions"]) == total, "havuz toplami tutmuyor"
    np.savez_compressed(out_path, **merged)
    print(f"HAVUZ: {out_path} -> {total} transition, "
          f"{len(merged['episode_returns'])} episode")
    return total


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    pool(sys.argv[1], sys.argv[2:])
