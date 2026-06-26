# Offline dataset

`offline_pool.npz` is the mixed-quality static dataset used by the offline-RL
experiments.

- Transitions: 420,103
- Episodes: 3,969
- SHA-256: `8BDA730DF861AA9757DA6649C6C9C7652CE860154825D89591C31995C7B3633E`

Validate it with:

```bash
python check_offline_npz.py offline_pool.npz
```

Reproduce the three-training-seed BC, naive-DQN, and CQL comparison with:

```bash
python code/run_offline_seeds.py
```

The three source datasets were archived outside the submission repository after
pooling. The pooled file contains all transitions required to reproduce the
submitted offline experiments.
