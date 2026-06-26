# run_all results table

Eval config = configs/eval_standard.yaml | seeds = [0, 1, 2] | primary metric = cost_per_order (lower better)

| policy | cost/order (mean±std) | success rate |
|---|---|---|
| random | 18.78 ± 1.27 | 0.653 |
| greedy_nearest | 4.57 ± 0.85 | 0.855 |
| milp_rolling | 4.72 ± 1.38 | 0.836 |
| DQN n=3 | 20.67 ± 6.57 | 0.495 |
| Double DQN n=3 | 6.76 ± 1.80 | 0.749 |
| Dueling DQN n=3 | 26.07 ± 3.49 | 0.428 |
| Factored Double DQN (demo warm-start) | 1.72 ± 0.05 | 0.914 |
| REINFORCE + GAE | 2.57 ± 0.86 | 0.903 |
| A2C | 1.09 ± 0.43 | 0.976 |
| Role C rollout depth=1 | 2.92 ± 0.40 | 0.881 |
| Offline CQL (joint) | 5.72 ± 1.38 | 0.754 |

## Role B continuous control

go_straight baseline: return = 26.37, success = 1.000, mean steps = 11.7.
DDPG on DroneControl-v0: return = 25.91, success = 1.000, mean steps = 18.0.

**Joint multi-agent** (DroneDispatchMA-v0, separate env): cost_per_order = 6.65, delivered/ep = 100.7, return = 793.8.
