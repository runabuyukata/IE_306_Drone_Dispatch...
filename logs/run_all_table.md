# run_all results table

Eval config = configs/eval_stress.yaml | seeds = [5, 6, 7] | primary metric = cost_per_order (lower better)

| policy | cost/order (mean±std) | success rate |
|---|---|---|
| random | 42.80 ± 6.80 | 0.377 |
| greedy_nearest | 12.02 ± 1.12 | 0.597 |
| milp_rolling | 12.23 ± 1.10 | 0.603 |
| DQN n=3 [config-incompatible] | nan ± nan | 0.000 |
| Double DQN n=3 [config-incompatible] | nan ± nan | 0.000 |
| Dueling DQN n=3 [config-incompatible] | nan ± nan | 0.000 |
| Factored Double DQN (demo warm-start) | 11.19 ± 0.56 | 0.583 |
| REINFORCE + GAE | 14.13 ± 2.02 | 0.546 |
| A2C | 19.53 ± 3.55 | 0.488 |
| Role C rollout depth=1 | 10.46 ± 3.05 | 0.625 |
| Offline CQL (joint) [config-incompatible] | nan ± nan | 0.000 |

## Role B continuous control

DDPG on DroneControl-v0: return = -126.42, success = 0.000, mean steps = 214.0.

**Joint multi-agent** (DroneDispatchMA-v0, separate env): cost_per_order = 31.28, delivered/ep = 74.0, return = -1187.3.
