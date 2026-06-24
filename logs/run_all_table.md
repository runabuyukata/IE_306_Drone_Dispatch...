# run_all results table

Eval config = configs/eval_standard.yaml | seeds = [0] | primary metric = cost_per_order (lower better)

| policy | cost/order (mean±std) | success rate |
|---|---|---|
| random | 19.98 ± 0.00 | 0.681 |
| greedy_nearest | 4.50 ± 0.00 | 0.854 |
| milp_rolling | 5.95 ± 0.00 | 0.793 |
| DQN n=3 | 21.07 ± 0.00 | 0.472 |
| Double DQN n=3 | 8.70 ± 0.00 | 0.678 |
| Dueling DQN n=3 | 30.46 ± 0.00 | 0.407 |
