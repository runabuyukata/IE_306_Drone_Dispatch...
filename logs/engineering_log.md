# Engineering Log

## Day 2 - charging diagnosis

The first DQN run stayed close to random performance: `cost_per_order` was 17.62 versus 18.78 for random and 4.57 for `greedy_nearest`. The main symptom was `charger_utilization = 0.0` with 8 depletion events.

Checks:
- Action space includes charging actions. With the standard config, assignment actions are 0-159, charge actions are 160-167, and no-op is 168.
- The environment mask opens charge actions for idle drones with `soc < 1.0`.
- In a diagnostic rollout over seeds 0, 1, and 2, random selected 49 charge actions and `greedy_nearest` selected 79, so the environment can charge.
- The trained DQN selected 0 charge actions while charge actions were available on all 99 decision steps.

The structural checks ruled out an action-index or masking bug. A second review of the training log showed the actual root cause: the run ended after only 5,664 environment steps with epsilon still at 0.73. Learning started at step 1,000 and ran every four steps, so the network received only about 1,166 gradient updates and never reached a low-exploration phase. Therefore, zero charge actions were evidence of an under-trained policy, not proof that delayed charging reward was the main cause.

Fix:
- Use a fixed 60,000-step training budget so DQN variants are compared with equal environment interaction.
- Decay epsilon to 0.05 over 40,000 steps, leaving the final 20,000 steps for low-exploration training.
- Keep gamma at 0.99 and the simulator reward unchanged. The existing depletion penalty is already large enough to support charging without reward shaping.
- Log action types, charge availability, and cumulative gradient updates for each episode.

Result after the fix:
- Training completed 60,000 environment steps, epsilon reached 0.05, and the network received 14,751 gradient updates.
- Charging was learned, but the policy overused charge and no-op actions. Evaluation selected 1,039 charge actions and 683 no-op actions over seeds 0, 1, and 2.
- Depletion fell from 8.0 to 3.67, but `cost_per_order` worsened to 29.39 because success rate fell to 0.385 and dropped orders increased to 83.67.
- The final 100 training episodes had a mean return of -693.57, so longer training alone did not produce a useful vanilla DQN policy.

Next test: train Double DQN with the same 60,000-step budget. The fixed budget keeps the comparison fair, and Double DQN directly tests whether max-Q overestimation is contributing to the charge/no-op policy collapse.

## Double DQN check

Double DQN was trained with the same seed, network, exploration schedule, and 60,000-step budget. It did not fix the collapse:
- `cost_per_order` increased to 37.51.
- Success rate remained low at 0.382.
- The policy selected 183 charge actions and 750 no-op actions.
- Depletion increased to 6.67.

Double DQN reduced the extreme charging seen in vanilla DQN but shifted the policy toward no-op, so max-Q overestimation is not the only cause. The next controlled test is the Dueling architecture, which separates state value from action advantages and may help in the 169-action space where many assignment actions are state-dependent.

## Stability check

The three costs were compared directly: random 18.78, DQN 29.39, and Double DQN 37.51. Since both learned policies were worse than random, the next test targeted training stability before adding Dueling.

The visualizer compared DQN and `greedy_nearest` on seed 0. DQN needed 799 recorded decision frames while greedy needed 159. Its reward stream contained -1530 from dropped orders and -150 from depletion. The replay also showed many lost-drone frames and long periods of charging or deferring while orders accumulated.

First change:
- Reduce learning rate from 0.0005 to 0.0001.
- Divide rewards stored in replay by 10 while keeping environment rewards and evaluation metrics unchanged.
- Keep relative reward magnitudes instead of clipping them.

This did not improve the policy. The scaled run produced `cost_per_order = 33.61`, success rate 0.365, 371 charge actions, and 714 no-op actions.

A review of the state vector found that positions, deadlines, and the grid were normalized, but time was still passed to the network as a raw value from 0 to 500. This feature could dominate the other inputs. Time normalization was added as an optional checkpoint setting so old checkpoints keep their original preprocessing.

With `time / T_max`, the same learning rate, reward scale, seed, and 60,000-step budget:
- `cost_per_order` improved to 22.33.
- Success rate improved to 0.494.
- Mean dropped orders fell to 69.33.
- Episode return improved to -364.50.

Time normalization fixed part of the shared problem, but the policy still did not beat random. The last 100 training episodes still averaged -570.61 and contained 12,899 no-op actions. The next test should target temporal credit assignment instead of another value-head architecture.

## Long-run benchmark

Before starting a long run, training now logs `episode_return` and the official `cost_per_order` formula side by side. The 20,000-step benchmark produced a Pearson correlation of -0.961 between them. Higher return therefore strongly corresponds to lower cost in the observed episodes, so the optimization target is not visibly misaligned with the grading metric.

The benchmark took 46.973 seconds, or about 425.8 environment steps per second. Estimated training times on the same machine:
- 600,000 steps: 23.5 minutes
- 3,000,000 steps: 117.4 minutes
- 6,000,000 steps: 234.9 minutes

Checkpoint evaluation over seeds 0, 1, and 2 improved from `cost_per_order = 60.83` at 10,000 steps to 37.03 at 20,000 steps. Both are still immature policies. A 6,000,000-step run was selected with checkpoints every 500,000 steps. The measured runtime estimate is about 3 hours 55 minutes.
