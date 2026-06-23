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
