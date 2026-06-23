# Engineering Log

## Day 2 - charging diagnosis

The first DQN run stayed close to random performance: `cost_per_order` was 17.62 versus 18.78 for random and 4.57 for `greedy_nearest`. The main symptom was `charger_utilization = 0.0` with 8 depletion events.

Checks:
- Action space includes charging actions. With the standard config, assignment actions are 0-159, charge actions are 160-167, and no-op is 168.
- The environment mask opens charge actions for idle drones with `soc < 1.0`.
- In a diagnostic rollout over seeds 0, 1, and 2, random selected 49 charge actions and `greedy_nearest` selected 79, so the environment can charge.
- The trained DQN selected 0 charge actions while charge actions were available on all 99 decision steps.

Current conclusion: the issue is not an action-index or masking bug. The DQN policy is learning Q-values that prefer assignments over charging, likely because charging has delayed benefit while assignment gives more immediate delivery reward. Added action-type logging to training and evaluation so future runs show whether charging behavior improves.
