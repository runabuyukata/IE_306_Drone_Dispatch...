# IE 306 Term Project — Reinforcement Learning for City-Scale Drone Delivery

**Team:** Sezen Balkan (Role A), Ozan Karhan (Role B), Tuba Nur Büyükata
(Role C). Offline RL and multi-agent control are joint components.

The primary metric is mean `cost_per_order`; lower is better. Unless stated
otherwise, the final table uses `configs/eval_standard.yaml` and environment
seeds 0, 1, and 2. It is reproduced with:

```bash
python run_all.py --config configs/eval_standard.yaml --seeds 0,1,2
```

## 1. Baselines

| Policy | cost/order, mean ± std | success rate |
|---|---:|---:|
| random | 18.78 ± 1.27 | 0.653 |
| **greedy_nearest** | **4.57 ± 0.85** | 0.855 |
| milp_rolling | 4.72 ± 1.38 | 0.836 |

`greedy_nearest` is the required bar. MILP is slightly worse on these seeds
because its single-epoch Manhattan-distance matching can make poor assignments
around no-fly geometry.

## 2. Role A — Value-based methods (Sezen Balkan)

### 2.1 Methods

The first implementation used a flat observation vector and a replay-buffer
DQN. Invalid actions were masked both during action selection and Bellman
backup. We then introduced Double DQN to separate next-action selection from
target evaluation, Dueling DQN to separate state value and action advantage,
and three-step returns to improve delayed credit assignment.

The flat network remained unstable and did not beat greedy. The final Role-A
method is therefore a **factored Double DQN**. Shared heads score every
drone-order assignment and every charging action. Features include routed
pickup/delivery distance, deadline feasibility, battery feasibility, current
SoC, and global demand. The network is warm-started by supervised imitation of
the Role-C depth-1 planner, then updated using replay, a target network, and
Double-DQN TD targets. The warm start is disclosed because presenting it as a
from-scratch DQN would be misleading.

### 2.2 Results

| Value-based method (3 training seeds) | cost/order | success |
|---|---:|---:|
| DQN n=3 | 20.67 ± 6.57 | 0.495 |
| Double DQN n=3, flat MLP | 6.76 ± 1.80 | 0.749 |
| Dueling DQN n=3 | 26.07 ± 3.49 | 0.428 |
| Factored, warm-start only (0 TD steps) | 3.28 ± 0.16 | 0.872 |
| **Factored Double DQN, best checkpoint** | **2.29 ± 0.41** | **0.883** |

The TD learning is not cosmetic on top of the warm start. With 0 TD steps the
network is a pure imitation of the Role-C depth-1 planner; it already beats
greedy at **3.28 ± 0.16**, but Double-DQN training lowers cost to
**2.29 ± 0.41** at 5,000 steps — a consistent ~30% reduction in *every* seed
(3.26→1.72, 3.09→2.65, 3.49→2.50), not a single lucky run. The value updates
add real improvement over the demonstrator rather than replaying it.

The weakness is stability, not contribution. Pushing all three seeds to 10,000
steps collapses them to ~32 mean cost (26.2, 33.1, 36.3) with negative return.
Training guards against this by saving only on validation improvement
(`save_policy` fires when `cost_per_order` drops), so the saved file is the
5,000-step checkpoint by construction — not a manual pick — and re-running
training reproduces the same selection. The single saved model (training seed 0)
scores **1.72 ± 0.05** over eval seeds 0–2, and `run_all.py` loads exactly that
file.

![Factored Double DQN learning curve](figures/factored_double_dqn_curve.png)

### 2.3 Required ablation: target network (on the shipped factored method)

The required ablation is run on the factored Double DQN we actually ship, not on
a discarded flat model. Both runs use the same seed and the identical Role-C
warm start (step-0 cost = 3.26); they differ only in the target-network delay —
ON copies the online weights to the target every 1,000 steps, OFF copies them
every step (no delayed target). Config: `configs/factored_double_dqn.yaml` vs
`configs/factored_double_dqn_notarget.yaml`.

| Setting | cost @ 5k steps | cost @ 10k steps |
|---|---:|---:|
| **Target network ON** (shipped) | **1.72** | 26.23 |
| Target network OFF | 26.57 | 24.65 |

The effect is decisive. With the delayed target ON, TD training improves the
warm start from 3.26 down to **1.72** (beating greedy) before its late
divergence, so a validation checkpoint is recoverable. With it OFF, training
**immediately destroys** the warm start — cost collapses to 26.57 by 5k steps
and never recovers, and the factored method produces no greedy-beating
checkpoint at all. The delayed target is therefore not a minor stabiliser but a
precondition for this method to learn. This isolates bootstrapping instability
as the failure mode, and is consistent with the same ablation on the earlier
flat Double-DQN (ON also beat OFF there, best 10.19 vs 13.18).

### 2.4 What broke

The original flat representation mixed a 20×20 grid with entity rows and a
large action head. It learned passive charging/no-op behavior and later
diverged. Increasing training to millions of steps did not fix the
representation. Factored action scoring supplied the relational information
needed for assignment. Even then, TD learning could destroy a good warm start,
which is why validation checkpoints and three training seeds are reported.

On individual ownership: the warm start borrows the Role-C planner as a
demonstrator, so the warm-start number (3.28) is not independent of Role C. The
individually-owned value-based contribution is the rest — the factored action
architecture, the **3.28 → 2.29 improvement the Double-DQN updates produce in
every seed**, and the instability-and-checkpointing diagnosis. Presenting the
warm start honestly is deliberate; the value learning is what is defended here.

### 2.5 Held-out robustness and why the flat architecture fails

To check that the factored policy is not tuned to eval seeds 0–2, we ran the
saved model on held-out seeds 5–7 it never saw during development. It scores
**1.69 ± 0.67** versus `greedy_nearest` at **3.19 ± 0.51** — essentially the
same as on the tuned seeds (1.72), so it generalizes rather than overfits. On
the held-out **stress config** (24×24 grid, `k_max=28`, higher demand, tighter
deadlines) the same weights still beat greedy, **11.19 ± 0.56** vs **12.02 ±
1.12**, because the factored network's parameters do not depend on grid size or
`k_max`. The flat DQN/Double/Dueling models, by contrast, are **config-
incompatible** there: their hard-wired flat observation and `N·K_max+N+1`
action head cannot even load on a different grid, so they fail the
dimension-robustness test outright.

This failure mode is not specific to our environment. A review of the
literature shows it is a known limitation of value-based methods: as the number
of discrete actions grows, a flat Q-head loses the ability to generalize across
actions and its complexity scales linearly with the action count (Dulac-Arnold
et al., 2015). The standard remedy is exactly what fixed our case — structured /
factored action representations that exploit the compositional structure of the
action space, which have been shown to improve substantially over flat
baselines (Sharma et al., 2017). Our flat-vs-factored result is a concrete
instance of this published pattern.

## 3. Role B — Policy-based methods (Ozan Karhan)

### 3.1 Methods

REINFORCE uses likelihood-ratio policy gradients with a learned value baseline
and GAE. A2C uses the same factored actor-critic but collects fixed-length
rollouts and bootstraps from the critic, reducing variance. Invalid dispatch
actions are masked. DDPG uses a deterministic continuous actor, Q critic,
replay buffer, target networks, Polyak averaging, and OU exploration on
`DroneControl-v0`.

### 3.2 Dispatch results

| Method | cost/order, mean ± std | success |
|---|---:|---:|
| REINFORCE + GAE | 2.72 ± 0.73 | 0.878 |
| **A2C** | **1.09 ± 0.43** | **0.976** |

A2C is the most reliable learned dispatcher. Its three saved training models
also beat greedy on seeds 0–4, whereas REINFORCE is highly seed-sensitive.
The difference is consistent with the lower-variance bootstrapped updates used
by A2C.

![Role B dispatch curves](figures/dispatch_curves.png)

### 3.3 Required ablation: GAE λ

The A2C sweep used λ ∈ {0, 0.9, 0.95, 0.99, 1.0}, two training seeds, and a
40k-step budget.

| λ | mean best validation cost |
|---:|---:|
| 0.0 | 13.82 |
| 0.9 | 0.77 |
| **0.95** | **0.76** |
| 0.99 | 0.89 |
| 1.0 | 0.90 |

One-step advantages are too myopic; λ around 0.9–0.95 balances bias and
variance. Although the main methods have three training seeds, this reduced
ablation has two seeds and should be interpreted as a controlled design check.

![GAE ablation](figures/ablation_gae.png)

### 3.4 DDPG result and failure

On seeds 0–2, the current go-straight controller obtains mean return **26.37**,
success **1.00**, and 11.7 steps. The selected DDPG checkpoint obtains return
**−69.66**, success **0.00**, and 234.3 steps. DDPG therefore does **not** beat
its controller baseline. It learned conservative movement that avoided some
large penalties but failed to reach the target. This remains an unresolved
Role-B weakness rather than a successful result.

The main engineering fix for A2C was reward scaling and Huber value loss:
unscaled value targets produced losses around 25,000 and suppressed useful
policy gradients.

## 4. Role C — Planning (Tuba Nur Büyükata)

Role C implements a deterministic decision-time planner. It evaluates every
valid assignment using routed pickup distance, full delivery distance,
remaining deadline, battery shortfall, order age, and charging readiness.
All coefficients are stored in `configs/role_c_rollout.yaml`. Routed distances
are computed by student-side BFS using only the frozen observation grid.

The depth ablation is:

- depth 0: nearest-pickup rule with a battery guard;
- depth 1: adds delivery distance, deadline risk, and battery feasibility;
- depth 2: adds post-delivery distance to a charger.

| Method | cost/order | success | on-time | delivered |
|---|---:|---:|---:|---:|
| depth 0 | 4.570 | 0.855 | 0.903 | 118.3 |
| **depth 1** | **2.923** | **0.881** | **0.982** | **126.3** |
| depth 2 | 3.331 | 0.869 | 0.982 | 124.3 |

Depth 1 is selected. Depth 2 is more conservative and sacrifices good current
assignments for future charger proximity. The implementation is a shallow
rollout-style scoring planner, not a full cloned-state MCTS tree; this is a
deliberate limitation and is stated explicitly.

## 5. Joint component — Offline RL

The pooled dataset contains **420,103 transitions from 3,969 episodes**. Its
checksum and validation command are in `DATASET.md`. It combines trajectories
from all three role policies and includes mixed-quality behavior.

Naive offline DQN performs Bellman regression over the static data. Because the
dataset does not store masks, its maximum ranges over all 169 actions,
including unsupported actions. CQL adds
`α(logsumexp Q(s,a) − Q(s,a_data))`; BC directly clones logged actions.

| Method | cost/order, 3 training seeds | success |
|---|---:|---:|
| BC | 18.45 ± 3.79 | 0.542 ± 0.058 |
| naive offline DQN | 13.96 ± 2.72 | 0.537 ± 0.055 |
| **CQL** | **7.06 ± 1.10** | **0.717 ± 0.030** |

CQL beats both required offline baselines. The selected CQL seed scores 5.72.
Naive final maximum Q-values were approximately 6,785, 4,170, and 5,982;
CQL reduced them to 839, 792, and 745. Thus the OOD over-estimation failure is
visible in every training seed, not only in one lucky run.

![Offline naive-DQN Q-value divergence vs CQL](figures/offline_q_divergence.png)

The figure plots `max Q` over training (seed 0, representative): naive Bellman
regression climbs past 6,000 as it bootstraps from over-optimistic
out-of-distribution actions, while CQL's conservative penalty holds it near 800
and BC stays near zero. This is the required visual demonstration of the
over-estimation failure and its fix.

CQL does **not** beat `greedy_nearest` (7.06 vs 4.57), and this is expected
rather than a failure of the method. Offline performance is upper-bounded by the
behavior policy that generated the data (~60% greedy, ~40% noisy/random), so
greedy-level cost is the realistic ceiling when learning purely from a
half-random log. The required offline comparison is against naive-DQN-offline
and BC — both of which CQL clears decisively.

## 6. Joint component — Multi-agent IDQN

Eight decentralized drone agents share one Q-network and one replay buffer on
`DroneDispatchMA-v0`. Each receives its local 59-dimensional observation and
chooses accept, move, charge, or idle. The cost metric now counts on-time
deliveries from their actual deadlines rather than using a reward threshold.

Three equal 30k-step training runs produced:

| training seed | cost/order | delivered/episode |
|---:|---:|---:|
| 0 | 55.33 | 26.0 |
| 1 | 44.17 | 30.3 |
| 2 | 17.90 | 71.7 |
| **mean ± std** | **39.14 ± 15.69** | **42.7 ± 20.6** |

The random MA baseline costs **9.23**, so these equal-budget runs do not
converge. An earlier extended 60k seed-0 checkpoint reaches **6.65**, delivers
100.7 orders, and beats random, but it is not presented as a robust three-seed
result. It also does not beat the best centralized A2C (1.09) or Role-C planner
(2.92). This corrects the earlier claim that IDQN generally beat the centralized
policy.

![Multi-agent three-seed curve](figures/ma_idqn_three_seed.png)

The high variance illustrates non-stationarity: from one drone's perspective,
the other seven agents change their behavior while the shared network learns.
Parameter sharing reduces but does not remove this moving-target problem.

## 7. Method origins

- **DQN:** Mnih et al., *Human-level control through deep reinforcement
  learning* (2015), chosen as the canonical replay/target-network value method.
- **Double DQN:** van Hasselt, Guez & Silver (2016), chosen to reduce
  max-operator over-estimation.
- **Dueling DQN:** Wang et al. (2016), chosen to separate state value from
  action-specific advantage.
- **REINFORCE:** Williams (1992), the likelihood-ratio policy-gradient baseline.
- **GAE:** Schulman et al. (2016), chosen for its explicit bias/variance λ knob.
- **A2C/A3C:** Mnih et al. (2016); we use the synchronous A2C variant.
- **DDPG:** Lillicrap et al. (2016), chosen for continuous speed/heading actions.
- **Rollout planning:** Sutton & Barto, Chapter 8, and Tesauro & Galperin
  (1996), motivating decision-time policy improvement.
- **CQL:** Kumar et al. (2020), chosen to penalize unsupported offline actions.
- **IDQN/parameter sharing:** Tampuu et al. (2017) and Gupta et al. (2017).

## 8. Reproducibility and final assessment

All experimental numbers are in YAML configs; dependencies are pinned.
`offline_pool.npz` is included with SHA-256 verification. `run_all.py` loads
the final saved policies for every role and both joint components. The simulator
tests pass **17/17**.

The strongest methods are A2C, factored Double DQN, and the depth-1 planner.
Offline CQL satisfies the required failure-and-fix comparison. DDPG and
three-seed IDQN remain honest negative results: they run end-to-end, but do not
meet their respective baselines reliably.

## References

External sources consulted on why the flat value-based architecture (Section
2.5) fails on the large discrete assignment action space and why a factored
representation fixes it:

- Dulac-Arnold, G. et al. (2015). *Deep Reinforcement Learning in Large
  Discrete Action Spaces.* arXiv:1512.07679.
  <https://arxiv.org/abs/1512.07679>
- Sharma, S., Suresh, A., Ramesh, R., Ravindran, B. (2017). *Learning to Factor
  Policies and Action-Value Functions: Factored Action Space Representations for
  Deep Reinforcement Learning.* arXiv:1705.07269.
  <https://arxiv.org/abs/1705.07269>
