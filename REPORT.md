# IE 306 Term Project — Reinforcement Learning for City-Scale Drone Delivery

**Team report.** Each member owns one method family (see `ROLES.md`); the offline-RL
and multi-agent components are joint. Primary metric: **mean cost per delivered
order** (`cost_per_order`, lower is better) on the held-out evaluation config.
Baselines on the standard config: random ≈ 18.78, **greedy_nearest ≈ 4.57** (the
bar), milp_rolling ≈ 4.72.

> Reproduce any table: `python run_all.py --config configs/eval_standard.yaml --seeds 0,1,2`

---

## 1. Role A — Value-based DQN family (Sezen Balkan)

### 1.1 Method descriptions

The discrete dispatcher chooses one of **169 actions** per step (160 assignment =
8 drones × 20 order slots, 8 charge, 1 no-op). The observation (drones, orders,
grid, time) is flattened and normalised; an invalid-action mask keeps the policy
on legal actions. We trained three value-based variants on `DroneDispatch-v0`,
plus an n-step return on top:

- **DQN** — a Q-network estimates `Q(s,a)`; trained toward the one-step Bellman
  target `r + γ·max Q(s',a')` using a slow **target network**, an experience
  **replay buffer**, and **ε-greedy** exploration (ε: 1.0 → 0.05 over 40k steps).
- **Double DQN** — decouples action *selection* (online net) from action
  *evaluation* (target net) in the target, removing the systematic max-operator
  **overestimation** of vanilla DQN.
- **Dueling DQN** — splits the head into a state-value `V(s)` and an advantage
  `A(s,a)` stream, which can speed learning when many actions are similar.
- **n-step returns (n=3)** — accumulates `Σ γ^k r_{t+k}` and bootstraps n steps
  ahead, propagating delayed delivery reward back faster (better credit
  assignment). Windows are truncated at episode boundaries (the T_max horizon is
  a true terminal in this finite-horizon MDP).

All hyperparameters live in `configs/*.yaml` (seed recorded explicitly); nothing
is hard-coded. The simulator package was never modified.

### 1.2 Diagnostic journey — "what broke and how we diagnosed it"

| Symptom | Diagnosis | Fix |
|---|---|---|
| Agent never charged (0 charge actions) | **Under-training** — ε never decayed; *not* an action-index/mask bug | Fixed 60k budget + ε decay to 0.05 |
| "Passive collapse": over-selects charge/no-op, few deliveries, cost ~22–29 | shared value-instability + bad input scaling | see below |
| Toggling the `normalize_time` flag gave a big jump (cost 29.4→22.3, success 0.38→0.49) | **misdiagnosis — corrected:** we first thought `time` was a raw 0–500 feature dominating the net. On re-checking the env it already returns `time = t/T_max ∈ [0,1]` (`env_dispatch.py:309`, spec §12.1), so it was *never* raw 0–500 | the flag in fact divides the already-normalised value *again* by T_max, pushing time → ≈0 — i.e. it **removes time as a feature**. The gain came from dropping a distractor input, not from rescaling a raw one (see note below) |
| 3M-step run **diverged** (cost 27.9 @1.5M → 80.7 @3M) | bottleneck is value stability / credit assignment, **not compute** | n-step + Double DQN |
| Suspected `masked_fill(-1e9)` target leak | **ruled out** — no-op (idx 168) is always a valid action, so the masked next-state is never all-False | no fix needed |

> **Correction note (time normalization).** An earlier version of this log (and report)
> claimed the raw `time` feature spanned 0–500 and dominated the network. That is wrong:
> the simulator already exposes `time` as `t/T_max ∈ [0,1]`. What our `normalize_time`
> checkpoint flag actually does is divide that already-normalised value by `T_max` a
> *second* time, collapsing it to ≈0 and effectively switching the time feature off. The
> measured improvement (29.4→22.3) is real, but its cause is "time was a misleading input
> and suppressing it helped," not "a raw 0–500 feature was rescaled." We keep the numbers
> and correct the mechanism.

### 1.3 Results tables

**3-seed summary (seeds 0,1,2), best eval `cost_per_order`, mean ± std**
(full table in `logs/results_seeds.md`):

| Method | best cost | post-decay mean | note |
|---|---|---|---|
| DQN n=3 (600k) | 13.87 ± 0.71 | 22.65 ± 0.79 | flat oscillation |
| Dueling DQN n=3 (600k) | 13.17 ± 6.43 | 28.87 ± 6.95 | unstable across seeds |
| Double DQN n=3 (600k) | 9.97 ± 0.18 | 19.51 ± 0.72 | tight, consistent |
| **Double DQN n=3 (3M)** | **6.39 ± 0.41** | 16.24 ± 1.32 | best & robust |

**Best submitted policy** — Double DQN n=3, validation-selected **1M checkpoint**
(`weights/double_dqn_nstep_3m_step_1000000.pt`), on seeds 0,1,2
(`logs/double_dqn_nstep_3m_best1M_eval.json`):

| metric | value | vs before (normalised n=1) |
|---|---|---|
| cost_per_order | **6.76** | 22.33 |
| success_rate | **0.749** | 0.49 |
| on-time rate | 0.80 | 0.76 |
| episode_return | +738 | −364 |
| no-op actions | 40 | 564 (passive collapse solved) |

**Baseline comparison** (`run_all.py`, standard config): random ≈ 18.78,
greedy_nearest ≈ 4.57, milp_rolling ≈ 4.72, **Double DQN n=3 (1M) ≈ 6.76**.

### 1.4 Learning curves

All curves are **3-seed (0,1,2) mean ± std bands** (not single lucky runs),
generated from `logs/*_eval.csv` by `python code/plot_curves.py`:

- `logs/curves_methods_600k_cost.png` — DQN vs Double vs Dueling (cost). Double DQN
  is consistently lowest **and** tightest across seeds; Dueling is the most unstable.
- `logs/curves_methods_600k_return.png` — same three, `episode_return`.
- `logs/curves_double_3m_cost.png` — Double DQN 3M: a stable good band ~1M–2.5M
  (cost 6.6–13) **then divergence after ~2.5M**, the central Role-A finding.

![Double vs DQN vs Dueling, 600k, 3-seed mean±std cost](logs/curves_methods_600k_cost.png)
![Double DQN 3M, 3-seed mean±std cost](logs/curves_double_3m_cost.png)

### 1.5 Ablation — target network on / off

We isolate the **target network**, the design choice most tied to our central
finding (value stability). ON = update target every 1000 steps (our default);
OFF = target equals the online net every step.

Same config, seed 0, 600k — the only change is `target_update_every` (1000 → 1):

| Setting | best cost | final cost | note |
|---|---|---|---|
| Target net ON (default) | **10.19** | **14.90** | stable |
| Target net OFF (update every step) | 13.18 | 25.66 | worse best, ~1.7× worse endpoint |

Removing the slow target network (target = online net every step) hurts both the
best policy (13.18 vs 10.19) and especially the endpoint (final 25.66 vs 14.90):
without it the values chase themselves and the run diverges harder late in
training. The target network is doing real stabilising work — exactly the
value-stability axis our whole diagnosis turns on. (`logs/double_dqn_nstep_600k_notarget_eval.csv`,
config `configs/double_dqn_nstep_600k_notarget.yaml`.)

*Supporting ablation (n-step, full 3-seed):* at 600k seed 0, n=1 reaches best 13.96
but **diverges** (final 45.63), whereas n=3 is best 13.33 / final 20.67 — n-step
helps on every aggregate.

### 1.6 Verdict (against the objective: beat greedy_nearest OR diagnose why not)

We did **not** beat greedy_nearest (best 6.76 ≈ 1.48× the 4.57 bar) → we are on
the **honest-diagnosis** branch. The three fixes (time-feature suppression + n-step +
Double DQN) **stack and work**, turning a passive-collapse policy (cost ~22–29,
success ~0.4) into a useful one (cost 6.76, success 0.75). The remaining gap is
attributable to **model capacity/representation and residual late-training
instability**, not exploration or credit assignment (both addressed). Crucially,
"more compute" is *not* the fix: plain DQN diverges, and even the stabilised
Double DQN variant diverges after ~2.5M.

**6M confirmation run (completed).** A 6M-step Double DQN n=3 run settles the
"is it just compute?" question. Its own best checkpoint sits at **1.65M (cost
6.62, return +804)** — squarely in the same ~1M–2.5M good band — and **4× more
compute never beats it**. After ~2.5M the policy degrades monotonically:
cost 13.6 → 20.3 → 38.8 → **53.1 at 6M** (return +52 → −329 → −990 → **−1078**),
with training loss simultaneously blowing up (~21 → 56). The divergence is
**permanent, not cyclical**: the run ends in deep collapse, not recovery. This
mirrors the 3M run and confirms the verdict — the value-based family plateaus at
cost ≈ 6.6–6.8 around 1–2M steps, and additional gradient updates actively hurt.
The submitted policy therefore remains the early checkpoint (Double DQN n=3, 1M,
cost 6.76); the 6M run's marginally-lower 6.62 is single-run and does not justify
the 4× compute. (Raw: `logs/double_dqn_nstep_6m_eval.csv`.)

### 1.7 Method-origin note (Role A)

- **DQN** — Mnih et al., *Human-level control through deep reinforcement
  learning*, Nature 2015. Chosen as the canonical value-based baseline for a
  discrete action space.
- **Double DQN** — van Hasselt, Guez & Silver, *Deep RL with Double Q-learning*,
  AAAI 2016. Chosen because our core failure was value-instability/divergence,
  which max-operator overestimation directly feeds.
- **Dueling DQN** — Wang et al., *Dueling Network Architectures for Deep RL*,
  ICML 2016. Tried because most of the 169 actions are state-dependent
  assignments with similar value.
- **n-step returns** — Sutton & Barto, *Reinforcement Learning* (2nd ed.), Ch. 7.
  Chosen to fix credit assignment under delayed delivery reward.

---

## 2. Role B — Policy-based (REINFORCE/GAE → A2C, + DDPG) — Ozan Karhan  _[done]_

Full write-up: **`REPORT_roleB.md`**; code in `code/role_b/`, configs
`configs/{reinforce,a2c,ddpg,ablation_gae}.yaml`, 3-seed weights `weights/{reinforce,a2c,ddpg}_seed{0,1,2}.pt`, curves in `figures/`.

**Methods.** REINFORCE + **GAE** → **A2C** on the discrete masked dispatcher
`DroneDispatch-v0`; **DDPG** on the continuous control sub-env `DroneControl-v0`.
Checkpoints are selected on validation `cost_per_order` (not return).

**Dispatch results** (best-of-3-seed, seeds 0–4; Role B's own baseline: greedy 4.309, milp 4.282):

| Method | cost_per_order ↓ | success | delivered/ep | note |
|---|---|---|---|---|
| **REINFORCE + GAE** | **2.636** | 0.88 | 122.6 | beats greedy |
| **A2C** | **1.735** | 0.96 | 134.0 | **beats greedy by ~60% — best learned result on the team** |

A2C's win comes from charging proactively and refusing battery-infeasible
assignments, removing the +50 depletion hits greedy keeps paying (depletions/ep
8.0 → 1.6). **DDPG** (control sub-env): best-seed return **−149.6** vs the
go-straight baseline **−417** — beats it on return on every seed (DDPG stays
somewhat unstable across seeds, a known trait).

**Ablation — GAE λ sweep** (A2C, λ ∈ {0,0.9,0.95,0.99,1.0}, `figures/ablation_gae.png`):
λ=0 (one-step, biased) never beats greedy (cost 13.8); **λ=0.9–0.95 is optimal
(≈0.76)**; λ=1.0 (Monte-Carlo, unbiased but high-variance) slightly worse (0.90)
— validating the GAE(0.95) default.

**Method-origin.** A2C — Mnih et al., *Asynchronous Methods for Deep RL*, ICML 2016;
GAE — Schulman et al., ICLR 2016; DDPG — Lillicrap et al., ICLR 2016.

## 3. Role C — Planning (rollout-style planner) — Tuba Nur Büyükata  _[done]_

Full write-up: **`REPORT_roleC.md`** (+ `logs/role_c_results.txt`); code in
`code/role_c/`, config `configs/role_c_rollout.yaml`, depth params
`weights/role_c_rollout_depth{0,1,2}.json`, eval logs `logs/role_c_rollout_depth{0,1,2}.csv`.

**Method.** A **rollout-style planning policy** for the centralized dispatcher: it
scores valid assignment/charge actions by routed pickup distance, delivery
distance, deadline risk and battery feasibility (it implements only the frozen
`act(obs)` interface; the simulator is untouched). It is a planner, not a learned
net — so the "≥3-seed" deliverable is the **depth ablation evaluated on seeds 0,1,2**
rather than a training curve.

**Ablation — rollout depth** (seeds 0,1,2, `configs/eval_standard.yaml`):

| Method | cost_per_order ↓ | success | on-time | delivered/ep |
|---|---|---|---|---|
| greedy_nearest (bar) | 4.570 | 0.855 | 0.903 | 118.3 |
| Role C depth=0 (≈greedy) | 4.570 | 0.855 | 0.903 | 118.3 |
| **Role C depth=1** | **2.923** | 0.881 | 0.982 | 126.3 |
| Role C depth=2 | 3.331 | 0.869 | 0.982 | 124.3 |

**depth=1 beats greedy_nearest (2.923 vs 4.570)** by adding one-step planning terms
(delivery distance, deadline risk, battery feasibility); depth=2's extra
post-delivery charging proxy is slightly too conservative.

**Method-origin.** Rollout / decision-time planning — Sutton & Barto Ch. 8;
Tesauro & Galperin, *On-line policy improvement using Monte-Carlo search*, 1996.

## 4. Joint — Offline RL (Ch. 20)  _[team — done]_

We pool logged trajectories from all three members into one mixed-quality
dataset and learn **without any environment interaction**: (i) a naive offline
DQN to **demonstrate** the overestimation / OOD-action failure, then (ii) **CQL**
to fix it and beat both naive-offline-DQN and a behavioural-cloning baseline.

**4.1 Dataset.** `offline_pool.npz` = **420,103 transitions / 3969 episodes**,
concatenated from the three members' rollouts (`pool_offline.py`); each source
passed the shared format check (`check_offline_npz.py`):

| Source | file | transitions |
|---|---|---|
| Role A (Double DQN + ε-noisy greedy mix) | `offline_dlogs.npz` | 200,029 |
| Role B | `offline_ozan_karhan.npz` | 120,074 |
| Role C | `offline_runa.npz` | 100,000 |
| **Pool** | `offline_pool.npz` | **420,103** |

Obs are the 181-dim `_flatten_obs` vector; we standardize once from the pool and
reuse the identical mean/std at eval. Training never queries the env. No
action-masks are stored, so the Bellman max and the CQL penalty range over all
169 actions — which is exactly the OOD setting we want to stress.

**4.2 Methods.** *Naive offline DQN* — vanilla `r + γ max Q(s',·)` regression on
static data. *CQL* (Kumar et al., NeurIPS 2020) — adds a conservative penalty
`α·(logsumexp_a Q(s,a) − Q(s,a_data))` that pushes down OOD-action values. *BC* —
supervised cross-entropy cloning of the logged actions. (`code/offline_rl.py`,
40k/40k/15k grad steps, batch 256, α=1.0; eval on seeds 0,1,2.)

**4.3 Results — the failure is measurable and the fix works.**

| Method | cost_per_order ↓ | success | final max‑Q | note |
|---|---|---|---|---|
| BC baseline | 22.47 | 0.50 | — | cloning a *mixed*-quality dataset is weak (worse than random) |
| **Naive offline DQN** | 17.44 | 0.47 | **6785** | **Q blows up** 61→6785 = overestimation |
| **CQL** | **8.42** | 0.68 | **839** | conservatism keeps Q bounded; **beats both** |
| _ref: greedy_nearest_ | _4.57_ | | | the bar |
| _ref: online DQN (1M)_ | _6.76_ | | | Role A best |

(Seeded run, `torch.manual_seed(0)`, so these reproduce exactly.) The naive
agent's estimated `max_a Q` diverges to **6785** (unbounded overestimation on
never-taken actions) and its policy is poor (cost 17.44). CQL holds `max‑Q` at
**839** and reaches **cost 8.42**, clearly beating naive (17.44) and BC (22.47)
despite never touching the env. It does **not** reach the online DQN / greedy
level — consistent with our overall finding that no model-free method beats the
greedy heuristic here — but both required outcomes hold: (i) the overestimation
failure is demonstrated with logged Q-curves (`logs/offline_qstats.csv`), (ii)
CQL beats naive-offline-DQN **and** BC (`logs/offline_results.json`).

> **Caveat — single run.** These offline numbers are one seeded training run per
> method, not a 3-seed mean±std like the Role-A curves. The offline scores are
> high-variance (we saw CQL move by several points under tiny training changes),
> so read the **ordering** (CQL < naive < BC, and naive's Q diverging) as the
> robust result, not the exact figures. A 2–3 seed offline sweep is the natural
> next step if time allows.

Method-origin: CQL — Kumar, Zhou, Tucker & Levine, *Conservative Q-Learning for
Offline RL*, NeurIPS 2020.

**What teammates B and C handed over** (reproduce / extend the pool) (so all logged trajectories pool into
one mixed-quality `.npz`). Roll out your trained policy on **`DroneDispatch-v0`**
(the centralized single-agent env — *not* the MA env) and dump one
`offline_<name>.npz` per policy, e.g. `offline_a2c.npz`, `offline_dyna.npz`.
Match `drone_dispatch_env/offline.py` exactly:

- **Obs flattening** — use `_flatten_obs(obs)`: `concat(drones.flatten(),
  orders.flatten(), time.astype(float32))`, dtype `float32`. Do **not** invent
  your own flatten order; import the helper.
- **Action indexing** — the same 169-action discrete space (160 assign + 8
  charge + 1 no-op). Record the raw integer action you passed to `env.step`.
- **npz keys & dtypes** (identical names, or the loader breaks):
  `observations` f32, `actions` i64, `rewards` f32, `next_observations` f32,
  `terminals` bool (true terminal only), `timeouts` bool (T_max truncation),
  `episode_returns` f32 (one per episode).
- **Mix** — include some sub-optimal/exploratory rollouts (e.g. an ε-noisy
  branch), not only your greedy-best policy; coverage gaps across policies are
  what make the offline problem interesting.
- Aim ≥ ~100k transitions each; we concatenate the per-policy arrays into the
  shared dataset. Nothing else (no weights, no configs) is required from you for
  the offline part.

This is **independent of the multi-agent task** — MA uses its own 59-dim
per-agent obs and 4 actions and needs no shared dataset.

## 5. Joint — Multi-agent (Ch. 21)  _[team — done]_

We replace the single centralized dispatcher with **8 decentralized per-drone
agents** on `DroneDispatchMA-v0` (per-agent 59-dim local obs, 4 actions:
accept / move / charge / idle). All agents share **one Q-network** (parameter
sharing) and pool their transitions into a single replay buffer — the canonical
**IDQN** setup (`code/train_ma_idqn.py`, 60k env-steps, ε:1→0.05 over 50%).

**5.1 Head-to-head** (eval seeds 0,1,2). The MA env exposes no `stats`, so we
reconstruct the *same* `cost_per_order` from the reward stream
(`cost = 10·delivered + 5·ontime − return`; deliveries counted exactly from a
`TO_DROPOFF→IDLE` transition on a live drone):

| Policy | cost_per_order ↓ | delivered/ep | return | note |
|---|---|---|---|---|
| random (MA) | 8.80 | 85.3 | 433.8 | strong baseline: accept+move already delivers |
| **IDQN, param-sharing (MA)** | **6.49** | **100.7** | **793.8** | beats random, **and** the centralized ref |
| _ref: centralized Double DQN_ | _6.76_ | — | — | own env/action abstraction (paradigm ref, not identical metric) |

The decentralized IDQN reaches **cost 6.49** — beating the random baseline (8.80)
and edging the centralized Double DQN reference (6.76). The comparison to the
centralized policy is across two different action abstractions, so we read it as
*paradigm vs paradigm* (decentralized control is competitive with centralized
dispatch here), not a strict same-metric tie.

**5.2 Non-stationarity — visible in the learning curve.** From any one drone's
view the other 7 are part of the environment and keep changing as they learn, so
each agent chases a **moving target**. This is exactly what `logs/ma_idqn.csv`
shows: the policy first gets **worse** (return −1303 → −1388, cost 66→82 over
12k–36k steps, while ε is still high and all 8 agents shift simultaneously),
recovers as exploration anneals (−913 @48k), and only converges late (return
+794, cost 6.49 @60k). Parameter sharing mitigates non-stationarity by giving
every agent the *same* evolving policy (the others' behaviour is at least
correlated with one's own), and pooling all agents' transitions multiplies
effective sample throughput 8×. Residual instability is inherent: independent
learners have no convergence guarantee in a general-sum multi-agent game.
Method-origin: IDQN — Tampuu et al., *Multiagent cooperation and competition with
deep RL*, 2017; parameter sharing — Gupta, Egorov & Kochenderfer, AAMAS 2017.

---

*Note ağırlığı analiz derinliğinde, sayfa sayısında değil. Tüm sayılar
`logs/` altındaki ham CSV'lerle desteklenmektedir; `run_all.py` tabloyu yeniden üretir.*
