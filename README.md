# AoI-V2X-DQN — Independent-DQN discrete baseline (clean lineage)

A from-scratch **Independent Double-DQN** re-implementation of the Parvini platoon
C-V2X AoI-MARL problem, built to address the discrete-action / mobility-cadence
critiques without inheriting the CMDP fork. The environment is **verbatim Parvini**;
only the learner is new.

## Lineage (why this repo exists)
- The CMDP repo (`AoI-V2X-CMDP`) has **no pre-CMDP commit** — its first commit already
  contains the soft/hard switch + cost critic. So the only clean discrete-baseline
  ancestor is the pristine Parvini repo (`AoI-V2X-IEEE-TVT-2023-main-original`, commit
  `d8f0c92`).
- **Commit 1** here vendors `Classes/Environment_Platoon.py` UNMODIFIED from that repo.
- **Commit 2** (this one) adds the Independent-DQN learner and the single deliberate env
  deviation: the import-time `np.random.seed(1376)` is removed so the RNG is seeded
  per-run in `Main.py` via `--seed` (paired seeds across arms).

## What is / isn't here
- **Environment**: identical AoI dynamics, channel model, mobility, and the **soft
  `-AoI/20` penalty in task-2** (so this run IS the discrete *soft baseline*).
- **Learner**: per-platoon Independent Double-DQN, **task-decomposed Q-heads**
  (`q_task1` V2V/CAM, `q_task2` V2I+AoI); greedy policy = `argmax_a [Q1(s,a)+Q2(s,a)]`.
  No global critic (pure independent learners), ε-greedy exploration, Polyak target nets.
- **Discrete action set**: `n_actions = n_RB * 2 * n_power` (default `3*2*30 = 180`);
  `decode_action` maps an index → (RB, inter/intra mode, power dBm in 1..30).
- Observation is the verbatim 19-d Parvini state; net defaults to **256/128**
  (Parvini's 1024/512 critic was ~54× over-parameterised for a 19-d input — this is the
  pre-emptive "shrink the net" step).

## Maps to the agreed plan
| step | status here |
|---|---|
| 1 DDPG→DQN | **done** (Independent Double-DQN, TDec Q-heads) |
| 2a per-episode renewal | **flag** `--renew_every` (default 20 = Parvini; set `1` for per-episode) |
| 2b 200 steps/ep | **intentionally NOT done** — horizon is physics-locked (100ms/1ms); bumping it desyncs the CAM period / AoI cap and would deflate the violation rate |
| 3 enlarge scenario | **flags** `--n_RB`, `--n_veh` (≤ 8 platoons: env hard-codes initial positions for 8) |
| 4 shrink net | **done** (256/128 default; tune `--fc1/--fc2`) |
| 5 per-platoon Lagrangian | **TODO** — add a `q_cost` head (Bellman on `1{AoI>τ}`) + per-platoon λ; select `argmax_a [Q1+Q2 − λ_j·Q^c]`; reuse the CMDP repo's PID/integral dual verbatim |

## Run
```
cd 1-IndependentDQN
python Main.py --smoke                                         # wiring test, seconds
python Main.py --episodes 600 --seed 2 --renew_every 20 --out_tag re20   # validate vs DDPG-soft
python Main.py --episodes 600 --seed 2 --renew_every 1  --out_tag re1    # step 2a (per-episode)
```
Outputs: `1-IndependentDQN/model/<label>/*.mat` (same names as the CMDP repo:
`viol_rate`, `AoI`, `AoI_evolution`, `reward_t1/t2`, `power`, `demand`, `V2I`, `V2V`,
`epsilon`). `viol_rate.mat` = per-platoon `P(AoI>τ)` per episode — the soft-baseline
quantity Finding-1 is about, so you can read the worst-platoon violation straight away.

## Validation order (change one axis at a time)
1. `--renew_every 20` first → confirm DQN converges and AoI levels / V2V success are in
   the same ballpark as the known DDPG-soft baseline.
2. then `--renew_every 1` → the per-episode-mobility result.
3. only then enlarge the scenario / add the Lagrangian.
