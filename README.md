# AoI-V2X-DQN — Independent Double-DQN for AoI-aware platoon C-V2X resource allocation

A clean-lineage **Independent Double-DQN** implementation of the platoon C-V2X
Age-of-Information (AoI) resource-allocation problem of Parvini et al. (IEEE TVT 2023),
extended with a **per-platoon Constrained-MDP (CMDP)** that turns AoI from a soft reward
penalty into a hard per-platoon constraint `P(AoI_j > τ) ≤ ε`.

This is the **discrete-control sibling** of `AoI-V2X-CMDP` (which uses continuous
DDPG/MADDPG): the same problem and the *same environment*, but a different, simpler learner.
It exists to (i) answer the "the action space is really discrete, so DQN suffices" critique,
and (ii) show the per-platoon CMDP result is **learner-agnostic** — it holds on discrete
control, not only on continuous DDPG.

Upstream environment & baseline: Parvini et al., *"AoI-Aware Resource Allocation for
Platoon-Based C-V2X Networks via Multi-Agent Multi-Task RL"*, IEEE TVT 2023.

## The problem (one paragraph)
Several vehicle platoons (1 leader + 3 followers each) share a few C-V2X sub-channels.
Each platoon **leader is an agent** that picks, per 1 ms slot, a `(sub-channel, mode, power)`
action: **inter-platoon V2I** (report to the base station, which resets the platoon's AoI)
or **intra-platoon V2V** (deliver the CAM payload to its followers). Only one mode per slot,
and with **5 platoons over 3 sub-channels** the channels must be reused → co-channel
interference. **AoI** = slots since the last successful V2I update (capped at 100); keeping
it fresh matters for platoon string stability. The core tension: every slot spent on V2V is
a slot in which AoI grows. The headline question: can we protect the **worst-served convoy's**
AoI, rather than only the network average?

## Method
- **Discrete action** per agent: `RB(K) × mode(2) × power(P)` enumerated (default
  `3 × 2 × 30 = 180`); `decode_action` maps an index → (sub-channel, inter/intra, power dBm).
- **Learner**: per-platoon Independent **Double-DQN** with **task-decomposed Q-heads** —
  `Q1` (V2V / CAM-delivery) and `Q2` (V2I-revenue + AoI); greedy policy `argmax_a [Q1+Q2]`.
  No global critic; ε-greedy exploration; Polyak (soft) target nets. (Parvini's task-decomposed
  structure, discretised.)
- **soft baseline** (`--mode soft`): AoI enters `Q2` as a `−AoI/20` reward penalty — the
  discrete analogue of the Parvini baseline.
- **hard / per-platoon CMDP** (`--mode hard`): enforce `P(AoI_j > τ) ≤ ε` per platoon with a
  per-platoon Lagrange multiplier `λ_j` on a two-timescale dual (PID-Lagrangian or integral,
  updated once per episode on `viol_rate_j − ε`, after a warmup). The constraint is injected by:
  - **RCPO** (`--cost_source raw`, **default**) — fold `−λ_j · 1{AoI>τ}` into `Q2`'s reward
    target; the greedy policy stays `argmax[Q1+Q2]`. Because λ multiplies a 0/1 cost it stays
    on the reward scale — stable.
  - **cost-critic** (`--cost_source critic`) — a separate cost head `Q^c` (Bellman regression
    on `1{AoI>τ}`); greedy `argmax[Q1+Q2 − λ_j·Q^c]`. *Combining raw Q-values is value-scale
    sensitive — keep `lam_max` small.*

## Headline result
On the discrete DQN, the per-platoon CMDP (RCPO; canonical cadence, seeds 2–7) pulls the
**worst-platoon** violation from **0.32 (soft) → 0.13 (≈ ε=0.10)**, holds **every** platoon
near ε, and lowers mean AoI (−14%), at **+5.6% transmit power** — matching the continuous
DDPG result (0.126) at lower cost. A single **global** multiplier, or a **fixed-weight**
AoI penalty, does not achieve this. Full numbers and the findings ledger: [`CLAUDE.md`](CLAUDE.md).

## Repository layout
```
1-IndependentDQN/
  Main.py                     training loop, argparse, .mat logging, per-episode dual update
  agent.py                    IndependentDQNAgent (Double-DQN; soft / hard; RCPO or cost-critic)
  Classes/
    Environment_Platoon.py    vendored Parvini env (AoI dynamics, channel model, mobility)
    buffer.py                 replay (discrete action indices + reward_cost channel)
    networks.py               QNetwork (256/128 LayerNorm MLP)
  model/
    plot_results.py           per-run figure generator
    <study>/<label>/*.mat     run outputs (tracked; *.png are gitignored)
CLAUDE.md                      current status, settled findings, config, data→claim map
REMOTE_RUNBOOK.md              running batches / committing on the GPU machine
```
**Lineage note.** `Classes/Environment_Platoon.py` is vendored **verbatim** from the pristine
Parvini repo; the only deliberate change is removing the import-time `np.random.seed(1376)`
(seeding moved to `Main.py --seed` so paired seeds reproduce across arms). The AoI dynamics,
channel model, and mobility are untouched. The learner is written from scratch. The hard-mode
AoI-penalty toggle and the `reward_cost` channel are the only other additions, tagged
`[RQ1-CMDP]` in-source.

## Quick start
```bash
cd 1-IndependentDQN
python Main.py --smoke                                         # end-to-end wiring test (seconds)

# soft baseline (canonical cadence re5):
python Main.py --episodes 600 --seed 2 --renew_every 5 --out_tag re5

# per-platoon CMDP (RCPO):
python Main.py --episodes 600 --seed 2 --renew_every 5 --mode hard --cost_source raw \
    --tau 8 --eps 0.10 --lam_max 5 --lam_warmup 150 --out_tag re5 --out_subdir Step5_RCPO

# figures for a run:
python model/plot_results.py model/Step5_RCPO/dqn_hard_seed2_re5
```
Requires Python ≥ 3.9, PyTorch, numpy, scipy (and matplotlib for plotting). Per-run outputs
land in `model/<out_subdir>/<label>/` as `.mat`; `viol_rate.mat` (per-platoon `P(AoI>τ)` per
episode) is the headline quantity. The **canonical** violation metric is
`(AoI_evolution > τ).mean(axis=(1,2))` per platoon.

## Key flags
| flag | meaning |
|---|---|
| `--mode {soft,hard}` | soft `−AoI/20` penalty vs per-platoon CMDP constraint |
| `--renew_every {1,5,20}` | mobility cadence (geometry renew interval; **re5** canonical, 1 = per-episode, 20 = Parvini) |
| `--cost_source {raw,critic}` | RCPO reward-fold (default) vs separate cost critic |
| `--tau --eps` | AoI threshold τ (slots) and target violation probability ε |
| `--lam_max --lam_warmup --dual {pid,integral}` | dual clip / warmup episodes / update rule |
| `--lam_scope {per_platoon,global_mean,global_max}` | per-platoon vs single global λ (ablation #3) |
| `--aoi_pen_type {raw,indicator} --aoi_pen_w` | fixed-weight threshold penalty, no dual (ablation #4) |
| `--n_RB --n_veh --n_power` | scenario size / power discretisation (env caps at 8 platoons) |
| `--episodes --seed --buffer --fc1 --fc2 --out_tag --out_subdir` | training / output control |

## Citation
This project builds directly on the environment of Parvini et al. (IEEE TVT 2023,
doi:10.1109/TVT.2023.3259688) — please cite the upstream paper and respect its license.
