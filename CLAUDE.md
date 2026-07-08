# CLAUDE.md — AoI-V2X-DQN (Independent-DQN discrete line) — read this first

> Sister repo to `AoI-V2X-CMDP` (the DDPG/MADDPG version). This is a **clean-lineage
> Independent Double-DQN** re-implementation of the Parvini AoI-MARL platoon C-V2X problem:
> the environment is **verbatim Parvini**, the learner is a fresh discrete DQN. Purpose:
> (i) answer the "the action space is really discrete, DQN suffices" critique, and
> (ii) show the per-platoon **CMDP** result is **learner-agnostic** (holds on discrete
> control, not only on continuous DDPG).
>
> **Status (2026-07-08):** the core result has LANDED. On the discrete DQN the per-platoon
> CMDP (RCPO) protects the worst convoy — worst-platoon violation **0.319 → 0.133 (−58%)**,
> every platoon pulled to ε, at **+5.6% transmit power** — matching DDPG (0.126) at lower
> cost. P2 necessity ablations now support the mechanism claim: a single global λ controls
> the network mean but not the worst platoon, and no tested fixed-weight indicator penalty
> matches the adaptive per-platoon dual.
>
> Active code: `1-IndependentDQN/`. Remote machine: `D:\Jinnan\CMDP\AoI-V2X-DQN`
> (`REMOTE_RUNBOOK.md`). Code overview for outsiders: `README.md`. **No Git-LFS** —
> code + small `.mat` are tracked as regular blobs; `*.png` under `model/` are gitignored.

---

## 1. Locked config (canonical)
Scenario **5 platoon × 4 veh × 3 RB**; discrete action per agent = RB(3) × mode(2) ×
power(30) = **180**. seeds **2–7**, **600** episodes, 100 steps/ep, **re5** (`--renew_every 5`)
the canonical cadence. DQN: lr 1e-3, γ 0.99, batch 64, buffer **50k**, net **256/128**,
target_tau 0.005, ε 1.0→0.05 over 50% of steps. CMDP (hard): τ=**8**, ε=**0.10**,
`--cost_source raw` (RCPO), `--lam_max 5`, `--lam_warmup 150`, PID `kp=ki=1 kd=0.5`.

## 2. Settled findings (each checked against raw `.mat`)
1. **DQN soft ≈ DDPG soft.** re20: net 0.182 / worst 0.365 / meanAoI 5.26 ≈ DDPG 0.18 /
   0.35–0.49 / 5.4. A completely different learner lands in the same AoI regime ⇒ the
   per-platoon starvation is **problem-intrinsic, not a DDPG artifact** (and DQN suffices).
2. **"Soft hides the worst convoy" is robust** (`Scan_Cadence/`, re1/re5/re20 × seeds 2–7):
   **18/18** cells have worst ≫ ε while net-mean ≈ 0.18. Cadence sets the **distribution**,
   not the mean — worst & skew rise as renewal slows (re1 0.27 → re5 0.32 → re20 0.38).
   **re5 is canonical**: strong + concentrated + tight cross-seed + dual-friendly (the
   bottleneck persists ~5 ep). re20 has high cross-seed variance (worst 0.28–0.51).
3. **Reward converges, ~matches DDPG.** re5/re20 reward slope ≈ DDPG; re1 noisier
   (per-episode geometry variance). DQN total reward ~0.10 below DDPG, **all in task1/V2V**
   (power discretization); task2/AoI tied. DQN's late reward is steadier (ε→0.05 vs DDPG's
   always-on σ=0.3).
4. **Buffer eviction is a LOSS-only artifact.** The 50k buffer fills at ep500; the ep500
   cost-loss drop is FIFO eviction, **not** learning — the violation OUTCOME is flat across
   it. Memory sweep (`Mem_Sweep/`, 25k/50k/100k × seeds 2–6): **25k (heaviest evict) ≈ 100k
   (no evict)** on worst-platoon ⇒ **no systematic eviction bias**; last-100@600 is not
   contaminated. ⇒ train to **600** (matches CMDP). Never claim convergence from the
   ep500–600 critic loss.
5. **Step-5: per-platoon CMDP works on DQN — but only via RCPO.** The naive critic-argmax
   port (`--cost_source critic`, lam_max=20, no warmup) **FAILED** (`Step5_CMDP/`: worst
   **0.41–0.66**, λ pinned at cap, AoI up) — **value-scale domination**: in the discrete
   argmax `Q1+Q2−λ·Q^c`, once λ>~2–3 the cost value (~10–20) dwarfs the reward and the
   policy abandons reward. **Fix = RCPO** (`--cost_source raw`: fold `−λ·1{AoI>τ}` into
   Q2's reward target so λ stays on the reward scale) + `--lam_max 5` + `--lam_warmup 150`.
   Result (`Step5_RCPO/`, re5, seeds 2–7): worst **0.319 → 0.133 (−58%)**, net 0.180→0.108,
   meanAoI 5.19→4.45, **+5.6% power**; **every** platoon pulled to ε (hard 0.08–0.17, **0%
   >2ε** vs soft's 33%); λ **selective** (large only on the bottleneck). Matches DDPG (0.126)
   at lower power ⇒ **per-platoon CMDP is learner-agnostic.**
6. **P2 necessity ablations support the CMDP mechanism.** `Ablation3_GlobalLambda/`
   (re5, seeds 2–7) shows a single global λ is the wrong granularity: `global_mean` keeps
   the network mean near ε (net **0.102**) but leaves the worst platoon high (worst
   **0.180**, all 6 seeds > ε); `global_max` spends much more power (12.8 dBm, λ often at
   cap) yet still has worst **0.185**. `Ablation4_FixedWeight/` shows fixed indicator
   penalties are not a substitute: the best tested weight (`w=2`) has worst **0.226**
   (max **0.444**), while `w=5/10/20` worsens AoI and power (worst ≈0.41–0.46, power
   ≈16 dBm). ⇒ The result is not just "tune a penalty"; per-platoon adaptive duals are
   necessary for worst-convoy protection.

## 3. Code structure (`1-IndependentDQN/`)
- `Main.py` — training loop, argparse, per-run `.mat` logging, per-episode dual update.
- `agent.py` — `IndependentDQNAgent`: Double-DQN, task-decomposed Q-heads (Q1=V2V/CAM,
  Q2=V2I+AoI). soft `argmax[Q1+Q2]`; hard-RCPO folds `−λ·c` into Q2's target; hard-critic
  adds a `Q^c` head. **Extra nets are built only when used ⇒ the soft path is byte-identical
  to the validated baseline** (no RNG perturbation).
- `Classes/Environment_Platoon.py` — vendored Parvini env. Only deviations: removed
  import-time `np.random.seed(1376)`; AoI penalty via `aoi_penalty_coef` (Main sets 0 in
  hard) and `aoi_pen_type` (raw/indicator, #4). Dynamics / channel / mobility untouched.
- `Classes/buffer.py` — joint-transition replay, discrete action indices, parallel
  `reward_cost` channel (`1{AoI>τ}`).
- `Classes/networks.py` — `QNetwork` (state → 180 Q, LayerNorm MLP).
- `model/plot_results.py` — per-run figures (canonical violation, AoI, lambda, mode_v2i, …).

## 4. `model/` studies
`Scan_Cadence/` (cadence sweep, soft) · `Step5_CMDP/` (FAILED critic-argmax, kept for the
record) · `Step5_RCPO/` (the working hard result) · `Mem_Sweep/` (eviction invariance) ·
`Ablation3_GlobalLambda/` + `Ablation4_FixedWeight/` (P2 necessity ablations, landed). Run-dir = `model/
<out_subdir>/dqn_<mode>_seed<S>_<out_tag>/`. `.mat` tracked; `*.png` gitignored (regenerate
with `plot_results.py`).

## 5. Outputs per run
`viol_rate` (P×E, per-platoon `P(AoI>τ)`/episode — the headline), `lambda` (P×E, 0 in soft),
`AoI`, `AoI_evolution` (P×100×100 — the **canonical-violation** source), `reward_t1/t2`,
`power`/`demand`/`V2I`/`V2V` (P×100×100), `mode_v2i`, `epsilon`. **Canonical violation** =
`(AoI_evolution > 8).mean(axis=(1,2))` per platoon (worst = max, net = mean).

## 6. Next
Optional fixed-weight fine sweep around `w≈2` if a reviewer asks for a denser penalty grid
→ cadence robustness of hard-RCPO on re1/re20 → ε-sweep {0.05,0.10,0.20} and `--cost_source`
critic` with the lam_max=5/warmup fix (cost-critic necessity, A1) → scale (`--n_RB`/`--n_veh`,
env caps at 8 platoons).
