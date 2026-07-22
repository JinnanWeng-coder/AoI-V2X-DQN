# Remote runbook: AoI-V2X-DQN

Run from `1-IndependentDQN/` on the GPU host. Stay inside the DQN repository. Never push from the GPU host.

## Current formal snapshot

The synchronized experiment snapshot is commit `6181e5d758c39532f7dfe023190cb6375db412cc`. It contains 130 registry rows, all seven studies S01--S07 marked complete, and the completed S05 cap-10/cap-20 artifacts.

```bash
cd /d/Jinnan/CMDP/AoI-V2X-DQN
git pull --ff-only origin main
git lfs install
git lfs pull
git rev-parse HEAD

cd 1-IndependentDQN
PY=/d/Jinnan/CMDP/AoI-V2X-CMDP/.venv/Scripts/python.exe
"$PY" analysis/audit_results.py
```

Expected audit result at this snapshot:

```text
AUDIT PASS: 130 registered rows; all formal runs, MAT files, and study references are valid.
```

## Golden rules

- Valid formal MAT data belongs only in `experiments/runs/`; registered negative prototypes remain under `experiments/archive/failed_prototypes/`.
- Pick a canonical `--run_name`; `Main.py` refuses to overwrite it.
- Give parallel jobs distinct `DQN_CKPT_SUBDIR` values. Values such as `tmp/<job>` resolve below `Classes/` and are ignored by Git.
- Use `scratch/` for smoke output. `Classes/tmp/`, `tmp/`, `checkpoints/`, and `scratch/` are ignored.
- Generated figures go to `run_dir/figures/` and are ignored. Preserve existing `metrics.txt`; the plotting tool does not overwrite it.
- Add a formal artifact to `experiments/registry.tsv` only after its required MAT files pass audit.

## Locked one-run commands

```bash
cd /d/Jinnan/CMDP/AoI-V2X-DQN/1-IndependentDQN
PY=/d/Jinnan/CMDP/AoI-V2X-CMDP/.venv/Scripts/python.exe

DQN_CKPT_SUBDIR=tmp/dqn_soft_s2_re5 "$PY" Main.py --episodes 600 --seed 2 --renew_every 5 \
  --out_subdir soft_raw_aoi --run_name dqn_soft_raw_re05_buf050k_seed02

DQN_CKPT_SUBDIR=tmp/dqn_hard_s2_re20 "$PY" Main.py --episodes 600 --seed 2 --renew_every 20 \
  --mode hard --cost_source raw --dual pid --tau 8 --eps 0.10 --lam_max 5 --lam_warmup 150 \
  --lam_scope per_platoon --out_subdir rcpo_per_platoon \
  --run_name dqn_rcpo_raw_per_pid_lmax05_re20_seed02
```

Completion is indicated by `done. run_dir=...`. The canonical directories above already exist in the synchronized snapshot, so `Main.py` will refuse to overwrite them.

## Formal batch runners

The three restart-safe batch runners and their completed studies are:

| script | study | formal matrix |
|---|---|---|
| `scripts/run_rcpo_re20.sh` | S04 | per-platoon RCPO, re20, seeds 2--7 |
| `scripts/run_re20_constraint_ablations.sh` | S07 | global mean/max and fixed `w={2,5,10,20}`, re20, seeds 2--7 |
| `scripts/run_re20_lambda_cap_sweep.sh` | S05 | per-platoon RCPO cap 10/20 plus reused cap 5, re20, seeds 2--7 |

Inspect the locked commands without launching training:

```bash
bash scripts/run_rcpo_re20.sh --dry-run
bash scripts/run_re20_constraint_ablations.sh --dry-run
bash scripts/run_re20_lambda_cap_sweep.sh --dry-run
```

Each runner audits and reuses complete result directories, refuses to overwrite incomplete directories, and writes its study metadata only after the complete formal matrix passes MAT validation.

## Analysis and verification

```bash
"$PY" analysis/plot_run.py experiments/runs/rcpo_per_platoon/dqn_rcpo_raw_per_pid_lmax05_re20_seed02
"$PY" analysis/summarize_study.py experiments/studies/S04_cadence_validation
"$PY" analysis/summarize_study.py experiments/studies/S07_re20_constraint_design
"$PY" analysis/audit_results.py
```

S05's committed aggregate and multiplier diagnostics are `experiments/studies/S05_lambda_cap/summary.tsv` and `lambda_diagnostic.tsv`. S07's committed per-seed summary is `experiments/studies/S07_re20_constraint_design/summary.tsv`.

Stage only explicit files or directories. Do not use `git add -A`; do not commit or push unless separately authorized.
