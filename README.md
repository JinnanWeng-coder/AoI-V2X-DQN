# AoI-V2X-DQN

Independent Double-DQN for AoI-aware platoon C-V2X resource allocation. The environment follows Parvini et al. (IEEE TVT 2023); the learner enumerates the discrete `(RB, mode, power)` action space and uses task-decomposed Q heads.

## Experiment snapshot

Snapshot `6181e5d` contains 130 registry rows: 124 `valid` formal runs and 6 archived `negative_result` prototype runs. The seven organized studies are complete, and `python analysis/audit_results.py` passes from `1-IndependentDQN/` for every registered run, MAT file, and study reference.

| study | status | scope |
|---|---|---|
| S01 soft cadence | complete | soft raw AoI, `renew_every={1,5,20}`, seeds 2--7 |
| S02 replay buffer | complete | re05, 25k/50k/100k, seeds 2--6 |
| S03 constraint design | complete | re05 soft, per/global RCPO, fixed `w={2,5,10,20}` |
| S04 cadence validation | complete | matched soft/per-platoon RCPO at re05 and re20 |
| S05 lambda cap | complete | re20 per-platoon RCPO, `lambda_max={5,10,20}` |
| S06 cross backbone | complete | external AoI-V2X-CMDP provenance reference |
| S07 re20 constraint design | complete | re20 global RCPO and fixed-weight controls |

## Experiment layout

All formal DQN artifacts live below `1-IndependentDQN/experiments/runs/`. MAT files are stored once only. Studies reference those run directories through `members.tsv`; a baseline or ablation is a study role, never a permanent run identity.

```text
1-IndependentDQN/
  analysis/                  # plot_run.py, summarize_study.py, audit_results.py
  scripts/                   # restart-safe formal batch runners
  experiments/
    registry.tsv             # authoritative formal-run registry
    runs/                    # only physical formal DQN run store
    studies/                 # members.tsv, summaries, manifests
    archive/                 # failed prototypes, verified duplicates, migration records
  checkpoints/               # ignored
  scratch/                   # ignored smoke output
```

Run `python analysis/audit_results.py` from `1-IndependentDQN/` after changing an experiment path, registry row, or study membership. MAT artifacts are managed by Git LFS under `1-IndependentDQN/experiments/**/*.mat`; generated PNG files, checkpoints, scratch output, logs, and `.out` files are ignored.

## Headline re20 aggregates

The table reports means over seeds 2--7. Metrics come from the final 100 training episodes; worst violation is computed per seed before averaging. Power is averaged in linear mW.

| method | worst-platoon | network mean | mean AoI | power (mW) |
|---|---:|---:|---:|---:|
| per-platoon RCPO, cap 5 | 0.146033 | 0.114167 | 4.485283 | 155.364813 |
| global-mean RCPO | 0.206150 | 0.104323 | 4.413677 | 150.393651 |
| global-max RCPO | 0.212050 | 0.114620 | 4.554076 | 193.038342 |
| fixed indicator `w=2` | 0.237350 | 0.136060 | 4.881960 | 154.599645 |
| soft raw AoI | 0.380267 | 0.197180 | 5.438109 | 126.354578 |
| fixed indicator `w=20` | 0.446650 | 0.235180 | 6.415855 | 192.409107 |
| fixed indicator `w=10` | 0.496017 | 0.267013 | 7.133211 | 183.948827 |
| fixed indicator `w=5` | 0.518517 | 0.282077 | 7.574691 | 168.118434 |

S05 varies only the per-platoon RCPO multiplier cap at re20:

| lambda_max | worst-platoon | network mean | mean AoI | power (mW) |
|---:|---:|---:|---:|---:|
| 5 | 0.146033 | 0.114167 | 4.485283 | 155.364813 |
| 10 | 0.140900 | 0.112660 | 4.466414 | 160.270919 |
| 20 | 0.127433 | 0.107973 | 4.391139 | 164.338083 |

Detailed seed rows and lambda diagnostics are stored in `1-IndependentDQN/experiments/studies/S05_lambda_cap/summary.tsv` and `lambda_diagnostic.tsv`; S07 seed rows are in `1-IndependentDQN/experiments/studies/S07_re20_constraint_design/summary.tsv`.

## Locked canonical command examples

```bash
cd 1-IndependentDQN
python Main.py --episodes 600 --seed 2 --renew_every 5 \
  --out_subdir soft_raw_aoi \
  --run_name dqn_soft_raw_re05_buf050k_seed02

python Main.py --episodes 600 --seed 2 --renew_every 20 \
  --mode hard --cost_source raw --dual pid --tau 8 --eps 0.10 \
  --lam_max 5 --lam_warmup 150 --lam_scope per_platoon \
  --out_subdir rcpo_per_platoon \
  --run_name dqn_rcpo_raw_per_pid_lmax05_re20_seed02
```

These two run names are already registered in the synchronized snapshot. The default output root is `experiments/runs/`, and `Main.py` refuses to overwrite an existing run directory. `--out_tag` and `--out_subdir` remain supported for legacy command compatibility; a distinct new formal run requires a canonical `--run_name` and a registry row.

## Formal batch runners

From `1-IndependentDQN/`, the completed formal matrices remain reproducible through restart-safe scripts:

```bash
bash scripts/run_rcpo_re20.sh --dry-run
bash scripts/run_re20_constraint_ablations.sh --dry-run
bash scripts/run_re20_lambda_cap_sweep.sh --dry-run
```

They correspond to S04 re20 per-platoon RCPO, S07 re20 constraint-design controls, and S05 lambda-cap sensitivity, respectively. Complete artifacts are audited and reused; incomplete directories are not overwritten.

## Analysis

Run these commands from `1-IndependentDQN/`:

```bash
python analysis/plot_run.py experiments/runs/rcpo_per_platoon/dqn_rcpo_raw_per_pid_lmax05_re20_seed02
python analysis/summarize_study.py experiments/studies/S04_cadence_validation
python analysis/summarize_study.py experiments/studies/S07_re20_constraint_design
python analysis/audit_results.py
```

`plot_run.py` writes PNG files to `run_dir/figures/` and prints metrics without overwriting `metrics.txt`.

## Evidence scope

The organized studies support empirical per-platoon AoI violation control during the final training window, not a finite-sample hard guarantee or deployment guarantee. Use seed as the inferential unit and linear mW for power aggregation. See `CLAUDE.md` and the study READMEs for the evidence ledger and provenance.

## Upstream reference

Parvini et al., “AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task RL,” IEEE Transactions on Vehicular Technology, 2023, doi:10.1109/TVT.2023.3259688.
