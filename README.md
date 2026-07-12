# AoI-V2X-DQN

Independent Double-DQN for AoI-aware platoon C-V2X resource allocation. The environment follows Parvini et al. (IEEE TVT 2023); the learner enumerates the discrete `(RB, mode, power)` action space and uses task-decomposed Q heads.

## Experiment layout

All formal DQN artifacts live below `1-IndependentDQN/experiments/runs/`. MAT files are stored once only. Studies reference those run directories through `members.tsv`; a baseline or ablation is a study role, never a permanent run identity.

```
1-IndependentDQN/
  analysis/                  # plot_run.py, summarize_study.py, audit_results.py
  experiments/
    registry.tsv              # authoritative formal-run registry
    runs/                     # only physical formal DQN run store
    studies/                  # members.tsv, summaries, optional figures
    archive/                  # failed prototypes, verified duplicates, migration records
  checkpoints/                # ignored
  scratch/                    # ignored smoke output
```

Run `python analysis/audit_results.py` from `1-IndependentDQN/` after changing an experiment path, registry row, or study membership. MAT artifacts are managed by Git LFS under `1-IndependentDQN/experiments/**/*.mat`; generated PNG files, checkpoints, scratch output, logs, and `.out` files are ignored.

## Run a new canonical artifact

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

The default output root is `experiments/runs/`. `Main.py` refuses to overwrite an existing run directory. `--out_tag` and `--out_subdir` remain supported for legacy command compatibility, but new formal runs should use a canonical `--run_name` and a registry row.

## Analysis

```bash
python analysis/plot_run.py experiments/runs/rcpo_per_platoon/dqn_rcpo_raw_per_pid_lmax05_re20_seed02
python analysis/summarize_study.py experiments/studies/S04_cadence_validation
python analysis/audit_results.py
```

`plot_run.py` writes PNG files to `run_dir/figures/` and prints metrics without overwriting `metrics.txt`.

## Scope of the current evidence

The formal result supports empirical per-platoon AoI violation control, not a finite-sample hard guarantee. In the matched re20 comparison, DQN soft and per-platoon RCPO have worst-platoon violation `0.3803` and `0.1460`, respectively. Power comparisons must use linear mW, not percentages of dBm averages. See `CLAUDE.md` and the study READMEs for the current evidence ledger.

## Upstream reference

Parvini et al., “AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task RL,” IEEE TVT 2023, doi:10.1109/TVT.2023.3259688.
