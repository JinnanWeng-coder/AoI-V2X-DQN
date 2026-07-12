# Remote runbook: AoI-V2X-DQN

Run from `1-IndependentDQN/` on the GPU host. Stay inside the DQN repository; do not run a formal training job until the worktree and the intended registry row have been reviewed. Never push from the GPU host.

## Golden rules

- Formal MAT data belongs only in `experiments/runs/`.
- Pick a canonical `--run_name`; `Main.py` refuses to overwrite it.
- Give parallel jobs distinct `DQN_CKPT_SUBDIR` values.
- Use `scratch/` for smoke output and `checkpoints/` for model state; both are ignored.
- Generated figures go to `run_dir/figures/` and are ignored. Preserve existing `metrics.txt`; the plotting tool does not overwrite it.

## One run

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

Completion is indicated by `done. run_dir=...`. Audit a completed artifact before adding it to `experiments/registry.tsv`.

## Batch and analysis

`bash run_rcpo_re20.sh --dry-run` prints the locked S04 command. The runner uses `experiments/runs/rcpo_per_platoon/`, writes study summaries under `experiments/studies/S04_cadence_validation/`, and preserves existing summaries.

```bash
"$PY" analysis/plot_run.py experiments/runs/rcpo_per_platoon/dqn_rcpo_raw_per_pid_lmax05_re20_seed02
"$PY" analysis/summarize_study.py experiments/studies/S04_cadence_validation
"$PY" analysis/audit_results.py
```

Stage only explicit files or directories. Do not use `git add -A`; do not commit or push unless separately authorized.
