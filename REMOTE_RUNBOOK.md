# REMOTE RUNBOOK — AoI-V2X-DQN on the GPU machine

> Repo on the remote: `D:\Jinnan\CMDP\AoI-V2X-DQN` (Git Bash: `/d/Jinnan/CMDP/AoI-V2X-DQN`).
> *What* the experiment is and the findings → `CLAUDE.md`. *How the code works* → `README.md`.

## 0. Environment
- Python **3.11.11**, torch **2.6.0+cu126**. Reuse the sister repo's venv:
  `PY=/d/Jinnan/CMDP/AoI-V2X-CMDP/.venv/Scripts/python.exe`. Only extra dep is
  `matplotlib` for plotting (`"$PY" -m pip install -q matplotlib` if missing).
- Always run from `1-IndependentDQN/`.

## 1. Golden rules (do not break)
- Stay inside `D:\Jinnan\CMDP\AoI-V2X-DQN`; never touch `AoI-V2X-CMDP` or other people's files.
- **Never `git push` from here** (no TTY credentials). Commit locally, report the hash, the
  human pushes.
- **Never `git add -A`.** Stage only the study folder(s) you produced. `.mat` and
  `metrics.txt` are tracked; `*.png` under `model/` are gitignored (regenerate locally).
- Give every run a **unique `DQN_CKPT_SUBDIR`** (else parallel runs overwrite each other's
  checkpoints under `Classes/tmp/`, which is gitignored).
- **Smoke first** for any new code path (`--smoke`, finishes in seconds). A run is **done**
  when its log ends with `done. label=`.

## 2. Run one
```
cd /d/Jinnan/CMDP/AoI-V2X-DQN/1-IndependentDQN
PY=/d/Jinnan/CMDP/AoI-V2X-CMDP/.venv/Scripts/python.exe

# soft baseline (canonical cadence re5):
DQN_CKPT_SUBDIR=tmp/dqn_soft_s2_re5 "$PY" Main.py --episodes 600 --seed 2 --renew_every 5 \
    --out_tag re5 --out_subdir MyStudy

# hard per-platoon CMDP (RCPO — the headline):
DQN_CKPT_SUBDIR=tmp/dqn_hard_s2_re5 "$PY" Main.py --episodes 600 --seed 2 --renew_every 5 \
    --mode hard --cost_source raw --dual pid --tau 8 --eps 0.10 --lam_max 5 --lam_warmup 150 \
    --out_tag re5 --out_subdir Step5_RCPO
```
Per-episode log line: `worst_viol / net_viol / meanAoI / power / v2v_succ / v2i_mode / lam`.

## 3. Run a batch (waves of ≤6, detached)
Nets are tiny (256/128) → 6 concurrent runs are fine (matches the sister repo's 6×).
```
mkdir -p logs
for S in 2 3 4 5 6 7; do
  DQN_CKPT_SUBDIR=tmp/dqn_hard_s${S}_re5 "$PY" Main.py --episodes 600 --seed $S --renew_every 5 \
    --mode hard --cost_source raw --tau 8 --eps 0.10 --lam_max 5 --lam_warmup 150 \
    --out_tag re5 --out_subdir Step5_RCPO > logs/hard_s${S}.out 2>&1 &
done
wait     # then start the next wave
```

## 4. Plot
```
"$PY" -c "import matplotlib" 2>/dev/null || "$PY" -m pip install -q matplotlib
for d in model/Step5_RCPO/dqn_*; do "$PY" model/plot_results.py "$d"; done
```
Writes `*.png` + `metrics.txt` into each run dir (canonical violation, AoI, lambda,
mode_v2i, reward, …). `metrics.txt` holds the headline numbers (`canonical.worst_platoon`,
`canonical.net_mean`, `mean_AoI_last100`, `mean_power_dBm`, …).

## 5. Commit (local only — human pushes)
```
cd /d/Jinnan/CMDP/AoI-V2X-DQN
git add 1-IndependentDQN/model/<study folder>
git status --short        # verify ONLY intended .mat/metrics.txt (no .png, tmp/, logs/)
git commit -m "DQN <study>: <N> runs"
# git push   <-- HUMAN ONLY
```
Report: repo status + commit hash + a compact per-run table (`canonical.worst_platoon` /
`net_mean` / `mean_AoI_last100`) from each `metrics.txt`. **Report numbers, don't draw
conclusions** — the operator cross-checks against raw `.mat`.

## 6. Key flags (full list: `Main.py` argparse)
| flag | values / default | meaning |
|---|---|---|
| `--mode` | soft / hard | `−AoI/20` penalty vs per-platoon CMDP constraint |
| `--renew_every` | 1 / 5 / 20 (re5 canonical) | mobility cadence (geometry renew interval) |
| `--cost_source` | raw / critic | RCPO reward-fold (default) vs separate cost critic |
| `--tau --eps` | 8 / 0.10 | AoI threshold τ, target violation ε |
| `--lam_max --lam_warmup --dual` | 5 / 150 / pid | dual clip / warmup episodes / rule |
| `--lam_scope` | per_platoon / global_mean / global_max | per-platoon vs global λ (ablation #3) |
| `--aoi_pen_type --aoi_pen_w` | raw / indicator, 5 | fixed-weight threshold penalty (ablation #4) |
| `--buffer` | 50000 | replay size (memory sweep) |
| `--n_RB --n_veh --n_power` | 3 / 20 / 30 | scenario size / power levels (≤8 platoons) |
| `--episodes --seed --fc1 --fc2 --out_tag --out_subdir` | — | training / output |

## 7. Troubleshooting
| symptom | cause → fix |
|---|---|
| hard: λ all pinned at `lam_max`, AoI blows up | value-scale domination → use `--cost_source raw`; lower `--lam_max`; ensure `--lam_warmup`>0 |
| soft shows ~0 violation for all platoons | τ too loose / cadence not binding → re5 with τ=8 binds |
| `import matplotlib` fails | `"$PY" -m pip install -q matplotlib` |
| checkpoints overwritten across parallel runs | give each run a unique `DQN_CKPT_SUBDIR` |
| numpy `np.int`/`np.bool` errors | not present here (code is numpy-2 clean); confirm the venv |
