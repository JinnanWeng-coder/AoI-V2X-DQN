#!/usr/bin/env bash
# Canonical DQN hard/RCPO replication at the original mobility cadence.
#
# Formal configuration:
#   renew_every=20, episodes=600, seeds=2..7, per-platoon PID dual,
#   raw-cost RCPO, tau=8, eps=0.10, lam_max=5, lam_warmup=150.
#
# Run from Git Bash on the GPU host. The script is deliberately idempotent:
# complete result directories are audited and reused, while incomplete result
# directories are never overwritten.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

PY_BIN="${PY:-/d/Jinnan/CMDP/AoI-V2X-CMDP/.venv/Scripts/python.exe}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
MONITOR_SECONDS="${MONITOR_SECONDS:-300}"
SKIP_SMOKE=0
SMOKE_ONLY=0
DRY_RUN=0

SEEDS=(2 3 4 5 6 7)
STUDY_NAME="S04_cadence_validation"
OUT_ROOT="$SCRIPT_DIR/experiments/runs/rcpo_per_platoon"
STUDY_DIR="$SCRIPT_DIR/experiments/studies/$STUDY_NAME"
LOG_ROOT="$REPO_ROOT/logs/$STUDY_NAME"
PIDS_FILE="$LOG_ROOT/pids.tsv"
REQUIRED_ANCESTOR="d7c7718"

usage() {
    cat <<'EOF'
Usage: bash run_rcpo_re20.sh [options]

Options:
  --python PATH          Python executable (default: remote CMDP venv)
  --max-parallel N       Concurrent training runs, 1..6 (default: 6)
  --monitor-seconds N    Compact status interval in seconds (default: 300)
  --skip-smoke           Skip the end-to-end smoke test (for a reviewed retry)
  --smoke-only           Run the smoke test and stop
  --dry-run              Print the locked configuration without executing it
  -h, --help             Show this help

Environment equivalents: PY, MAX_PARALLEL, MONITOR_SECONDS.
EOF
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

while (($# > 0)); do
    case "$1" in
        --python)
            (($# >= 2)) || die "--python requires a path"
            PY_BIN="$2"
            shift 2
            ;;
        --max-parallel)
            (($# >= 2)) || die "--max-parallel requires an integer"
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --monitor-seconds)
            (($# >= 2)) || die "--monitor-seconds requires an integer"
            MONITOR_SECONDS="$2"
            shift 2
            ;;
        --skip-smoke)
            SKIP_SMOKE=1
            shift
            ;;
        --smoke-only)
            SMOKE_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

is_uint "$MAX_PARALLEL" || die "MAX_PARALLEL must be an integer"
((MAX_PARALLEL >= 1 && MAX_PARALLEL <= 6)) || die "MAX_PARALLEL must be in 1..6"
is_uint "$MONITOR_SECONDS" || die "MONITOR_SECONDS must be an integer"
((MONITOR_SECONDS >= 10)) || die "MONITOR_SECONDS must be at least 10"
((SKIP_SMOKE == 0 || SMOKE_ONLY == 0)) || die "--skip-smoke and --smoke-only are mutually exclusive"

print_locked_command() {
    cat <<EOF
PYTHONUNBUFFERED=1 DQN_CKPT_SUBDIR=tmp/dqn_hard_s<SEED>_re20 \\
  "$PY_BIN" Main.py --episodes 600 --seed <SEED> --renew_every 20 \\
  --mode hard --cost_source raw --dual pid --tau 8 --eps 0.10 \\
  --lam_max 5 --lam_warmup 150 --lam_scope per_platoon \\
  --kp 1.0 --ki 1.0 --kd 0.5 \\
  --out_subdir rcpo_per_platoon --run_name dqn_rcpo_raw_per_pid_lmax05_re20_seed<SEED>
EOF
}

if ((DRY_RUN == 1)); then
    log "dry run; source commit=$SOURCE_COMMIT"
    log "script_dir=$SCRIPT_DIR"
    log "python=$PY_BIN max_parallel=$MAX_PARALLEL monitor_seconds=$MONITOR_SECONDS"
    log "seeds=$(IFS=,; printf '%s' "${SEEDS[*]}") output=experiments/runs/rcpo_per_platoon"
    print_locked_command
    exit 0
fi

ACTIVE_PIDS=()

remove_active_pid() {
    local target="$1"
    local pid
    local -a kept=()
    for pid in "${ACTIVE_PIDS[@]}"; do
        [[ "$pid" == "$target" ]] || kept+=("$pid")
    done
    ACTIVE_PIDS=("${kept[@]}")
}

terminate_children() {
    local pid
    for pid in "${ACTIVE_PIDS[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "terminating task-owned child pid=$pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${ACTIVE_PIDS[@]}"; do
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done
    ACTIVE_PIDS=()
}

on_signal() {
    log "received a termination signal"
    terminate_children
    exit 130
}

on_exit() {
    local rc=$?
    if ((rc != 0)) && ((${#ACTIVE_PIDS[@]} > 0)); then
        terminate_children
    fi
}

trap on_signal INT TERM HUP
trap on_exit EXIT

cd "$SCRIPT_DIR"

[[ -f Main.py ]] || die "Main.py is missing from $SCRIPT_DIR"
[[ -f analysis/plot_run.py ]] || die "analysis/plot_run.py is missing"
git -C "$REPO_ROOT" diff --quiet -- || die "tracked unstaged changes exist; refusing a formal run"
git -C "$REPO_ROOT" diff --cached --quiet -- || die "staged changes exist; refusing a formal run"
git -C "$REPO_ROOT" merge-base --is-ancestor "$REQUIRED_ANCESTOR" HEAD \
    || die "HEAD does not contain required ancestor $REQUIRED_ANCESTOR"

mkdir -p "$LOG_ROOT"

if [[ -s "$PIDS_FILE" ]]; then
    while IFS=$'\t' read -r old_seed old_pid _old_log; do
        [[ "$old_seed" == "seed" ]] && continue
        if is_uint "${old_pid:-}" && kill -0 "$old_pid" 2>/dev/null; then
            die "a prior task-owned process may still be alive: seed=$old_seed pid=$old_pid"
        fi
    done <"$PIDS_FILE"
fi

log "source_commit=$SOURCE_COMMIT"
log "python=$PY_BIN"
"$PY_BIN" - <<'PY'
import sys
import numpy
import scipy
import torch

print("python=" + sys.version.replace("\n", " "))
print("numpy=" + numpy.__version__)
print("scipy=" + scipy.__version__)
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the formal run")
print("gpu=" + torch.cuda.get_device_name(0))
PY

validate_run() {
    local run_dir="$1"
    "$PY_BIN" - "$run_dir" <<'PY'
import sys
from pathlib import Path

import numpy as np
import scipy.io

run_dir = Path(sys.argv[1])
required = {
    "reward_t1.mat": "reward_t1",
    "reward_t2.mat": "reward_t2",
    "AoI.mat": "AoI",
    "viol_rate.mat": "viol_rate",
    "AoI_evolution.mat": "AoI_evolution",
    "demand.mat": "demand",
    "V2I.mat": "V2I",
    "V2V.mat": "V2V",
    "power.mat": "power",
    "epsilon.mat": "epsilon",
    "mode_v2i.mat": "mode_v2i",
    "lambda.mat": "lambda",
}
expected_shapes = {
    "viol_rate": (5, 600),
    "lambda": (5, 600),
    "AoI_evolution": (5, 100, 100),
    "power": (5, 100, 100),
}

if not run_dir.is_dir():
    raise SystemExit(f"missing result directory: {run_dir}")

loaded = {}
for filename, key in required.items():
    path = run_dir / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty: {path}")
    mat = scipy.io.loadmat(path)
    if key not in mat:
        raise SystemExit(f"missing key {key!r} in {path}")
    arr = np.asarray(mat[key])
    if not np.isfinite(arr).all():
        raise SystemExit(f"NaN/Inf in {path}")
    loaded[key] = arr

for key, expected in expected_shapes.items():
    if loaded[key].shape != expected:
        raise SystemExit(f"bad shape for {key}: {loaded[key].shape}, expected {expected}")

print(f"valid: {run_dir}")
PY
}

missing_seeds=()
for seed in "${SEEDS[@]}"; do
    run_dir="$OUT_ROOT/dqn_rcpo_raw_per_pid_lmax05_re20_seed$(printf '%02d' "$seed")"
    if [[ -d "$run_dir" ]]; then
        if audit_message="$(validate_run "$run_dir" 2>&1)"; then
            log "reuse seed=$seed; $audit_message"
        else
            printf '%s\n' "$audit_message" >&2
            die "seed=$seed has an incomplete existing result; refusing to overwrite $run_dir"
        fi
    else
        missing_seeds+=("$seed")
    fi
done

run_smoke() {
    local stamp smoke_root smoke_log expected_label run_name
    stamp="$(date '+%Y%m%d_%H%M%S')_$$"
    smoke_root="scratch/smoke/Smoke_RCPO_re20/$stamp"
    run_name="dqn_rcpo_raw_per_pid_lmax05_re20_smoke_seed02"
    smoke_log="$LOG_ROOT/smoke_${stamp}.out"
    expected_label="$smoke_root/$run_name"

    log "starting smoke test; log=$smoke_log"
    PYTHONUNBUFFERED=1 DQN_CKPT_SUBDIR="tmp/codex_smoke_rcpo_re20_$stamp" \
        "$PY_BIN" Main.py \
        --smoke \
        --seed 2 \
        --renew_every 20 \
        --mode hard \
        --cost_source raw \
        --dual pid \
        --tau 8 \
        --eps 0.10 \
        --lam_max 5 \
        --lam_warmup 150 \
        --lam_scope per_platoon \
        --kp 1.0 --ki 1.0 --kd 0.5 \
        --output_root "$smoke_root" \
        --run_name "$run_name" \
        >"$smoke_log" 2>&1

    if ! grep -Fq "done. run_dir=$expected_label" "$smoke_log"; then
        tail -n 30 "$smoke_log" >&2 || true
        die "smoke test did not reach its completion marker"
    fi
    if grep -Eqi 'Traceback|CUDA out of memory|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$smoke_log"; then
        tail -n 30 "$smoke_log" >&2 || true
        die "smoke test log contains an error marker"
    fi
    log "smoke test passed"
}

if ((SKIP_SMOKE == 0)) && ((${#missing_seeds[@]} > 0 || SMOKE_ONLY == 1)); then
    run_smoke
fi
if ((SMOKE_ONLY == 1)); then
    log "smoke-only run complete"
    exit 0
fi

LAST_PID=""
LAST_LOG=""

launch_seed() {
    local seed="$1"
    local log_file="$LOG_ROOT/hard_s${seed}.out"
    local checkpoint="tmp/dqn_hard_s${seed}_re20"
    local run_name="dqn_rcpo_raw_per_pid_lmax05_re20_seed$(printf '%02d' "$seed")"

    if [[ -e "$log_file" ]]; then
        mv -- "$log_file" "$log_file.previous_$(date '+%Y%m%d_%H%M%S')"
    fi

    log "launching seed=$seed log=$log_file checkpoint=$checkpoint"
    (
        export PYTHONUNBUFFERED=1
        export DQN_CKPT_SUBDIR="$checkpoint"
        exec "$PY_BIN" Main.py \
            --episodes 600 \
            --seed "$seed" \
            --renew_every 20 \
            --mode hard \
            --cost_source raw \
            --dual pid \
            --tau 8 \
            --eps 0.10 \
            --lam_max 5 \
            --lam_warmup 150 \
            --lam_scope per_platoon \
            --kp 1.0 --ki 1.0 --kd 0.5 \
            --out_subdir rcpo_per_platoon \
            --run_name "$run_name"
    ) >"$log_file" 2>&1 &

    LAST_PID=$!
    LAST_LOG="$log_file"
    ACTIVE_PIDS+=("$LAST_PID")
    printf '%s\t%s\t%s\n' "$seed" "$LAST_PID" "$log_file" >>"$PIDS_FILE"
}

run_wave() {
    local -a wave_seeds=("$@")
    local -a wave_pids=()
    local -a wave_logs=()
    local -a wave_done=()
    local seed pid log_file episode status_line expected_label
    local i running rc wave_failed=0

    for seed in "${wave_seeds[@]}"; do
        launch_seed "$seed"
        wave_pids+=("$LAST_PID")
        wave_logs+=("$LAST_LOG")
        wave_done+=(0)
    done

    while true; do
        running=0
        local -a status_parts=()
        for i in "${!wave_seeds[@]}"; do
            seed="${wave_seeds[$i]}"
            pid="${wave_pids[$i]}"
            log_file="${wave_logs[$i]}"

            if [[ "${wave_done[$i]}" == 1 ]]; then
                status_parts+=("s${seed}=done")
                continue
            fi

            if kill -0 "$pid" 2>/dev/null; then
                running=$((running + 1))
                episode="$(grep -oE '\[dqn hard ep [0-9]+\]' "$log_file" 2>/dev/null \
                    | tail -n 1 | grep -oE '[0-9]+' || true)"
                status_parts+=("s${seed}=ep${episode:-starting}")
                continue
            fi

            if wait "$pid"; then
                rc=0
            else
                rc=$?
            fi
            remove_active_pid "$pid"
            wave_done[$i]=1
            expected_label="experiments/runs/rcpo_per_platoon/dqn_rcpo_raw_per_pid_lmax05_re20_seed$(printf '%02d' "$seed")"

            if ((rc != 0)) || ! grep -Fq "done. run_dir=$expected_label" "$log_file"; then
                log "seed=$seed failed rc=$rc; last log lines follow"
                tail -n 25 "$log_file" >&2 || true
                status_parts+=("s${seed}=FAILED")
                wave_failed=1
            else
                status_parts+=("s${seed}=done")
                log "seed=$seed reached completion marker"
            fi
        done

        status_line="$(IFS=' '; printf '%s' "${status_parts[*]}")"
        log "monitor $status_line"
        ((running == 0)) && break
        sleep "$MONITOR_SECONDS"
    done

    return "$wave_failed"
}

if ((${#missing_seeds[@]} > 0)); then
    printf 'seed\tpid\tlog\n' >"$PIDS_FILE"
    overall_failed=0
    for ((start = 0; start < ${#missing_seeds[@]}; start += MAX_PARALLEL)); do
        wave=("${missing_seeds[@]:start:MAX_PARALLEL}")
        log "starting wave seeds=$(IFS=,; printf '%s' "${wave[*]}")"
        if ! run_wave "${wave[@]}"; then
            overall_failed=1
        fi
    done
    ((overall_failed == 0)) || die "one or more formal runs failed; inspect $LOG_ROOT"
else
    log "all six formal results already exist and passed the pre-run audit"
fi

log "auditing all formal results"
for seed in "${SEEDS[@]}"; do
    validate_run "$OUT_ROOT/dqn_rcpo_raw_per_pid_lmax05_re20_seed$(printf '%02d' "$seed")"
done

if ! "$PY_BIN" -c "import matplotlib" >/dev/null 2>&1; then
    log "matplotlib is missing; installing the runbook-approved plotting dependency"
    "$PY_BIN" -m pip install -q matplotlib
fi

for seed in "${SEEDS[@]}"; do
    log "generating plots for seed=$seed"
    "$PY_BIN" analysis/plot_run.py "$OUT_ROOT/dqn_rcpo_raw_per_pid_lmax05_re20_seed$(printf '%02d' "$seed")"
done

log "writing linear-power summary and run manifest when they do not already exist"
mkdir -p "$STUDY_DIR"
summary_path="$STUDY_DIR/summary.tsv"
manifest_path="$STUDY_DIR/rcpo_re20_run_manifest.txt"
if [[ ! -e "$summary_path" && ! -e "$manifest_path" ]]; then
"$PY_BIN" - "$OUT_ROOT" "$SOURCE_COMMIT" "$STUDY_DIR" <<'PY'
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

root = Path(sys.argv[1])
source_commit = sys.argv[2]
study_dir = Path(sys.argv[3])
rows = []
for seed in range(2, 8):
    run = root / f"dqn_rcpo_raw_per_pid_lmax05_re20_seed{seed:02d}"
    ev = np.asarray(scipy.io.loadmat(run / "AoI_evolution.mat")["AoI_evolution"], dtype=np.float64)
    aoi = np.asarray(scipy.io.loadmat(run / "AoI.mat")["AoI"], dtype=np.float64)
    power_dbm = np.asarray(scipy.io.loadmat(run / "power.mat")["power"], dtype=np.float64)
    viol_pp = (ev > 8.0).mean(axis=(1, 2))
    rows.append(
        (
            seed,
            float(viol_pp.max()),
            float(viol_pp.mean()),
            float(aoi[:, -100:].mean()),
            float(power_dbm.mean()),
            float(np.power(10.0, power_dbm / 10.0).mean()),
        )
    )

header = (
    "seed\tcanonical_worst_platoon\tcanonical_net_mean\t"
    "mean_AoI_last100\tmean_power_dBm\tmean_power_mW"
)
lines = [header]
for row in rows:
    lines.append(
        f"{row[0]}\t{row[1]:.6f}\t{row[2]:.6f}\t"
        f"{row[3]:.6f}\t{row[4]:.6f}\t{row[5]:.6f}"
    )
(study_dir / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest = [
    "study=S04_cadence_validation",
    f"created_at={dt.datetime.now().astimezone().isoformat()}",
    f"source_commit={source_commit}",
    "episodes=600",
    "seeds=2,3,4,5,6,7",
    "renew_every=20",
    "mode=hard",
    "cost_source=raw",
    "dual=pid",
    "tau=8",
    "eps=0.10",
    "lam_max=5",
    "lam_warmup=150",
    "lam_scope=per_platoon",
    "kp=1.0",
    "ki=1.0",
    "kd=0.5",
    f"python={sys.version.replace(chr(10), ' ')}",
    f"torch={torch.__version__}",
    f"cuda_available={torch.cuda.is_available()}",
    f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}",
    "power_note=mean_power_mW is mean(10**(power_dBm/10)); do not percentage-average dBm",
]
(study_dir / "rcpo_re20_run_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

print("\n".join(lines))
PY
elif [[ -e "$summary_path" && -e "$manifest_path" ]]; then
    log "summary and manifest already exist; preserving them without overwrite"
else
    die "partial study metadata: expected both summary.tsv and rcpo_re20_run_manifest.txt; summary=$summary_path manifest=$manifest_path"
fi

log "formal experiment complete"
log "results=$OUT_ROOT"
log "summary=$STUDY_DIR/summary.tsv"
log "logs=$LOG_ROOT"
log "next: stage only experiments/runs/rcpo_per_platoon and experiments/studies/$STUDY_NAME; never push from the GPU host"
