#!/usr/bin/env bash
# Run the missing renew_every=20 constraint-design ablations.
#
# Formal matrix (36 runs total):
#   methods: RCPO global-mean, RCPO global-max, fixed indicator w={2,5,10,20}
#   seeds: 2..7; episodes=600; renew_every=20; buffer=50000; tau=8
#
# The script is restart-safe: complete runs are audited and reused, while an
# existing incomplete run directory is never overwritten. It never retries a
# failed training process. After all 36 runs pass MAT audit, it writes S07 study
# metadata and registers the runs atomically in experiments/registry.tsv.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DQN_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
REPO_ROOT="$(git -C "$DQN_DIR" rev-parse --show-toplevel)"
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

PY_BIN="${PY:-/d/Jinnan/CMDP/AoI-V2X-CMDP/.venv/Scripts/python.exe}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
MONITOR_SECONDS="${MONITOR_SECONDS:-300}"
SKIP_SMOKE=0
SMOKE_ONLY=0
DRY_RUN=0

SEEDS=(2 3 4 5 6 7)
METHODS=(global_mean global_max fixed_w02 fixed_w05 fixed_w10 fixed_w20)
STUDY_NAME="S07_re20_constraint_design"
STUDY_DIR="$DQN_DIR/experiments/studies/$STUDY_NAME"
LOG_ROOT="$REPO_ROOT/logs/$STUDY_NAME"
PIDS_FILE="$LOG_ROOT/pids.tsv"
REGISTRY="$DQN_DIR/experiments/registry.tsv"
REQUIRED_ANCESTOR="f64f492"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_re20_constraint_ablations.sh [options]

Options:
  --python PATH          Python executable (default: remote CMDP venv)
  --max-parallel N       Concurrent training runs, 1..6 (default: 6)
  --monitor-seconds N    Compact status interval in seconds (default: 300)
  --skip-smoke           Skip six configuration smoke tests
  --smoke-only           Run the six smoke tests and stop
  --dry-run              Print the complete locked 36-run matrix and stop
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

configure_job() {
    local method="$1"
    local seed="$2"
    local seed02
    seed02="$(printf '%02d' "$seed")"

    JOB_METHOD="$method"
    JOB_SEED="$seed"
    JOB_WEIGHT="n/a"
    case "$method" in
        global_mean)
            JOB_MODE="hard"
            JOB_SCOPE="global_mean"
            JOB_OUT_SUBDIR="rcpo_global_mean"
            JOB_RUN_NAME="dqn_rcpo_raw_global_mean_pid_lmax05_re20_seed${seed02}"
            ;;
        global_max)
            JOB_MODE="hard"
            JOB_SCOPE="global_max"
            JOB_OUT_SUBDIR="rcpo_global_max"
            JOB_RUN_NAME="dqn_rcpo_raw_global_max_pid_lmax05_re20_seed${seed02}"
            ;;
        fixed_w02|fixed_w05|fixed_w10|fixed_w20)
            JOB_MODE="soft"
            JOB_SCOPE="n/a"
            JOB_WEIGHT="${method#fixed_w}"
            JOB_WEIGHT="$((10#$JOB_WEIGHT))"
            JOB_OUT_SUBDIR="fixed_indicator"
            JOB_RUN_NAME="dqn_fixed_ind_w$(printf '%02d' "$JOB_WEIGHT")_re20_seed${seed02}"
            ;;
        *)
            die "unsupported method: $method"
            ;;
    esac
    JOB_ID="${method}_seed${seed02}"
    JOB_RUN_DIR="$DQN_DIR/experiments/runs/$JOB_OUT_SUBDIR/$JOB_RUN_NAME"
    JOB_LABEL="experiments/runs/$JOB_OUT_SUBDIR/$JOB_RUN_NAME"
    JOB_CHECKPOINT="tmp/re20_${method}_seed${seed02}"
}

print_job_command() {
    local method="$1"
    local seed="$2"
    configure_job "$method" "$seed"
    if [[ "$JOB_MODE" == "hard" ]]; then
        printf 'PYTHONUNBUFFERED=1 DQN_CKPT_SUBDIR=%q %q Main.py --episodes 600 --seed %s --renew_every 20 --buffer 50000 --mode hard --cost_source raw --dual pid --tau 8 --eps 0.10 --lam_max 5 --lam_warmup 150 --lam_scope %s --kp 1.0 --ki 1.0 --kd 0.5 --out_subdir %s --run_name %s\n' \
            "$JOB_CHECKPOINT" "$PY_BIN" "$seed" "$JOB_SCOPE" "$JOB_OUT_SUBDIR" "$JOB_RUN_NAME"
    else
        printf 'PYTHONUNBUFFERED=1 DQN_CKPT_SUBDIR=%q %q Main.py --episodes 600 --seed %s --renew_every 20 --buffer 50000 --mode soft --tau 8 --aoi_pen_type indicator --aoi_pen_w %s --out_subdir %s --run_name %s\n' \
            "$JOB_CHECKPOINT" "$PY_BIN" "$seed" "$JOB_WEIGHT" "$JOB_OUT_SUBDIR" "$JOB_RUN_NAME"
    fi
}

if ((DRY_RUN == 1)); then
    log "dry run; source_commit=$SOURCE_COMMIT"
    log "python=$PY_BIN max_parallel=$MAX_PARALLEL monitor_seconds=$MONITOR_SECONDS"
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            print_job_command "$method" "$seed"
        done
    done
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

cd "$DQN_DIR"
[[ -f Main.py ]] || die "Main.py is missing from $DQN_DIR"
[[ -f "$REGISTRY" ]] || die "registry is missing: $REGISTRY"
git -C "$REPO_ROOT" diff --quiet -- || die "tracked unstaged changes exist; refusing a formal run"
git -C "$REPO_ROOT" diff --cached --quiet -- || die "staged changes exist; refusing a formal run"
git -C "$REPO_ROOT" merge-base --is-ancestor "$REQUIRED_ANCESTOR" HEAD \
    || die "HEAD does not contain required ancestor $REQUIRED_ANCESTOR"

mkdir -p "$LOG_ROOT"
if [[ -s "$PIDS_FILE" ]]; then
    while IFS=$'\t' read -r old_job old_pid _old_log; do
        [[ "$old_job" == "job" ]] && continue
        if is_uint "${old_pid:-}" && kill -0 "$old_pid" 2>/dev/null; then
            die "a prior task-owned process may still be alive: job=$old_job pid=$old_pid"
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

run_main() {
    local method="$1"
    local seed="$2"
    local output_root="$3"
    local run_name="$4"
    local smoke_flag="$5"
    local -a common_args=(
        Main.py --episodes 600 --seed "$seed" --renew_every 20 --buffer 50000 --tau 8
        --output_root "$output_root" --out_subdir "$JOB_OUT_SUBDIR" --run_name "$run_name"
    )
    [[ "$smoke_flag" == "1" ]] && common_args+=(--smoke)

    if [[ "$JOB_MODE" == "hard" ]]; then
        "$PY_BIN" "${common_args[@]}" \
            --mode hard --cost_source raw --dual pid --eps 0.10 \
            --lam_max 5 --lam_warmup 150 --lam_scope "$JOB_SCOPE" \
            --kp 1.0 --ki 1.0 --kd 0.5
    else
        "$PY_BIN" "${common_args[@]}" \
            --mode soft --aoi_pen_type indicator --aoi_pen_w "$JOB_WEIGHT"
    fi
}

missing_jobs=()
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        configure_job "$method" "$seed"
        if [[ -d "$JOB_RUN_DIR" ]]; then
            if audit_message="$(validate_run "$JOB_RUN_DIR" 2>&1)"; then
                log "reuse job=$JOB_ID; $audit_message"
            else
                printf '%s\n' "$audit_message" >&2
                die "job=$JOB_ID has an incomplete existing result; refusing to overwrite $JOB_RUN_DIR"
            fi
        else
            missing_jobs+=("$method:$seed")
        fi
    done
done

run_smokes() {
    local stamp method smoke_root smoke_name smoke_log expected_label
    stamp="$(date '+%Y%m%d_%H%M%S')_$$"
    smoke_root="scratch/smoke/$STUDY_NAME/$stamp"
    for method in "${METHODS[@]}"; do
        configure_job "$method" 2
        smoke_name="${JOB_RUN_NAME}_smoke"
        smoke_log="$LOG_ROOT/smoke_${method}_${stamp}.out"
        expected_label="$smoke_root/$JOB_OUT_SUBDIR/$smoke_name"
        log "starting smoke method=$method log=$smoke_log"
        PYTHONUNBUFFERED=1 DQN_CKPT_SUBDIR="tmp/smoke_${STUDY_NAME}_${method}_$stamp" \
            run_main "$method" 2 "$smoke_root" "$smoke_name" 1 >"$smoke_log" 2>&1
        if ! grep -Fq "done. run_dir=$expected_label" "$smoke_log"; then
            tail -n 30 "$smoke_log" >&2 || true
            die "smoke method=$method did not reach its completion marker"
        fi
        if grep -Eqi 'Traceback|CUDA out of memory|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$smoke_log"; then
            tail -n 30 "$smoke_log" >&2 || true
            die "smoke method=$method contains an error marker"
        fi
        log "smoke method=$method passed"
    done
}

if ((SKIP_SMOKE == 0)) && ((${#missing_jobs[@]} > 0 || SMOKE_ONLY == 1)); then
    run_smokes
fi
if ((SMOKE_ONLY == 1)); then
    log "all six smoke tests passed"
    exit 0
fi

LAST_PID=""
LAST_LOG=""

launch_job() {
    local spec="$1"
    local method="${spec%%:*}"
    local seed="${spec##*:}"
    configure_job "$method" "$seed"
    local log_file="$LOG_ROOT/${JOB_ID}.out"
    if [[ -e "$log_file" ]]; then
        mv -- "$log_file" "$log_file.previous_$(date '+%Y%m%d_%H%M%S')"
    fi
    log "launching job=$JOB_ID log=$log_file checkpoint=$JOB_CHECKPOINT"
    (
        export PYTHONUNBUFFERED=1
        export DQN_CKPT_SUBDIR="$JOB_CHECKPOINT"
        run_main "$method" "$seed" "experiments/runs" "$JOB_RUN_NAME" 0
    ) >"$log_file" 2>&1 &
    LAST_PID=$!
    LAST_LOG="$log_file"
    ACTIVE_PIDS+=("$LAST_PID")
    printf '%s\t%s\t%s\n' "$JOB_ID" "$LAST_PID" "$log_file" >>"$PIDS_FILE"
}

run_wave() {
    local -a wave_specs=("$@")
    local -a wave_pids=()
    local -a wave_logs=()
    local -a wave_done=()
    local i spec method seed pid log_file episode expected_label job_id
    local running rc wave_failed=0

    for spec in "${wave_specs[@]}"; do
        launch_job "$spec"
        wave_pids+=("$LAST_PID")
        wave_logs+=("$LAST_LOG")
        wave_done+=(0)
    done

    while true; do
        running=0
        local -a status_parts=()
        for i in "${!wave_specs[@]}"; do
            spec="${wave_specs[$i]}"
            method="${spec%%:*}"
            seed="${spec##*:}"
            pid="${wave_pids[$i]}"
            log_file="${wave_logs[$i]}"
            configure_job "$method" "$seed"
            job_id="$JOB_ID"
            expected_label="$JOB_LABEL"

            if [[ "${wave_done[$i]}" == 1 ]]; then
                status_parts+=("$job_id=done")
                continue
            fi
            if kill -0 "$pid" 2>/dev/null; then
                running=$((running + 1))
                episode="$(grep -oE '\[dqn (hard|soft) ep [0-9]+\]' "$log_file" 2>/dev/null \
                    | tail -n 1 | grep -oE '[0-9]+' || true)"
                status_parts+=("$job_id=ep${episode:-starting}")
                continue
            fi

            if wait "$pid"; then rc=0; else rc=$?; fi
            remove_active_pid "$pid"
            wave_done[$i]=1
            if ((rc != 0)) || ! grep -Fq "done. run_dir=$expected_label" "$log_file"; then
                log "job=$job_id failed rc=$rc; last log lines follow"
                tail -n 25 "$log_file" >&2 || true
                status_parts+=("$job_id=FAILED")
                wave_failed=1
            else
                status_parts+=("$job_id=done")
                log "job=$job_id reached completion marker"
            fi
        done
        log "monitor $(IFS=' '; printf '%s' "${status_parts[*]}")"
        ((running == 0)) && break
        sleep "$MONITOR_SECONDS"
    done
    return "$wave_failed"
}

if ((${#missing_jobs[@]} > 0)); then
    printf 'job\tpid\tlog\n' >"$PIDS_FILE"
    overall_failed=0
    for ((start = 0; start < ${#missing_jobs[@]}; start += MAX_PARALLEL)); do
        wave=("${missing_jobs[@]:start:MAX_PARALLEL}")
        log "starting wave $(IFS=,; printf '%s' "${wave[*]}")"
        if ! run_wave "${wave[@]}"; then
            overall_failed=1
        fi
    done
    ((overall_failed == 0)) || die "one or more formal runs failed; inspect $LOG_ROOT; no retry was attempted"
else
    log "all 36 formal results already exist and passed the pre-run audit"
fi

log "auditing all 36 formal results"
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        configure_job "$method" "$seed"
        validate_run "$JOB_RUN_DIR"
    done
done

log "writing S07 metadata and registering audited runs"
mkdir -p "$STUDY_DIR"
"$PY_BIN" - "$DQN_DIR" "$SOURCE_COMMIT" "$STUDY_DIR" "$REGISTRY" <<'PY'
import csv
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

dqn_dir = Path(sys.argv[1])
source_commit = sys.argv[2]
study_dir = Path(sys.argv[3])
registry_path = Path(sys.argv[4])
seeds = range(2, 8)
configs = [
    ("A10", "global_mean", "rcpo_global_mean", "ablation_global_mean", None),
    ("A20", "global_max", "rcpo_global_max", "ablation_global_max", None),
    ("A32", "fixed_w02", "fixed_indicator", "ablation_fixed_w02", 2),
    ("A35", "fixed_w05", "fixed_indicator", "ablation_fixed_w05", 5),
    ("A40", "fixed_w10", "fixed_indicator", "ablation_fixed_w10", 10),
    ("A50", "fixed_w20", "fixed_indicator", "ablation_fixed_w20", 20),
]

def run_name(method, seed):
    if method in {"global_mean", "global_max"}:
        return f"dqn_rcpo_raw_{method}_pid_lmax05_re20_seed{seed:02d}"
    weight = int(method.removeprefix("fixed_w"))
    return f"dqn_fixed_ind_w{weight:02d}_re20_seed{seed:02d}"

summary_rows = []
member_rows = []
registry_rows = []
for arm_id, method, out_subdir, role, weight in configs:
    for seed in seeds:
        name = run_name(method, seed)
        relative = Path("runs") / out_subdir / name
        run = dqn_dir / "experiments" / relative
        ev = np.asarray(scipy.io.loadmat(run / "AoI_evolution.mat")["AoI_evolution"], dtype=np.float64)
        aoi = np.asarray(scipy.io.loadmat(run / "AoI.mat")["AoI"], dtype=np.float64)
        power_dbm = np.asarray(scipy.io.loadmat(run / "power.mat")["power"], dtype=np.float64)
        viol_pp = (ev > 8.0).mean(axis=(1, 2))
        summary_rows.append([
            method, seed, f"{viol_pp.max():.6f}", f"{viol_pp.mean():.6f}",
            f"{aoi[:, -100:].mean():.6f}", f"{power_dbm.mean():.6f}",
            f"{np.power(10.0, power_dbm / 10.0).mean():.6f}",
        ])
        member_rows.append([arm_id, role, seed, f"../../{relative.as_posix()}"])
        if method in {"global_mean", "global_max"}:
            registry_rows.append([
                name, relative.as_posix(), f"rcpo_{method}", "valid", seed, 600, 20, 50000,
                "raw", method, "pid", 8, 0.1, 5, "n/a", source_commit, "n/a",
            ])
        else:
            registry_rows.append([
                name, relative.as_posix(), "fixed_indicator", "valid", seed, 600, 20, 50000,
                "n/a", "n/a", "n/a", 8, "n/a", "n/a", weight, source_commit, "n/a",
            ])

def atomic_tsv(path, header, rows):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temp, path)

atomic_tsv(
    study_dir / "summary.tsv",
    ["method", "seed", "worst_platoon", "net_mean", "mean_AoI_last100", "mean_power_dBm", "mean_power_mW"],
    summary_rows,
)
atomic_tsv(study_dir / "members.tsv", ["arm_id", "role", "seed", "run_path"], member_rows)

manifest = [
    "study=S07_re20_constraint_design",
    f"created_at={dt.datetime.now().astimezone().isoformat()}",
    f"source_commit={source_commit}",
    "episodes=600",
    "seeds=2,3,4,5,6,7",
    "renew_every=20",
    "methods=global_mean,global_max,fixed_w02,fixed_w05,fixed_w10,fixed_w20",
    "global_rcpo=cost_source:raw,dual:pid,tau:8,eps:0.10,lam_max:5,lam_warmup:150,kp:1,ki:1,kd:0.5",
    "fixed_indicator=mode:soft,tau:8,weights:2,5,10,20",
    "buffer=50000",
    f"python={sys.version.replace(chr(10), ' ')}",
    f"torch={torch.__version__}",
    f"cuda_available={torch.cuda.is_available()}",
    f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}",
    "power_note=mean_power_mW is mean(10**(power_dBm/10)); do not percentage-average dBm",
]
(study_dir / "run_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
(study_dir / "README.md").write_text(
    "# S07: re20 constraint design\n\n"
    "Status: complete. Compares RCPO global-mean, RCPO global-max, and fixed "
    "indicator weights 2, 5, 10, and 20 at `renew_every=20`, each with seeds "
    "2--7. See `members.tsv`, `summary.tsv`, and `run_manifest.txt`.\n",
    encoding="utf-8",
)

with registry_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle, delimiter="\t")
    existing = list(reader)
header, old_rows = existing[0], existing[1:]
expected_header = [
    "run_id", "relative_path", "method", "status", "seed", "episodes", "renew_every",
    "buffer_capacity", "cost_source", "lambda_scope", "dual", "tau", "epsilon",
    "lambda_max", "penalty_weight", "source_commit", "legacy_path",
]
if header != expected_header:
    raise SystemExit(f"unexpected registry header: {header}")
by_id = {row[0]: row for row in old_rows}
for row in registry_rows:
    row = [str(value) for value in row]
    if row[0] in by_id and by_id[row[0]] != row:
        raise SystemExit(f"conflicting registry row: {row[0]}")
    by_id.setdefault(row[0], row)
ordered = old_rows + [[str(value) for value in row] for row in registry_rows if str(row[0]) not in {r[0] for r in old_rows}]
atomic_tsv(registry_path, header, ordered)

print(f"registered={len(registry_rows)} summary_rows={len(summary_rows)}")
PY

log "running repository experiment audit"
"$PY_BIN" analysis/audit_results.py
log "formal experiment complete"
log "results=experiments/runs/{rcpo_global_mean,rcpo_global_max,fixed_indicator}"
log "study=experiments/studies/$STUDY_NAME"
log "logs=$LOG_ROOT"
log "next: inspect git status, verify LFS attributes, then commit and push the audited artifacts"
