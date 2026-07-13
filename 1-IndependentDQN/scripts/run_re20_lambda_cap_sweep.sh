#!/usr/bin/env bash
# Minimal lambda-cap sensitivity sweep for per-platoon RCPO at renew_every=20.
#
# Existing baseline: lambda_max=5, seeds 2..7.
# New formal runs:   lambda_max={10,20}, seeds 2..7 (12 runs).
# All other training parameters are locked to the existing re20 per-platoon run.

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
NEW_CAPS=(10 20)
ALL_CAPS=(5 10 20)
STUDY_NAME="S05_lambda_cap"
STUDY_DIR="$DQN_DIR/experiments/studies/$STUDY_NAME"
OUT_ROOT="$DQN_DIR/experiments/runs/rcpo_per_platoon"
LOG_ROOT="$REPO_ROOT/logs/$STUDY_NAME"
PIDS_FILE="$LOG_ROOT/pids.tsv"
REGISTRY="$DQN_DIR/experiments/registry.tsv"
REQUIRED_ANCESTOR="5c1dbbd"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_re20_lambda_cap_sweep.sh [options]

Options:
  --python PATH          Python executable (default: remote CMDP venv)
  --max-parallel N       Concurrent runs, 1..6 (default: 6)
  --monitor-seconds N    Status interval in seconds (default: 300)
  --skip-smoke           Skip cap=10 and cap=20 smoke tests
  --smoke-only           Run the two smoke tests and stop
  --dry-run              Print the locked 12-run matrix and stop
  -h, --help             Show this help

Environment equivalents: PY, MAX_PARALLEL, MONITOR_SECONDS.
EOF
}

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

while (($# > 0)); do
    case "$1" in
        --python)
            (($# >= 2)) || die "--python requires a path"
            PY_BIN="$2"; shift 2 ;;
        --max-parallel)
            (($# >= 2)) || die "--max-parallel requires an integer"
            MAX_PARALLEL="$2"; shift 2 ;;
        --monitor-seconds)
            (($# >= 2)) || die "--monitor-seconds requires an integer"
            MONITOR_SECONDS="$2"; shift 2 ;;
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        --smoke-only) SMOKE_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

is_uint "$MAX_PARALLEL" || die "MAX_PARALLEL must be an integer"
((MAX_PARALLEL >= 1 && MAX_PARALLEL <= 6)) || die "MAX_PARALLEL must be in 1..6"
is_uint "$MONITOR_SECONDS" || die "MONITOR_SECONDS must be an integer"
((MONITOR_SECONDS >= 10)) || die "MONITOR_SECONDS must be at least 10"
((SKIP_SMOKE == 0 || SMOKE_ONLY == 0)) || die "--skip-smoke and --smoke-only are mutually exclusive"

configure_job() {
    local cap="$1" seed="$2" seed02
    seed02="$(printf '%02d' "$seed")"
    JOB_CAP="$cap"
    JOB_SEED="$seed"
    JOB_ID="lmax${cap}_seed${seed02}"
    JOB_RUN_NAME="dqn_rcpo_raw_per_pid_lmax$(printf '%02d' "$cap")_re20_seed${seed02}"
    JOB_RUN_DIR="$OUT_ROOT/$JOB_RUN_NAME"
    JOB_LABEL="experiments/runs/rcpo_per_platoon/$JOB_RUN_NAME"
    JOB_CHECKPOINT="tmp/re20_per_lmax${cap}_seed${seed02}"
}

print_job_command() {
    local cap="$1" seed="$2"
    configure_job "$cap" "$seed"
    printf 'PYTHONUNBUFFERED=1 DQN_CKPT_SUBDIR=%q %q Main.py --episodes 600 --seed %s --renew_every 20 --buffer 50000 --mode hard --cost_source raw --dual pid --tau 8 --eps 0.10 --lam_max %s --lam_warmup 150 --lam_scope per_platoon --kp 1.0 --ki 1.0 --kd 0.5 --out_subdir rcpo_per_platoon --run_name %s\n' \
        "$JOB_CHECKPOINT" "$PY_BIN" "$seed" "$cap" "$JOB_RUN_NAME"
}

if ((DRY_RUN == 1)); then
    log "dry run; source_commit=$SOURCE_COMMIT"
    log "python=$PY_BIN max_parallel=$MAX_PARALLEL monitor_seconds=$MONITOR_SECONDS"
    for cap in "${NEW_CAPS[@]}"; do
        for seed in "${SEEDS[@]}"; do print_job_command "$cap" "$seed"; done
    done
    exit 0
fi

ACTIVE_PIDS=()
remove_active_pid() {
    local target="$1" pid
    local -a kept=()
    for pid in "${ACTIVE_PIDS[@]}"; do [[ "$pid" == "$target" ]] || kept+=("$pid"); done
    ACTIVE_PIDS=("${kept[@]}")
}
terminate_children() {
    local pid
    for pid in "${ACTIVE_PIDS[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
    done
    for pid in "${ACTIVE_PIDS[@]}"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
    ACTIVE_PIDS=()
}
on_signal() { log "received termination signal"; terminate_children; exit 130; }
on_exit() { local rc=$?; if ((rc != 0)) && ((${#ACTIVE_PIDS[@]} > 0)); then terminate_children; fi; }
trap on_signal INT TERM HUP
trap on_exit EXIT

cd "$DQN_DIR"
[[ -f Main.py ]] || die "Main.py is missing from $DQN_DIR"
[[ -f "$REGISTRY" ]] || die "registry is missing"
git -C "$REPO_ROOT" diff --quiet -- || die "tracked unstaged changes exist; refusing formal run"
git -C "$REPO_ROOT" diff --cached --quiet -- || die "staged changes exist; refusing formal run"
git -C "$REPO_ROOT" merge-base --is-ancestor "$REQUIRED_ANCESTOR" HEAD \
    || die "HEAD does not contain required ancestor $REQUIRED_ANCESTOR"

mkdir -p "$LOG_ROOT"
if [[ -s "$PIDS_FILE" ]]; then
    while IFS=$'\t' read -r old_job old_pid _old_log; do
        [[ "$old_job" == "job" ]] && continue
        if is_uint "${old_pid:-}" && kill -0 "$old_pid" 2>/dev/null; then
            die "prior task process may still be alive: job=$old_job pid=$old_pid"
        fi
    done <"$PIDS_FILE"
fi

log "source_commit=$SOURCE_COMMIT"
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
    raise SystemExit("CUDA is required")
print("gpu=" + torch.cuda.get_device_name(0))
PY

validate_run() {
    local run_dir="$1"
    "$PY_BIN" - "$run_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np
import scipy.io

run = Path(sys.argv[1])
required = {
    "reward_t1.mat":"reward_t1", "reward_t2.mat":"reward_t2", "AoI.mat":"AoI",
    "viol_rate.mat":"viol_rate", "AoI_evolution.mat":"AoI_evolution", "demand.mat":"demand",
    "V2I.mat":"V2I", "V2V.mat":"V2V", "power.mat":"power", "epsilon.mat":"epsilon",
    "mode_v2i.mat":"mode_v2i", "lambda.mat":"lambda",
}
shapes = {
    "reward_t1":(5,600), "reward_t2":(5,600), "AoI":(5,600), "viol_rate":(5,600),
    "mode_v2i":(5,600), "lambda":(5,600), "AoI_evolution":(5,100,100),
    "demand":(5,100,100), "V2I":(5,100,100), "V2V":(5,100,100), "power":(5,100,100),
}
if not run.is_dir(): raise SystemExit(f"missing run: {run}")
for filename, key in required.items():
    path = run / filename
    if not path.is_file() or path.stat().st_size == 0: raise SystemExit(f"missing/empty: {path}")
    mat = scipy.io.loadmat(path)
    if key not in mat: raise SystemExit(f"missing key {key} in {path}")
    arr = np.asarray(mat[key])
    if not np.isfinite(arr).all(): raise SystemExit(f"NaN/Inf in {path}")
    if key in shapes and arr.shape != shapes[key]: raise SystemExit(f"bad {key} shape {arr.shape}")
print(f"valid: {run}")
PY
}

# The lambda_max=5 baseline is required for the final three-level study.
for seed in "${SEEDS[@]}"; do
    configure_job 5 "$seed"
    validate_run "$JOB_RUN_DIR" >/dev/null || die "invalid baseline: $JOB_RUN_DIR"
done

missing_jobs=()
for cap in "${NEW_CAPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        configure_job "$cap" "$seed"
        if [[ -d "$JOB_RUN_DIR" ]]; then
            if message="$(validate_run "$JOB_RUN_DIR" 2>&1)"; then
                log "reuse job=$JOB_ID; $message"
            else
                printf '%s\n' "$message" >&2
                die "incomplete existing result; refusing overwrite: $JOB_RUN_DIR"
            fi
        else
            missing_jobs+=("$cap:$seed")
        fi
    done
done

run_main() {
    local cap="$1" seed="$2" output_root="$3" run_name="$4" smoke="$5"
    local -a args=(
        Main.py --episodes 600 --seed "$seed" --renew_every 20 --buffer 50000
        --mode hard --cost_source raw --dual pid --tau 8 --eps 0.10
        --lam_max "$cap" --lam_warmup 150 --lam_scope per_platoon
        --kp 1.0 --ki 1.0 --kd 0.5
        --output_root "$output_root" --out_subdir rcpo_per_platoon --run_name "$run_name"
    )
    [[ "$smoke" == 1 ]] && args+=(--smoke)
    "$PY_BIN" "${args[@]}"
}

run_smokes() {
    local stamp cap smoke_root smoke_name smoke_log expected
    stamp="$(date '+%Y%m%d_%H%M%S')_$$"
    smoke_root="scratch/smoke/$STUDY_NAME/$stamp"
    for cap in "${NEW_CAPS[@]}"; do
        configure_job "$cap" 2
        smoke_name="${JOB_RUN_NAME}_smoke"
        smoke_log="$LOG_ROOT/smoke_lmax${cap}_${stamp}.out"
        expected="$smoke_root/rcpo_per_platoon/$smoke_name"
        log "starting smoke cap=$cap"
        PYTHONUNBUFFERED=1 DQN_CKPT_SUBDIR="tmp/smoke_${STUDY_NAME}_lmax${cap}_$stamp" \
            run_main "$cap" 2 "$smoke_root" "$smoke_name" 1 >"$smoke_log" 2>&1
        grep -Fq "done. run_dir=$expected" "$smoke_log" || { tail -n 30 "$smoke_log" >&2; die "smoke cap=$cap incomplete"; }
        if grep -Eqi 'Traceback|CUDA out of memory|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$smoke_log"; then
            tail -n 30 "$smoke_log" >&2; die "smoke cap=$cap contains error marker"
        fi
        log "smoke cap=$cap passed"
    done
}

if ((SKIP_SMOKE == 0)) && ((${#missing_jobs[@]} > 0 || SMOKE_ONLY == 1)); then run_smokes; fi
if ((SMOKE_ONLY == 1)); then log "both smoke tests passed"; exit 0; fi

LAST_PID=""; LAST_LOG=""
launch_job() {
    local spec="$1" cap="${spec%%:*}" seed="${spec##*:}"
    configure_job "$cap" "$seed"
    local log_file="$LOG_ROOT/${JOB_ID}.out"
    [[ ! -e "$log_file" ]] || mv -- "$log_file" "$log_file.previous_$(date '+%Y%m%d_%H%M%S')"
    log "launching job=$JOB_ID log=$log_file"
    (
        export PYTHONUNBUFFERED=1 DQN_CKPT_SUBDIR="$JOB_CHECKPOINT"
        run_main "$cap" "$seed" "experiments/runs" "$JOB_RUN_NAME" 0
    ) >"$log_file" 2>&1 &
    LAST_PID=$!; LAST_LOG="$log_file"; ACTIVE_PIDS+=("$LAST_PID")
    printf '%s\t%s\t%s\n' "$JOB_ID" "$LAST_PID" "$log_file" >>"$PIDS_FILE"
}

run_wave() {
    local -a specs=("$@") pids=() logs=() done_flags=()
    local spec cap seed i pid log_file episode running rc failed=0 job_id expected
    for spec in "${specs[@]}"; do launch_job "$spec"; pids+=("$LAST_PID"); logs+=("$LAST_LOG"); done_flags+=(0); done
    while true; do
        running=0
        local -a states=()
        for i in "${!specs[@]}"; do
            spec="${specs[$i]}"; cap="${spec%%:*}"; seed="${spec##*:}"
            pid="${pids[$i]}"; log_file="${logs[$i]}"; configure_job "$cap" "$seed"
            job_id="$JOB_ID"; expected="$JOB_LABEL"
            if [[ "${done_flags[$i]}" == 1 ]]; then states+=("$job_id=done"); continue; fi
            if kill -0 "$pid" 2>/dev/null; then
                running=$((running+1))
                episode="$(grep -oE '\[dqn hard ep [0-9]+\]' "$log_file" 2>/dev/null | tail -n1 | grep -oE '[0-9]+' || true)"
                states+=("$job_id=ep${episode:-starting}"); continue
            fi
            if wait "$pid"; then rc=0; else rc=$?; fi
            remove_active_pid "$pid"; done_flags[$i]=1
            if ((rc != 0)) || ! grep -Fq "done. run_dir=$expected" "$log_file"; then
                log "job=$job_id failed rc=$rc"; tail -n25 "$log_file" >&2 || true
                states+=("$job_id=FAILED"); failed=1
            else
                states+=("$job_id=done"); log "job=$job_id reached completion marker"
            fi
        done
        log "monitor $(IFS=' '; printf '%s' "${states[*]}")"
        ((running == 0)) && break
        sleep "$MONITOR_SECONDS"
    done
    return "$failed"
}

if ((${#missing_jobs[@]} > 0)); then
    printf 'job\tpid\tlog\n' >"$PIDS_FILE"
    overall_failed=0
    for ((start=0; start<${#missing_jobs[@]}; start+=MAX_PARALLEL)); do
        wave=("${missing_jobs[@]:start:MAX_PARALLEL}")
        log "starting wave $(IFS=,; printf '%s' "${wave[*]}")"
        run_wave "${wave[@]}" || overall_failed=1
    done
    ((overall_failed == 0)) || die "one or more runs failed; no retry attempted"
else
    log "all 12 new runs already exist and pass audit"
fi

for cap in "${ALL_CAPS[@]}"; do
    for seed in "${SEEDS[@]}"; do configure_job "$cap" "$seed"; validate_run "$JOB_RUN_DIR" >/dev/null; done
done

log "writing S05 metadata and registering 12 new runs"
mkdir -p "$STUDY_DIR"
"$PY_BIN" - "$DQN_DIR" "$SOURCE_COMMIT" "$STUDY_DIR" "$REGISTRY" <<'PY'
import csv, datetime as dt, os, sys
from pathlib import Path
import numpy as np
import scipy.io
import torch

dqn = Path(sys.argv[1]); source = sys.argv[2]; study = Path(sys.argv[3]); registry = Path(sys.argv[4])
caps = (5, 10, 20); seeds = range(2, 8)

def name(cap, seed): return f"dqn_rcpo_raw_per_pid_lmax{cap:02d}_re20_seed{seed:02d}"
def atomic_tsv(path, header, rows):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n"); w.writerow(header); w.writerows(rows)
    os.replace(temp, path)

summary=[]; diagnostics=[]; members=[]; new_registry=[]
for cap in caps:
    for seed in seeds:
        run_name=name(cap,seed); rel=Path("runs/rcpo_per_platoon")/run_name; run=dqn/"experiments"/rel
        ev=np.asarray(scipy.io.loadmat(run/"AoI_evolution.mat")["AoI_evolution"],dtype=float)
        aoi=np.asarray(scipy.io.loadmat(run/"AoI.mat")["AoI"],dtype=float)
        pwr=np.asarray(scipy.io.loadmat(run/"power.mat")["power"],dtype=float)
        lam=np.asarray(scipy.io.loadmat(run/"lambda.mat")["lambda"],dtype=float)
        viol=np.asarray(scipy.io.loadmat(run/"viol_rate.mat")["viol_rate"],dtype=float)
        vpp=(ev>8.0).mean(axis=(1,2))
        summary.append([cap,seed,f"{vpp.max():.6f}",f"{vpp.mean():.6f}",f"{aoi[:,-100:].mean():.6f}",f"{np.power(10,pwr/10).mean():.6f}"])
        members.append([f"L{cap:02d}","baseline" if cap==5 else "cap_sensitivity",seed,f"../../{rel.as_posix()}"])
        for platoon in range(5):
            diagnostics.append([
                cap,seed,platoon+1,f"{viol[platoon,-100:].mean():.6f}",f"{lam[platoon,-1]:.6f}",
                f"{lam[platoon,-100:].mean():.6f}",f"{lam[platoon,-100:].max():.6f}",
                f"{(lam[platoon,-100:]>=cap-1e-6).mean():.3f}",f"{(lam[platoon,-100:]>=0.98*cap).mean():.3f}",
            ])
        if cap in (10,20):
            new_registry.append([
                run_name,rel.as_posix(),"rcpo_per_platoon","valid",seed,600,20,50000,"raw","per_platoon","pid",8,0.1,cap,"n/a",source,"n/a"
            ])

atomic_tsv(study/"summary.tsv",["lambda_max","seed","worst_platoon","net_mean","mean_AoI_last100","mean_power_mW"],summary)
atomic_tsv(study/"lambda_diagnostic.tsv",["lambda_max","seed","platoon","viol_last100","lambda_final","lambda_mean100","lambda_max100","cap_fraction100","near_cap_fraction100"],diagnostics)
atomic_tsv(study/"members.tsv",["arm_id","role","seed","run_path"],members)

manifest=[
    "study=S05_lambda_cap",f"created_at={dt.datetime.now().astimezone().isoformat()}",f"source_commit={source}",
    "lambda_max_values=5,10,20","new_lambda_max_values=10,20","episodes=600","seeds=2,3,4,5,6,7",
    "renew_every=20","buffer=50000","mode=hard","cost_source=raw","dual=pid","tau=8","eps=0.10",
    "lam_warmup=150","lam_scope=per_platoon","kp=1.0","ki=1.0","kd=0.5",
    f"python={sys.version.replace(chr(10),' ')}",f"torch={torch.__version__}",
    f"cuda_available={torch.cuda.is_available()}",f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}",
]
(study/"run_manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
(study/"README.md").write_text(
    "# S05: lambda-cap sensitivity\n\nStatus: complete. Compares per-platoon RCPO at "
    "`lambda_max` 5, 10, and 20 under `renew_every=20`, each with seeds 2--7. "
    "See `summary.tsv`, `lambda_diagnostic.tsv`, `members.tsv`, and `run_manifest.txt`.\n",encoding="utf-8")

with registry.open("r",encoding="utf-8",newline="") as f: existing=list(csv.reader(f,delimiter="\t"))
header,old=existing[0],existing[1:]
expected=["run_id","relative_path","method","status","seed","episodes","renew_every","buffer_capacity","cost_source","lambda_scope","dual","tau","epsilon","lambda_max","penalty_weight","source_commit","legacy_path"]
if header!=expected: raise SystemExit(f"unexpected registry header: {header}")
by_id={r[0]:r for r in old}; additions=[]
for row in ([str(x) for x in r] for r in new_registry):
    if row[0] in by_id:
        # Preserve the original training commit on a later idempotent audit.
        keep = [i for i in range(len(row)) if i != header.index("source_commit")]
        if any(by_id[row[0]][i] != row[i] for i in keep):
            raise SystemExit(f"conflicting registry row: {row[0]}")
    else:
        additions.append(row)
atomic_tsv(registry,header,old+additions)
print(f"summary_rows={len(summary)} diagnostic_rows={len(diagnostics)} registry_additions={len(additions)}")
PY

"$PY_BIN" analysis/audit_results.py
log "S05 complete: results=$OUT_ROOT study=$STUDY_DIR"
log "next: verify LFS, stage only new lmax10/lmax20 runs plus registry and S05, then commit"
