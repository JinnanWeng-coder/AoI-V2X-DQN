"""Audit the experiment registry, study references, and formal MAT artifacts.

Usage: python analysis/audit_results.py
"""
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT / "experiments"
REGISTRY = EXPERIMENTS / "registry.tsv"
BASE_REQUIRED = {
    "reward_t1.mat": "reward_t1", "reward_t2.mat": "reward_t2", "AoI.mat": "AoI",
    "viol_rate.mat": "viol_rate", "AoI_evolution.mat": "AoI_evolution", "demand.mat": "demand",
    "V2I.mat": "V2I", "V2V.mat": "V2V", "power.mat": "power", "epsilon.mat": "epsilon",
    "mode_v2i.mat": "mode_v2i",
}


def fail(errors, message):
    errors.append(message)


def audit_run(row, errors):
    run = EXPERIMENTS / row["relative_path"]
    if not run.is_dir():
        fail(errors, f"missing registry path: {row['relative_path']}")
        return
    loaded = {}
    for filename, key in BASE_REQUIRED.items():
        path = run / filename
        if not path.is_file():
            fail(errors, f"{row['run_id']}: missing {filename}")
            continue
        try:
            loaded[key] = np.asarray(loadmat(path)[key])
        except Exception as exc:
            fail(errors, f"{row['run_id']}: unreadable {filename}: {exc}")
            continue
        if not np.isfinite(loaded[key]).all():
            fail(errors, f"{row['run_id']}: NaN/Inf in {filename}")
    hard = row["cost_source"] in {"raw", "critic"}
    lambda_path = run / "lambda.mat"
    if hard and not lambda_path.is_file():
        fail(errors, f"{row['run_id']}: hard run lacks lambda.mat")
    if lambda_path.is_file():
        try:
            loaded["lambda"] = np.asarray(loadmat(lambda_path)["lambda"])
            if not np.isfinite(loaded["lambda"]).all():
                fail(errors, f"{row['run_id']}: NaN/Inf in lambda.mat")
        except Exception as exc:
            fail(errors, f"{row['run_id']}: unreadable lambda.mat: {exc}")
    if "AoI" not in loaded or "viol_rate" not in loaded or "AoI_evolution" not in loaded:
        return
    try:
        episodes = int(row["episodes"])
    except ValueError:
        fail(errors, f"{row['run_id']}: invalid episodes value")
        return
    p = loaded["AoI"].shape[0]
    expected_2d = (p, episodes)
    expected_3d = (p, 100, 100)
    for key in ("AoI", "viol_rate", "reward_t1", "reward_t2", "mode_v2i"):
        if key in loaded and loaded[key].shape != expected_2d:
            fail(errors, f"{row['run_id']}: {key} shape {loaded[key].shape}, expected {expected_2d}")
    if "epsilon" in loaded and loaded["epsilon"].shape not in {(episodes,), (1, episodes)}:
        fail(errors, f"{row['run_id']}: epsilon shape {loaded['epsilon'].shape}")
    if "lambda" in loaded and loaded["lambda"].shape != expected_2d:
        fail(errors, f"{row['run_id']}: lambda shape {loaded['lambda'].shape}, expected {expected_2d}")
    for key in ("AoI_evolution", "demand", "V2I", "V2V", "power"):
        if key in loaded and loaded[key].shape != expected_3d:
            fail(errors, f"{row['run_id']}: {key} shape {loaded[key].shape}, expected {expected_3d}")


def audit_studies(errors):
    studies = EXPERIMENTS / "studies"
    for study in sorted(path for path in studies.iterdir() if path.is_dir()):
        members = study / "members.tsv"
        if not members.exists():
            continue
        with members.open(newline="", encoding="utf-8") as handle:
            fields = csv.DictReader(handle, delimiter="\t")
            required = {"arm_id", "role", "seed", "run_path"}
            if set(fields.fieldnames or []) != required:
                fail(errors, f"{study.name}: bad members.tsv header")
                continue
            for row in fields:
                if not (study / row["run_path"]).resolve().is_dir():
                    fail(errors, f"{study.name}: unresolved member {row['run_path']}")


def main():
    errors = []
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        fail(errors, "registry run_id values are not unique")
    registry_paths = {str((EXPERIMENTS / row["relative_path"]).resolve()) for row in rows}
    for row in rows:
        if row["status"] not in {"valid", "negative_result", "legacy_duplicate", "planned"}:
            fail(errors, f"{row['run_id']}: invalid status {row['status']}")
        if row["status"] in {"valid", "negative_result"}:
            audit_run(row, errors)
    for run in sorted({path.parent for path in (EXPERIMENTS / "runs").rglob("*.mat")}):
        if str(run.resolve()) not in registry_paths:
            fail(errors, f"unregistered formal run: {run.relative_to(EXPERIMENTS)}")
    audit_studies(errors)
    if errors:
        print("AUDIT FAILED")
        for error in errors:
            print("- " + error)
        raise SystemExit(1)
    print(f"AUDIT PASS: {len(rows)} registered rows; all formal runs, MAT files, and study references are valid.")


if __name__ == "__main__":
    main()

