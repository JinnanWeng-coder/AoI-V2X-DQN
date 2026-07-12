"""Summarize a study defined only by its members.tsv references.

Usage: python analysis/summarize_study.py experiments/studies/S04_cadence_validation
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def metrics(run_dir):
    ev = np.asarray(loadmat(run_dir / "AoI_evolution.mat")["AoI_evolution"], dtype=float)
    aoi = np.asarray(loadmat(run_dir / "AoI.mat")["AoI"], dtype=float)
    power = np.asarray(loadmat(run_dir / "power.mat")["power"], dtype=float)
    per = (ev > 8.0).mean(axis=(1, 2))
    return {
        "canonical_worst_platoon": float(per.max()),
        "canonical_net_mean": float(per.mean()),
        "mean_AoI_last100": float(aoi[:, -100:].mean()),
        "mean_power_mW": float(np.power(10.0, power / 10.0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--output", type=Path, help="optional TSV output; refuses to overwrite")
    args = parser.parse_args()
    study = args.study_dir.resolve()
    members_path = study / "members.tsv"
    if not members_path.is_file():
        raise SystemExit(f"missing members.tsv: {members_path}")
    rows = []
    with members_path.open(newline="", encoding="utf-8") as handle:
        for member in csv.DictReader(handle, delimiter="\t"):
            run_dir = (study / member["run_path"]).resolve()
            if not run_dir.is_dir():
                raise SystemExit(f"unresolved study member: {member['run_path']}")
            rows.append({**member, **metrics(run_dir)})
    fields = ["arm_id", "role", "seed", "run_path", "canonical_worst_platoon", "canonical_net_mean", "mean_AoI_last100", "mean_power_mW"]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row[field]) for field in fields))
    text = "\n".join(lines) + "\n"
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise SystemExit(f"refusing to overwrite: {output}")
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

