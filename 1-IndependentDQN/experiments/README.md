# Experiment data layout

`runs/` is the only physical store for formal DQN run artifacts. A study never copies MAT files; it references runs through `members.tsv` paths resolved relative to the study directory.

- `registry.tsv` is the single run registry. Only `valid` and `negative_result` rows are registered formal artifacts.
- `archive/duplicate_legacy/` holds verified duplicate or incomplete legacy copies and is intentionally absent from the registry.
- `archive/failed_prototypes/` preserves negative results without presenting them as canonical CMDP runs.
- `studies/` contains analysis membership, existing summaries, and optional generated figures.

Run `python analysis/audit_results.py` from `1-IndependentDQN/` after changing the registry or a study membership file.


