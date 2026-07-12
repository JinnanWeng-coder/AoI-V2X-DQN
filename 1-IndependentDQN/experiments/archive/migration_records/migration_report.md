# Result-directory migration report

Date: 2026-07-12

## Inventory and migration

- Pre-migration inventory: 90 run directories, 1,057 MAT files, 189,867,264 MAT bytes.
- Formal registry: 76 `valid` runs and 6 `negative_result` runs.
- Verified duplicate archive: 8 legacy run directories. No MAT files were deleted.
- Formal physical store: `experiments/runs/` contains 76 run directories and 894 MAT files.
- Failed prototype store: `archive/failed_prototypes/F01_unscaled_cost_critic/` contains 6 run directories and 72 MAT files.
- Duplicate store: `archive/duplicate_legacy/` contains 8 run directories and 91 MAT files.

## Deduplication decisions

Three root-level soft copies were exact duplicates of `Scan_Cadence` runs or
lacked only `mode_v2i.mat`. Five `Mem_Sweep` 50k runs were exact matches of the
corresponding `Scan_Cadence` re05 run on every shared MAT array; their extra
`lambda.mat` arrays were all zero. Details are in `duplicate_detection.tsv` and
the canonical targets are recorded in `../../legacy_path_map.tsv`.

## Validation

`python analysis/audit_results.py` passed after migration. It validates registry
paths, unregistered formal runs, study reference resolution, MAT presence, shape,
finite values, and unique run IDs. Key worst-platoon means were recomputed from
the new paths:

| result | value |
|---|---:|
| DQN soft re20 | 0.3802667 |
| DQN RCPO re20 | 0.1460333 |
| DQN soft re05 | 0.3188833 |
| DQN RCPO re05 | 0.1334000 |

## Known local limitation

Python compilation, analysis audit, and result summaries were executed locally.
The batch runner's `--dry-run` could not be executed because this workstation's
only `bash.exe` is a broken WSL launcher and Git Bash is not installed. No formal
training was run during the migration.
