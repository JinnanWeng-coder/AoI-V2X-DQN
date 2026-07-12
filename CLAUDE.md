# AoI-V2X-DQN working ledger

## Current structure

The authoritative run catalogue is `1-IndependentDQN/experiments/registry.tsv`. Formal MAT artifacts appear only under `experiments/runs/`; archived duplicates and failed prototypes are not registered as valid runs. Run `python analysis/audit_results.py` after changing the layout.

## Formal evidence

- **S01_soft_cadence:** raw-AoI soft runs, renew_every 1/5/20, seeds 2--7.
- **S02_replay_buffer:** re05 soft runs at 25k/50k/100k. The 50k arm reuses S01 runs after exact MAT comparison; no data is copied.
- **S03_constraint_design:** re05 soft, global mean/max RCPO, fixed indicator penalties, and per-platoon RCPO.
- **S04_cadence_validation:** matched soft/per-platoon RCPO at re05 and re20. The re20 DQN comparison is worst violation `0.3803 -> 0.1460` across seeds 2--7. This is violation control, not a guarantee that every seed reaches epsilon 0.10.
- **S05_lambda_cap:** planned only; no formal result exists.
- **S06_cross_backbone:** external reference to AoI-V2X-CMDP without copying any MAT artifacts.

## Important reporting constraints

- Use seed (`n=6`) as the inferential replication unit, not 30 platoon-seed values.
- The primary endpoint is per-seed worst-platoon `P(AoI > 8)` from `AoI_evolution.mat`.
- Report power in mW or dB differences. Do not compute percentages or multiples from mean dBm values.
- Do not call the current finite-training result a hard constraint guarantee, deployment guarantee, or 3GPP-compliance proof.
- `F01_unscaled_cost_critic` is a negative prototype, not a standard CMDP arm.

## Legacy migration

`experiments/legacy_path_map.tsv` records every migrated result path. The pre-migration inventory and exact duplicate comparisons are retained in `experiments/archive/migration_records/`.
