# AoI-V2X-DQN working ledger

Repository snapshot: `6181e5d758c39532f7dfe023190cb6375db412cc` (`Add re20 lambda cap sensitivity results`). At this snapshot, `1-IndependentDQN/experiments/registry.tsv` contains 130 rows: 124 `valid` formal runs and 6 archived `negative_result` prototype runs. The repository audit reports:

```text
AUDIT PASS: 130 registered rows; all formal runs, MAT files, and study references are valid.
```

## Current structure

The authoritative run catalogue is `1-IndependentDQN/experiments/registry.tsv`. Valid formal MAT artifacts appear only under `experiments/runs/`; studies reference those artifacts through `members.tsv` and never copy them. Archived duplicates, migration records, and the six registered negative prototypes live under `experiments/archive/`.

Run `python analysis/audit_results.py` from `1-IndependentDQN/` after changing a formal run path, registry row, or study reference.

## Formal experiment status

- **S01_soft_cadence — complete:** raw-AoI soft runs at `renew_every={1,5,20}`, seeds 2--7.
- **S02_replay_buffer — complete:** re05 soft runs at 25k/50k/100k replay capacity, seeds 2--6. The 50k arm reuses S01 artifacts after exact MAT comparison.
- **S03_constraint_design — complete:** re05 comparison of soft raw AoI, per-platoon RCPO, global-mean/global-max RCPO, and fixed indicator penalties `w={2,5,10,20}`, seeds 2--7.
- **S04_cadence_validation — complete:** matched soft/per-platoon RCPO comparison at re05 and re20, seeds 2--7.
- **S05_lambda_cap — complete:** re20 per-platoon RCPO at `lambda_max={5,10,20}`, seeds 2--7. Cap 5 reuses S04; caps 10 and 20 add 12 formal runs.
- **S06_cross_backbone — complete external-reference study:** references AoI-V2X-CMDP provenance without copying its MAT artifacts.
- **S07_re20_constraint_design — complete:** re20 global-mean/global-max RCPO and fixed indicator penalties `w={2,5,10,20}`, seeds 2--7.

## Locked aggregate results

All values below are seed means over seeds 2--7. `worst` is the per-run maximum platoon `P(AoI > 8)` computed from `AoI_evolution.mat`; `net` is the corresponding platoon mean. AoI and power use the final 100 training episodes, and power is averaged in linear mW.

### Cadence validation

| arm | renew_every | worst | net | mean AoI | power (mW) |
|---|---:|---:|---:|---:|---:|
| soft raw AoI | 5 | 0.318883 | 0.180303 | 5.193278 | 120.023246 |
| per-platoon RCPO, cap 5 | 5 | 0.133400 | 0.107513 | 4.450459 | 139.154313 |
| soft raw AoI | 20 | 0.380267 | 0.197180 | 5.438109 | 126.354578 |
| per-platoon RCPO, cap 5 | 20 | 0.146033 | 0.114167 | 4.485283 | 155.364813 |

### re20 constraint-design comparison

| method | worst | net | mean AoI | power (mW) |
|---|---:|---:|---:|---:|
| per-platoon RCPO, cap 5 | 0.146033 | 0.114167 | 4.485283 | 155.364813 |
| global-mean RCPO | 0.206150 | 0.104323 | 4.413677 | 150.393651 |
| global-max RCPO | 0.212050 | 0.114620 | 4.554076 | 193.038342 |
| fixed indicator `w=2` | 0.237350 | 0.136060 | 4.881960 | 154.599645 |
| soft raw AoI | 0.380267 | 0.197180 | 5.438109 | 126.354578 |
| fixed indicator `w=20` | 0.446650 | 0.235180 | 6.415855 | 192.409107 |
| fixed indicator `w=10` | 0.496017 | 0.267013 | 7.133211 | 183.948827 |
| fixed indicator `w=5` | 0.518517 | 0.282077 | 7.574691 | 168.118434 |

The method ordering by mean worst-platoon violation is identical in S03/re05 and S07/re20: per-platoon RCPO, global mean, global max, fixed `w=2`, soft raw AoI, fixed `w=20`, fixed `w=10`, fixed `w=5`.

### re20 lambda-cap sensitivity

| lambda_max | worst | net | mean AoI | power (mW) |
|---:|---:|---:|---:|---:|
| 5 | 0.146033 | 0.114167 | 4.485283 | 155.364813 |
| 10 | 0.140900 | 0.112660 | 4.466414 | 160.270919 |
| 20 | 0.127433 | 0.107973 | 4.391139 | 164.338083 |

The S05 diagnostics use 30 platoon-seed units per cap. At cap 20, 3/30 units touch the cap during the final 100 episodes, 0/30 spend at least half of that window at the cap, 1/30 spend at least half near the cap, and 16/30 remain above `epsilon=0.10`. The cap sweep therefore records improved aggregate violation together with residual violations after persistent cap saturation has disappeared.

## Evidence and reporting constraints

- Use seed (`n=6`) as the inferential replication unit, not the 30 platoon-seed values.
- The primary endpoint is per-seed worst-platoon `P(AoI > 8)` from `AoI_evolution.mat`.
- Current study summaries describe the final training window: exploration ends at epsilon 0.05, and learning and dual updates remain active. They are not registered frozen-deployment results.
- Report power in mW or as dB differences. Do not compute percentages or multiples from mean dBm values.
- Do not call the finite-training result a hard-constraint guarantee, deployment guarantee, or 3GPP-compliance proof.
- `F01_unscaled_cost_critic` is a registered negative prototype, not a standard CMDP arm.

## Legacy migration

`experiments/legacy_path_map.tsv` records every migrated result path. The pre-migration inventory and exact duplicate comparisons are retained in `experiments/archive/migration_records/`.
