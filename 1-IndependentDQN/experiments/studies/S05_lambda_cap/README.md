# S05: lambda-cap sensitivity

Status: planned. The existing `lambda_max=5` re20 runs show persistent cap binding in 5 of 30 platoon-seed units during the final 100 episodes. The formal sensitivity study reuses that baseline and adds `lambda_max=10` and `20` for seeds 2--7 via `scripts/run_re20_lambda_cap_sweep.sh`. Results are registered only after all MAT artifacts pass audit.

