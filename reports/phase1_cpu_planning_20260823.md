# Phase 1 CPU Planning Report

- Planning seed: `20260823`
- Split version: `phase1-splits-v1`
- GSM8K: 50 calibration / 100 held-out queries
- MATH-500: 50 calibration / 100 held-out queries
- The 50 GSM8K calibration queries are exactly the completed Phase 0 signal set.

## Power Resolution

- Per-task held-out correlation MDE (n=100, two-sided alpha=.05, 80% power): 0.277
- Task-stratified pooled correlation MDE (n=200 approximation): 0.197
- Per-task standardized paired-effect MDE (n=100): 0.280
- These are planning approximations, not replacements for query-clustered bootstrap intervals in the final analysis.

## Resource Budget

- New generations: 41,600
- Phase-0-calibrated GPU-hours: 330.6
- GPU-hours with 30% MATH/runtime contingency: 429.8
- Approximate wall time on 4 GPUs: 82.7 hours before contingency
- Approximate wall time on 8 GPUs: 41.3 hours before contingency
- Estimated compressed trace storage: 3.9 GiB

## Design Consequence

All five schedules receive 16 screening rollouts. Only the frozen AR and AO-B32 primary comparison is extended to 64 rollouts on held-out queries. Calibration outcomes may define the BCI and matching rule, but held-out outcomes may not modify them.
