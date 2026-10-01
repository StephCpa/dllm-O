# Phase 1 Runner CPU Readiness

**Date:** 2026-08-23  
**Status:** CPU implementation complete; GPU smoke not yet run

## Implemented Contracts

- Separate Phase 1 runner; Phase 0 code and output semantics remain unchanged.
- Exact split-manifest and artifact-schema hash validation.
- Five frozen policies: AR-B1, confidence B8/B16/B32, and random-B32.
- Three explicit run kinds: 10-generation smoke, 16-rollout screening, and held-out AR/AO primary extension from rollouts 16 through 63.
- Task-dispatched GSM8K and MATH-500 prompts and pinned JustGRPO graders.
- Task-specific, policy-independent seed keys.
- Atomic gzip traces, schema-validated result records, strict resume checks, and atomic shard completion markers.
- Held-out hard gate requiring a complete BCI-freeze record bound to the split manifest and current runner commit.
- Primary-extension hard gate requiring all corresponding screening artifacts.
- Clean-repository requirement for every real inference run.

## CPU Verification

- 45 tests pass.
- Work counts: smoke 10; calibration screening 6,400; held-out screening 16,000; primary extension 19,200.
- Three-way held-out sharding is disjoint and complete.
- Toy end-to-end generation produces a trace, schema-valid result, successful trace-equivalence flag, and schema-valid shard marker.
- The JSON Schema and all frozen design hashes validate.

## Remaining Launch Gate

Run the ten-generation smoke on one GPU using the already verified base-model snapshot and the two pinned parquet files. Accept only if:

1. all ten results complete and parse without infrastructure errors;
2. all ten traced outputs exactly match their untraced repeats;
3. prompt, trace, split-manifest, schema, model, and external-repository bindings validate;
4. the shard completion marker reports ten expected and observed work keys;
5. no orphan or temporary trace/result files remain.

Passing this smoke authorizes Phase 1A calibration only. It does not authorize held-out inference, forced-token intervention, or private aggregation.
