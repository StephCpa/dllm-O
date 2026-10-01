# Phase 0 CPU Readiness Report

**Date:** 2026-08-20
**Status:** CPU implementation ready; no model inference launched

## Completed

- Pinned JustGRPO at `1a2fddb5c6655597e63081c0af5ebb718a849f39`.
- Pinned `GSAI-ML/LLaDA-8B-Instruct` at revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- Pinned `openai/gsm8k` at revision `740312add88f781978c0658806c59bc2815b9866`.
- Froze 50 query indices, 32 rollouts, AR `B=1`, AO `B=32`, 256 steps, 256 generated tokens, and temperature 0.6.
- Implemented policy-independent SHA-256 rollout seeds for common-random-number pairing.
- Implemented a traced decoder with explicit RNG and eligible-to-commit entropy/margin snapshots.
- Implemented atomic gzip JSONL traces, checksums, result records, and resume checks.
- Implemented query-level Pass@k, paired query bootstrap, solved-set counts, closure summaries, and Spearman analysis before seeing GPU outcomes.
- Implemented static multi-GPU sharding at the query-rollout level.

## Verification

### Unit tests

```text
29 passed
```

Coverage includes:

- `B=1`, `B=32`, and `B=256` eligibility and commitment invariants;
- exact completion reconstruction;
- zero AR closure by construction;
- traced/untraced stochastic equivalence;
- seed stability and variation;
- Pass@k edge cases;
- atomic artifact publication and failure cleanup;
- frozen query/policy counts and disjoint sharding;
- paired-bootstrap behavior.

### Public-reference parity

The independent decoder was compared token-for-token with the pinned public
`utils/generate.py` on the same toy masked model and global random stream.

| Block length | Temperature | Exact output match |
|---:|---:|:---:|
| 1 | 0.0 | Yes |
| 1 | 0.6 | Yes |
| 32 | 0.0 | Yes |
| 32 | 0.6 | Yes |
| 256 | 0.0 | Yes |
| 256 | 0.6 | Yes |

### Dry-run counts

- Stage 0A: 20 query-rollout work items, 40 policy generations, 80 decoder executions because each traced output is repeated untraced.
- Stage 0B: 1,600 query-rollout work items, 3,200 policy generations and decoder executions.
- A three-way Stage 0B split assigns 534, 533, and 533 work items, with no overlap.

## Remaining Before GPU Launch

1. Commit and synchronize this independent module to the server.
2. Install the pinned JustGRPO requirements. The current local environment lacks at least `word2number`, so the source grader was not imported end-to-end locally.
3. Confirm the server has enough free storage for Stage 0A before estimating Stage 0B trace volume.
4. Run Stage 0A only.
5. Audit all 40 records for exact trace equivalence, completion count, parser failures, wall time, peak memory, and compressed trace size.
6. Re-estimate the Stage 0B GPU-hour and storage budgets from measured Stage 0A values.

## Known Launch Risks

- Entropy requires a full-vocabulary reduction at every step. It does not add a second model forward pass, but it can increase latency and temporary memory.
- Stage 0B contains 819,200 denoising forward passes in total. Its cost must be measured rather than inferred from the toy model.
- The source math grader has additional symbolic-math dependencies and must be tested on the server before a run is accepted.
- Token-level connector matching is descriptive and tokenizer-dependent; it is not the prospective causal endpoint.
- The analysis loads the pinned tokenizer to normalize selected token IDs. An unavailable revision must block analysis rather than fall back to `main`.

## Launch Decision

**Ready for Stage 0A after commit/synchronization and dependency installation.**

Stage 0B is not yet authorized. Its launch depends on Stage 0A output equivalence and measured resource costs.
