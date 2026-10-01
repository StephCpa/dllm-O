# Phase 1 Runner Gap Audit

**Date:** 2026-08-23  
**Scope:** Read-only audit of the Phase 0 implementation against the frozen Phase 1 protocol  
**Verdict:** Phase 1 GPU launch is not yet authorized. The decoder and trace core are reusable, but the experiment orchestration remains Phase-0-specific.

## Reusable Without Scientific Change

| Capability | Current evidence | Phase 1 disposition |
|---|---|---|
| Output-preserving tracing | Phase 0 equivalence: 40/40 traced outputs matched untraced outputs | Reuse and repeat on one item per new policy/task |
| Decode schedules | `DecodeConfig` supports arbitrary block divisors and `low_confidence`/`random` remasking | Reuse for B=1/8/16/32 and random-B32 |
| Atomic compressed traces | Gzip JSONL is published only after successful close | Reuse |
| Resume checks | Completed results are skipped only when trace hashes match | Generalize to Phase 1 keys |
| Stable seed derivation | Seed key is independent of policy | Generalize protocol prefix and record matched-seed semantics |
| Trace invariants | Existing tests cover reconstruction, block membership, finite metrics, and logging invariance | Extend to every Phase 1 policy |

## Blocking Gaps

| Gap | Why it blocks launch | Required implementation |
|---|---|---|
| Phase 0 constants are hard-coded | Only two policies, one dataset, and fixed stage sizes can be run | Add an immutable Phase 1 config loader bound to the split-manifest hash |
| GSM8K-only input loader | MATH-500 cannot be loaded or prompt-hash-validated | Add pinned MATH-500 parquet loading and exact row/schema checks |
| GSM8K-only grader | MATH answers cannot be parsed with the pinned JustGRPO logic | Add a task-dispatched grader using `extract_answer(..., timeout=True)` semantics from the frozen external commit |
| No split-role boundary | Calibration and held-out queries could be mixed accidentally | Require `calibration` or `heldout` and resolve work only from the checked manifest |
| Uniform rollout count | Screening 16 and primary extension 16→64 cannot be expressed safely | Generate explicit work keys from task, split role, policy, query, and rollout; make extension unconditional for primary policies |
| Result schema mismatch | Existing records do not satisfy the Phase 1 source/environment contract | Emit and validate `phase1_artifact_schema_v1.json` records |
| No shard completion manifest | Partial shards can look complete after a job exits | Hash expected and observed work-key sets and atomically publish a completion marker |
| Analysis hard-coded to Phase 0 | It assumes two policies, GSM query IDs, and Pass@32 contrast | Build a separate Phase 1 analyzer; do not mutate Phase 0 analysis semantics |
| No calibration freeze validator | Held-out inference could begin before the BCI formula is frozen | Require a valid `phase1_bci_freeze_v1.json` hash for held-out jobs |

## Required Pre-Launch Tests

1. Manifest checksum and prompt-hash rejection tests for both datasets.
2. Work-key cardinality, disjoint-shard, and resume tests for calibration, held-out screening, and primary extension.
3. Parser fixtures covering correct, incorrect, malformed, truncated, and timeout-prone MATH expressions.
4. Trace reconstruction and logging-invariance smoke for all five policies on one GSM8K and one MATH-500 query.
5. Random-B32 stochasticity test showing that matched seed labels are recorded without claiming stepwise common random numbers.
6. JSON-Schema validation for generation results and shard completion manifests.
7. Dry-run output that reports exact generations, expected trace files, model/data revisions, split-manifest hash, and whether a BCI freeze is required.

## Recommended Implementation Order

1. Introduce Phase 1 config and manifest validation without touching the Phase 0 runner.
2. Add task-dispatched dataset loading and grading.
3. Add the five-policy work planner and shard completion contract.
4. Add a Phase 1 runner that reuses `generate_with_trace` and atomic writers.
5. Run CPU tests and dry runs.
6. Run a 10-generation GPU smoke covering both tasks and all policies.
7. Audit smoke artifacts before authorizing Phase 1A calibration.

The private-aggregation module, forced-token intervention, semantic rationale clustering, and second-model replication are intentionally out of scope until Gate 1.
