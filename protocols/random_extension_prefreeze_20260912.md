# Random-B32 64-Rollout Extension: Pre-Freeze Protocol

**Protocol version:** `random-extension-freeze-v1`
**Status:** pre-outcome-read specification; the machine-readable freeze is
emitted only after runner/analyzer tests pass and their commit is fixed.
**Selection status:** post-unblinding targeted prospective follow-up.

## 1. Why this extension was selected

The frozen Phase 1 comparison gave `random_b32` only 16 rollouts per held-out
query. A post-unblinding per-k decomposition showed a crossing between
confidence-first `ao_b32` and `random_b32`: AO was resolvedly better at low k,
while random had directionally higher coverage at k=8 and k=16. The original
equal-weight aggregate canceled these effects. The random policy is therefore
the only policy in the intended frontier without k=32 and k=64 observations.

This rationale was chosen after observing Phase 1, Option B, and Condition A
outcomes. The extension is prospective only with respect to the 9,600 new
generations. It is not part of the original confirmatory chain and cannot alter
the frozen Phase 1, Option B, or Condition A verdicts.

## 2. Frozen data and generation design

- Model: `GSAI-ML/LLaDA-8B-Instruct`, revision
  `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- Decoder/evaluator: `LeapLabTHU/JustGRPO`, commit
  `1a2fddb5c6655597e63081c0af5ebb718a849f39`.
- Frozen held-out split: 100 GSM8K and 100 MATH-500 queries from
  `phase1_split_manifest_v1.json`.
- Policy: `random_b32` only; block length 32, random eligible-position
  selection, 256 denoising steps, 256 generated positions, temperature 0.6,
  CFG scale 0.
- Existing records: rollout indices 0--15 from frozen Phase 1 screening.
- New records: rollout indices 16--63, using the existing policy-independent
  seed derivation. These indices are common-random-number matched to existing
  AO, AR, and EF rollouts where those policies are compared.
- New workload: `2 tasks x 100 queries x 48 rollouts = 9,600 generations`.
- Formal execution: three deterministic shards of 3,200 records each.
- New run kind: `random-extension`; outputs cannot share a run-kind directory
  or completion marker with any earlier phase.

## 3. Primary endpoint

For each query, estimate Pass@k from all 64 random rollouts and all 64 AO
rollouts. Define

`HighBudgetCoverage = mean(Pass@16, Pass@32, Pass@64)`.

The sole primary contrast is the paired per-query

`HighBudgetCoverage(random_b32) - HighBudgetCoverage(ao_b32)`.

Aggregation is the equal-weight mean of the GSM8K and MATH-500 task means.
The uncertainty interval is a 10,000-replicate task-stratified query-bootstrap
percentile interval with seed `20260912`.

### Frozen primary taxonomy

- CI entirely above zero: resolved high-budget random-ranking coverage benefit
  relative to confidence-first AO in this two-task panel.
- CI straddles zero: no resolved high-budget difference at this design's
  resolution.
- CI entirely below zero: resolved high-budget harm from random ranking.

This taxonomy is estimation-oriented. No minimum effect-size threshold and no
equivalence margin are introduced.

## 4. Secondary endpoints

All secondary results are reported regardless of direction and cannot replace
or modify the primary verdict:

1. Per-task primary contrasts.
2. Paired random-minus-AO Pass@k for k in `{1,2,4,8,16,32,64}`.
3. Paired random-minus-AR and random-minus-EF Pass@k on the same grid.
4. The declared low-budget grid `{2,4,8,16}` and full grid
   `{4,8,16,32,64}`, with every grid printed in the report.
5. Solved-set discordance at k=64.
6. Parsed-answer unique count, entropy, and modal share at k=16 and k=64.
7. Parser-status, empty-response, decoded-length, and artifact-completeness
   summaries.

No multiple-testing-adjusted confirmatory claim is attached to secondary
intervals. They characterize the frontier and identify whether an apparent
benefit is task-specific or ceiling-limited.

## 5. Artifact and launch gates

Every new result must validate against
`random_extension_artifact_schema_v1.json` and contain:

- the random-extension freeze SHA-256;
- the original BCI freeze SHA-256 and split-manifest SHA-256;
- runner commit, external decoder commit, model revision, source row/hash,
  rollout index, seed key, trace hash, and environment fingerprint;
- `run_kind=random-extension`, `policy=random_b32`, and completion status.

Before formal launch:

1. all repository tests must pass;
2. dry-run cardinality must equal 9,600 and each of three shards must equal
   3,200;
3. one GSM8K and one MATH-500 canary record may be generated after freezing;
4. canary validation must read no correctness, parsed answer, or response;
5. no formal shard marker may be written by a truncated canary invocation.

Formal analysis is blocked unless 9,600/9,600 new records and all three shard
markers validate. Parser failures and empty responses remain incorrect; no
query or rollout is removed after outcomes are read.

## 6. Recovery and provenance

- Existing valid records are validated and skipped on resume.
- Only an interrupted work key may be regenerated, with the same seed and work
  key. Orphan temporary traces are quarantined and recorded before recovery.
- Changing GPU device does not change the work key and is permitted if all
  other command arguments and the frozen code remain identical.
- Inference must run from the clean commit named in the machine freeze.
- Validation-only mode must log `outcome_fields_read=false` before analysis.
- The first analysis invocation must log the timestamp at which correctness is
  first read.

## 7. Not permitted

- Calling this extension part of the original pre-registered Gate-1 evidence.
- Reinterpreting the frozen AO-minus-random aggregate null.
- Claiming universal superiority of random ranking, a universal crossover, or
  portability beyond this model, two tasks, decoder, and temperature.
- Treating parsed-answer diversity as semantic reasoning diversity.
- Adding another policy, task, k-grid, endpoint, or exclusion after reading the
  48 new rollouts.
- Using the asymmetric AO-64/random-16 exploratory comparison as evidence for
  the 64-rollout result.

## 8. Decision after completion

Regardless of the primary result, the experimental program stops after this
extension. A positive result fills the random policy's high-budget frontier; a
straddling or negative result is equally reportable and rules out the proposed
coverage advantage at the achieved resolution. No repair arm follows.
