# Intermediate Block Sizes B=2 and B=4: Pre-Freeze Draft

**Protocol version:** `intermediate-block-size-freeze-v1` (proposed)  
**Priority (2026-10-03 UTC review):** P1. The validation on unseen problems
(E1) and the CF-resample control (E2) come first; see
`unseen_problem_validation_prefreeze_draft_20261003.md` and
`cf_resample_control_prefreeze_draft_20261003.md`. A second model alone does not
resolve the novelty concern.  
**Status:** draft for author review. Not frozen and not authorized for GPU launch.
Items marked **[author decision]** or **[bind at freeze]** must be resolved before
any new generation.  
**Selection status:** post-review prospective follow-up. It was proposed after the
Phase 2 shape (only `B=1 -> 8` resolved) was read. It is prospective only for the
new `B=2` and `B=4` outcomes. The reused `B=1` and `B=8` cells were already
unblinded, and that is disclosed wherever results are reported.

## 1. Question

The resolved coverage cost lies between `B=1` and `B=8`, and the paper cannot
distinguish an immediate onset of order choice from a gradual change, because
`B=2` and `B=4` were never measured. This study asks: **where inside `[1, 8]` does
the matched-NFE coverage cost appear?**

## 2. Generation design

- Model and decoder: unchanged (`GSAI-ML/LLaDA-8B-Instruct`, revision
  `08b83a6feb34df1a6011b80c3c00c7563e963b07`; JustGRPO commit
  `1a2fddb5c6655597e63081c0af5ebb718a849f39`).
- Data: the existing frozen 200 held-out queries.
- New cells: confidence-first `B=2` (`ao_b2`) and `B=4` (`ao_b4`), with confidence
  remasking. With 256 positions and 256 steps, these settings use 128 and 64
  blocks, so one token is committed per NFE as in every existing cell.
- Fixed settings: temperature 0.6, CFG 0, the existing parser and grader.
- Rollouts: 64 per query and cell, with the existing policy-independent seed keys
  (`seeding.rollout_seed_key`). The new cells are therefore seed-matched to
  `ar_b1`, `ao_b8`, `ao_b16`, and `ao_b32`.
- Workload: `2 tasks x 100 queries x 2 cells x 64 = 25,600 generations`.
- Reused controls: `ar_b1` and `ao_b8` (64 rollouts each), bound by a
  no-outcome-read, checksummed inventory.

## 3. Frozen primary family

Three adjacent pooled contrasts on the original grid `K64 = {4,8,16,32,64}`:

- `E12 = CoverageAUC(B=2) - CoverageAUC(B=1)`;
- `E24 = CoverageAUC(B=4) - CoverageAUC(B=2)`;
- `E48 = CoverageAUC(B=8) - CoverageAUC(B=4)`.

Their sum is the already-observed `B=1 -> 8` contrast (−0.038), so the family
partitions that drop. Uncertainty: a 10,000-replicate task-stratified
paired-query bootstrap, seed `20261004`, with nonstudentized maximum-deviation
simultaneous 95% bands over the three endpoints.

### Frozen readings

- `onset_at_first_choice`: `E12` is resolved below zero, and neither `E24` nor
  `E48` is resolved below zero.
- `late_onset`: `E48` is resolved below zero, and neither `E12` nor `E24` is.
- `distributed`: two or more of the three contrasts are resolved below zero.
- `not_localized`: every other outcome, reported with estimates. This includes the
  case in which no single contrast resolves even though their sum does. It does
  not mean the cost is absent.

## 4. Secondary endpoints

All are reported regardless of direction and cannot change the readings:

1. The share of the `B=1 -> 8` drop reached at `B=2` and at `B=4`, with percentile
   intervals (descriptive; the denominator interval excludes zero).
2. The full curve over `B in {1,2,4,8,16,32}` for CoverageAUC, Pass@1, and Pass@16,
   updating Figure 2 without its hatched gap.
3. Zero-success query counts per cell.
4. Parser status, empty responses, decoded length, and artifact completeness.

## 5. Launch gates

1. Register `ao_b2` and `ao_b4` as new policies under a new run kind. Commit the
   runner, analyzer, schema, and tests; the inference checkout must be clean.
2. A test must confirm that `B=2` and `B=4` transfer exactly one token per step,
   in every block.
3. Build the reused-control inventory without reading correctness, then dry-run
   all shards (25,600 unique work keys) and emit the machine freeze.
4. Run a smoke on one query per task, validated without reading correctness.
5. Record a validation-only invocation before the first correctness read.

## 6. Power note

The `B=1 -> 8` interval implies a query-level standard error of about 0.0095 for
one adjacent contrast. Assuming similar variance and the three-endpoint critical
value (about 2.39 SE), the approximate power to resolve `E12` is:

| share of the −0.038 drop at `B=2` | power |
|---:|---:|
| all of it | 0.95 |
| 75% | 0.73 |
| 50% | 0.35 |

The design is decisive if the cost appears immediately. A gradual change will
probably produce `not_localized`, which should be reported as such and not
over-read.

## 7. Resource estimate

About 185–228 GPU-hours at 26–32 seconds per generation (about 31–38 hours on six
GPUs).

## 8. Scope boundary

The readings concern the confidence-first schedule at temperature 0.6 on this
model and query panel. They cannot by themselves separate the existence of a
choice from the confidence ranking used to make it; random selection at `B=2` would
be a separate design.
