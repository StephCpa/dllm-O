# Cold Sequential Decoding Versus Confidence-First: Pre-Freeze Draft

**Protocol version:** `cold-sequential-freeze-v1` (proposed)  
**Status:** **deferred** after independent review (2026-10-03). Launch is not
recommended at this stage, because the design needs a new non-inferiority margin
and is probably underpowered on the current panel (Section 6). Kept for reference
only. It is not frozen and not authorized for GPU launch.
Items marked **[author decision]** or **[bind at freeze]** must be resolved before
any new generation.  
**Selection status:** post-review prospective follow-up. It was proposed after the
temperature-extension outcomes were read. It is prospective only for the new
sequential outcomes.

## 1. Question

Across the tested temperatures, confidence-first `B=32` never reaches the
high-budget coverage of sequential `B=1` at `T=0.6`: its Pass@32 is 0.790–0.815,
against 0.850 for sequential. Heating confidence-first therefore does not buy
coverage. The reverse question is open: **can a colder sequential decoder match
confidence-first's single-sample accuracy while keeping a high-budget coverage
advantage?** If it can, confidence-first `B=32` is dominated at every tested budget
in this setting.

Reference values under the 32-rollout functional at `T=0.6`, from the committed
temperature report: sequential minus CF is −0.019 at Pass@1 and +0.054 on
`mean(Pass@16, Pass@32)`.

## 2. Generation design

- Model, decoder, data, and fixed settings: unchanged from the temperature
  extension (256 positions, 256 steps, CFG 0, one token per NFE).
- New cell: sequential `B=1` at `T=0.3` **[author decision: 0.3 is proposed;
  choose one value before freeze]**.
- Rollouts: 32 per query, with the same seed keys as the temperature extension.
- Workload: `2 tasks x 100 queries x 1 cell x 32 = 6,400 generations`.
- Reused controls: the first 32 rollouts of `ao_b32` and `ar_b1` at `T=0.6`, which
  are already bound in `temperature_control_inventory_v1.json`.

Only one temperature is proposed. Adding a second temperature to the frozen family
raises the simultaneous critical value and lowers power substantially (Section 6).

## 3. Frozen primary family

Two paired query-level contrasts against confidence-first at `T=0.6`:

- `D1 = Pass@1(seq, T=0.3) - Pass@1(CF, T=0.6)`;
- `D2 = HB(seq, T=0.3) - HB(CF, T=0.6)`, with `HB = mean(Pass@16, Pass@32)`.

Uncertainty: a 10,000-replicate task-stratified paired-query bootstrap, seed
`20261005`, with nonstudentized maximum-deviation simultaneous 95% bands over the
two endpoints.

Non-inferiority margin for `D1`: `delta = 0.02` (two Pass@1 points) **[author
decision]**. This is the first margin used in the project. It must be justified as
a substantive tolerance before the freeze and must not be tuned to the existing
`T=0.6` estimates.

### Frozen verdict

- `sequential_dominates`: the `D2` band lies entirely above zero, and the lower
  bound of the `D1` band is above `-delta`.
- `coverage_advantage_only`: the `D2` band lies entirely above zero, and the lower
  bound of the `D1` band is at or below `-delta`.
- `not_resolved`: every other outcome, reported with estimates. It does not mean
  the two decoders are equivalent.

## 4. Secondary endpoints

1. Per-task estimates.
2. The sequential frontier (Pass@1 versus HB) across `T in {0.3, 0.6, 0.9, 1.2}`,
   plotted against confidence-first at `T in {0.6, 0.9, 1.2}`.
3. Per-k curves for `k in {1,2,4,8,16,32}`; parsed-answer diversity; tie-aware
   majority vote.
4. Parser status, empty responses, decoded length, and artifact completeness.

## 5. Launch gates

These are the same as the temperature extension, applied to a new run kind: a
committed runner, analyzer, schema, and tests; a no-outcome-read control
inventory; a dry run (6,400 unique work keys); a smoke validated without reading
correctness; the machine freeze; and a validation-only invocation before the first
correctness read.

## 6. Power note

The temperature-extension high-budget interval implies a query-level standard error
of about 0.020 for an HB contrast. Under the two-endpoint critical value (about
2.24 SE), the approximate power to resolve `D2` is:

| true `D2` | power |
|---:|---:|
| +0.054 (sequential keeps its full `T=0.6` advantage) | 0.66 |
| +0.040 | 0.39 |
| +0.027 | 0.18 |

The `sequential_dominates` verdict also requires the `D1` condition, so its joint
power is lower still. With the existing 200-query panel, this study is cheap but
will probably end `not_resolved`. Its realistic value is one more descriptive
frontier point. A decisive version needs a larger fresh query panel, with
confidence-first generated on the same new queries, which roughly triples the cost
**[author decision]**.

## 7. Resource estimate

About 46–57 GPU-hours at 26–32 seconds per generation.

## 8. Scope boundary

Even `sequential_dominates` would hold only for this model, task panel, and pair of
temperatures. It would not show that order freedom is never useful, because
throughput-oriented decoders that commit several tokens per NFE are outside this
design.
