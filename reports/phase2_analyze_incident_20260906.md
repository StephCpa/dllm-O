# Phase 2 First-Analysis Incident and Repair Scope

**Date:** 2026-09-06  
**Status:** Post-unblinding implementation incident; no statistical report was
produced by the failed invocation.

## Incident

The first frozen `analyze` invocation was recorded at
`2026-09-06T13:07:05.794369+00:00` with
`reads_heldout_correctness=true`. Artifact validation completed, after which the
analyzer stopped with `KeyError: ('gsm8k', 649, 32)` before computing or writing
P1, P2, bootstrap intervals, or the Condition A verdict. The console log SHA-256
is `ae14af992bc2d5e51531957bae39e63352c0c3de8dc7b94819e56391b1bcde7f`.

## Root cause

`_load_successes` used a `defaultdict(int)` but created a key only when a rollout
was correct. A valid query-policy cell with 0 correct rollouts out of 64 was
therefore absent from the returned plain dictionary. GSM8K query 649 exposed the
bug: `ao_b8`, `ao_b16`, and `ao_b32` each had 0/64 successes. Artifact validation
correctly passed because no file was missing or invalid.

## Permitted repair

The sole statistical-path repair is to initialize every frozen
`(task, query, B)` cell to zero before reading its 64 correctness values. This
maps an all-false vector to its mathematically specified success count of zero.
It changes no endpoint, query, policy, rollout, `k` grid, CoverageAUC formula,
bootstrap seed or replicate count, confidence-interval rule, coding, or verdict
taxonomy. No experimental artifact is modified.

Because the original Phase 2 freeze binds the pre-repair analyzer commit and its
SHA is embedded in all Phase 2 artifacts, the freeze is not rewritten. A separate
`phase2-analysis-repair-v1` amendment must bind the original freeze SHA, original
analyzer commit, repair commit, failed invocation, console-log SHA, and exact
repair scope. The repaired analyzer rejects any amendment that changes statistical
definitions, artifacts, or decisions based on observed outcomes. The amendment
SHA is recorded in the next invocation log and report.

The missing CSV writer noticed during preflight is a packaging issue and is not
part of this repair. Canonical analysis outputs remain JSON and Markdown; any CSV
will be derived mechanically from the canonical JSON after analysis.
