# Phase 1 Screening Malformed-Output Audit

- Audit version: `phase1-screening-malformed-audit-v1`
- Created (UTC): `2026-08-29T17:46:36.989507+00:00`
- Scope: held-out screening artifacts only; runs before extension completion and reads no correctness values
- Correctness fields read: **False**; fields read by design: parser_status, parsed_answer
- output is invariant to 'correct' and 'response' fields; proven by sentinel-substitution test
- Malformed definition: `parser_status != 'ok' or parsed_answer is null`
- Frozen sensitivity threshold: 1% (trigger is strictly greater)

## Counts

- Expected screening records: 16,000
- Present and valid: 16,000
- Missing: 0
- Invalid: 0

## Per task-policy cell

| task | policy | expected | valid | missing | invalid | malformed | rate | >1%? |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | ao_b16 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |
| gsm8k | ao_b32 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |
| gsm8k | ao_b8 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |
| gsm8k | ar_b1 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |
| gsm8k | random_b32 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |
| math500 | ao_b16 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |
| math500 | ao_b32 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |
| math500 | ao_b8 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |
| math500 | ar_b1 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |
| math500 | random_b32 | 1600 | 1600 | 0 | 0 | 0 | 0.0000 | no |

Triggered cells: none

## Differential missingness (descriptive, no p-values)

| task | ar_b1 - ao_b32 rate | flagged (>=1pp) | max |pairwise| gap |
|---|---:|---:|---|---|
| gsm8k | 0.0000 | no | ar_b1 vs ao_b8: 0.0000 |
| math500 | 0.0000 | no | ar_b1 vs ao_b8: 0.0000 |

A flagged differential would confound the primary contrast independently of any coverage mechanism; the frozen valid-output sensitivity is the designated remedy.

Operational missingness audit, not a Gate 1 analysis. Any cell above the frozen threshold means the frozen valid-output sensitivity will run in the final analysis.
