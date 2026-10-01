# Post-Review CPU Audits

**Status:** post-outcome exploratory analyses. No frozen endpoint, interval,
or verdict is replaced by this report.

## Random-extension independence checks

| k | New-48 RND - CF | Old-16 minus new-48 RND |
|---:|---:|---:|
| 1 | -0.0916 [-0.1185, -0.0648] | -0.0006 [-0.0140, +0.0125] |
| 2 | -0.0405 [-0.0684, -0.0131] | +0.0038 [-0.0097, +0.0176] |
| 4 | -0.0037 [-0.0327, +0.0252] | +0.0084 [-0.0072, +0.0244] |
| 8 | +0.0215 [-0.0098, +0.0531] | +0.0137 [-0.0044, +0.0338] |
| 16 | +0.0358 [+0.0027, +0.0694] | +0.0151 [-0.0123, +0.0437] |
| 32 | +0.0399 [+0.0045, +0.0769] | n/a |

New-48 mean over k={16,32}: +0.0378 [+0.0048, +0.0717]

## Block-size grid sensitivity

| Grid | B32 - B1 |
|---|---:|
| `low_k1_2_4_8` | -0.0196 [-0.0402, +0.0000] |
| `frozen_k4_8_16_32_64` | -0.0442 [-0.0698, -0.0203] |
| `high_k16_32_64` | -0.0470 [-0.0790, -0.0162] |

## MATH-500 difficulty stratification

Descriptive only; strata come from the frozen split manifest.

| Level | n | Sequential CAUC | CF CAUC | RND CAUC | EF CAUC |
|---|---:|---:|---:|---:|---:|
| Level 1 | 10 | 0.998 | 1.000 | 0.995 | 0.987 |
| Level 2 | 19 | 0.754 | 0.769 | 0.766 | 0.715 |
| Level 3 | 20 | 0.716 | 0.704 | 0.681 | 0.683 |
| Level 4 | 23 | 0.513 | 0.386 | 0.460 | 0.433 |
| Level 5 | 28 | 0.458 | 0.412 | 0.443 | 0.356 |

## Generation accounting

| Stage | New generations |
|---|---:|
| `phase0_signal` | 3,200 |
| `phase1_calibration` | 6,400 |
| `phase1_heldout_screening` | 16,000 |
| `phase1_primary_extension` | 19,200 |
| `phase2_block_size_extension` | 19,200 |
| `condition_a_entropy_first` | 12,800 |
| `random_policy_extension` | 9,600 |
| **Total** | **86,400** |

The JSON additionally contains per-k block contrasts, split-half checks,
tie-aware majority-vote accuracy, parser/empty-response summaries, decoded
lengths, and per-query success-count distributions.
