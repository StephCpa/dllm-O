# Query-Level Exploratory Audits

**Status:** post-outcome exploratory, CPU-only. These analyses do not
replace or reinterpret any frozen endpoint.

Bootstrap seed: `20261001`; replicates: `10000`.

## Block-size effect distribution

For each held-out query, this report averages $B=32-B=1$ over $k\in\{4,8,16,32,64\}$. The pooled query-bootstrap estimate is -0.0442 [-0.0697, -0.0199].

| quantity | value |
|---|---:|
| queries | 200 |
| mean / median | -0.0442 / +0.0000 |
| q10 / q25 | -0.2602 / -0.0000 |
| q75 / q90 | +0.0000 / +0.0282 |
| positive / negative / zero | 72 / 55 / 73 |

This describes heterogeneity; it is not a query-level significance test.

## Influence

The largest absolute leave-one-query-out change is `0.003930` on the pooled primary-grid estimate. The ten largest cases are retained in JSON for audit, but are not treated as outlier exclusions.

## Dispersion diagnostic

The variance ratio compares between-query success-fraction variance with the binomial reference at 64 rollouts. The beta-binomial moment rho is descriptive only; no beta-binomial null is fitted or used for inference.

| task | policy | mean success | variance ratio | rho | zero-success queries |
|---|---|---:|---:|---:|---:|
| gsm8k | ar_b1 | 0.725 | 32.24 | +0.4959 | 0 |
| gsm8k | ao_b32 | 0.732 | 42.63 | +0.6608 | 4 |
| gsm8k | random_b32 | 0.644 | 27.58 | +0.4219 | 1 |
| gsm8k | ef_b32 | 0.498 | 20.26 | +0.3057 | 1 |
| math500 | ar_b1 | 0.360 | 42.05 | +0.6517 | 25 |
| math500 | ao_b32 | 0.385 | 49.85 | +0.7754 | 30 |
| math500 | random_b32 | 0.286 | 35.40 | +0.5460 | 25 |
| math500 | ef_b32 | 0.222 | 28.63 | +0.4386 | 31 |

## Majority-vote paired intervals

Unadjusted query-bootstrap intervals for descriptive, tie-aware majority-vote differences:

| k | Sequential - RND | CF - RND |
|---:|---:|---:|
| 1 | +0.0750 [+0.0050, +0.1400] | +0.0900 [+0.0300, +0.1550] |
| 16 | +0.0208 [-0.0167, +0.0600] | -0.0039 [-0.0497, +0.0442] |
| 32 | +0.0050 [-0.0350, +0.0450] | -0.0275 [-0.0750, +0.0200] |
| 64 | +0.0190 [-0.0175, +0.0555] | -0.0260 [-0.0675, +0.0140] |

## Same-policy A/A split

Each policy is split into rollout indices 0--31 versus 32--63. These are
calibration diagnostics, not independent replication or evidence of a
policy effect.

| policy | k=1 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|
| ar_b1 | +0.0023 [-0.0066, +0.0116] | +0.0074 [-0.0068, +0.0223] | +0.0066 [-0.0118, +0.0264] | +0.0049 [-0.0175, +0.0284] |
| ao_b32 | +0.0084 [+0.0005, +0.0164] | +0.0104 [-0.0021, +0.0240] | +0.0091 [-0.0083, +0.0272] | +0.0030 [-0.0198, +0.0270] |
| random_b32 | +0.0053 [-0.0064, +0.0169] | +0.0014 [-0.0151, +0.0182] | -0.0014 [-0.0216, +0.0192] | -0.0130 [-0.0382, +0.0122] |
