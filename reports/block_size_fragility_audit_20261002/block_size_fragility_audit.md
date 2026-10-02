# Block-Size Primary: Fragility Audit

**Status:** post-outcome exploratory, CPU-only. Queries are removed in
order of their observed effect, so this describes how concentrated the
frozen effect is. It is not a test, an outlier rule, or a replacement for
the frozen endpoint.

Contrast: `CoverageAUC_K64(ao_b32) - CoverageAUC_K64(ar_b1)`. Bootstrap: task-stratified percentile,
10,000 replicates, seed `20261002` (+ number removed).

Full panel (200 queries): -0.0442 [-0.0702, -0.0197].

## Concentration

- Queries with |effect| < 0.01: 134 of 200.
- Queries with effect < -0.05: 37; with effect > +0.05: 16.

| most negative queries | share of the summed effect |
|---:|---:|
| 1 | 0.09 |
| 5 | 0.41 |
| 10 | 0.71 |
| 20 | 1.12 |
| 40 | 1.40 |

Shares above 1 mean those queries carry more than the net effect, which
queries favoring $B=32$ partly offset.

## Removing the most negative queries first

- The interval first reaches zero after removing **9** queries.
- The point estimate first becomes nonnegative after removing **17** queries.

| removed | GSM8K / MATH-500 removed | last removed effect | pooled estimate [95% CI] |
|---:|---:|---:|---:|
| 0 | 0 / 0 |  | -0.0442 [-0.0702, -0.0197] |
| 1 | 1 / 0 | -0.8252 | -0.0403 [-0.0641, -0.0175] |
| 2 | 2 / 0 | -0.8027 | -0.0364 [-0.0591, -0.0146] |
| 3 | 3 / 0 | -0.7063 | -0.0329 [-0.0551, -0.0118] |
| 4 | 3 / 1 | -0.7063 | -0.0296 [-0.0510, -0.0091] |
| 5 | 4 / 1 | -0.5958 | -0.0266 [-0.0471, -0.0066] |
| 6 | 4 / 2 | -0.5958 | -0.0237 [-0.0436, -0.0045] |
| 7 | 5 / 2 | -0.5489 | -0.0209 [-0.0393, -0.0022] |
| 8 | 5 / 3 | -0.5105 | -0.0184 [-0.0365, -0.0004] |
| 9 | 5 / 4 | -0.5105 | -0.0159 [-0.0336, +0.0017] |
| 10 | 5 / 5 | -0.5105 | -0.0133 [-0.0309, +0.0042] |
| 11 | 6 / 5 | -0.4152 | -0.0112 [-0.0278, +0.0057] |
| 12 | 6 / 6 | -0.3887 | -0.0092 [-0.0253, +0.0071] |
| 13 | 6 / 7 | -0.3875 | -0.0072 [-0.0230, +0.0088] |
| 14 | 6 / 8 | -0.3875 | -0.0051 [-0.0206, +0.0103] |
| 15 | 6 / 9 | -0.3875 | -0.0030 [-0.0177, +0.0125] |
| 16 | 6 / 10 | -0.3875 | -0.0008 [-0.0152, +0.0147] |
| 17 | 6 / 11 | -0.3587 | +0.0013 [-0.0123, +0.0163] |
| 18 | 6 / 12 | -0.3263 | +0.0032 [-0.0101, +0.0177] |
| 19 | 6 / 13 | -0.2922 | +0.0049 [-0.0079, +0.0196] |
| 20 | 7 / 13 | -0.2848 | +0.0064 [-0.0063, +0.0204] |
| 21 | 8 / 13 | -0.2574 | +0.0078 [-0.0046, +0.0215] |
| 22 | 8 / 14 | -0.2313 | +0.0093 [-0.0028, +0.0232] |
| 23 | 9 / 14 | -0.2068 | +0.0104 [-0.0014, +0.0241] |
| 24 | 10 / 14 | -0.1915 | +0.0114 [-0.0001, +0.0249] |
| 25 | 10 / 15 | -0.1804 | +0.0126 [+0.0013, +0.0261] |
| 30 | 12 / 18 | -0.1179 | +0.0171 [+0.0059, +0.0309] |
| 35 | 15 / 20 | -0.0708 | +0.0206 [+0.0097, +0.0338] |
| 40 | 16 / 24 | -0.0214 | +0.0227 [+0.0114, +0.0371] |

## Interpretation boundary

A heavy-tailed effect is expected to be fragile under outcome-selected
removal, so this number does not by itself weaken the frozen interval,
which already accounts for query-level sampling variability. It does
show that the effect is carried by a small subset of the 200 queries.
Therefore the most informative robustness checks are new queries or a
second model, not further reanalysis of this panel.

Once enough negative queries are removed, the remaining contrast becomes
positive and eventually resolves above zero. This is produced by the
selection itself: dropping the most negative values always raises the
mean. It must not be read as evidence that $B=32$ helps most queries.
The opposite reference curve (removing the most positive queries first)
is retained in the JSON.
