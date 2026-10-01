# Frozen Recipe Resolution Check (Pre-Unblinding Planning Note)

- Simulation seed: `20260820`
- Replicates: 10,000
- Latent bivariate-normal correlation: 0.3
- Queries per task: 100
- Status: pre-unblinding planning documentation; not a frozen endpoint and cannot modify Gate 1

## Empirical half-widths of the frozen stratified bootstrap CI

| scope | point | CI low | CI high | half-width |
|---|---:|---:|---:|---:|
| gsm8k (n=100) | 0.3308 | 0.1502 | 0.4944 | 0.1721 |
| math500 (n=100) | 0.3227 | 0.1307 | 0.4917 | 0.1805 |
| pooled stratified recipe | 0.3267 | 0.1985 | 0.4420 | 0.1218 |

## Fisher-z planning MDEs (for comparison)

- Per-task n=100: 0.277
- Pooled n=200 approximation: 0.197

Half-widths are for the frozen average-rank stratified statistic under a bivariate-normal copula. The Fisher-z MDEs are single-sample approximations, not calibrated to the stratified recipe; the empirical half-widths here measure the actual recipe's resolution at the stated latent correlation.
