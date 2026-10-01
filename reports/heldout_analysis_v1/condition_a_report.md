# Condition A Results

- Freeze SHA-256: `a1205bd8bc5b6d9d81fdfc63c842efc3ede8cc59f192f80523b61151bf8d7be5`
- Valid artifacts: 12800/12800

## Primary

- EF-B32 minus AO-B32 CoverageAUC64: -0.00986 (95% CI [-0.04200, 0.02377])
- Classification: `straddles_zero`
- Pre-registered reading: no resolved entropy-first benefit; confidence-guided position choice is not identified as the source of the coverage cost

## Matched 16-rollout random reference

- EF-B32 minus random-B32 ScreeningCoverageAUC16: -0.06206 (95% CI [-0.09139, -0.03297])

## Length guard

- Triggered: `True`
- All policies commit exactly 256 generated positions by design; the guard tests decoded-response whitespace length and does not remove outcomes.
