# dLLM Order Transmission Analysis Code

CPU-tested research infrastructure for the prospectively frozen Phase 0 study and
the pre-launch Phase 1 design in:

The corresponding protocols are in `protocols/`; reported analyses are in
`reports/`.

The decoder follows the public JustGRPO sampling semantics at commit
`1a2fddb5c6655597e63081c0af5ebb718a849f39`, while adding explicit random
generators and output-preserving trace callbacks. No model weights or benchmark
outputs are stored in this module.

Temperature-extension trace note: `candidate_probabilities` is a policy-dependent
selection score. For confidence-first it is the sampled candidate token's model
probability; for random selection it is a random score used to choose an eligible
position. It must not be compared across policies as a calibrated probability.

`low_confidence_resample` is the CF-resample control used by the E2 draft
protocol. It ranks and selects positions exactly as `low_confidence` does, then
commits a fresh draw from the same tempered distribution at the selected
position instead of the winning candidate. At temperature 0 it is identical to
`low_confidence`. No frozen protocol uses it.

Run the local tests with:

```bash
python -m pytest
```

## Server setup

Pin the external evaluator and install its grading dependencies:

```bash
git clone https://github.com/LeapLabTHU/JustGRPO.git external/JustGRPO
git -C external/JustGRPO checkout 1a2fddb5c6655597e63081c0af5ebb718a849f39
python -m pip install -r external/JustGRPO/requirements.txt
python -m pip install -e .
```

The model and dataset are loaded at the revisions frozen in `protocol.py`.
On servers without Hugging Face connectivity, pass verified local snapshots with
`--model-path` and `--dataset-path`. These flags change only input transport;
the frozen model/dataset IDs and revisions remain in every result record.

## Stage 0A

Run instrumentation equivalence on one GPU before any signal run:

```bash
CUDA_VISIBLE_DEVICES=0 dllm-phase0 \
  --stage equivalence \
  --justgrpo-root dllm_order_transmission/external/JustGRPO \
  --output dllm_order_transmission/outputs/dllm_order_phase0 \
  --model-path /path/to/verified/LLaDA-8B-Instruct \
  --dataset-path /path/to/gsm8k-740312 \
  --device cuda:0
```

Do not begin Stage 0B until all 40 result records report
`trace_equivalent=true` and the trace invariant audit passes.

## Stage 0B sharding

Each shard owns complete query-rollout pairs and runs both policies with the
same seed. For three GPUs, launch one process per visible GPU with shard indices
0, 1, and 2 and `--num-shards 3`.

## Frozen analysis

```bash
dllm-phase0-analyze \
  --input dllm_order_transmission/outputs/dllm_order_phase0 \
  --stage signal \
  --output-json dllm_order_transmission/outputs/dllm_order_phase0/phase0_analysis.json \
  --output-md dllm_order_transmission/outputs/dllm_order_phase0/phase0_analysis.md
```

Do not use `--allow-incomplete` for the final Phase 0 decision.

## Phase 1 planning

The Phase 1 split manifest and resource report are generated deterministically
from the pinned parquet files:

```bash
dllm-phase1-plan \
  --gsm8k-parquet /path/to/gsm8k-test-740312.parquet \
  --math500-parquet /path/to/math500-test-412eb03.parquet \
  --output-manifest dllm_order_transmission/protocols/phase1_split_manifest_v1.json \
  --output-report dllm_order_transmission/reports/phase1_cpu_planning_20260823.md
```

Verify the frozen design before implementation or inference:

```bash
cd dllm_order_transmission/protocols
shasum -a 256 -c phase1_design_hashes_v1.sha256
```

The current Phase 0 runner is not a Phase 1 launcher. Complete the blocking items
in `reports/phase1_runner_gap_audit_20260823.md` before running Phase 1A on GPUs.

## Phase 1 runner

Audit a launch without loading a model:

```bash
dllm-phase1 \
  --split-manifest dllm_order_transmission/protocols/phase1_split_manifest_v1.json \
  --artifact-schema dllm_order_transmission/protocols/phase1_artifact_schema_v1.json \
  --split-role calibration \
  --run-kind smoke \
  --task all \
  --justgrpo-root dllm_order_transmission/external/JustGRPO \
  --output dllm_order_transmission/outputs/dllm_order_phase1 \
  --dry-run
```

The pre-launch smoke is exactly ten generations: one calibration query from each
task under all five policies. Every generation is repeated without tracing and
must match exactly:

```bash
CUDA_VISIBLE_DEVICES=0 dllm-phase1 \
  --split-manifest dllm_order_transmission/protocols/phase1_split_manifest_v1.json \
  --artifact-schema dllm_order_transmission/protocols/phase1_artifact_schema_v1.json \
  --split-role calibration \
  --run-kind smoke \
  --task all \
  --justgrpo-root dllm_order_transmission/external/JustGRPO \
  --output dllm_order_transmission/outputs/dllm_order_phase1 \
  --model-path /path/to/verified/LLaDA-8B-Instruct \
  --gsm8k-parquet /path/to/gsm8k-test-740312.parquet \
  --math500-parquet /path/to/math500-test-412eb03.parquet \
  --device cuda:0
```

Held-out commands require `--bci-freeze`. Primary-extension jobs also refuse to
start unless all 16 matching screening rollouts already exist and pass schema and
trace-hash validation.

## Phase 1 calibration freeze

After all Phase 1A calibration shards complete, fit the calibration-only BCI and
predictors before any held-out inference:

```bash
dllm-phase1-calibrate \
  --phase0-root dllm_order_transmission/outputs/dllm_order_phase0 \
  --phase1-root dllm_order_transmission/outputs/dllm_order_phase1 \
  --split-manifest dllm_order_transmission/protocols/phase1_split_manifest_v1.json \
  --repo-root . \
  --output-report-json dllm_order_transmission/outputs/dllm_order_phase1/phase1-v1/calibration-analysis/report.json \
  --output-report-md dllm_order_transmission/outputs/dllm_order_phase1/phase1-v1/calibration-analysis/report.md \
  --output-freeze dllm_order_transmission/outputs/dllm_order_phase1/phase1-v1/calibration-analysis/phase1_bci_freeze_v1.json
```

The command validates every AR-B1/AO-B32 result and every AO-B32 trace used by
the analysis, refuses to overwrite an existing report or freeze, and records
aggregate hashes for both artifact sources. It deliberately does not inspect
secondary-policy outcomes while constructing the freeze. Calibration fits use
the explicitly named three-point
`ScreeningCoverageAUC` (`Pass@4/8/16`); the formal held-out endpoint remains the
five-point `CoverageAUC` (`Pass@4/8/16/32/64`). The generated freeze belongs in
the ignored output tree and binds the clean analysis-code commit. Keep its
printed SHA-256 with the launch record, and pass the same file to every held-out
runner command with `--bci-freeze`.

Candidate-control matching is within task/query/rollout/block and balances both
initial eligible entropy and negative-log eligible margin, in addition to block
position and commitment rank. This prevents the initial terms that define the
candidate score from mechanically determining the closure contrast.
