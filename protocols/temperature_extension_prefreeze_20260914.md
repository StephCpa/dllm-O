# Temperature-by-Schedule Extension: Pre-Freeze Protocol

**Protocol version:** `temperature-extension-freeze-v1`  
**Status:** pre-outcome-read specification; not yet frozen or authorized for GPU launch.  
**Selection status:** post-review prospective follow-up.

## 1. Question and evidence status

The existing fixed-NFE study found that confidence-first and random eligible-position
selection have different precision--coverage profiles at temperature 0.6. This
follow-up tests the strongest alternative explanation: whether an ordinary change in
sampling temperature can account for the apparent schedule effect.

The study was selected after the original outcomes and reviewer critique were known.
It is prospective only for the new temperature 0.9 and 1.2 outcomes, is not part of
the original Gate-1 chain, and cannot revise any earlier frozen verdict.

## 2. Frozen generation design

- Model: `GSAI-ML/LLaDA-8B-Instruct`, revision
  `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- Decoder/evaluator: `LeapLabTHU/JustGRPO`, commit
  `1a2fddb5c6655597e63081c0af5ebb718a849f39`.
- Data: the existing frozen 100 GSM8K and 100 MATH-500 held-out queries.
- New temperatures: `T in {0.9, 1.2}`.
- Schedules at each temperature:
  - sequential `B=1`, confidence remasking;
  - confidence-first `B=32`, confidence remasking;
  - random `B=32`, random remasking.
- Fixed generation settings: 256 generated positions, 256 denoising steps, CFG 0,
  one committed token per NFE, and the existing parser and grader.
- New rollout budget: 32 per query and cell.
- New workload: `2 tasks x 100 queries x 2 temperatures x 3 schedules x 32 =
  38,400 generations`.
- Formal partition: six deterministic shards of 6,400 generations.
- Existing temperature-0.6 records: the first 32 rollouts of `ar_b1`, `ao_b32`,
  and `random_b32`; these are reused only after a no-outcome-read, checksummed
  inventory is generated and bound into the machine freeze.

The same Phase-1 seed key is used for a task/query/rollout across all cells. This is
paired seed construction, not a claim that the random trajectory remains identical
after temperature or schedule decisions cause decoding paths to diverge.

## 3. Frozen primary family

For each new temperature, define paired query-level random-minus-confidence-first
contrasts at two endpoints:

- low-budget endpoint: `Pass@1`;
- high-budget endpoint: `mean(Pass@16, Pass@32)`.

The four endpoints (two temperatures by two budget summaries) form one family.
Aggregation is the equal-weight mean of the GSM8K and MATH-500 task means. Uncertainty
uses a 10,000-replicate task-stratified paired-query bootstrap, seed `20260914`, with
nonstudentized maximum-absolute-deviation simultaneous 95% bands across all four
endpoints.

### Frozen verdict per temperature

- `resolved_reversal`: the simultaneous interval for the low-budget contrast lies
  entirely below zero and the simultaneous interval for the high-budget contrast
  lies entirely above zero.
- `not_resolved_as_reversal`: every other outcome, reported with estimates and
  intervals. This label does not mean equivalence or no effect.

No aggregate coverage AUC spanning the crossing is used as a primary endpoint.

## 4. Secondary endpoints

All secondary endpoints are reported regardless of direction and cannot modify the
primary verdict:

1. Per-task primary-family point estimates.
2. Pass@k curves for all schedules and temperatures at
   `k in {1,2,4,8,16,32}`.
3. Within-temperature per-k schedule contrasts.
4. Cross-temperature empirical frontier comparisons, including higher-temperature
   confidence-first versus temperature-0.6 random.
5. Parsed-answer diversity and majority-vote accuracy; ties among modal parsed
   answers receive fractional correctness across the tied modes.
6. Parser-status, empty-response, decoded-length, runtime, and artifact-completeness
   summaries.

The study does not use an equivalence margin and cannot establish that temperature
and schedule are universally independent.

## 5. Artifact and launch gates

Every new result must validate against
`temperature_extension_artifact_schema_v1.json` and bind the machine freeze,
split manifest, model revision, external decoder commit, runner commit, source row
and prompt hash, policy, temperature, rollout index, seed key, trace hash, and
environment fingerprint.

Before formal launch:

1. Commit the runner, analyzer, schema, tests, and this protocol; the inference
   checkout must be clean at that commit.
2. Run the full test suite.
3. Build and verify the 19,200-record temperature-0.6 control inventory without
   accessing `correct`, `response`, or `parsed_answer` as analysis fields.
4. Emit the immutable machine freeze bound to the committed runner and inventory.
5. Dry-run all six formal shards; total cardinality must be 38,400 and every shard
   cardinality 6,400.
6. Run the 12-record calibration smoke (six new cells on one query per task), including
   traced/untraced output equivalence.
7. Validate all 12 smoke records and its single marker without reading correctness.
8. Launch formal shards only after the smoke audit passes.

The implementation-level work plan currently closes at 38,400 unique keys with
global SHA-256 `8c856aa5728249147804cdcc18791745844362a4349b713e26f691c8ef53d765`.
The six 6,400-key shard hashes are, in index order:

1. `81b6daa39d3f69e3629bd0081c495bb5c0bbec883a11d0a11b432e47a2cf6d11`;
2. `de10c6a922fa3628154b843337d74930b466093a32274a10607686b3f5702d54`;
3. `4212803e255b38f908e7ac7d20b3c1bd0f53c44cfe534f867ad22b8137f4b646`;
4. `2dd7a246a30bac04b43bc06d44687f0012d800dceb660f41f99b470dae799aac`;
5. `fb391fb7e2a1f2ff0ee563f3c89f0049a7082aa1e9a324a70c9009ad4e29adbe`;
6. `e51132515eb1b16fa7ca2da2cc3d50ee23f2e78e04f5c265292a5ba5efe8b03e`.

These hashes must be recomputed after checkout on the server and must match before
launch.

Formal analysis is blocked unless all 38,400 new records, all six formal markers, and
all 19,200 bound control artifacts validate. Parser failures and empty responses are
retained as incorrect; no query or rollout is removed after outcomes are read.

## 6. Recovery and scope boundaries

- Existing valid new records are validated and skipped on resume.
- Orphan traces are quarantined before rerunning the same work key.
- Changing only `--device` is allowed for recovery; work key, seed, frozen commit,
  data, model, and all other semantic arguments remain unchanged.
- The first `analyze` invocation must append its correctness-read timestamp before
  outcomes are loaded.
- No new temperature, policy, task, endpoint, exclusion, or repair arm may be added
  after the first outcome read.
- Conclusions are scoped to this model, these two tasks, this temperature grid, and
  fixed-256-NFE decoding.

## 7. Resource estimate

At the previously observed 26--32 seconds per generation, 38,400 generations require
approximately 277--341 aggregate GPU-hours. This corresponds to about 46--57 hours
on six continuously available GPUs, 92--114 hours on three GPUs, or 12--14 days on
one GPU. Smoke timing should replace this estimate before formal scheduling.

## 8. Command sequence after code commit

The following is a template. Paths must be set to the pinned server locations before
execution.

```bash
export PYTHONPATH="$PWD/dllm_order_transmission/src"
export DT=dllm_order_transmission
export PROTO="$DT/protocols"
export SPLIT="$PROTO/phase1_split_manifest_v1.json"
export SCHEMA="$PROTO/temperature_extension_artifact_schema_v1.json"
export CONTROL_ROOT="$DT/outputs/dllm_order_temperature_extension/control"
export INVENTORY="$CONTROL_ROOT/temperature_control_inventory_v1.json"
export FREEZE="$CONTROL_ROOT/temperature_extension_freeze_v1.json"
export PHASE1_ROOT="$DT/outputs/dllm_order_phase1"
export TEMP_ROOT="$DT/outputs/dllm_order_temperature_extension"
export JUSTGRPO_ROOT=/path/to/JustGRPO
export MODEL_PATH=/path/to/LLaDA-8B-Instruct
export GSM8K_PARQUET=/path/to/gsm8k.parquet
export MATH500_PARQUET=/path/to/math500.parquet
mkdir -p "$CONTROL_ROOT"
```

The generated inventory and operative freeze deliberately live outside the tracked
source tree. This lets the inference checkout remain clean at the runner commit named
inside the freeze. After inference, immutable copies can be committed from a separate
control-plane worktree without changing the operative checkout.

Build the reused-control inventory and emit the freeze:

```bash
python -m dllm_order_transmission.temperature_extension_analyzer \
  --mode build-control-inventory --repo-root "$PWD" \
  --split-manifest "$SPLIT" --artifact-schema "$SCHEMA" \
  --control-inventory "$INVENTORY" --phase1-root "$PHASE1_ROOT" \
  --invocation-log "$CONTROL_ROOT/invocation.jsonl"

python -m dllm_order_transmission.temperature_extension_analyzer \
  --mode emit-freeze --repo-root "$PWD" \
  --split-manifest "$SPLIT" --artifact-schema "$SCHEMA" \
  --control-inventory "$INVENTORY" --freeze "$FREEZE"
```

Dry-run a formal shard and run the smoke after the freeze exists:

```bash
python -m dllm_order_transmission.temperature_extension_runner \
  --split-manifest "$SPLIT" --artifact-schema "$SCHEMA" \
  --temperature-freeze "$FREEZE" --run-kind temperature-formal \
  --task all --output "$TEMP_ROOT" --justgrpo-root "$JUSTGRPO_ROOT" \
  --shard-index 0 --num-shards 6 --dry-run

python -m dllm_order_transmission.temperature_extension_runner \
  --split-manifest "$SPLIT" --artifact-schema "$SCHEMA" \
  --temperature-freeze "$FREEZE" --run-kind temperature-smoke \
  --task all --output "$TEMP_ROOT" --justgrpo-root "$JUSTGRPO_ROOT" \
  --model-path "$MODEL_PATH" --gsm8k-parquet "$GSM8K_PARQUET" \
  --math500-parquet "$MATH500_PARQUET" --device cuda:0
```

Validate the smoke without reading correctness:

```bash
python -m dllm_order_transmission.temperature_extension_analyzer \
  --mode validate-smoke --repo-root "$PWD" \
  --split-manifest "$SPLIT" --artifact-schema "$SCHEMA" \
  --control-inventory "$INVENTORY" --freeze "$FREEZE" \
  --temperature-root "$TEMP_ROOT" \
  --invocation-log "$TEMP_ROOT/invocation.jsonl"
```

Formal shards use the same runner command with `--run-kind temperature-formal`,
`--num-shards 6`, and paired `--shard-index 0..5` / authorized `--device` values.
