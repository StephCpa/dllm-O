"""Auditable runner for the frozen temperature-by-schedule extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator

from .artifacts import sha256_file, write_json_atomic
from .external import Gsm8kGrader, Math500Grader, validate_justgrpo_root
from .phase1_protocol import (
    JUSTGRPO_COMMIT,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_SEED,
    SPLIT_MANIFEST_SHA256,
    Phase1Policy,
    load_split_manifest,
)
from .phase1_runner import (
    _environment_fingerprint,
    _generate_once,
    _load_datasets,
    _load_model,
    _prompt_reference_and_hash,
    _repo_state,
    load_gsm8k_grader,
    load_math500_grader,
)
from .seeding import derive_rollout_seed, rollout_seed_key
from .temperature_extension_protocol import (
    ARTIFACT_SCHEMA_SHA256,
    POLICY_BY_NAME,
    PROTOCOL_VERSION,
    TemperaturePolicy,
    TemperatureWorkItem,
    build_work_items,
    validate_freeze,
)


def _hash_keys(keys: list[str]) -> str:
    payload = "\n".join(sorted(keys)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_key(item: TemperatureWorkItem) -> str:
    return rollout_seed_key(
        item.row_index,
        item.rollout_index,
        global_seed=PROTOCOL_SEED,
        protocol_prefix=f"dllm-order-phase1-v1|{item.task}",
    )


def _seed(item: TemperatureWorkItem) -> int:
    return derive_rollout_seed(
        item.row_index,
        item.rollout_index,
        global_seed=PROTOCOL_SEED,
        protocol_prefix=f"dllm-order-phase1-v1|{item.task}",
    )


def _result_paths(output_root: Path, item: TemperatureWorkItem) -> tuple[Path, Path]:
    directory = (
        output_root
        / PROTOCOL_VERSION
        / item.run_kind
        / item.task
        / item.policy_name
        / f"q{item.row_index:04d}"
    )
    stem = f"r{item.rollout_index:02d}"
    return directory / f"{stem}.json", directory / f"{stem}.trace.jsonl.gz"


def _load_validator(path: Path) -> Draft202012Validator:
    observed = sha256_file(path)
    if observed != ARTIFACT_SCHEMA_SHA256:
        raise RuntimeError(
            f"temperature schema hash mismatch: expected {ARTIFACT_SCHEMA_SHA256}, "
            f"found {observed}"
        )
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _policy_payload(policy: TemperaturePolicy) -> dict[str, Any]:
    config = asdict(policy.decode_config)
    return {
        "name": policy.name,
        "schedule": policy.schedule,
        "steps": config["steps"],
        "generation_length": config["gen_length"],
        "block_length": config["block_length"],
        "temperature": config["temperature"],
        "cfg_scale": config["cfg_scale"],
        "remasking": config["remasking"],
    }


def _validate_existing(
    *,
    result_path: Path,
    trace_path: Path,
    item: TemperatureWorkItem,
    freeze_sha256: str,
    validator: Draft202012Validator,
) -> None:
    record = json.loads(result_path.read_text(encoding="utf-8"))
    validator.validate(record)
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": item.run_kind,
        "temperature_extension_freeze_sha256": freeze_sha256,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "rollout_index": item.rollout_index,
        "seed_key": _seed_key(item),
        "policy": _policy_payload(POLICY_BY_NAME[item.policy_name]),
        "completion_status": "complete",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"existing result has mismatched {key}: {result_path}")
    source = record["source"]
    if (
        int(source["row_index"]) != item.row_index
        or source["split_role"] != item.split_role
        or source["prompt_sha256"] != item.prompt_sha256
    ):
        raise RuntimeError(f"existing result belongs to another item: {result_path}")
    if not trace_path.is_file() or sha256_file(trace_path) != record["trace_sha256"]:
        raise RuntimeError(f"existing result has invalid trace: {result_path}")


def _run_item(
    *,
    item: TemperatureWorkItem,
    rows: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    model: Any,
    graders: dict[str, Gsm8kGrader | Math500Grader],
    device: torch.device,
    output_root: Path,
    repo_state: dict[str, Any],
    environment: dict[str, Any],
    manifest: dict[str, Any],
    freeze_sha256: str,
    validator: Draft202012Validator,
) -> str:
    result_path, trace_path = _result_paths(output_root, item)
    if result_path.exists():
        _validate_existing(
            result_path=result_path,
            trace_path=trace_path,
            item=item,
            freeze_sha256=freeze_sha256,
            validator=validator,
        )
        return "skipped"
    if trace_path.exists():
        raise RuntimeError(f"orphan trace exists without result: {trace_path}")

    row = rows[item.task][item.row_index]
    prompt, reference, prompt_hash = _prompt_reference_and_hash(item.task, row)
    if prompt_hash != item.prompt_sha256:
        raise RuntimeError(f"prompt hash mismatch for {item.task} row {item.row_index}")
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    seed = _seed(item)
    temperature_policy = POLICY_BY_NAME[item.policy_name]
    policy = Phase1Policy(
        name=temperature_policy.name,
        decode_config=temperature_policy.decode_config,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output, event_count = _generate_once(
        model,
        prompt_ids,
        policy,
        seed,
        trace_path,
        full_distribution=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    item_environment = dict(environment)
    if item.run_kind == "temperature-smoke":
        untraced, _ = _generate_once(model, prompt_ids, policy, seed, None)
        equivalent = bool(torch.equal(output, untraced))
        item_environment["trace_equivalence_verified_for_this_result"] = equivalent
        if not equivalent:
            raise RuntimeError(f"trace changed temperature smoke output: {item.key}")

    completion_ids = output[:, prompt_ids.shape[1] :]
    response = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]
    parser_status = "ok"
    try:
        parsed_answer, correct = graders[item.task].grade(reference, response)
    except Exception as error:
        parser_status = f"error:{type(error).__name__}"
        parsed_answer = None
        correct = False

    source = manifest["sources"][item.task]
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": item.run_kind,
        "temperature_extension_freeze_sha256": freeze_sha256,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "source": {
            "dataset_id": source["dataset_id"],
            "dataset_revision": source["revision"],
            "dataset_split": source["split"],
            "row_index": item.row_index,
            "split_role": item.split_role,
            "prompt_sha256": item.prompt_sha256,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "policy": _policy_payload(temperature_policy),
        "rollout_index": item.rollout_index,
        "seed_key": _seed_key(item),
        "seed": seed,
        "response": response,
        "parsed_answer": parsed_answer,
        "parser_status": parser_status,
        "correct": bool(correct),
        "wall_time_seconds": elapsed,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "trace_event_count": event_count,
        "trace_sha256": sha256_file(trace_path),
        "runner_repository": repo_state,
        "external_repository_commit": JUSTGRPO_COMMIT,
        "environment_fingerprint": item_environment,
        "completion_status": "complete",
    }
    validator.validate(record)
    write_json_atomic(result_path, record)
    return "completed"


def _marker_path(
    output_root: Path,
    *,
    run_kind: str,
    task_selection: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    return (
        output_root
        / PROTOCOL_VERSION
        / "manifests"
        / run_kind
        / task_selection
        / f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    )


def _write_marker(
    output_root: Path,
    items: list[TemperatureWorkItem],
    *,
    run_kind: str,
    task_selection: str,
    shard_index: int,
    num_shards: int,
    validator: Draft202012Validator,
) -> Path:
    keys = [item.key for item in items]
    marker = {
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": run_kind,
        "task_selection": task_selection,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "expected_keys_sha256": _hash_keys(keys),
        "observed_keys_sha256": _hash_keys(keys),
        "result_count": len(keys),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    validator.validate(marker)
    path = _marker_path(
        output_root,
        run_kind=run_kind,
        task_selection=task_selection,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in marker.keys() - {"completed_at_utc"}:
            if existing.get(key) != marker[key]:
                raise RuntimeError(f"existing marker disagrees on {key}: {path}")
        return path
    write_json_atomic(path, marker)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--temperature-freeze", type=Path, required=True)
    parser.add_argument(
        "--run-kind",
        choices=["temperature-smoke", "temperature-formal"],
        required=True,
    )
    parser.add_argument("--task", choices=["all", "gsm8k", "math500"], default="all")
    parser.add_argument("--justgrpo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--gsm8k-parquet", type=Path)
    parser.add_argument("--math500-parquet", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_split_manifest(args.split_manifest)
    validator = _load_validator(args.artifact_schema)
    repo_root = Path(__file__).resolve().parents[3]
    repo_state = _repo_state(repo_root)
    freeze_sha = validate_freeze(
        args.temperature_freeze,
        runner_commit=None if args.dry_run else repo_state["commit"],
    )
    items = build_work_items(
        manifest,
        run_kind=args.run_kind,
        task_selection=args.task,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.max_items is not None:
        if args.max_items < 1:
            raise RuntimeError("--max-items must be positive")
        if not args.dry_run:
            raise RuntimeError(
                "--max-items is dry-run only; truncated executions must not write markers"
            )
        items = items[: args.max_items]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_kind": args.run_kind,
                    "task_selection": args.task,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "generation_count": len(items),
                    "decoder_execution_count": len(items)
                    * (2 if args.run_kind == "temperature-smoke" else 1),
                    "policies": sorted(
                        {
                            _policy_payload(POLICY_BY_NAME[item.policy_name])["name"]
                            for item in items
                        }
                    ),
                    "temperature_extension_freeze_sha256": freeze_sha,
                    "expected_work_keys_sha256": _hash_keys(
                        [item.key for item in items]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if repo_state["dirty"]:
        raise RuntimeError("temperature runner repository must be clean")
    justgrpo_root = validate_justgrpo_root(args.justgrpo_root)
    rows = _load_datasets(
        manifest,
        items,
        gsm8k_parquet=args.gsm8k_parquet,
        math500_parquet=args.math500_parquet,
    )
    device = torch.device(args.device)
    tokenizer, model, model_source_mode = _load_model(device, args.model_path)
    graders: dict[str, Gsm8kGrader | Math500Grader] = {}
    if "gsm8k" in rows:
        graders["gsm8k"] = load_gsm8k_grader(justgrpo_root)
    if "math500" in rows:
        graders["math500"] = load_math500_grader(justgrpo_root)
    environment = _environment_fingerprint(
        device,
        {
            "model": model_source_mode,
            "gsm8k": "local_pinned_parquet" if "gsm8k" in rows else "unused",
            "math500": "local_pinned_parquet" if "math500" in rows else "unused",
        },
    )

    completed = skipped = 0
    for item in items:
        status = _run_item(
            item=item,
            rows=rows,
            tokenizer=tokenizer,
            model=model,
            graders=graders,
            device=device,
            output_root=args.output,
            repo_state=repo_state,
            environment=environment,
            manifest=manifest,
            freeze_sha256=freeze_sha,
            validator=validator,
        )
        completed += status == "completed"
        skipped += status == "skipped"
        print(
            f"work_key={item.key} completed={completed} skipped={skipped}",
            flush=True,
        )

    marker = _write_marker(
        args.output,
        items,
        run_kind=args.run_kind,
        task_selection=args.task,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        validator=validator,
    )
    print(f"shard_completion={marker}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
