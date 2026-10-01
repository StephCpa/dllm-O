"""Auditable runner for the prospectively frozen Phase 1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator

from .artifacts import AtomicJsonlTraceSink, sha256_file, write_json_atomic
from .decoding import generate_with_trace
from .external import (
    Gsm8kGrader,
    Math500Grader,
    load_gsm8k_grader,
    load_math500_grader,
    validate_justgrpo_root,
)
from .phase1_protocol import (
    ARTIFACT_SCHEMA_SHA256,
    JUSTGRPO_COMMIT,
    MODEL_ID,
    MODEL_REVISION,
    PHASE2_ARTIFACT_SCHEMA_SHA256,
    POLICY_BY_NAME,
    PROTOCOL_SEED,
    PROTOCOL_VERSION,
    RANDOM_EXTENSION_ARTIFACT_SCHEMA_SHA256,
    SPLIT_MANIFEST_SHA256,
    Phase1Policy,
    Phase1WorkItem,
    build_phase1_work_items,
    load_split_manifest,
    validate_bci_freeze,
    validate_phase2_freeze,
    validate_random_extension_freeze,
)
from .seeding import derive_rollout_seed, rollout_seed_key


MATH_INSTRUCTION = (
    r"(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_keys(keys: list[str]) -> str:
    payload = "\n".join(sorted(keys)) + "\n"
    return _sha256_text(payload)


def _repo_state(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _environment_fingerprint(
    device: torch.device, input_modes: dict[str, str]
) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "pyarrow", "jsonschema", "sympy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "input_source_modes": input_modes,
    }


def _load_artifact_validator(
    path: Path, run_kind: str = "primary-extension"
) -> Draft202012Validator:
    if run_kind == "phase2-extension":
        expected_hash = PHASE2_ARTIFACT_SCHEMA_SHA256
    elif run_kind == "random-extension":
        expected_hash = RANDOM_EXTENSION_ARTIFACT_SCHEMA_SHA256
    else:
        expected_hash = ARTIFACT_SCHEMA_SHA256
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            "artifact-schema hash mismatch for run kind "
            f"{run_kind}: expected {expected_hash}, found {observed_hash}"
        )
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _load_parquet_rows(
    path: Path, expected_hash: str, expected_count: int
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"dataset parquet hash mismatch: {path}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required for Phase 1 dataset loading") from error
    rows = parquet.read_table(path).to_pylist()
    if len(rows) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} rows in {path}, found {len(rows)}"
        )
    return rows


def _load_datasets(
    manifest: dict[str, Any],
    items: list[Phase1WorkItem],
    *,
    gsm8k_parquet: Path | None,
    math500_parquet: Path | None,
) -> dict[str, list[dict[str, Any]]]:
    required_tasks = {item.task for item in items}
    paths = {"gsm8k": gsm8k_parquet, "math500": math500_parquet}
    rows: dict[str, list[dict[str, Any]]] = {}
    for task in sorted(required_tasks):
        path = paths[task]
        if path is None:
            raise ValueError(f"--{task}-parquet is required for selected work")
        source = manifest["sources"][task]
        rows[task] = _load_parquet_rows(
            path.resolve(), source["parquet_sha256"], source["row_count"]
        )
    return rows


def _load_model(device: torch.device, model_path: Path | None) -> tuple[Any, Any, str]:
    from transformers import AutoModel, AutoTokenizer

    source = str(model_path.resolve()) if model_path is not None else MODEL_ID
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if model_path is None:
        kwargs["revision"] = MODEL_REVISION
        source_mode = "hub_pinned_revision"
    else:
        if not model_path.resolve().is_dir():
            raise FileNotFoundError(f"missing model snapshot: {model_path}")
        kwargs["local_files_only"] = True
        source_mode = "local_verified_snapshot"
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(source, torch_dtype=dtype, **kwargs)
    model.eval().requires_grad_(False).to(device)
    return tokenizer, model, source_mode


def _prompt_reference_and_hash(task: str, row: dict[str, Any]) -> tuple[str, str, str]:
    if task == "gsm8k":
        raw_prompt = str(row["question"])
        return raw_prompt, str(row["answer"]), _sha256_text(raw_prompt)
    if task == "math500":
        raw_prompt = str(row["problem"])
        return (
            raw_prompt + MATH_INSTRUCTION,
            str(row["solution"]),
            _sha256_text(raw_prompt),
        )
    raise ValueError(f"unknown task: {task}")


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def _generate_once(
    model: Any,
    prompt_ids: torch.Tensor,
    policy: Phase1Policy,
    seed: int,
    trace_path: Path | None,
    *,
    full_distribution: bool = False,
) -> tuple[torch.Tensor, int]:
    generator = _make_generator(prompt_ids.device, seed)
    if trace_path is None:
        return (
            generate_with_trace(
                model,
                prompt_ids,
                config=policy.decode_config,
                generator=generator,
            ),
            0,
        )
    with AtomicJsonlTraceSink(trace_path) as sink:
        output = generate_with_trace(
            model,
            prompt_ids,
            config=policy.decode_config,
            generator=generator,
            trace_callback=sink,
            full_distribution=full_distribution,
        )
    return output, sink.event_count


def _result_paths(output_root: Path, item: Phase1WorkItem) -> tuple[Path, Path]:
    directory = (
        output_root
        / PROTOCOL_VERSION
        / item.split_role
        / item.run_kind
        / item.task
        / item.policy_name
        / f"q{item.row_index:04d}"
    )
    stem = f"r{item.rollout_index:02d}"
    return directory / f"{stem}.json", directory / f"{stem}.trace.jsonl.gz"


def _policy_payload(policy: Phase1Policy) -> dict[str, Any]:
    config = asdict(policy.decode_config)
    return {
        "name": policy.name,
        "steps": config["steps"],
        "generation_length": config["gen_length"],
        "block_length": config["block_length"],
        "temperature": config["temperature"],
        "cfg_scale": config["cfg_scale"],
        "remasking": config["remasking"],
    }


def _seed_key(item: Phase1WorkItem) -> str:
    return rollout_seed_key(
        item.row_index,
        item.rollout_index,
        global_seed=PROTOCOL_SEED,
        protocol_prefix=f"dllm-order-phase1-v1|{item.task}",
    )


def _seed(item: Phase1WorkItem) -> int:
    return derive_rollout_seed(
        item.row_index,
        item.rollout_index,
        global_seed=PROTOCOL_SEED,
        protocol_prefix=f"dllm-order-phase1-v1|{item.task}",
    )


def _validate_existing_result(
    result_path: Path,
    trace_path: Path,
    item: Phase1WorkItem,
    *,
    bci_freeze_sha256: str | None,
    random_extension_freeze_sha256: str | None = None,
    validator: Draft202012Validator,
) -> dict[str, Any]:
    record = json.loads(result_path.read_text(encoding="utf-8"))
    validator.validate(record)
    expected = {
        "run_kind": item.run_kind,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "bci_freeze_sha256": bci_freeze_sha256,
        "rollout_index": item.rollout_index,
        "seed_key": _seed_key(item),
        "completion_status": "complete",
    }
    if random_extension_freeze_sha256 is not None:
        expected["random_extension_freeze_sha256"] = (
            random_extension_freeze_sha256
        )
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"existing result has mismatched {key}: {result_path}")
    source = record["source"]
    if (
        source["row_index"] != item.row_index
        or source["split_role"] != item.split_role
        or source["prompt_sha256"] != item.prompt_sha256
        or record["policy"]["name"] != item.policy_name
    ):
        raise RuntimeError(
            f"existing result is bound to another work item: {result_path}"
        )
    if not trace_path.is_file() or sha256_file(trace_path) != record["trace_sha256"]:
        raise RuntimeError(
            f"existing result has a missing or invalid trace: {result_path}"
        )
    return record


def _validate_primary_extension_prerequisite(
    output_root: Path,
    item: Phase1WorkItem,
    *,
    bci_freeze_sha256: str,
    validator: Draft202012Validator,
) -> None:
    for rollout_index in range(16):
        prerequisite = Phase1WorkItem(
            task=item.task,
            split_role="heldout",
            run_kind="screening",
            row_index=item.row_index,
            prompt_sha256=item.prompt_sha256,
            policy_name=item.policy_name,
            rollout_index=rollout_index,
        )
        result_path, trace_path = _result_paths(output_root, prerequisite)
        if not result_path.is_file():
            raise RuntimeError(
                f"primary extension requires completed screening result: {result_path}"
            )
        _validate_existing_result(
            result_path,
            trace_path,
            prerequisite,
            bci_freeze_sha256=bci_freeze_sha256,
            validator=validator,
        )


def run_work_item(
    *,
    item: Phase1WorkItem,
    rows: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    model: Any,
    graders: dict[str, Gsm8kGrader | Math500Grader],
    device: torch.device,
    output_root: Path,
    repo_state: dict[str, Any],
    environment: dict[str, Any],
    manifest: dict[str, Any],
    bci_freeze_sha256: str | None,
    random_extension_freeze_sha256: str | None,
    validator: Draft202012Validator,
    prerequisite_validator: Draft202012Validator,
    validated_prerequisites: set[tuple[str, int, str]],
) -> str:
    result_path, trace_path = _result_paths(output_root, item)
    if result_path.exists():
        _validate_existing_result(
            result_path,
            trace_path,
            item,
            bci_freeze_sha256=bci_freeze_sha256,
            random_extension_freeze_sha256=random_extension_freeze_sha256,
            validator=validator,
        )
        return "skipped"
    if trace_path.exists():
        raise RuntimeError(f"orphan trace exists without result record: {trace_path}")
    if item.run_kind in ("primary-extension", "random-extension"):
        if bci_freeze_sha256 is None:
            raise RuntimeError(f"{item.run_kind} requires a BCI freeze")
        if (
            item.run_kind == "random-extension"
            and random_extension_freeze_sha256 is None
        ):
            raise RuntimeError("random extension requires its operative freeze")
        prerequisite_key = (item.task, item.row_index, item.policy_name)
        if prerequisite_key not in validated_prerequisites:
            _validate_primary_extension_prerequisite(
                output_root,
                item,
                bci_freeze_sha256=bci_freeze_sha256,
                validator=prerequisite_validator,
            )
            validated_prerequisites.add(prerequisite_key)

    row = rows[item.task][item.row_index]
    prompt, reference, observed_prompt_hash = _prompt_reference_and_hash(item.task, row)
    if observed_prompt_hash != item.prompt_sha256:
        raise RuntimeError(f"prompt hash mismatch for {item.task} row {item.row_index}")
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    policy = POLICY_BY_NAME[item.policy_name]
    resolved_seed = _seed(item)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    traced_output, event_count = _generate_once(
        model,
        prompt_ids,
        policy,
        resolved_seed,
        trace_path,
        full_distribution=(item.run_kind == "phase2-extension"),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    item_environment = dict(environment)
    if item.run_kind == "smoke":
        untraced_output, _ = _generate_once(
            model, prompt_ids, policy, resolved_seed, None
        )
        trace_equivalent = bool(torch.equal(traced_output, untraced_output))
        item_environment["trace_equivalence_verified_for_this_result"] = (
            trace_equivalent
        )
        if not trace_equivalent:
            raise RuntimeError(f"trace changed the smoke output for {item.key}")

    completion_ids = traced_output[:, prompt_ids.shape[1] :]
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
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "bci_freeze_sha256": bci_freeze_sha256,
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
        "policy": _policy_payload(policy),
        "rollout_index": item.rollout_index,
        "seed_key": _seed_key(item),
        "seed": resolved_seed,
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
    if random_extension_freeze_sha256 is not None:
        record["random_extension_freeze_sha256"] = (
            random_extension_freeze_sha256
        )
    validator.validate(record)
    write_json_atomic(result_path, record)
    return "completed"


def _completion_path(
    output_root: Path,
    *,
    split_role: str,
    run_kind: str,
    task_selection: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    return (
        output_root
        / PROTOCOL_VERSION
        / "manifests"
        / split_role
        / run_kind
        / task_selection
        / f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    )


def _write_shard_completion(
    output_root: Path,
    items: list[Phase1WorkItem],
    *,
    split_role: str,
    run_kind: str,
    task_selection: str,
    shard_index: int,
    num_shards: int,
    validator: Draft202012Validator,
) -> Path:
    keys = [item.key for item in items]
    marker = {
        "protocol_version": PROTOCOL_VERSION,
        "split_role": split_role,
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
    path = _completion_path(
        output_root,
        split_role=split_role,
        run_kind=run_kind,
        task_selection=task_selection,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "protocol_version",
            "split_role",
            "run_kind",
            "task_selection",
            "shard_index",
            "num_shards",
            "expected_keys_sha256",
            "observed_keys_sha256",
            "result_count",
        ):
            if existing.get(key) != marker[key]:
                raise RuntimeError(f"existing shard marker disagrees on {key}: {path}")
        return path
    write_json_atomic(path, marker)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument(
        "--split-role", choices=["calibration", "heldout"], required=True
    )
    parser.add_argument(
        "--run-kind",
        choices=[
            "smoke",
            "screening",
            "primary-extension",
            "phase2-extension",
            "random-extension",
        ],
        required=True,
    )
    parser.add_argument("--task", choices=["all", "gsm8k", "math500"], default="all")
    parser.add_argument("--justgrpo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--gsm8k-parquet", type=Path)
    parser.add_argument("--math500-parquet", type=Path)
    parser.add_argument("--bci-freeze", type=Path)
    parser.add_argument("--phase2-freeze", type=Path)
    parser.add_argument("--random-extension-freeze", type=Path)
    parser.add_argument(
        "--policies",
        help=(
            "comma-separated policy override (phase2 smoke cells); default is "
            "the run-kind default"
        ),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="truncate this shard's work items (phase2 smoke)",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_split_manifest(args.split_manifest)
    validator = _load_artifact_validator(args.artifact_schema, args.run_kind)
    prerequisite_validator = _load_artifact_validator(
        args.artifact_schema.parent / "phase1_artifact_schema_v1.json",
        "primary-extension",
    )
    policy_override = None
    if args.policies is not None:
        policy_override = tuple(p.strip() for p in args.policies.split(","))
        invalid = [p for p in policy_override if p not in POLICY_BY_NAME]
        if invalid:
            raise RuntimeError(f"unknown policy names: {invalid}")
    items = build_phase1_work_items(
        manifest,
        split_role=args.split_role,
        run_kind=args.run_kind,
        task_selection=args.task,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        policies=policy_override,
    )
    if args.max_items is not None:
        if args.max_items < 1:
            raise RuntimeError("--max-items must be positive")
        items = items[: args.max_items]
    bci_freeze_sha256 = None
    if args.bci_freeze is not None:
        bci_freeze_sha256 = validate_bci_freeze(args.bci_freeze, SPLIT_MANIFEST_SHA256)
    if args.run_kind == "phase2-extension":
        if args.phase2_freeze is None:
            if not args.dry_run:
                raise RuntimeError("phase2-extension requires --phase2-freeze")
        else:
            # Documented field reuse: phase2 records stamp the operative
            # Phase 2 freeze SHA in the bci_freeze_sha256 slot; the Phase 2
            # freeze itself records the BCI freeze as provenance.
            bci_freeze_sha256 = validate_phase2_freeze(
                args.phase2_freeze, SPLIT_MANIFEST_SHA256
            )
    random_extension_freeze_sha256 = None
    if args.run_kind == "random-extension":
        if args.bci_freeze is None or args.random_extension_freeze is None:
            if not args.dry_run:
                raise RuntimeError(
                    "random-extension requires --bci-freeze and "
                    "--random-extension-freeze"
                )
        else:
            random_extension_freeze_sha256 = validate_random_extension_freeze(
                args.random_extension_freeze, SPLIT_MANIFEST_SHA256
            )
    bci_required = args.split_role == "heldout"
    if bci_required and bci_freeze_sha256 is None and not args.dry_run:
        raise RuntimeError(
            "held-out inference requires --bci-freeze or --phase2-freeze"
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
                    "artifact_schema_sha256": sha256_file(args.artifact_schema),
                    "split_role": args.split_role,
                    "run_kind": args.run_kind,
                    "task_selection": args.task,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "generation_count": len(items),
                    "decoder_execution_count": len(items)
                    * (2 if args.run_kind == "smoke" else 1),
                    "bci_freeze_required": bci_required,
                    "bci_freeze_present": bci_freeze_sha256 is not None,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "external_repository_commit": JUSTGRPO_COMMIT,
                    "policy_names": sorted({item.policy_name for item in items}),
                    "task_names": sorted({item.task for item in items}),
                    "expected_work_keys_sha256": _hash_keys(
                        [item.key for item in items]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    justgrpo_root = validate_justgrpo_root(args.justgrpo_root)
    repo_root = Path(__file__).resolve().parents[3]
    repo_state = _repo_state(repo_root)
    if repo_state["dirty"]:
        raise RuntimeError("Phase 1 runner repository must be clean")
    if args.run_kind == "phase2-extension":
        operative_freeze = args.phase2_freeze
    elif args.run_kind == "random-extension":
        operative_freeze = args.random_extension_freeze
    else:
        operative_freeze = args.bci_freeze
    if operative_freeze is not None:
        freeze_payload = json.loads(operative_freeze.read_text(encoding="utf-8"))
        if freeze_payload["runner_commit"] != repo_state["commit"]:
            raise RuntimeError(
                "operative freeze is bound to a different runner commit"
            )

    datasets = _load_datasets(
        manifest,
        items,
        gsm8k_parquet=args.gsm8k_parquet,
        math500_parquet=args.math500_parquet,
    )
    device = torch.device(args.device)
    tokenizer, model, model_source_mode = _load_model(device, args.model_path)
    graders: dict[str, Gsm8kGrader | Math500Grader] = {}
    if "gsm8k" in datasets:
        graders["gsm8k"] = load_gsm8k_grader(justgrpo_root)
    if "math500" in datasets:
        graders["math500"] = load_math500_grader(justgrpo_root)
    environment = _environment_fingerprint(
        device,
        {
            "model": model_source_mode,
            "gsm8k": "local_pinned_parquet" if "gsm8k" in datasets else "unused",
            "math500": "local_pinned_parquet" if "math500" in datasets else "unused",
        },
    )

    completed = 0
    skipped = 0
    validated_prerequisites: set[tuple[str, int, str]] = set()
    for item in items:
        status = run_work_item(
            item=item,
            rows=datasets,
            tokenizer=tokenizer,
            model=model,
            graders=graders,
            device=device,
            output_root=args.output,
            repo_state=repo_state,
            environment=environment,
            manifest=manifest,
            bci_freeze_sha256=bci_freeze_sha256,
            random_extension_freeze_sha256=random_extension_freeze_sha256,
            validator=validator,
            prerequisite_validator=prerequisite_validator,
            validated_prerequisites=validated_prerequisites,
        )
        completed += status == "completed"
        skipped += status == "skipped"
        print(
            f"work_key={item.key} completed={completed} skipped={skipped}",
            flush=True,
        )

    if args.run_kind == "random-extension" and args.max_items is not None:
        print(
            "random_extension_canary_complete=true shard_marker_written=false",
            flush=True,
        )
    else:
        marker = _write_shard_completion(
            args.output,
            items,
            split_role=args.split_role,
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
