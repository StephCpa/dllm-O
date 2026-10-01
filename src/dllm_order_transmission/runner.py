"""CLI runner for the prospectively frozen Phase 0 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import AtomicJsonlTraceSink, sha256_file, write_json_atomic
from .decoding import generate_with_trace
from .external import Gsm8kGrader, load_gsm8k_grader, validate_justgrpo_root
from .protocol import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    EXPECTED_DATASET_SIZE,
    JUSTGRPO_COMMIT,
    MODEL_ID,
    MODEL_REVISION,
    POLICIES,
    PROTOCOL_VERSION,
    Policy,
    stage_query_indices,
    stage_rollout_count,
)
from .seeding import derive_rollout_seed, rollout_seed_key


@dataclass(frozen=True)
class WorkItem:
    query_index: int
    rollout_index: int


def build_work_items(stage: str, shard_index: int, num_shards: int) -> list[WorkItem]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= index < num_shards")
    all_items = [
        WorkItem(query_index=query_index, rollout_index=rollout_index)
        for query_index in stage_query_indices(stage)
        for rollout_index in range(stage_rollout_count(stage))
    ]
    return [item for index, item in enumerate(all_items) if index % num_shards == shard_index]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _load_inputs(
    device: torch.device,
    *,
    model_path: Path | None = None,
    dataset_path: Path | None = None,
) -> tuple[Any, Any, Any, dict[str, str]]:
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    if dataset_path is None:
        dataset = load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
            revision=DATASET_REVISION,
        )
        dataset_source = "hub_pinned_revision"
    else:
        dataset_file = (
            dataset_path.resolve()
            / DATASET_CONFIG
            / f"{DATASET_SPLIT}-00000-of-00001.parquet"
        )
        if not dataset_file.is_file():
            raise FileNotFoundError(f"missing frozen GSM8K parquet: {dataset_file}")
        dataset = load_dataset(
            "parquet",
            data_files={DATASET_SPLIT: str(dataset_file)},
            split=DATASET_SPLIT,
        )
        dataset_source = "local_pinned_snapshot"
    if len(dataset) != EXPECTED_DATASET_SIZE:
        raise RuntimeError(
            f"expected {EXPECTED_DATASET_SIZE} GSM8K test rows, found {len(dataset)}"
        )

    model_source = str(model_path.resolve()) if model_path is not None else MODEL_ID
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if model_path is None:
        model_kwargs["revision"] = MODEL_REVISION
        model_source_mode = "hub_pinned_revision"
    else:
        if not model_path.resolve().is_dir():
            raise FileNotFoundError(f"missing local model snapshot: {model_path}")
        model_kwargs["local_files_only"] = True
        model_source_mode = "local_verified_snapshot"
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        **model_kwargs,
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_source,
        torch_dtype=dtype,
        **model_kwargs,
    )
    model.eval().requires_grad_(False).to(device)
    return dataset, tokenizer, model, {
        "model": model_source_mode,
        "dataset": dataset_source,
    }


def _result_paths(
    output_root: Path,
    stage: str,
    policy: Policy,
    item: WorkItem,
) -> tuple[Path, Path]:
    directory = output_root / stage / policy.name / f"q{item.query_index:04d}"
    stem = f"r{item.rollout_index:02d}"
    return directory / f"{stem}.json", directory / f"{stem}.trace.jsonl.gz"


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def _generate_once(
    model: Any,
    prompt_ids: torch.Tensor,
    policy: Policy,
    seed: int,
    trace_path: Path | None,
) -> tuple[torch.Tensor, int]:
    generator = _make_generator(prompt_ids.device, seed)
    if trace_path is None:
        output = generate_with_trace(
            model,
            prompt_ids,
            config=policy.decode_config,
            generator=generator,
        )
        return output, 0
    with AtomicJsonlTraceSink(trace_path) as sink:
        output = generate_with_trace(
            model,
            prompt_ids,
            config=policy.decode_config,
            generator=generator,
            trace_callback=sink,
        )
    return output, sink.event_count


def run_item_policy(
    *,
    dataset: Any,
    tokenizer: Any,
    model: Any,
    grader: Gsm8kGrader,
    device: torch.device,
    repo_root: Path,
    output_root: Path,
    stage: str,
    item: WorkItem,
    policy: Policy,
    input_source_modes: dict[str, str],
) -> str:
    result_path, trace_path = _result_paths(output_root, stage, policy, item)
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("completion_status") == "complete":
            if trace_path.exists() and sha256_file(trace_path) == existing.get("trace_sha256"):
                return "skipped"
        raise RuntimeError(f"invalid or incomplete existing result: {result_path}")

    row = dataset[item.query_index]
    question = str(row["question"])
    reference_answer = str(row["answer"])
    messages = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

    seed = derive_rollout_seed(item.query_index, item.rollout_index)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    traced_output, event_count = _generate_once(
        model, prompt_ids, policy, seed, trace_path
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    trace_equivalent: bool | None = None
    if stage == "equivalence":
        untraced_output, _ = _generate_once(model, prompt_ids, policy, seed, None)
        trace_equivalent = bool(torch.equal(traced_output, untraced_output))
        if not trace_equivalent:
            raise RuntimeError(
                f"trace changed output for {policy.name}, query={item.query_index}, "
                f"rollout={item.rollout_index}"
            )

    completion_ids = traced_output[:, prompt_ids.shape[1] :]
    response = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]
    parser_status = "ok"
    parsed_answer: str | None
    try:
        parsed_answer, correct = grader.grade(reference_answer, response)
    except Exception as error:
        parser_status = f"error:{type(error).__name__}"
        parsed_answer = None
        correct = False

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "runner_repository": _repo_state(repo_root),
        "external_repo_commit": JUSTGRPO_COMMIT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "input_source_modes": input_source_modes,
        "query_index": item.query_index,
        "question_sha256": _sha256_text(question),
        "reference_answer_sha256": _sha256_text(reference_answer),
        "policy": policy.name,
        "decode_config": asdict(policy.decode_config),
        "rollout_index": item.rollout_index,
        "seed_key": rollout_seed_key(item.query_index, item.rollout_index),
        "resolved_seed": seed,
        "generated_token_ids": completion_ids[0].detach().cpu().tolist(),
        "decoded_text": response,
        "parsed_answer": parsed_answer,
        "correct": correct,
        "parser_status": parser_status,
        "trace_equivalent": trace_equivalent,
        "trace_event_count": event_count,
        "trace_sha256": sha256_file(trace_path),
        "wall_time_seconds": elapsed,
        "peak_memory_bytes": peak_memory,
        "completion_status": "complete",
    }
    write_json_atomic(result_path, record)
    return "completed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["equivalence", "signal"], required=True)
    parser.add_argument("--justgrpo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    justgrpo_root = validate_justgrpo_root(args.justgrpo_root)
    items = build_work_items(args.stage, args.shard_index, args.num_shards)
    if args.dry_run:
        policy_generations = len(items) * len(POLICIES)
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "work_items": len(items),
                    "policy_generations": policy_generations,
                    "decoder_executions": policy_generations
                    * (2 if args.stage == "equivalence" else 1),
                    "external_commit": JUSTGRPO_COMMIT,
                    "model_source": (
                        "local_verified_snapshot" if args.model_path else "hub_pinned_revision"
                    ),
                    "dataset_source": (
                        "local_pinned_snapshot" if args.dataset_path else "hub_pinned_revision"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    device = torch.device(args.device)
    grader = load_gsm8k_grader(justgrpo_root)
    dataset, tokenizer, model, input_source_modes = _load_inputs(
        device,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
    )
    repo_root = Path(__file__).resolve().parents[3]

    completed = 0
    skipped = 0
    for item in items:
        for policy in POLICIES:
            status = run_item_policy(
                dataset=dataset,
                tokenizer=tokenizer,
                model=model,
                grader=grader,
                device=device,
                repo_root=repo_root,
                output_root=args.output,
                stage=args.stage,
                item=item,
                policy=policy,
                input_source_modes=input_source_modes,
            )
            completed += status == "completed"
            skipped += status == "skipped"
        print(
            f"query={item.query_index} rollout={item.rollout_index} "
            f"completed={completed} skipped={skipped}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
