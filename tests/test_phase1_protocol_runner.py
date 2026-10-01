from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest
import torch

from dllm_order_transmission.external import Math500Grader
from dllm_order_transmission.phase1_protocol import (
    PHASE1_POLICIES,
    Phase1WorkItem,
    SPLIT_MANIFEST_SHA256,
    build_phase1_work_items,
    load_split_manifest,
    validate_bci_freeze,
)
from dllm_order_transmission.phase1_runner import (
    _hash_keys,
    _load_artifact_validator,
    _prompt_reference_and_hash,
    _seed_key,
    _write_shard_completion,
    parse_args,
    run_work_item,
)


PROTOCOL_ROOT = Path(__file__).parents[1] / "protocols"
SPLIT_MANIFEST = PROTOCOL_ROOT / "phase1_split_manifest_v1.json"
ARTIFACT_SCHEMA = PROTOCOL_ROOT / "phase1_artifact_schema_v1.json"
PHASE2_ARTIFACT_SCHEMA = PROTOCOL_ROOT / "phase2_artifact_schema_v1.json"
RANDOM_EXTENSION_ARTIFACT_SCHEMA = (
    PROTOCOL_ROOT / "random_extension_artifact_schema_v1.json"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_split_manifest(SPLIT_MANIFEST)


def test_phase1_policy_grid_is_frozen() -> None:
    assert [policy.name for policy in PHASE1_POLICIES] == [
        "ar_b1",
        "ao_b8",
        "ao_b16",
        "ao_b32",
        "random_b32",
    ]
    assert [policy.decode_config.block_length for policy in PHASE1_POLICIES] == [
        1,
        8,
        16,
        32,
        32,
    ]
    assert PHASE1_POLICIES[-1].decode_config.remasking == "random"


def test_frozen_work_counts(manifest: dict) -> None:
    smoke = build_phase1_work_items(
        manifest, split_role="calibration", run_kind="smoke"
    )
    calibration = build_phase1_work_items(
        manifest, split_role="calibration", run_kind="screening"
    )
    heldout = build_phase1_work_items(
        manifest, split_role="heldout", run_kind="screening"
    )
    extension = build_phase1_work_items(
        manifest, split_role="heldout", run_kind="primary-extension"
    )
    random_extension = build_phase1_work_items(
        manifest, split_role="heldout", run_kind="random-extension"
    )
    assert len(smoke) == 10
    assert len(calibration) == 6_400
    assert len(heldout) == 16_000
    assert len(extension) == 19_200
    assert len(random_extension) == 9_600
    assert {item.policy_name for item in random_extension} == {"random_b32"}
    assert {item.rollout_index for item in random_extension} == set(range(16, 64))
    assert len({item.key for item in smoke}) == len(smoke)
    assert {item.task for item in smoke} == {"gsm8k", "math500"}
    assert {item.policy_name for item in smoke} == {
        "ar_b1",
        "ao_b8",
        "ao_b16",
        "ao_b32",
        "random_b32",
    }


def test_calibration_reuses_gsm_primary_conditions(manifest: dict) -> None:
    gsm = build_phase1_work_items(
        manifest,
        split_role="calibration",
        run_kind="screening",
        task_selection="gsm8k",
    )
    assert len(gsm) == 2_400
    assert {item.policy_name for item in gsm} == {
        "ao_b8",
        "ao_b16",
        "random_b32",
    }


def test_phase1_shards_are_disjoint_and_complete(manifest: dict) -> None:
    complete = set(
        build_phase1_work_items(manifest, split_role="heldout", run_kind="screening")
    )
    shards = [
        set(
            build_phase1_work_items(
                manifest,
                split_role="heldout",
                run_kind="screening",
                shard_index=index,
                num_shards=3,
            )
        )
        for index in range(3)
    ]
    assert not (shards[0] & shards[1])
    assert not (shards[0] & shards[2])
    assert not (shards[1] & shards[2])
    assert set.union(*shards) == complete


def test_phase2_completion_marker_satisfies_phase2_schema(
    manifest: dict, tmp_path: Path
) -> None:
    item = build_phase1_work_items(
        manifest,
        split_role="heldout",
        run_kind="phase2-extension",
        task_selection="gsm8k",
        policies=("ao_b8",),
    )[0]
    validator = _load_artifact_validator(
        PHASE2_ARTIFACT_SCHEMA, run_kind="phase2-extension"
    )
    marker_path = _write_shard_completion(
        tmp_path,
        [item],
        split_role="heldout",
        run_kind="phase2-extension",
        task_selection="gsm8k",
        shard_index=0,
        num_shards=1,
        validator=validator,
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    validator.validate(marker)
    assert marker["run_kind"] == "phase2-extension"


def test_primary_extension_cannot_target_calibration(manifest: dict) -> None:
    with pytest.raises(ValueError, match="held-out only"):
        build_phase1_work_items(
            manifest,
            split_role="calibration",
            run_kind="primary-extension",
        )

    with pytest.raises(ValueError, match="held-out only"):
        build_phase1_work_items(
            manifest,
            split_role="calibration",
            run_kind="random-extension",
        )


def test_random_extension_shards_and_marker_schema(
    manifest: dict, tmp_path: Path
) -> None:
    complete = set(
        build_phase1_work_items(
            manifest, split_role="heldout", run_kind="random-extension"
        )
    )
    shards = [
        set(
            build_phase1_work_items(
                manifest,
                split_role="heldout",
                run_kind="random-extension",
                shard_index=index,
                num_shards=3,
            )
        )
        for index in range(3)
    ]
    assert all(len(shard) == 3_200 for shard in shards)
    assert not (shards[0] & shards[1] or shards[0] & shards[2] or shards[1] & shards[2])
    assert set.union(*shards) == complete

    validator = _load_artifact_validator(
        RANDOM_EXTENSION_ARTIFACT_SCHEMA, run_kind="random-extension"
    )
    item = next(iter(shards[0]))
    marker_path = _write_shard_completion(
        tmp_path,
        [item],
        split_role="heldout",
        run_kind="random-extension",
        task_selection="all",
        shard_index=0,
        num_shards=3,
        validator=validator,
    )
    validator.validate(json.loads(marker_path.read_text()))


def test_seed_keys_are_policy_independent_but_task_specific(manifest: dict) -> None:
    items = build_phase1_work_items(
        manifest, split_role="calibration", run_kind="smoke"
    )
    gsm = [item for item in items if item.task == "gsm8k"]
    math = [item for item in items if item.task == "math500"]
    assert len({_seed_key(item) for item in gsm}) == 1
    assert len({_seed_key(item) for item in math}) == 1
    assert _seed_key(gsm[0]) != _seed_key(math[0])


def test_prompt_hash_excludes_public_math_instruction() -> None:
    prompt, reference, prompt_hash = _prompt_reference_and_hash(
        "math500", {"problem": "Solve x=1.", "solution": "x=1"}
    )
    assert prompt.startswith("Solve x=1.")
    assert "\\boxed" in prompt
    assert reference == "x=1"
    assert prompt_hash == hashlib.sha256(b"Solve x=1.").hexdigest()


def test_math_grader_uses_extracted_answers_and_timeout_equality() -> None:
    calls: list[tuple[str, str]] = []

    def extract(value: str) -> str:
        return value.removeprefix("answer:")

    def equals(target: str, prediction: str) -> bool:
        calls.append((target, prediction))
        return target == prediction

    grader = Math500Grader(extract_answer=extract, equals_with_timeout=equals)
    prediction, correct = grader.grade("answer:4", "answer:4")
    assert prediction == "4"
    assert correct is True
    assert calls == [("4", "4")]


def test_bci_freeze_requires_all_frozen_fields(tmp_path) -> None:
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"protocol_version": "phase1-bci-freeze-v1"}))
    with pytest.raises(RuntimeError, match="missing required fields"):
        validate_bci_freeze(path, SPLIT_MANIFEST_SHA256)


def test_bci_freeze_binds_split_manifest(tmp_path) -> None:
    payload = {
        "protocol_version": "phase1-bci-freeze-v1",
        "created_at_utc": "2026-08-23T00:00:00+00:00",
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "calibration_artifact_hashes": ["a" * 64],
        "candidate_boundary_rule": {},
        "matching_rule": {},
        "bci_formula": {},
        "baseline_predictor": {},
        "augmented_predictor": {},
        "normalization": {},
        "parser_policy": {},
        "exclusion_policy": {},
        "target_policy": {},
        "failure_thresholds": {},
        "bootstrap_seed": 20260823,
        "runner_commit": "b" * 40,
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    assert len(validate_bci_freeze(path, SPLIT_MANIFEST_SHA256)) == 64


def test_phase1_cli_parses_dry_run(tmp_path) -> None:
    args = parse_args(
        [
            "--split-manifest",
            str(SPLIT_MANIFEST),
            "--artifact-schema",
            str(ARTIFACT_SCHEMA),
            "--split-role",
            "calibration",
            "--run-kind",
            "smoke",
            "--justgrpo-root",
            str(tmp_path / "JustGRPO"),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert args.run_kind == "smoke"
    assert args.split_role == "calibration"
    assert args.dry_run is True


def test_result_and_shard_marker_satisfy_frozen_schema(tmp_path, monkeypatch) -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

        def batch_decode(self, token_ids, **kwargs):
            return ["42"]

    def fake_generate(model, prompt_ids, policy, seed, trace_path, **kwargs):
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(trace_path, mode="wt", encoding="utf-8") as handle:
                handle.write("{}\n")
        completion = torch.full((1, 256), 3, dtype=torch.long)
        return torch.cat([prompt_ids, completion], dim=1), 256 if trace_path else 0

    monkeypatch.setattr(
        "dllm_order_transmission.phase1_runner._generate_once", fake_generate
    )
    prompt = "What is six times seven?"
    item = Phase1WorkItem(
        task="gsm8k",
        split_role="calibration",
        run_kind="smoke",
        row_index=0,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        policy_name="ar_b1",
        rollout_index=0,
    )
    manifest = {
        "sources": {
            "gsm8k": {
                "dataset_id": "openai/gsm8k",
                "revision": "740312add88f781978c0658806c59bc2815b9866",
                "split": "test",
            }
        }
    }
    grader = Math500Grader(
        extract_answer=lambda value: value.split()[-1],
        equals_with_timeout=lambda target, prediction: target == prediction,
    )
    validator = _load_artifact_validator(ARTIFACT_SCHEMA)
    status = run_work_item(
        item=item,
        rows={"gsm8k": [{"question": prompt, "answer": "42"}]},
        tokenizer=FakeTokenizer(),
        model=object(),
        graders={"gsm8k": grader},
        device=torch.device("cpu"),
        output_root=tmp_path,
        repo_state={"commit": "a" * 40, "dirty": False},
        environment={"test": True},
        manifest=manifest,
        bci_freeze_sha256=None,
        random_extension_freeze_sha256=None,
        validator=validator,
        prerequisite_validator=validator,
        validated_prerequisites=set(),
    )
    assert status == "completed"
    result_paths = list(tmp_path.rglob("r00.json"))
    assert len(result_paths) == 1
    result = json.loads(result_paths[0].read_text())
    validator.validate(result)
    assert (
        result["environment_fingerprint"]["trace_equivalence_verified_for_this_result"]
        is True
    )

    marker_path = _write_shard_completion(
        tmp_path,
        [item],
        split_role="calibration",
        run_kind="smoke",
        task_selection="gsm8k",
        shard_index=0,
        num_shards=1,
        validator=validator,
    )
    marker = json.loads(marker_path.read_text())
    validator.validate(marker)
    assert marker["result_count"] == 1
