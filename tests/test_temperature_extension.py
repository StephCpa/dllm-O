from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from dllm_order_transmission.artifacts import sha256_file
from dllm_order_transmission.phase1_protocol import load_split_manifest
from dllm_order_transmission.temperature_extension_analyzer import (
    _fractional_majority_correct,
    _simultaneous_primary,
    build_freeze_payload,
)
from dllm_order_transmission.temperature_extension_protocol import (
    ARTIFACT_SCHEMA_SHA256,
    DEFAULT_NUM_SHARDS,
    EXPECTED_FORMAL_GENERATIONS,
    TEMPERATURE_POLICIES,
    build_work_items,
    validate_freeze,
)
from dllm_order_transmission.temperature_extension_runner import (
    _policy_payload,
    _seed_key,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "protocols" / "phase1_split_manifest_v1.json"
SCHEMA_PATH = ROOT / "protocols" / "temperature_extension_artifact_schema_v1.json"


def test_temperature_extension_cardinality_and_shards() -> None:
    manifest = load_split_manifest(MANIFEST_PATH)
    items = build_work_items(manifest, run_kind="temperature-formal")
    assert len(items) == EXPECTED_FORMAL_GENERATIONS
    assert len({item.key for item in items}) == EXPECTED_FORMAL_GENERATIONS
    assert {item.policy_name for item in items} == {
        policy.name for policy in TEMPERATURE_POLICIES
    }
    assert {item.rollout_index for item in items} == set(range(32))

    shards = [
        build_work_items(
            manifest,
            run_kind="temperature-formal",
            shard_index=index,
            num_shards=DEFAULT_NUM_SHARDS,
        )
        for index in range(DEFAULT_NUM_SHARDS)
    ]
    assert {len(shard) for shard in shards} == {6_400}
    assert {item.key for shard in shards for item in shard} == {
        item.key for item in items
    }


def test_temperature_smoke_covers_every_new_cell() -> None:
    manifest = load_split_manifest(MANIFEST_PATH)
    items = build_work_items(manifest, run_kind="temperature-smoke")
    assert len(items) == 12
    assert {item.task for item in items} == {"gsm8k", "math500"}
    assert {item.policy_name for item in items} == {
        policy.name for policy in TEMPERATURE_POLICIES
    }
    assert {item.rollout_index for item in items} == {0}


def test_temperature_cells_reuse_phase1_seed_pairing() -> None:
    manifest = load_split_manifest(MANIFEST_PATH)
    items = build_work_items(
        manifest,
        run_kind="temperature-formal",
        task_selection="gsm8k",
    )
    first_query = [item for item in items if item.row_index == items[0].row_index]
    rollout_zero = [item for item in first_query if item.rollout_index == 0]
    assert len({_seed_key(item) for item in rollout_zero}) == 1
    assert _seed_key(rollout_zero[0]).startswith("dllm-order-phase1-v1|gsm8k|20260823|")


def test_temperature_schema_hash_and_policy_payloads() -> None:
    assert sha256_file(SCHEMA_PATH) == ARTIFACT_SCHEMA_SHA256
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    payloads = {
        _policy_payload(policy)["name"]: _policy_payload(policy)
        for policy in TEMPERATURE_POLICIES
    }
    assert payloads["sequential_b1_t09"]["block_length"] == 1
    assert payloads["sequential_b1_t09"]["temperature"] == 0.9
    assert payloads["cf_b32_t12"]["remasking"] == "low_confidence"
    assert payloads["rnd_b32_t12"]["remasking"] == "random"


def test_schema_rejects_policy_name_temperature_disagreement() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    policy_schema = schema["$defs"]["policy"]
    validator = Draft202012Validator(policy_schema)
    invalid = {
        "name": "cf_b32_t09",
        "schedule": "confidence_first",
        "steps": 256,
        "generation_length": 256,
        "block_length": 32,
        "temperature": 1.2,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
    }
    with pytest.raises(ValidationError):
        validator.validate(invalid)


def test_freeze_payload_and_validation(tmp_path: Path) -> None:
    payload = build_freeze_payload(
        runner_commit="a" * 40,
        control_inventory_sha256="b" * 64,
    )
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    validate_freeze(path, runner_commit="a" * 40)
    assert payload["primary_family"]["uncertainty"]["family_size"] == 4
    assert payload["cardinality"]["new_generations"] == 38_400


def test_simultaneous_primary_detects_frozen_reversal() -> None:
    values = {
        endpoint: {
            "gsm8k": [value] * 20,
            "math500": [value] * 20,
        }
        for endpoint, value in {
            "t09_low": -0.2,
            "t09_high": 0.2,
            "t12_low": -0.1,
            "t12_high": 0.1,
        }.items()
    }
    result = _simultaneous_primary(values)
    assert result["verdicts"] == {
        "t09": "resolved_reversal",
        "t12": "resolved_reversal",
    }


def test_majority_vote_ties_receive_fractional_credit() -> None:
    records = [
        {"parsed_answer": "a", "correct": True},
        {"parsed_answer": "b", "correct": False},
        {"parsed_answer": "a", "correct": True},
        {"parsed_answer": "b", "correct": False},
    ]
    assert _fractional_majority_correct(records, 4) == 0.5
