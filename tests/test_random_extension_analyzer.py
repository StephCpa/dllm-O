from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator

from dllm_order_transmission.artifacts import sha256_file
from dllm_order_transmission.phase1_protocol import (
    RANDOM_EXTENSION_ARTIFACT_SCHEMA_SHA256,
    SPLIT_MANIFEST_SHA256,
    validate_random_extension_freeze,
)
from dllm_order_transmission.random_extension_analyzer import (
    EXPECTED_NEW_GENERATIONS,
    PRIMARY_GRID,
    _classify,
    _coverage,
    _freeze_payload,
)


ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "protocols/random_extension_artifact_schema_v1.json"


def test_random_extension_schema_is_frozen_and_valid() -> None:
    assert sha256_file(SCHEMA) == RANDOM_EXTENSION_ARTIFACT_SCHEMA_SHA256
    Draft202012Validator.check_schema(json.loads(SCHEMA.read_text()))


def test_freeze_payload_discloses_selection_and_single_primary(tmp_path: Path) -> None:
    payload = _freeze_payload("a" * 40)
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload))
    validate_random_extension_freeze(path, SPLIT_MANIFEST_SHA256)
    assert payload["selection_status"].startswith("post-unblinding")
    assert payload["primary_endpoint"]["contrast"] == "random_b32 - ao_b32"
    assert payload["cardinality"]["new_generations"] == EXPECTED_NEW_GENERATIONS
    assert tuple(PRIMARY_GRID) == (16, 32, 64)


def test_freeze_validation_rejects_changed_design(tmp_path: Path) -> None:
    payload = _freeze_payload("a" * 40)
    changed = deepcopy(payload)
    changed["primary_endpoint"]["contrast"] = "random_b32 - ar_b1"
    path = tmp_path / "changed-freeze.json"
    path.write_text(json.dumps(changed))
    try:
        validate_random_extension_freeze(path, SPLIT_MANIFEST_SHA256)
    except RuntimeError as error:
        assert "primary endpoint" in str(error)
    else:
        raise AssertionError("changed primary endpoint was accepted")


def test_primary_metric_and_taxonomy() -> None:
    assert _coverage(0, PRIMARY_GRID) == 0.0
    assert _coverage(64, PRIMARY_GRID) == 1.0
    assert _classify({"ci_low": 0.01, "ci_high": 0.02}) == "entirely_above_zero"
    assert _classify({"ci_low": -0.01, "ci_high": 0.02}) == "straddles_zero"
