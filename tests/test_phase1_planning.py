from __future__ import annotations

import json
from pathlib import Path

import pytest

from dllm_order_transmission.phase1_planning import (
    _largest_remainder_allocation,
    minimum_detectable_correlation,
    minimum_detectable_paired_effect,
    resource_budget,
)


def test_largest_remainder_allocation_respects_total_and_capacities() -> None:
    capacities = {"a": 2, "b": 3, "c": 5}
    allocation = _largest_remainder_allocation(capacities, 7)
    assert sum(allocation.values()) == 7
    assert all(allocation[key] <= capacities[key] for key in capacities)
    assert allocation == _largest_remainder_allocation(capacities, 7)


def test_power_resolution_matches_frozen_design() -> None:
    assert minimum_detectable_correlation(100) == pytest.approx(0.276, abs=0.002)
    assert minimum_detectable_correlation(200) == pytest.approx(0.197, abs=0.002)
    assert minimum_detectable_paired_effect(100) == pytest.approx(0.280, abs=0.002)


def test_resource_budget_counts_all_new_generations() -> None:
    budget = resource_budget()
    assert budget["calibration_gsm8k_generations"] == 2_400
    assert budget["calibration_math500_generations"] == 4_000
    assert budget["heldout_screening_generations"] == 16_000
    assert budget["heldout_primary_extension_generations"] == 19_200
    assert budget["total_new_generations"] == 41_600


def test_phase1_artifact_schema_is_valid_json() -> None:
    schema_path = (
        Path(__file__).parents[1] / "protocols" / "phase1_artifact_schema_v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("2020-12/schema")
    assert set(schema["$defs"]) == {
        "source",
        "policy",
        "repository_state",
        "generation_result",
        "trace_event",
        "shard_completion",
    }
