"""Deterministic split, power, and resource planning for Phase 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import norm

from .artifacts import sha256_file, write_json_atomic, write_text_atomic
from .protocol import QUERY_INDICES


PLANNING_SEED = 20260823
SPLIT_VERSION = "phase1-splits-v1"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_FILE_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
MATH500_REVISION = "412eb03a7193aed402c5e4009fc958c66e61b937"
MATH500_FILE_SHA256 = "a0bcfdab7dfeca2917d287e54e7ea1199bf216307ce880340e626b7d0288e10a"
PHASE0_MEAN_SECONDS_PER_GENERATION = 28.613567122056265
PHASE0_MEAN_TRACE_BYTES = 101_900

POLICIES = ("ar_b1", "ao_b8", "ao_b16", "ao_b32", "random_b32")
PRIMARY_POLICIES = ("ar_b1", "ao_b32")
CALIBRATION_QUERIES_PER_TASK = 50
HELDOUT_QUERIES_PER_TASK = 100
SCREENING_ROLLOUTS = 16
FINAL_PRIMARY_ROLLOUTS = 64


def minimum_detectable_correlation(
    sample_size: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Approximate two-sided correlation MDE using Fisher's z transform."""
    if sample_size <= 3:
        raise ValueError("sample_size must exceed 3")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie strictly between zero and one")
    critical = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(math.tanh(critical / math.sqrt(sample_size - 3)))


def minimum_detectable_paired_effect(
    sample_size: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Normal-approximation MDE for a standardized paired mean contrast."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    critical = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(critical / math.sqrt(sample_size))


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{PLANNING_SEED}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _row_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _largest_remainder_allocation(
    capacities: dict[str, int], total: int
) -> dict[str, int]:
    if total < 0 or total > sum(capacities.values()):
        raise ValueError("allocation total must fit within aggregate capacity")
    if total == 0:
        return {key: 0 for key in capacities}
    denominator = sum(capacities.values())
    quotas = {
        key: total * capacity / denominator for key, capacity in capacities.items()
    }
    allocation = {
        key: min(capacities[key], math.floor(quota))
        for key, quota in quotas.items()
    }
    remaining = total - sum(allocation.values())
    order = sorted(
        capacities,
        key=lambda key: (-(quotas[key] - math.floor(quotas[key])), key),
    )
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] >= capacities[key]:
                continue
            allocation[key] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("unable to complete bounded allocation")
    return allocation


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to build the split manifest") from error
    return parquet.read_table(path).to_pylist()


def _gsm8k_splits(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if len(rows) != 1319:
        raise ValueError(f"expected 1319 GSM8K test rows, found {len(rows)}")
    calibration_indices = list(QUERY_INDICES)
    candidates = [index for index in range(len(rows)) if index not in QUERY_INDICES]
    random.Random(_stable_seed("gsm8k-heldout")).shuffle(candidates)
    heldout_indices = sorted(candidates[:HELDOUT_QUERIES_PER_TASK])

    def entry(index: int) -> dict[str, Any]:
        return {
            "row_index": index,
            "prompt_sha256": _row_hash(str(rows[index]["question"])),
        }

    return {
        "calibration": [entry(index) for index in calibration_indices],
        "heldout": [entry(index) for index in heldout_indices],
    }


def _math500_splits(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if len(rows) != 500:
        raise ValueError(f"expected 500 MATH-500 test rows, found {len(rows)}")
    strata: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        stratum = f"{row['type']}|{row['level']}"
        strata[stratum].append(index)
    for stratum, indices in strata.items():
        random.Random(_stable_seed(f"math500|{stratum}")).shuffle(indices)

    sample_total = CALIBRATION_QUERIES_PER_TASK + HELDOUT_QUERIES_PER_TASK
    sample_allocation = _largest_remainder_allocation(
        {stratum: len(indices) for stratum, indices in strata.items()}, sample_total
    )
    calibration_allocation = _largest_remainder_allocation(
        sample_allocation, CALIBRATION_QUERIES_PER_TASK
    )

    calibration_indices: list[int] = []
    heldout_indices: list[int] = []
    for stratum in sorted(strata):
        selected = strata[stratum][: sample_allocation[stratum]]
        cutoff = calibration_allocation[stratum]
        calibration_indices.extend(selected[:cutoff])
        heldout_indices.extend(selected[cutoff:])

    def entry(index: int) -> dict[str, Any]:
        row = rows[index]
        return {
            "row_index": index,
            "prompt_sha256": _row_hash(str(row["problem"])),
            "stratum": f"{row['type']}|{row['level']}",
        }

    return {
        "calibration": [entry(index) for index in sorted(calibration_indices)],
        "heldout": [entry(index) for index in sorted(heldout_indices)],
    }


def build_split_manifest(
    gsm8k_path: Path, math500_path: Path
) -> dict[str, Any]:
    observed_hashes = {
        "gsm8k": sha256_file(gsm8k_path),
        "math500": sha256_file(math500_path),
    }
    expected_hashes = {
        "gsm8k": GSM8K_FILE_SHA256,
        "math500": MATH500_FILE_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise ValueError(
            f"dataset checksum mismatch: expected {expected_hashes}, observed {observed_hashes}"
        )
    gsm8k_rows = _load_parquet(gsm8k_path)
    math500_rows = _load_parquet(math500_path)
    return {
        "split_version": SPLIT_VERSION,
        "planning_seed": PLANNING_SEED,
        "sources": {
            "gsm8k": {
                "dataset_id": "openai/gsm8k",
                "config": "main",
                "split": "test",
                "revision": GSM8K_REVISION,
                "parquet_sha256": GSM8K_FILE_SHA256,
                "row_count": len(gsm8k_rows),
            },
            "math500": {
                "dataset_id": "ankner/math-500",
                "split": "test",
                "revision": MATH500_REVISION,
                "parquet_sha256": MATH500_FILE_SHA256,
                "row_count": len(math500_rows),
            },
        },
        "splits": {
            "gsm8k": _gsm8k_splits(gsm8k_rows),
            "math500": _math500_splits(math500_rows),
        },
    }


def resource_budget() -> dict[str, float | int]:
    calibration_gsm8k = (
        CALIBRATION_QUERIES_PER_TASK
        * (len(POLICIES) - len(PRIMARY_POLICIES))
        * SCREENING_ROLLOUTS
    )
    calibration_math500 = (
        CALIBRATION_QUERIES_PER_TASK * len(POLICIES) * SCREENING_ROLLOUTS
    )
    heldout_screening = (
        2 * HELDOUT_QUERIES_PER_TASK * len(POLICIES) * SCREENING_ROLLOUTS
    )
    heldout_primary_extension = (
        2
        * HELDOUT_QUERIES_PER_TASK
        * len(PRIMARY_POLICIES)
        * (FINAL_PRIMARY_ROLLOUTS - SCREENING_ROLLOUTS)
    )
    total = (
        calibration_gsm8k
        + calibration_math500
        + heldout_screening
        + heldout_primary_extension
    )
    gpu_hours = total * PHASE0_MEAN_SECONDS_PER_GENERATION / 3600
    return {
        "calibration_gsm8k_generations": calibration_gsm8k,
        "calibration_math500_generations": calibration_math500,
        "heldout_screening_generations": heldout_screening,
        "heldout_primary_extension_generations": heldout_primary_extension,
        "total_new_generations": total,
        "phase0_calibrated_gpu_hours": gpu_hours,
        "gpu_hours_with_30_percent_contingency": gpu_hours * 1.3,
        "estimated_trace_gib": total * PHASE0_MEAN_TRACE_BYTES / 1024**3,
        "four_gpu_wall_hours": gpu_hours / 4,
        "eight_gpu_wall_hours": gpu_hours / 8,
    }


def render_planning_report(manifest: dict[str, Any]) -> str:
    budget = resource_budget()
    gsm = manifest["splits"]["gsm8k"]
    math500 = manifest["splits"]["math500"]
    return "\n".join(
        [
            "# Phase 1 CPU Planning Report",
            "",
            f"- Planning seed: `{PLANNING_SEED}`",
            f"- Split version: `{SPLIT_VERSION}`",
            f"- GSM8K: {len(gsm['calibration'])} calibration / {len(gsm['heldout'])} held-out queries",
            f"- MATH-500: {len(math500['calibration'])} calibration / {len(math500['heldout'])} held-out queries",
            "- The 50 GSM8K calibration queries are exactly the completed Phase 0 signal set.",
            "",
            "## Power Resolution",
            "",
            f"- Per-task held-out correlation MDE (n=100, two-sided alpha=.05, 80% power): {minimum_detectable_correlation(100):.3f}",
            f"- Task-stratified pooled correlation MDE (n=200 approximation): {minimum_detectable_correlation(200):.3f}",
            f"- Per-task standardized paired-effect MDE (n=100): {minimum_detectable_paired_effect(100):.3f}",
            "- These are planning approximations, not replacements for query-clustered bootstrap intervals in the final analysis.",
            "",
            "## Resource Budget",
            "",
            f"- New generations: {budget['total_new_generations']:,}",
            f"- Phase-0-calibrated GPU-hours: {budget['phase0_calibrated_gpu_hours']:.1f}",
            f"- GPU-hours with 30% MATH/runtime contingency: {budget['gpu_hours_with_30_percent_contingency']:.1f}",
            f"- Approximate wall time on 4 GPUs: {budget['four_gpu_wall_hours']:.1f} hours before contingency",
            f"- Approximate wall time on 8 GPUs: {budget['eight_gpu_wall_hours']:.1f} hours before contingency",
            f"- Estimated compressed trace storage: {budget['estimated_trace_gib']:.1f} GiB",
            "",
            "## Design Consequence",
            "",
            "All five schedules receive 16 screening rollouts. Only the frozen AR and AO-B32 primary comparison is extended to 64 rollouts on held-out queries. Calibration outcomes may define the BCI and matching rule, but held-out outcomes may not modify them.",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k-parquet", type=Path, required=True)
    parser.add_argument("--math500-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_split_manifest(args.gsm8k_parquet, args.math500_parquet)
    write_json_atomic(args.output_manifest, manifest)
    write_text_atomic(args.output_report, render_planning_report(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
