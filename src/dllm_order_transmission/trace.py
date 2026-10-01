"""Trace schema and invariants for masked-diffusion decoding."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StepTrace:
    """One batch element at one pre-mutation denoising step."""

    batch_index: int
    global_step: int
    block_index: int
    step_in_block: int
    active_block_start: int
    active_block_end: int
    eligible_positions: tuple[int, ...]
    entropy: tuple[float, ...]
    top1_token_ids: tuple[int, ...]
    top2_token_ids: tuple[int, ...]
    top1_logits: tuple[float, ...]
    top2_logits: tuple[float, ...]
    logit_margins: tuple[float, ...]
    probability_margins: tuple[float, ...]
    candidate_token_ids: tuple[int, ...]
    candidate_probabilities: tuple[float, ...]
    eligible_entropy: tuple[float, ...]
    eligible_logit_margins: tuple[float, ...]
    selected_positions: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    state_checksum_before: str
    state_checksum_after: str
    # Phase 2 Option B logging additions (only for phase2-extension runs):
    # per eligible position, the top-64 token ids/logits, the full softmax
    # normalizer, and the full entropy. Absent from Phase 1 traces.
    top64_token_ids: tuple[tuple[int, ...], ...] | None = None
    top64_logits: tuple[tuple[float, ...], ...] | None = None
    full_logsumexp: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


class ListTraceSink:
    """In-memory trace sink used by tests and small smoke runs."""

    def __init__(self) -> None:
        self.events: list[StepTrace] = []

    def __call__(self, event: StepTrace) -> None:
        self.events.append(event)


def reconstruct_completion(
    events: Iterable[StepTrace],
    *,
    prompt_length: int,
    gen_length: int,
    batch_index: int = 0,
) -> list[int]:
    """Reconstruct committed completion tokens from step events."""
    completion: list[int | None] = [None] * gen_length
    for event in events:
        if event.batch_index != batch_index:
            continue
        for position, token_id in zip(
            event.selected_positions, event.selected_token_ids, strict=True
        ):
            offset = position - prompt_length
            if not 0 <= offset < gen_length:
                raise ValueError(f"committed position {position} is outside completion")
            if completion[offset] is not None:
                raise ValueError(f"position {position} was committed more than once")
            completion[offset] = token_id
    if any(token is None for token in completion):
        missing = [i for i, token in enumerate(completion) if token is None]
        raise ValueError(f"missing completion offsets: {missing[:10]}")
    return [int(token) for token in completion]


def validate_trace(
    events: Iterable[StepTrace],
    *,
    prompt_length: int,
    gen_length: int,
    total_steps: int,
    block_length: int,
    batch_size: int = 1,
) -> list[str]:
    """Return all protocol-invariant violations found in a trace."""
    records = list(events)
    errors: list[str] = []
    expected_blocks = gen_length // block_length

    for batch_index in range(batch_size):
        batch_events = sorted(
            (event for event in records if event.batch_index == batch_index),
            key=lambda event: event.global_step,
        )
        if len(batch_events) != total_steps:
            errors.append(
                f"batch {batch_index}: expected {total_steps} events, "
                f"found {len(batch_events)}"
            )
        observed_steps = [event.global_step for event in batch_events]
        if observed_steps != list(range(total_steps)):
            errors.append(f"batch {batch_index}: global steps are not contiguous")

        committed: dict[int, int] = {}
        for event in batch_events:
            if not 0 <= event.block_index < expected_blocks:
                errors.append(f"invalid block index at step {event.global_step}")
            fields = (
                event.entropy,
                event.top1_token_ids,
                event.top2_token_ids,
                event.top1_logits,
                event.top2_logits,
                event.logit_margins,
                event.probability_margins,
                event.candidate_token_ids,
                event.candidate_probabilities,
                event.eligible_entropy,
                event.eligible_logit_margins,
            )
            if any(len(field) != len(event.eligible_positions) for field in fields):
                errors.append(f"unaligned trace fields at step {event.global_step}")

            numeric_fields = (
                event.entropy,
                event.top1_logits,
                event.top2_logits,
                event.logit_margins,
                event.probability_margins,
                event.candidate_probabilities,
                event.eligible_entropy,
                event.eligible_logit_margins,
            )
            if any(not math.isfinite(value) for field in numeric_fields for value in field):
                errors.append(f"non-finite metric at step {event.global_step}")

            eligible_set = set(event.eligible_positions)
            for position, token_id in zip(
                event.selected_positions, event.selected_token_ids, strict=True
            ):
                if position not in eligible_set:
                    errors.append(
                        f"step {event.global_step}: selected ineligible position {position}"
                    )
                if not event.active_block_start <= position < event.active_block_end:
                    errors.append(
                        f"step {event.global_step}: selected position outside active block"
                    )
                if position in committed:
                    errors.append(
                        f"position {position} committed at steps "
                        f"{committed[position]} and {event.global_step}"
                    )
                committed[position] = event.global_step
                if token_id < 0:
                    errors.append(f"negative token ID committed at step {event.global_step}")

        expected_positions = set(range(prompt_length, prompt_length + gen_length))
        if set(committed) != expected_positions:
            missing = sorted(expected_positions - set(committed))
            extra = sorted(set(committed) - expected_positions)
            errors.append(
                f"batch {batch_index}: completion coverage mismatch; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

    return errors
