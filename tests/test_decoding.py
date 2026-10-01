from __future__ import annotations

import math

import pytest
import torch

from conftest import ToyMaskedModel
from dllm_order_transmission.decoding import (
    DecodeConfig,
    generate_with_trace,
    get_num_transfer_tokens,
)
from dllm_order_transmission.trace import (
    ListTraceSink,
    reconstruct_completion,
    validate_trace,
)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def test_transfer_counts_preserve_total() -> None:
    mask = torch.ones((2, 7), dtype=torch.bool)
    counts = get_num_transfer_tokens(mask, steps=4)
    assert counts.tolist() == [[2, 2, 2, 1], [2, 2, 2, 1]]
    assert torch.equal(counts.sum(dim=1), torch.tensor([7, 7]))


@pytest.mark.parametrize("block_length", [1, 32, 256])
def test_protocol_block_schedules_satisfy_trace_invariants(block_length: int) -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=256,
        gen_length=256,
        block_length=block_length,
        temperature=0.0,
        mask_id=7,
    )
    sink = ListTraceSink()
    output = generate_with_trace(model, prompt, config=config, trace_callback=sink)

    errors = validate_trace(
        sink.events,
        prompt_length=prompt.shape[1],
        gen_length=config.gen_length,
        total_steps=config.steps,
        block_length=config.block_length,
    )
    assert errors == []
    reconstructed = reconstruct_completion(
        sink.events,
        prompt_length=prompt.shape[1],
        gen_length=config.gen_length,
    )
    assert reconstructed == output[0, prompt.shape[1] :].tolist()


def test_strict_ar_closure_is_zero_by_construction() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=1,
        temperature=0.0,
        mask_id=7,
    )
    sink = ListTraceSink()
    generate_with_trace(model, prompt, config=config, trace_callback=sink)
    for event in sink.events:
        selected = event.selected_positions[0]
        index = event.eligible_positions.index(selected)
        assert event.entropy[index] == event.eligible_entropy[index]
        assert event.logit_margins[index] == event.eligible_logit_margins[index]


@pytest.mark.parametrize("block_length", [1, 4, 8])
def test_trace_callback_does_not_change_stochastic_output(block_length: int) -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=block_length,
        temperature=0.6,
        mask_id=7,
    )
    untraced = generate_with_trace(
        model,
        prompt,
        config=config,
        generator=make_generator(1234),
    )
    sink = ListTraceSink()
    traced = generate_with_trace(
        model,
        prompt,
        config=config,
        generator=make_generator(1234),
        trace_callback=sink,
    )
    assert torch.equal(untraced, traced)


def test_different_rollout_seeds_change_stochastic_generation() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=8,
        temperature=0.6,
        mask_id=7,
    )
    first = generate_with_trace(
        model, prompt, config=config, generator=make_generator(10)
    )
    second = generate_with_trace(
        model, prompt, config=config, generator=make_generator(11)
    )
    assert not torch.equal(first, second)


def test_full_distribution_trace_fields() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=256,
        gen_length=256,
        block_length=32,
        temperature=0.0,
        mask_id=7,
    )
    sink = ListTraceSink()
    generate_with_trace(
        model,
        prompt,
        config=config,
        trace_callback=sink,
        full_distribution=True,
    )
    assert len(sink.events) == config.steps
    for event in sink.events:
        payload = event.to_dict()
        assert "top64_token_ids" in payload
        assert "top64_logits" in payload
        assert "full_logsumexp" in payload
        assert len(payload["top64_token_ids"]) == len(event.eligible_positions)
        for row in payload["top64_token_ids"]:
            assert len(row) == min(64, model.vocab_size)
        for row in payload["top64_logits"]:
            assert len(row) == min(64, model.vocab_size)
            assert all(math.isfinite(value) for value in row)
        assert all(
            math.isfinite(value) for value in payload["full_logsumexp"]
        )
    # Plain Phase 1-style traces must not carry the new fields.
    plain_sink = ListTraceSink()
    generate_with_trace(
        model, prompt, config=config, trace_callback=plain_sink
    )
    for event in plain_sink.events:
        payload = event.to_dict()
        assert "top64_token_ids" not in payload
        assert "full_logsumexp" not in payload


def test_full_distribution_leaves_output_unchanged() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=256,
        gen_length=256,
        block_length=16,
        temperature=0.0,
        mask_id=7,
    )
    plain = generate_with_trace(
        model, prompt, config=config, generator=make_generator(7)
    )
    with_fields = generate_with_trace(
        model,
        prompt,
        config=config,
        generator=make_generator(7),
        trace_callback=ListTraceSink(),
        full_distribution=True,
    )
    assert torch.equal(plain, with_fields)


def test_high_entropy_selects_most_uncertain_eligible_position() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=8,
        temperature=0.0,
        remasking="high_entropy",
        mask_id=7,
    )
    sink = ListTraceSink()
    generate_with_trace(model, prompt, config=config, trace_callback=sink)

    for event in sink.events:
        best_entropy = max(event.entropy)
        expected = min(
            position
            for position, entropy in zip(
                event.eligible_positions, event.entropy, strict=True
            )
            if entropy == best_entropy
        )
        assert event.selected_positions == (expected,)


def test_high_entropy_trace_does_not_change_output() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=8,
        temperature=0.6,
        remasking="high_entropy",
        mask_id=7,
    )
    plain = generate_with_trace(
        model, prompt, config=config, generator=make_generator(20260906)
    )
    traced = generate_with_trace(
        model,
        prompt,
        config=config,
        generator=make_generator(20260906),
        trace_callback=ListTraceSink(),
        full_distribution=True,
    )
    assert torch.equal(plain, traced)
