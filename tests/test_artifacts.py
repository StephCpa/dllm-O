from __future__ import annotations

import gzip
import json
from dataclasses import replace

import pytest

from dllm_order_transmission.artifacts import (
    AtomicJsonlTraceSink,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)
from dllm_order_transmission.trace import StepTrace


def sample_event() -> StepTrace:
    return StepTrace(
        batch_index=0,
        global_step=0,
        block_index=0,
        step_in_block=0,
        active_block_start=2,
        active_block_end=3,
        eligible_positions=(2,),
        entropy=(1.0,),
        top1_token_ids=(1,),
        top2_token_ids=(2,),
        top1_logits=(2.0,),
        top2_logits=(1.0,),
        logit_margins=(1.0,),
        probability_margins=(0.3,),
        candidate_token_ids=(1,),
        candidate_probabilities=(0.6,),
        eligible_entropy=(1.0,),
        eligible_logit_margins=(1.0,),
        selected_positions=(2,),
        selected_token_ids=(1,),
        state_checksum_before="before",
        state_checksum_after="after",
    )


def test_atomic_trace_sink_publishes_complete_gzip(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    with AtomicJsonlTraceSink(path) as sink:
        sink(sample_event())
        sink(replace(sample_event(), global_step=1))
    assert sink.event_count == 2
    assert len(sha256_file(path)) == 64
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert [row["global_step"] for row in rows] == [0, 1]


def test_atomic_trace_sink_does_not_publish_partial_output(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    with pytest.raises(RuntimeError):
        with AtomicJsonlTraceSink(path) as sink:
            sink(sample_event())
            raise RuntimeError("simulated failure")
    assert not path.exists()
    assert not (tmp_path / ".trace.jsonl.gz.tmp").exists()


def test_atomic_json_writer(tmp_path) -> None:
    path = tmp_path / "record.json"
    write_json_atomic(path, {"complete": True, "count": 3})
    assert json.loads(path.read_text()) == {"complete": True, "count": 3}


def test_atomic_text_writer(tmp_path) -> None:
    path = tmp_path / "report.md"
    write_text_atomic(path, "# Report")
    assert path.read_text() == "# Report\n"
