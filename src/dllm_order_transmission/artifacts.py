"""Atomic, checksummed artifact writers for long-running inference jobs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import IO, Any

from .trace import StepTrace


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class AtomicJsonlTraceSink:
    """Stream trace events to gzip JSONL and publish only on successful close."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.suffix != ".gz":
            raise ValueError("trace path must end in .gz")
        self.temp_path = self.path.with_name(f".{self.path.name}.tmp")
        self._handle: IO[str] | None = None
        self.event_count = 0

    def __enter__(self) -> "AtomicJsonlTraceSink":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.temp_path.exists():
            raise FileExistsError(f"stale temporary trace exists: {self.temp_path}")
        self._handle = gzip.open(self.temp_path, mode="wt", encoding="utf-8")
        return self

    def __call__(self, event: StepTrace) -> None:
        if self._handle is None:
            raise RuntimeError("trace sink must be opened as a context manager")
        json.dump(event.to_dict(), self._handle, sort_keys=True, separators=(",", ":"))
        self._handle.write("\n")
        self.event_count += 1

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if exc_type is None:
            os.replace(self.temp_path, self.path)
        elif self.temp_path.exists():
            self.temp_path.unlink()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a compact JSON object through an adjacent temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    if temp_path.exists():
        raise FileExistsError(f"stale temporary JSON exists: {temp_path}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write UTF-8 text through an adjacent temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    if temp_path.exists():
        raise FileExistsError(f"stale temporary text exists: {temp_path}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise
