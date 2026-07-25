from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO


def discover_metadata_files(path: Path) -> list[Path]:
    """Return deterministic JSON/JSONL inputs for a file or directory."""
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Input must be a JSON/JSONL file or directory: {path}")
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(f"No .json or .jsonl files found in: {path}")
    return files


def iter_metadata_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream object records from JSONL, JSON arrays, or frame-map JSON files."""
    for input_path in discover_metadata_files(path):
        if input_path.suffix.lower() == ".jsonl":
            yield from _iter_jsonl(input_path)
        else:
            yield from _iter_json(input_path)


def iter_keyframe_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream keyframes, preferring the repository's keyframe filename contract."""
    if path.is_dir():
        preferred = sorted(path.glob("keyframes_*.jsonl"))
        preferred = [
            candidate
            for candidate in preferred
            if candidate.is_file()
            and not candidate.name.endswith(("_report.jsonl", "_validation.jsonl"))
        ]
        if preferred:
            for input_path in preferred:
                yield from _iter_jsonl(input_path)
            return
    yield from iter_metadata_records(path)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path} at line {line_number}")
            yield value


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if isinstance(value, list):
        for offset, record in enumerate(value):
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path} at array offset {offset}")
            yield record
        return
    if isinstance(value, dict):
        # A single metadata record has identifying fields. Otherwise this is a
        # frame map whose values are records keyed by FAISS offset.
        if any(key in value for key in ("frame_id", "keyframe_id", "video_id")):
            yield value
            return
        for key in sorted(value, key=_natural_mapping_key):
            record = value[key]
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path} at key {key!r}")
            yield record
        return
    raise ValueError(f"Expected a JSON object or array in {path}")


def _natural_mapping_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


@contextmanager
def atomic_record_writer(path: Path) -> Iterator["RecordWriter"]:
    """Write compact records and replace the destination only after success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    writer: RecordWriter | None = None
    try:
        handle = temporary.open("w", encoding="utf-8", newline="\n")
        writer = RecordWriter(handle, as_json_array=path.suffix.lower() == ".json")
        writer.start()
        yield writer
        writer.finish()
        handle.close()
        os.replace(temporary, path)
    except Exception:
        if writer is not None and not writer.handle.closed:
            writer.handle.close()
        temporary.unlink(missing_ok=True)
        raise


class RecordWriter:
    """Compact deterministic JSON/JSONL record writer."""

    def __init__(self, handle: TextIO, *, as_json_array: bool) -> None:
        self.handle = handle
        self.as_json_array = as_json_array
        self._first = True

    def start(self) -> None:
        if self.as_json_array:
            self.handle.write("[")

    def write(self, record: dict[str, Any]) -> None:
        payload = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        if self.as_json_array:
            if not self._first:
                self.handle.write(",")
            self.handle.write(payload)
        else:
            self.handle.write(payload + "\n")
        self._first = False

    def finish(self) -> None:
        if self.as_json_array:
            self.handle.write("]\n")


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records atomically and return the number written."""
    count = 0
    with atomic_record_writer(path) as writer:
        for record in records:
            writer.write(record)
            count += 1
    return count
