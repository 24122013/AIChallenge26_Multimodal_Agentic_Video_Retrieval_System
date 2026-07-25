from __future__ import annotations

import argparse
import json
import math
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.indexing.io_utils import atomic_record_writer, iter_keyframe_records


SCHEMA_VERSION = "1.0"


def _required_text(record: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Missing required field; expected one of: {', '.join(names)}")


def _optional_frame_index(record: dict[str, Any]) -> int | None:
    value = record.get("frame_index")
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("frame_index must be an integer, not boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid frame_index: {value!r}") from exc
    if result < 0 or float(value) != result:
        raise ValueError(f"frame_index must be a non-negative integer: {value!r}")
    return result


def _timestamp(
    record: dict[str, Any],
    *,
    frame_index: int | None,
    fallback_fps: float | None,
) -> tuple[float, str]:
    raw_timestamp = record.get("timestamp")
    if raw_timestamp is not None and raw_timestamp != "":
        try:
            timestamp = float(raw_timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid timestamp: {raw_timestamp!r}") from exc
        source = str(record.get("timestamp_source") or "metadata")
    else:
        raw_fps = record.get("fps")
        if raw_fps is None:
            raw_fps = record.get("video_fps")
        if raw_fps is None:
            raw_fps = fallback_fps
        if frame_index is None or raw_fps is None:
            raise ValueError(
                "Missing timestamp. Supply frame_index plus per-record fps/video_fps "
                "or the --fps fallback."
            )
        try:
            fps = float(raw_fps)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid FPS: {raw_fps!r}") from exc
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"FPS must be finite and > 0: {raw_fps!r}")
        timestamp = frame_index / fps
        source = "frame_index_fps"
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"timestamp must be finite and non-negative: {timestamp!r}")
    return (round(timestamp, 6), source)


def _create_staging_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE keyframes (
            video_id TEXT NOT NULL,
            frame_id TEXT NOT NULL,
            frame_index INTEGER,
            timestamp REAL NOT NULL,
            timestamp_source TEXT NOT NULL,
            PRIMARY KEY (video_id, frame_id)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX keyframes_video_time
        ON keyframes(video_id, timestamp, frame_index, frame_id)
        """
    )


def _stage_records(
    connection: sqlite3.Connection,
    input_path: Path,
    *,
    fallback_fps: float | None,
) -> tuple[int, int]:
    inserted = duplicate_count = 0
    for ordinal, record in enumerate(iter_keyframe_records(input_path), start=1):
        try:
            video_id = _required_text(record, ("video_id",))
            frame_id = _required_text(record, ("frame_id", "keyframe_id"))
            frame_index = _optional_frame_index(record)
            timestamp, timestamp_source = _timestamp(
                record,
                frame_index=frame_index,
                fallback_fps=fallback_fps,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid keyframe record {ordinal}: {exc}") from exc
        existing = connection.execute(
            """
            SELECT frame_index, timestamp, timestamp_source
            FROM keyframes WHERE video_id = ? AND frame_id = ?
            """,
            (video_id, frame_id),
        ).fetchone()
        if existing is not None:
            expected = (frame_index, timestamp, timestamp_source)
            if tuple(existing) != expected:
                raise ValueError(
                    f"Conflicting duplicate keyframe ({video_id!r}, {frame_id!r})"
                )
            duplicate_count += 1
            continue
        connection.execute(
            """
            INSERT INTO keyframes(
                video_id, frame_id, frame_index, timestamp, timestamp_source
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (video_id, frame_id, frame_index, timestamp, timestamp_source),
        )
        inserted += 1
    connection.commit()
    if inserted == 0:
        raise ValueError(f"No keyframe records found in: {input_path}")
    return inserted, duplicate_count


def _neighbor_rows(
    connection: sqlite3.Connection,
    *,
    video_id: str,
    frame_id: str,
    timestamp: float,
    window_seconds: float,
    before: bool,
) -> list[sqlite3.Row]:
    comparator = "<" if before else ">"
    lower = max(0.0, timestamp - window_seconds)
    upper = timestamp + window_seconds
    query = f"""
        SELECT frame_id, frame_index, timestamp
        FROM keyframes
        WHERE video_id = ?
          AND frame_id != ?
          AND timestamp {comparator} ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp, COALESCE(frame_index, -1), frame_id
    """
    return list(
        connection.execute(query, (video_id, frame_id, timestamp, lower, upper))
    )


def _compact_neighbor(row: sqlite3.Row, center_timestamp: float) -> dict[str, Any]:
    # The canonical frame map already owns frame_index/timestamp/path metadata.
    # Keeping only the reference and signed offset avoids copying those fields
    # into every neighbor occurrence.
    return {
        "frame_id": str(row["frame_id"]),
        "delta_seconds": round(float(row["timestamp"]) - center_timestamp, 6),
    }


def iter_neighbor_records(
    connection: sqlite3.Connection,
    *,
    window_seconds: float,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic compact neighbor records from a staged database."""
    centers = connection.execute(
        """
        SELECT video_id, frame_id, frame_index, timestamp, timestamp_source
        FROM keyframes
        ORDER BY video_id, timestamp, COALESCE(frame_index, -1), frame_id
        """
    )
    for center in centers:
        timestamp = float(center["timestamp"])
        before = _neighbor_rows(
            connection,
            video_id=str(center["video_id"]),
            frame_id=str(center["frame_id"]),
            timestamp=timestamp,
            window_seconds=window_seconds,
            before=True,
        )
        after = _neighbor_rows(
            connection,
            video_id=str(center["video_id"]),
            frame_id=str(center["frame_id"]),
            timestamp=timestamp,
            window_seconds=window_seconds,
            before=False,
        )
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "video_id": str(center["video_id"]),
            "frame_id": str(center["frame_id"]),
            "timestamp": round(timestamp, 6),
            "timestamp_source": str(center["timestamp_source"]),
            "neighbors_before": [
                _compact_neighbor(row, timestamp) for row in before
            ],
            "neighbors_after": [
                _compact_neighbor(row, timestamp) for row in after
            ],
        }
        if center["frame_index"] is not None:
            result["frame_index"] = int(center["frame_index"])
        yield result


def build_neighbor_index(
    input_path: Path,
    output_path: Path,
    *,
    window_seconds: float,
    fps: float | None = None,
) -> dict[str, Any]:
    """Build a compact, deterministic temporal-neighbor index.

    JSONL input is streamed into a temporary SQLite table, so the implementation
    does not retain the full metadata collection in Python memory. Duplicate
    identities are ignored only when their normalized values are identical.
    """
    if not math.isfinite(window_seconds) or window_seconds < 0:
        raise ValueError("window_seconds must be finite and >= 0")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be finite and > 0")
    if input_path.is_file() and input_path.resolve() == output_path.resolve():
        raise ValueError("output must not overwrite the frame-level input metadata")

    with tempfile.TemporaryDirectory(prefix="neighbor-index-") as temporary_dir:
        database_path = Path(temporary_dir) / "staging.sqlite3"
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        try:
            _create_staging_database(connection)
            record_count, duplicate_count = _stage_records(
                connection,
                input_path,
                fallback_fps=fps,
            )
            with atomic_record_writer(output_path) as writer:
                for record in iter_neighbor_records(
                    connection,
                    window_seconds=window_seconds,
                ):
                    writer.write(record)
        finally:
            connection.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "window_seconds": window_seconds,
        "record_count": record_count,
        "duplicate_input_count": duplicate_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic timestamp-window neighbor index."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSON/JSONL file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Compact .jsonl or .json output")
    parser.add_argument("--window-seconds", type=float, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Fallback FPS only when a record has no timestamp or per-record FPS.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = build_neighbor_index(
            args.input,
            args.output,
            window_seconds=args.window_seconds,
            fps=args.fps,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
