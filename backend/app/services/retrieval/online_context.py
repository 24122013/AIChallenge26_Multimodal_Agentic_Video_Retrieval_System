"""Read canonical neighbor and segment artifacts for online retrieval.

This module is deliberately read-only.  It consumes artifacts published by the
offline pipeline without importing or invoking any offline stage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.app.services.metadata.metadata_store import MetadataStore


DEFAULT_NEIGHBOR_PATH = Path("data/metadata/neighbors_all.jsonl")
DEFAULT_SEGMENT_PATH = Path("data/metadata/segments_all.jsonl")


@dataclass(frozen=True)
class ContextLookup:
    """Canonical context resolved for one retrieved frame."""

    segment_id: str
    neighbors: tuple[dict[str, Any], ...]
    segment: dict[str, Any] | None
    sources: tuple[str, ...]


class OnlineContextIndex:
    """In-memory lookup over canonical online context artifacts."""

    def __init__(
        self,
        *,
        neighbor_records: Iterable[Mapping[str, Any]] = (),
        segment_records: Iterable[Mapping[str, Any]] = (),
        metadata_store: MetadataStore | None = None,
        neighbor_path: Path | None = None,
        segment_path: Path | None = None,
        frame_map_path: Path | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.neighbor_path = neighbor_path
        self.segment_path = segment_path
        self.frame_map_path = frame_map_path
        self._neighbors: dict[tuple[str, str], dict[str, Any]] = {}
        self._segments: dict[tuple[str, str], dict[str, Any]] = {}
        self._segment_by_frame: dict[tuple[str, str], str] = {}

        for raw in neighbor_records:
            record = dict(raw)
            video_id = _required_text(record, "video_id", "neighbor")
            frame_id = _required_text(record, "frame_id", "neighbor")
            key = (video_id, frame_id)
            if key in self._neighbors:
                raise ValueError(f"Duplicate canonical neighbor record: {key}")
            _validate_neighbor_list(record.get("neighbors_before"), key)
            _validate_neighbor_list(record.get("neighbors_after"), key)
            self._neighbors[key] = record

        for raw in segment_records:
            record = dict(raw)
            video_id = _required_text(record, "video_id", "segment")
            segment_id = _required_text(record, "segment_id", "segment")
            key = (video_id, segment_id)
            if key in self._segments:
                raise ValueError(f"Duplicate canonical segment record: {key}")
            self._segments[key] = record
            keyframe_ids = record.get("keyframe_ids") or ()
            if not isinstance(keyframe_ids, (list, tuple)):
                raise ValueError(f"segment keyframe_ids must be a list: {key}")
            for raw_frame_id in keyframe_ids:
                frame_id = str(raw_frame_id).strip()
                if not frame_id:
                    continue
                frame_key = (video_id, frame_id)
                existing = self._segment_by_frame.get(frame_key)
                if existing is not None and existing != segment_id:
                    raise ValueError(
                        f"Canonical frame belongs to multiple segments: {frame_key}"
                    )
                self._segment_by_frame[frame_key] = segment_id

    @classmethod
    def from_artifacts(
        cls,
        *,
        neighbor_path: str | Path = DEFAULT_NEIGHBOR_PATH,
        segment_path: str | Path = DEFAULT_SEGMENT_PATH,
        frame_map_path: str | Path | None = None,
        require_neighbors: bool = True,
        require_segments: bool = True,
    ) -> "OnlineContextIndex":
        """Load selected canonical artifacts without modifying them."""

        resolved_neighbors = Path(neighbor_path)
        resolved_segments = Path(segment_path)
        resolved_frame_map = Path(frame_map_path) if frame_map_path is not None else None
        neighbor_records = _load_jsonl(
            resolved_neighbors,
            required=require_neighbors,
            label="neighbor",
        )
        segment_records = _load_jsonl(
            resolved_segments,
            required=require_segments,
            label="segment",
        )
        metadata_store = None
        if resolved_frame_map is not None and resolved_frame_map.is_file():
            metadata_store = MetadataStore.from_frame_map(resolved_frame_map)
        return cls(
            neighbor_records=neighbor_records,
            segment_records=segment_records,
            metadata_store=metadata_store,
            neighbor_path=resolved_neighbors,
            segment_path=resolved_segments,
            frame_map_path=resolved_frame_map,
        )

    def lookup(
        self,
        *,
        video_id: str,
        frame_id: str,
        timestamp: float,
        segment_id: str = "",
        existing_neighbors: Iterable[Mapping[str, Any]] = (),
    ) -> ContextLookup:
        """Resolve canonical neighbors and segment context for a candidate."""

        key = (str(video_id), str(frame_id))
        canonical_segment_id = self._segment_by_frame.get(key, "")
        resolved_segment_id = canonical_segment_id or str(segment_id or "")
        sources: list[str] = []

        neighbors = [dict(item) for item in existing_neighbors]
        record = self._neighbors.get(key)
        if record is not None:
            neighbors = self._canonical_neighbors(
                video_id=key[0],
                center_timestamp=float(timestamp),
                record=record,
            )
            sources.append("neighbors_all")
        neighbors = _dedupe_neighbors(neighbors)

        segment = self._segments.get((key[0], resolved_segment_id))
        if segment is not None:
            sources.append("segments_all")
        return ContextLookup(
            segment_id=resolved_segment_id,
            neighbors=tuple(neighbors),
            segment=dict(segment) if segment is not None else None,
            sources=tuple(sources),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "neighbor_record_count": len(self._neighbors),
            "segment_record_count": len(self._segments),
            "mapped_keyframe_count": len(self._segment_by_frame),
            "frame_map_loaded": self.metadata_store is not None,
            "neighbor_path": str(self.neighbor_path) if self.neighbor_path else None,
            "segment_path": str(self.segment_path) if self.segment_path else None,
            "frame_map_path": str(self.frame_map_path) if self.frame_map_path else None,
        }

    def _canonical_neighbors(
        self,
        *,
        video_id: str,
        center_timestamp: float,
        record: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for field, direction in (
            ("neighbors_before", "before"),
            ("neighbors_after", "after"),
        ):
            for reference in record.get(field) or ():
                frame_id = str(reference.get("frame_id") or "")
                delta = float(reference.get("delta_seconds") or 0.0)
                payload: dict[str, Any]
                frame = (
                    self.metadata_store.get_by_frame_id(frame_id)
                    if self.metadata_store is not None
                    else None
                )
                if frame is not None and frame.video_id == video_id:
                    payload = frame.to_dict()
                else:
                    payload = {
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "timestamp": round(center_timestamp + delta, 6),
                        "segment_id": self._segment_by_frame.get(
                            (video_id, frame_id),
                            "",
                        ),
                    }
                payload["delta_seconds"] = delta
                payload["direction"] = direction
                output.append(payload)
        return output


def _load_jsonl(path: Path, *, required: bool, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Canonical {label} artifact not found: {path}")
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _required_text(record: Mapping[str, Any], field: str, label: str) -> str:
    value = str(record.get(field) or "").strip()
    if not value:
        raise ValueError(f"Canonical {label} record is missing {field}")
    return value


def _validate_neighbor_list(value: object, key: tuple[str, str]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(f"Canonical neighbor list must be an array: {key}")
    for item in value:
        if not isinstance(item, Mapping) or not str(item.get("frame_id") or "").strip():
            raise ValueError(f"Invalid canonical neighbor reference: {key}")
        try:
            float(item.get("delta_seconds"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid neighbor delta_seconds: {key}") from exc


def _dedupe_neighbors(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for value in values:
        payload = dict(value)
        key = (
            str(payload.get("video_id") or ""),
            str(payload.get("frame_id") or ""),
        )
        if key[1]:
            merged[key] = payload
    return sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("timestamp") or 0.0),
            str(item.get("frame_id") or ""),
        ),
    )


__all__ = [
    "ContextLookup",
    "DEFAULT_NEIGHBOR_PATH",
    "DEFAULT_SEGMENT_PATH",
    "OnlineContextIndex",
]
