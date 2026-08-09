from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.indexing.io_utils import (
    iter_keyframe_records,
    iter_metadata_records,
    write_records,
)


SCHEMA_VERSION = "1.0"
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(value: object) -> str:
    """Normalize Unicode and whitespace without changing display casing."""
    return _WHITESPACE.sub(
        " ",
        unicodedata.normalize("NFKC", str(value or "")).strip(),
    )


def comparison_text(value: object) -> str:
    """Return a stable comparison key for text deduplication."""
    return _WHITESPACE.sub(
        " ",
        _NON_WORD.sub(" ", normalize_text(value).casefold()),
    ).strip()


def _finite_non_negative(value: object, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative: {value!r}")
    return round(result, 6)


def _frame_index(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("frame_index must be an integer, not boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame_index must be an integer: {value!r}") from exc
    if result < 0 or float(value) != result:
        raise ValueError(f"frame_index must be a non-negative integer: {value!r}")
    return result


def _record_timestamp(record: dict[str, Any], fallback_fps: float | None) -> float:
    value = record.get("timestamp")
    if value is not None and value != "":
        return _finite_non_negative(value, "timestamp")
    frame_index = _frame_index(record.get("frame_index"))
    raw_fps = record.get("fps")
    if raw_fps is None:
        raw_fps = record.get("video_fps")
    if raw_fps is None:
        raw_fps = fallback_fps
    if frame_index is None or raw_fps is None:
        raise ValueError(
            "missing timestamp; supply frame_index plus fps/video_fps or --fps"
        )
    fps = _finite_non_negative(raw_fps, "fps")
    if fps == 0:
        raise ValueError("fps must be > 0")
    return round(frame_index / fps, 6)


@dataclass(frozen=True)
class Keyframe:
    """Minimal keyframe representation retained while aggregating segments."""

    video_id: str
    frame_id: str
    timestamp: float
    frame_index: int | None
    segment_id: str
    boundary_start: float | None
    boundary_end: float | None
    selection_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    """Mutable segment group assembled from keyframe metadata."""

    video_id: str
    segment_id: str
    keyframes: list[Keyframe] = field(default_factory=list)

    def ordered_keyframes(self) -> list[Keyframe]:
        return sorted(
            self.keyframes,
            key=lambda item: (
                item.timestamp,
                item.frame_index if item.frame_index is not None else -1,
                item.frame_id,
            ),
        )

    def time_range(self) -> tuple[float, float]:
        starts = [
            item.boundary_start
            for item in self.keyframes
            if item.boundary_start is not None
        ]
        ends = [
            item.boundary_end
            for item in self.keyframes
            if item.boundary_end is not None
        ]
        ordered = self.ordered_keyframes()
        start = min(starts) if starts else ordered[0].timestamp
        end = max(ends) if ends else ordered[-1].timestamp
        if start < 0 or end < 0 or start > end:
            raise ValueError(
                f"Invalid segment range for {self.segment_id}: start={start}, end={end}"
            )
        return round(start, 6), round(end, 6)


def _source_id(
    record: dict[str, Any],
    *,
    preferred: Sequence[str],
    fallback_prefix: str,
    ordinal: int,
) -> str:
    for name in preferred:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    frame_id = record.get("frame_id") or record.get("keyframe_id")
    if frame_id:
        return str(frame_id)
    return f"{fallback_prefix}_{ordinal:08d}"


def _selection_metadata(record: dict[str, Any], frame_id: str) -> dict[str, Any]:
    """Retain selection audit fields while segmenting final keyframes."""

    value: dict[str, Any] = {"frame_id": frame_id}
    for name in (
        "candidate_id",
        "candidate_index",
        "candidate_reasons",
        "keyframe_strategy",
        "selection_phase",
        "selection_rank",
        "selection_reasons",
        "covered_event_ids",
        "selection_score",
        "protected",
        "coverage_added",
        "importance_score",
        "semantic_novelty",
        "component_scores",
        "available_modalities",
        "protected_event_ids",
        "selection_provenance",
    ):
        if name in record:
            value[name] = record[name]
    return value


def _load_artifacts(
    path: Path | None,
    *,
    filename_prefix: str,
) -> list[dict[str, Any]]:
    if path is None:
        return []
    if path.is_dir():
        candidates = sorted(path.glob(f"{filename_prefix}_*.jsonl"))
        if filename_prefix == "asr":
            candidates = [
                candidate
                for candidate in candidates
                if not candidate.name.startswith("asr_segments_")
            ]
        if not candidates:
            raise FileNotFoundError(
                f"No {filename_prefix}_*.jsonl artifacts found in: {path}"
            )
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            records.extend(iter_metadata_records(candidate))
        return records
    return list(iter_metadata_records(path))


def build_segments(
    keyframe_records: Iterable[dict[str, Any]],
    *,
    strategy: str = "auto",
    fixed_duration_seconds: float = 10.0,
    fallback_fps: float | None = None,
) -> list[Segment]:
    """Group keyframes using existing boundaries, with fixed windows as fallback."""
    if strategy not in {"auto", "boundary", "fixed"}:
        raise ValueError("strategy must be one of: auto, boundary, fixed")
    if not math.isfinite(fixed_duration_seconds) or fixed_duration_seconds <= 0:
        raise ValueError("fixed_duration_seconds must be finite and > 0")
    if fallback_fps is not None and (
        not math.isfinite(fallback_fps) or fallback_fps <= 0
    ):
        raise ValueError("fallback_fps must be finite and > 0")

    grouped: dict[tuple[str, str], Segment] = {}
    seen: dict[tuple[str, str], Keyframe] = {}
    for ordinal, record in enumerate(keyframe_records, start=1):
        video_id = normalize_text(record.get("video_id"))
        frame_id = normalize_text(record.get("frame_id") or record.get("keyframe_id"))
        if not video_id:
            raise ValueError(f"Keyframe record {ordinal} is missing video_id")
        if not frame_id:
            raise ValueError(
                f"Keyframe record {ordinal} is missing frame_id/keyframe_id"
            )
        try:
            timestamp = _record_timestamp(record, fallback_fps)
            frame_index = _frame_index(record.get("frame_index"))
            boundary_start = (
                _finite_non_negative(record["shot_start"], "shot_start")
                if record.get("shot_start") is not None
                else None
            )
            boundary_end = (
                _finite_non_negative(record["shot_end"], "shot_end")
                if record.get("shot_end") is not None
                else None
            )
        except ValueError as exc:
            raise ValueError(f"Invalid keyframe record {ordinal}: {exc}") from exc
        if (
            boundary_start is not None
            and boundary_end is not None
            and boundary_start > boundary_end
        ):
            raise ValueError(
                f"Invalid keyframe record {ordinal}: shot_start is after shot_end"
            )

        explicit_id = normalize_text(record.get("segment_id") or record.get("shot_id"))
        use_boundary = strategy == "boundary" or (strategy == "auto" and explicit_id)
        if use_boundary:
            if not explicit_id:
                raise ValueError(
                    f"Keyframe record {ordinal} has no segment_id/shot_id for "
                    "boundary strategy"
                )
            segment_id = explicit_id
        else:
            bucket = int(math.floor(timestamp / fixed_duration_seconds))
            segment_id = f"SEG_{video_id}_FIXED_{bucket:06d}"
            boundary_start = round(bucket * fixed_duration_seconds, 6)
            boundary_end = round((bucket + 1) * fixed_duration_seconds, 6)

        keyframe = Keyframe(
            video_id=video_id,
            frame_id=frame_id,
            timestamp=timestamp,
            frame_index=frame_index,
            segment_id=segment_id,
            boundary_start=boundary_start,
            boundary_end=boundary_end,
            selection_metadata=_selection_metadata(record, frame_id),
        )
        identity = (video_id, frame_id)
        if identity in seen:
            if seen[identity] != keyframe:
                raise ValueError(f"Conflicting duplicate keyframe: {identity}")
            continue
        seen[identity] = keyframe
        group_key = (video_id, segment_id)
        segment = grouped.setdefault(
            group_key,
            Segment(video_id=video_id, segment_id=segment_id),
        )
        segment.keyframes.append(keyframe)

    if not grouped:
        raise ValueError("No keyframe records were supplied")
    return sorted(
        grouped.values(),
        key=lambda item: (item.video_id, item.time_range()[0], item.segment_id),
    )


def _artifact_matches_segment(
    record: dict[str, Any],
    *,
    segment: Segment,
    frame_ids: set[str],
    start_time: float,
    end_time: float,
) -> bool:
    record_video = record.get("video_id")
    if record_video is not None and str(record_video) != segment.video_id:
        return False
    record_frame = record.get("frame_id") or record.get("keyframe_id")
    if record_frame is not None:
        return str(record_frame) in frame_ids
    record_segment = record.get("segment_id") or record.get("shot_id")
    if record_segment is not None:
        return str(record_segment) == segment.segment_id
    raw_time = record.get("timestamp")
    if raw_time is not None:
        timestamp = _finite_non_negative(raw_time, "artifact timestamp")
        return start_time <= timestamp <= end_time
    return False


def aggregate_captions(
    records: Iterable[dict[str, Any]],
    *,
    near_duplicate_threshold: float = 0.92,
) -> tuple[str, list[str]]:
    """Aggregate captions in time order with exact and near-duplicate removal."""
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    prepared: list[tuple[float, str, str]] = []
    for ordinal, record in enumerate(records):
        if record.get("status") not in {None, "success"}:
            continue
        text = normalize_text(record.get("caption") or record.get("segment_caption"))
        if not text:
            continue
        timestamp = float(record.get("timestamp") or 0.0)
        source_id = _source_id(
            record,
            preferred=("caption_id",),
            fallback_prefix="CAPTION",
            ordinal=ordinal,
        )
        prepared.append((timestamp, source_id, text))
    prepared.sort(key=lambda item: (item[0], item[1], item[2]))
    unique: list[str] = []
    comparison_keys: list[str] = []
    source_ids: list[str] = []
    for _, source_id, text in prepared:
        if source_id not in source_ids:
            source_ids.append(source_id)
        key = comparison_text(text)
        if not key:
            continue
        duplicate = any(
            key == previous
            or SequenceMatcher(None, key, previous).ratio() >= near_duplicate_threshold
            for previous in comparison_keys
        )
        if duplicate:
            continue
        unique.append(text)
        comparison_keys.append(key)
    return " ".join(unique), sorted(source_ids)


def aggregate_ocr(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge repeated OCR text and retain confidence, time span, and sources."""
    merged: dict[str, dict[str, Any]] = {}
    for ordinal, record in enumerate(records):
        if record.get("status") not in {None, "success"}:
            continue
        timestamp_value = record.get("timestamp")
        timestamp = (
            _finite_non_negative(timestamp_value, "OCR timestamp")
            if timestamp_value is not None
            else None
        )
        source_id = _source_id(
            record,
            preferred=("ocr_id",),
            fallback_prefix="OCR",
            ordinal=ordinal,
        )
        regions = record.get("text_regions")
        if not isinstance(regions, list) or not regions:
            text = normalize_text(record.get("ocr_text"))
            regions = [{"text": text}] if text else []
        for region in regions:
            if not isinstance(region, dict):
                continue
            text = normalize_text(region.get("text"))
            key = comparison_text(text)
            if not key:
                continue
            confidence_value = region.get("confidence", record.get("confidence"))
            confidence = (
                float(confidence_value) if confidence_value is not None else None
            )
            current = merged.setdefault(
                key,
                {
                    "text": text,
                    "confidence": confidence,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "source_ids": [],
                },
            )
            if confidence is not None:
                previous = current["confidence"]
                current["confidence"] = confidence if previous is None else max(previous, confidence)
            if timestamp is not None:
                current["first_seen"] = (
                    timestamp
                    if current["first_seen"] is None
                    else min(current["first_seen"], timestamp)
                )
                current["last_seen"] = (
                    timestamp
                    if current["last_seen"] is None
                    else max(current["last_seen"], timestamp)
                )
            if source_id not in current["source_ids"]:
                current["source_ids"].append(source_id)
    output = []
    for value in merged.values():
        if value["confidence"] is None:
            value.pop("confidence")
        else:
            value["confidence"] = round(float(value["confidence"]), 6)
        if value["first_seen"] is None:
            value.pop("first_seen")
            value.pop("last_seen")
        value["source_ids"].sort()
        output.append(value)
    return sorted(
        output,
        key=lambda item: (
            item.get("first_seen", float("inf")),
            comparison_text(item["text"]),
        ),
    )


def aggregate_asr(
    records: Iterable[dict[str, Any]],
    *,
    segment_start: float,
    segment_end: float,
) -> list[dict[str, Any]]:
    """Select overlapping ASR chunks and merge duplicate overlap chunks."""
    prepared: list[tuple[float, float, str, str, dict[str, Any]]] = []
    for ordinal, record in enumerate(records):
        if record.get("status") not in {None, "success"}:
            continue
        text = normalize_text(record.get("text") or record.get("transcript_text"))
        if not text:
            continue
        raw_start = record.get("start", record.get("start_time"))
        raw_end = record.get("end", record.get("end_time"))
        if raw_start is None or raw_end is None:
            continue
        start = _finite_non_negative(raw_start, "ASR start")
        end = _finite_non_negative(raw_end, "ASR end")
        if start > end:
            raise ValueError(f"ASR start is after end: {start} > {end}")
        if end <= segment_start or start >= segment_end:
            continue
        source_id = _source_id(
            record,
            preferred=("transcript_segment_id", "asr_id"),
            fallback_prefix="ASR",
            ordinal=ordinal,
        )
        prepared.append((start, end, source_id, text, record))
    prepared.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    output: list[dict[str, Any]] = []
    for start, end, source_id, text, record in prepared:
        key = comparison_text(text)
        duplicate = next(
            (
                value
                for value in output
                if value["_key"] == key
                and end > value["start_time"]
                and start < value["end_time"]
            ),
            None,
        )
        interval = {"start_time": start, "end_time": end, "source_id": source_id}
        if duplicate is not None:
            duplicate["start_time"] = min(duplicate["start_time"], start)
            duplicate["end_time"] = max(duplicate["end_time"], end)
            if interval not in duplicate["source_intervals"]:
                duplicate["source_intervals"].append(interval)
            if source_id not in duplicate["source_ids"]:
                duplicate["source_ids"].append(source_id)
            continue
        value: dict[str, Any] = {
            "_key": key,
            "text": text,
            "start_time": start,
            "end_time": end,
            "source_ids": [source_id],
            "source_intervals": [interval],
        }
        if record.get("language") is not None:
            value["language"] = record["language"]
        output.append(value)
    for value in output:
        value.pop("_key")
        value["source_ids"].sort()
        value["source_intervals"].sort(
            key=lambda item: (
                item["start_time"],
                item["end_time"],
                item["source_id"],
            )
        )
    return output


def aggregate_objects(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate normalized labels without claiming detections are unique objects."""
    grouped: dict[str, dict[str, Any]] = {}
    for ordinal, record in enumerate(records):
        if record.get("status") not in {None, "success"}:
            continue
        source_id = _source_id(
            record,
            preferred=("object_record_id",),
            fallback_prefix="OBJECT",
            ordinal=ordinal,
        )
        detections = record.get("objects")
        if not isinstance(detections, list):
            continue
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            label = normalize_text(
                detection.get("label")
                or detection.get("class_name")
                or detection.get("class")
            ).casefold()
            if not label:
                continue
            confidence_value = detection.get("confidence")
            confidence = (
                float(confidence_value) if confidence_value is not None else None
            )
            track_id = detection.get("track_id")
            current = grouped.setdefault(
                label,
                {
                    "label": label,
                    "max_confidence": confidence,
                    "_occurrences": set(),
                    "_has_track": False,
                    "_has_detection": False,
                    "source_ids": set(),
                },
            )
            if confidence is not None:
                previous = current["max_confidence"]
                current["max_confidence"] = (
                    confidence if previous is None else max(previous, confidence)
                )
            if track_id is not None and str(track_id) != "":
                occurrence = ("track", str(track_id))
                current["_has_track"] = True
            else:
                occurrence = ("detection", source_id, detection_index)
                current["_has_detection"] = True
            current["_occurrences"].add(occurrence)
            current["source_ids"].add(source_id)
    output: list[dict[str, Any]] = []
    for value in grouped.values():
        has_track = value.pop("_has_track")
        has_detection = value.pop("_has_detection")
        occurrences = value.pop("_occurrences")
        value["occurrence_count"] = len(occurrences)
        if has_track and not has_detection:
            semantics = "unique_track"
        elif has_detection and not has_track:
            semantics = "detection_occurrence"
        else:
            semantics = "unique_track_or_detection_occurrence"
        value["occurrence_count_semantics"] = semantics
        value["source_ids"] = sorted(value["source_ids"])
        if value["max_confidence"] is None:
            value.pop("max_confidence")
        else:
            value["max_confidence"] = round(float(value["max_confidence"]), 6)
        output.append(value)
    return sorted(output, key=lambda item: item["label"])


@dataclass
class ArtifactIndex:
    """Lookup index that avoids scanning every frame artifact for every segment."""

    records: Sequence[dict[str, Any]]
    by_frame: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    by_segment: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    unlinked: list[int] = field(default_factory=list)

    @classmethod
    def build(cls, records: Sequence[dict[str, Any]]) -> "ArtifactIndex":
        index = cls(records=records)
        for ordinal, record in enumerate(records):
            frame_id = record.get("frame_id") or record.get("keyframe_id")
            segment_id = record.get("segment_id") or record.get("shot_id")
            linked = False
            if frame_id is not None:
                index.by_frame[str(frame_id)].append(ordinal)
                linked = True
            if segment_id is not None:
                index.by_segment[str(segment_id)].append(ordinal)
                linked = True
            if not linked:
                index.unlinked.append(ordinal)
        return index

    def matching(
        self,
        *,
        segment: Segment,
        frame_ids: set[str],
        start_time: float,
        end_time: float,
    ) -> list[dict[str, Any]]:
        candidate_indices: set[int] = set(self.by_segment.get(segment.segment_id, []))
        for frame_id in frame_ids:
            candidate_indices.update(self.by_frame.get(frame_id, []))
        candidate_indices.update(self.unlinked)
        return [
            self.records[index]
            for index in sorted(candidate_indices)
            if _artifact_matches_segment(
                self.records[index],
                segment=segment,
                frame_ids=frame_ids,
                start_time=start_time,
                end_time=end_time,
            )
        ]


def build_segment_records(
    segments: Sequence[Segment],
    *,
    captions: Sequence[dict[str, Any]] = (),
    ocr: Sequence[dict[str, Any]] = (),
    asr: Sequence[dict[str, Any]] = (),
    objects: Sequence[dict[str, Any]] = (),
    caption_similarity_threshold: float = 0.92,
) -> list[dict[str, Any]]:
    """Build deterministic search-oriented segment records with provenance."""
    output: list[dict[str, Any]] = []
    caption_index = ArtifactIndex.build(captions)
    ocr_index = ArtifactIndex.build(ocr)
    object_index = ArtifactIndex.build(objects)
    asr_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in asr:
        video_id = str(record.get("video_id") or "")
        asr_by_video[video_id].append(record)

    for segment in segments:
        ordered = segment.ordered_keyframes()
        frame_ids = {item.frame_id for item in ordered}
        start_time, end_time = segment.time_range()
        caption_records = caption_index.matching(
            segment=segment,
            frame_ids=frame_ids,
            start_time=start_time,
            end_time=end_time,
        )
        ocr_records = ocr_index.matching(
            segment=segment,
            frame_ids=frame_ids,
            start_time=start_time,
            end_time=end_time,
        )
        object_records = object_index.matching(
            segment=segment,
            frame_ids=frame_ids,
            start_time=start_time,
            end_time=end_time,
        )
        caption_text, caption_source_ids = aggregate_captions(
            caption_records,
            near_duplicate_threshold=caption_similarity_threshold,
        )
        first, last = ordered[0], ordered[-1]
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "segment_id": segment.segment_id,
            "video_id": segment.video_id,
            "start_time": start_time,
            "end_time": end_time,
            "start_keyframe": first.frame_id,
            "end_keyframe": last.frame_id,
            "keyframe_ids": [item.frame_id for item in ordered],
            "keyframe_selection": [item.selection_metadata for item in ordered],
            "covered_event_ids": sorted(
                {
                    str(event_id)
                    for item in ordered
                    for event_id in item.selection_metadata.get(
                        "covered_event_ids",
                        [],
                    )
                }
            ),
            "protected": any(
                bool(item.selection_metadata.get("protected")) for item in ordered
            ),
            "captions_aggregated": caption_text,
            "caption_source_ids": caption_source_ids,
            "ocr": aggregate_ocr(ocr_records),
            "asr": aggregate_asr(
                asr_by_video.get(segment.video_id, []),
                segment_start=start_time,
                segment_end=end_time,
            ),
            "objects": aggregate_objects(object_records),
        }
        if first.frame_index is not None:
            value["start_frame"] = first.frame_index
        if last.frame_index is not None:
            value["end_frame"] = last.frame_index
        output.append(value)
    return output


def build_segment_metadata(
    input_path: Path,
    output_path: Path,
    *,
    captions_path: Path | None = None,
    ocr_path: Path | None = None,
    asr_path: Path | None = None,
    objects_path: Path | None = None,
    strategy: str = "auto",
    fixed_duration_seconds: float = 10.0,
    fps: float | None = None,
    caption_similarity_threshold: float = 0.92,
) -> dict[str, Any]:
    """Build segment-level metadata without mutating frame-level artifacts."""
    protected_inputs = [
        path
        for path in (
            input_path,
            captions_path,
            ocr_path,
            asr_path,
            objects_path,
        )
        if path is not None and path.is_file()
    ]
    if any(path.resolve() == output_path.resolve() for path in protected_inputs):
        raise ValueError("output must not overwrite a frame-level source artifact")
    segments = build_segments(
        iter_keyframe_records(input_path),
        strategy=strategy,
        fixed_duration_seconds=fixed_duration_seconds,
        fallback_fps=fps,
    )
    records = build_segment_records(
        segments,
        captions=_load_artifacts(captions_path, filename_prefix="captions"),
        ocr=_load_artifacts(ocr_path, filename_prefix="ocr"),
        asr=_load_artifacts(asr_path, filename_prefix="asr"),
        objects=_load_artifacts(objects_path, filename_prefix="objects"),
        caption_similarity_threshold=caption_similarity_threshold,
    )
    record_count = write_records(output_path, records)
    return {
        "schema_version": SCHEMA_VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "strategy": strategy,
        "fixed_duration_seconds": fixed_duration_seconds,
        "record_count": record_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build boundary-aware segment-level multimodal metadata."
    )
    parser.add_argument("--input", type=Path, required=True, help="Keyframe JSON/JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Segment .jsonl or .json")
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--ocr", type=Path)
    parser.add_argument("--asr", type=Path)
    parser.add_argument("--objects", type=Path)
    parser.add_argument(
        "--strategy",
        choices=("auto", "boundary", "fixed"),
        default="auto",
    )
    parser.add_argument("--fixed-duration-seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--caption-similarity-threshold", type=float, default=0.92)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = build_segment_metadata(
            args.input,
            args.output,
            captions_path=args.captions,
            ocr_path=args.ocr,
            asr_path=args.asr,
            objects_path=args.objects,
            strategy=args.strategy,
            fixed_duration_seconds=args.fixed_duration_seconds,
            fps=args.fps,
            caption_similarity_threshold=args.caption_similarity_threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
