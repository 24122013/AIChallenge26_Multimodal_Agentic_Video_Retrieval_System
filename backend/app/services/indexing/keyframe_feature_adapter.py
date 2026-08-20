"""Deterministic multimodal feature adapter for keyframe selection.

The adapter consumes already-produced in-memory metadata.  It performs no
model inference and no filesystem I/O.  Frame-level artifacts are joined by
``(video_id, candidate_id)``; legacy artifacts may use ``frame_id`` instead.
Raw OCR regions and object detections are intentionally preferred over
segment aggregates because the latter discard geometry needed for conservative
hard-event qualification.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from numbers import Integral, Real
from typing import Any

import numpy as np

from .keyframe_candidates import KeyframeCandidate
from .keyframe_selection import ProtectedEvent, SelectionCandidate


COMPONENT_ORDER = ("ocr", "objects", "transition", "caption")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class FeatureAdapterConfig:
    """Conservative, configurable thresholds for metadata-only adaptation."""

    ocr_min_confidence: float = 0.75
    ocr_extreme_confidence: float = 0.95
    ocr_min_alnum_chars: int = 4
    ocr_min_area_ratio: float = 0.002
    ocr_persistence_candidates: int = 2
    ocr_episode_gap_seconds: float = 1.25
    ocr_reappearance_cooldown_seconds: float = 5.0
    ocr_similarity_threshold: float = 0.82
    ocr_spatial_tolerance: float = 0.08
    ocr_common_frame_fraction: float = 0.20
    ocr_common_shot_fraction: float = 0.20
    ocr_common_min_frames: int = 5
    ocr_rare_frame_fraction: float = 0.08
    ocr_subtitle_band_start: float = 0.78
    ocr_subtitle_max_area_ratio: float = 0.03

    object_min_confidence: float = 0.65
    object_extreme_confidence: float = 0.90
    object_min_area_ratio: float = 0.01
    object_extreme_min_area_ratio: float = 0.02
    object_persistence_candidates: int = 2
    object_episode_gap_seconds: float = 1.25
    object_reappearance_cooldown_seconds: float = 5.0
    object_rare_frame_fraction: float = 0.05

    transition_absolute_floor: float = 0.18
    transition_mad_multiplier: float = 2.5
    transition_max_pair_gap_seconds: float = 0.75
    transition_nms_seconds: float = 1.0

    ocr_weight: float = 0.25
    object_weight: float = 0.25
    transition_weight: float = 0.35
    caption_weight: float = 0.05

    ocr_event_priority: int = 2
    object_event_priority: int = 1
    transition_event_priority: int = 3

    def __post_init__(self) -> None:
        probability_fields = (
            "ocr_min_confidence",
            "ocr_extreme_confidence",
            "ocr_min_area_ratio",
            "ocr_similarity_threshold",
            "ocr_spatial_tolerance",
            "ocr_common_frame_fraction",
            "ocr_common_shot_fraction",
            "ocr_rare_frame_fraction",
            "ocr_subtitle_band_start",
            "ocr_subtitle_max_area_ratio",
            "object_min_confidence",
            "object_extreme_confidence",
            "object_min_area_ratio",
            "object_extreme_min_area_ratio",
            "object_rare_frame_fraction",
            "transition_absolute_floor",
        )
        for name in probability_fields:
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))

        positive_time_fields = (
            "ocr_episode_gap_seconds",
            "ocr_reappearance_cooldown_seconds",
            "object_episode_gap_seconds",
            "object_reappearance_cooldown_seconds",
            "transition_max_pair_gap_seconds",
        )
        for name in positive_time_fields:
            value = _finite_real(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
            object.__setattr__(self, name, value)
        nms = _finite_real(self.transition_nms_seconds, "transition_nms_seconds")
        if nms < 0:
            raise ValueError("transition_nms_seconds must be >= 0")
        object.__setattr__(self, "transition_nms_seconds", nms)
        multiplier = _finite_real(
            self.transition_mad_multiplier,
            "transition_mad_multiplier",
        )
        if multiplier < 0:
            raise ValueError("transition_mad_multiplier must be >= 0")
        object.__setattr__(self, "transition_mad_multiplier", multiplier)

        count_fields = (
            "ocr_min_alnum_chars",
            "ocr_persistence_candidates",
            "ocr_common_min_frames",
            "object_persistence_candidates",
            "ocr_event_priority",
            "object_event_priority",
            "transition_event_priority",
        )
        for name in count_fields:
            value = _non_negative_int(getattr(self, name), name)
            if name in {
                "ocr_min_alnum_chars",
                "ocr_persistence_candidates",
                "ocr_common_min_frames",
                "object_persistence_candidates",
            } and value < 1:
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, value)

        weight_fields = (
            "ocr_weight",
            "object_weight",
            "transition_weight",
            "caption_weight",
        )
        weights = []
        for name in weight_fields:
            value = _finite_real(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)
            weights.append(value)
        if not math.isfinite(sum(weights)) or sum(weights) <= 0:
            raise ValueError("at least one finite positive modality weight is required")


@dataclass(frozen=True)
class CandidateComponentScore:
    candidate_id: str
    importance_score: float
    component_scores: tuple[tuple[str, float], ...]
    available_modalities: tuple[str, ...]
    protected_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "importance_score": self.importance_score,
            "component_scores": dict(self.component_scores),
            "available_modalities": list(self.available_modalities),
            "protected_event_ids": list(self.protected_event_ids),
        }


@dataclass(frozen=True)
class FeatureAdapterReport:
    video_id: str
    candidate_count: int
    modality_available_counts: tuple[tuple[str, int], ...]
    ocr_event_count: int
    object_event_count: int
    transition_boundary_count: int
    suppressed_ocr_common_tracks: int
    suppressed_ocr_subtitle_observations: int
    suppressed_ocr_weak_episodes: int
    suppressed_object_weak_episodes: int
    transition_threshold: float | None
    missing_embedding_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "candidate_count": self.candidate_count,
            "modality_available_counts": dict(self.modality_available_counts),
            "ocr_event_count": self.ocr_event_count,
            "object_event_count": self.object_event_count,
            "transition_boundary_count": self.transition_boundary_count,
            "suppressed_ocr_common_tracks": self.suppressed_ocr_common_tracks,
            "suppressed_ocr_subtitle_observations": (
                self.suppressed_ocr_subtitle_observations
            ),
            "suppressed_ocr_weak_episodes": self.suppressed_ocr_weak_episodes,
            "suppressed_object_weak_episodes": self.suppressed_object_weak_episodes,
            "transition_threshold": self.transition_threshold,
            "missing_embedding_count": self.missing_embedding_count,
        }


@dataclass(frozen=True)
class FeatureAdapterResult:
    selection_candidates: tuple[SelectionCandidate, ...]
    protected_events: tuple[ProtectedEvent, ...]
    candidate_scores: tuple[CandidateComponentScore, ...]
    report: FeatureAdapterReport


@dataclass(frozen=True)
class _BaseCandidate:
    candidate_id: str
    video_id: str
    timestamp: float
    frame_index: int
    shot_index: int
    source_reasons: tuple[str, ...]
    shot_start_sec: float | None
    shot_end_sec: float | None
    duplicate_group: str | None
    legacy_frame_id: str | None


@dataclass(frozen=True)
class _OcrObservation:
    candidate_id: str
    timestamp: float
    shot_index: int
    text: str
    comparison: str
    confidence: float
    area_ratio: float | None
    center: tuple[float, float] | None
    quality: float
    subtitle_like: bool


@dataclass(frozen=True)
class _ObjectObservation:
    candidate_id: str
    timestamp: float
    class_key: str
    class_name: str
    confidence: float
    area_ratio: float | None
    quality: float


def adapt_feature_records(
    candidates: Iterable[KeyframeCandidate | Mapping[str, Any]],
    *,
    ocr_records: Iterable[Mapping[str, Any]] = (),
    object_records: Iterable[Mapping[str, Any]] = (),
    caption_records: Iterable[Mapping[str, Any]] = (),
    embeddings: np.ndarray | None = None,
    embedding_records: Iterable[Mapping[str, Any]] = (),
    config: FeatureAdapterConfig | None = None,
) -> FeatureAdapterResult:
    """Adapt existing feature artifacts into the pure Phase 1 selector API."""

    config = config or FeatureAdapterConfig()
    base = _normalize_candidates(candidates)
    video_id = base[0].video_id if base else ""
    alias_to_id = _candidate_aliases(base)

    ocr_index = _index_frame_records(
        ocr_records,
        modality="ocr",
        video_id=video_id,
        alias_to_id=alias_to_id,
    )
    object_index = _index_frame_records(
        object_records,
        modality="objects",
        video_id=video_id,
        alias_to_id=alias_to_id,
    )
    caption_index = _index_frame_records(
        caption_records,
        modality="caption",
        video_id=video_id,
        alias_to_id=alias_to_id,
    )
    embedding_by_id = _map_embeddings(
        embeddings,
        tuple(embedding_records),
        video_id=video_id,
        alias_to_id=alias_to_id,
    )

    component_scores = {
        candidate.candidate_id: {name: 0.0 for name in COMPONENT_ORDER}
        for candidate in base
    }
    availability: dict[str, set[str]] = {
        candidate.candidate_id: set() for candidate in base
    }

    ocr_events, ocr_stats = _adapt_ocr(
        base,
        ocr_index,
        object_index,
        component_scores,
        availability,
        config,
    )
    object_events, object_stats = _adapt_objects(
        base,
        object_index,
        component_scores,
        availability,
        config,
    )
    _adapt_captions(base, caption_index, component_scores, availability)
    transition_events, transition_stats = _adapt_transitions(
        base,
        embedding_by_id,
        component_scores,
        availability,
        config,
    )

    events = tuple(
        sorted(
            (*ocr_events, *object_events, *transition_events),
            key=lambda event: (-event.priority, event.event_type, event.event_id),
        )
    )
    protected_by_candidate: dict[str, list[str]] = defaultdict(list)
    for event in events:
        for candidate_id in event.candidate_ids:
            protected_by_candidate[candidate_id].append(event.event_id)

    weights = {
        "ocr": config.ocr_weight,
        "objects": config.object_weight,
        "transition": config.transition_weight,
        "caption": config.caption_weight,
    }
    selection_candidates: list[SelectionCandidate] = []
    score_records: list[CandidateComponentScore] = []
    for candidate in base:
        candidate_id = candidate.candidate_id
        available = tuple(
            name for name in COMPONENT_ORDER if name in availability[candidate_id]
        )
        denominator = sum(weights[name] for name in available)
        importance = (
            sum(weights[name] * component_scores[candidate_id][name] for name in available)
            / denominator
            if denominator > 0
            else 0.0
        )
        importance = _clamp01(importance)
        embedding = embedding_by_id.get(candidate_id)
        selection_candidates.append(
            SelectionCandidate(
                candidate_id=candidate_id,
                timestamp=candidate.timestamp,
                frame_index=candidate.frame_index,
                shot_index=candidate.shot_index,
                importance_score=importance,
                semantic_embedding=tuple(float(value) for value in embedding)
                if embedding is not None
                else (),
                duplicate_group=candidate.duplicate_group,
                source_reasons=candidate.source_reasons,
                shot_start_sec=candidate.shot_start_sec,
                shot_end_sec=candidate.shot_end_sec,
            )
        )
        score_records.append(
            CandidateComponentScore(
                candidate_id=candidate_id,
                importance_score=importance,
                component_scores=tuple(
                    (name, _clamp01(component_scores[candidate_id][name]))
                    for name in COMPONENT_ORDER
                ),
                available_modalities=available,
                protected_event_ids=tuple(sorted(protected_by_candidate[candidate_id])),
            )
        )

    available_counts = tuple(
        (
            name,
            sum(name in availability[candidate.candidate_id] for candidate in base),
        )
        for name in COMPONENT_ORDER
    )
    report = FeatureAdapterReport(
        video_id=video_id,
        candidate_count=len(base),
        modality_available_counts=available_counts,
        ocr_event_count=len(ocr_events),
        object_event_count=len(object_events),
        transition_boundary_count=transition_stats["boundary_count"],
        suppressed_ocr_common_tracks=ocr_stats["common_tracks"],
        suppressed_ocr_subtitle_observations=ocr_stats["subtitle_observations"],
        suppressed_ocr_weak_episodes=ocr_stats["weak_episodes"],
        suppressed_object_weak_episodes=object_stats["weak_episodes"],
        transition_threshold=transition_stats["threshold"],
        missing_embedding_count=len(base) - len(embedding_by_id),
    )
    return FeatureAdapterResult(
        selection_candidates=tuple(selection_candidates),
        protected_events=events,
        candidate_scores=tuple(score_records),
        report=report,
    )


def _normalize_candidates(
    candidates: Iterable[KeyframeCandidate | Mapping[str, Any]],
) -> tuple[_BaseCandidate, ...]:
    values: list[_BaseCandidate] = []
    for position, value in enumerate(candidates):
        if isinstance(value, KeyframeCandidate):
            base = _BaseCandidate(
                candidate_id=value.candidate_id,
                video_id=value.video_id,
                timestamp=value.timestamp_sec,
                frame_index=value.frame_index,
                shot_index=value.shot_index,
                source_reasons=value.reasons,
                shot_start_sec=value.shot_start_sec,
                shot_end_sec=value.shot_end_sec,
                duplicate_group=None,
                legacy_frame_id=None,
            )
        elif isinstance(value, Mapping):
            base = _base_from_mapping(value, position)
        else:
            raise TypeError("candidates must contain KeyframeCandidate or mapping values")
        # Reuse the selector's strict numeric and bounds validation.
        SelectionCandidate(
            candidate_id=base.candidate_id,
            timestamp=base.timestamp,
            frame_index=base.frame_index,
            shot_index=base.shot_index,
            source_reasons=base.source_reasons,
            shot_start_sec=base.shot_start_sec,
            shot_end_sec=base.shot_end_sec,
            duplicate_group=base.duplicate_group,
        )
        if not base.video_id.strip():
            raise ValueError(f"candidates[{position}].video_id must not be empty")
        values.append(base)

    candidate_ids = [value.candidate_id for value in values]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id values must be unique")
    video_ids = {value.video_id for value in values}
    if len(video_ids) > 1:
        raise ValueError(f"one adapter call must contain one video_id, got {sorted(video_ids)}")
    return tuple(
        sorted(
            values,
            key=lambda item: (item.timestamp, item.frame_index, item.candidate_id),
        )
    )


def _base_from_mapping(record: Mapping[str, Any], position: int) -> _BaseCandidate:
    raw_candidate_id = record.get("candidate_id") or record.get("frame_id")
    if not isinstance(raw_candidate_id, str) or not raw_candidate_id.strip():
        raise ValueError(f"candidates[{position}] requires candidate_id or frame_id")
    video_id = record.get("video_id")
    if not isinstance(video_id, str):
        raise TypeError(f"candidates[{position}].video_id must be a string")
    raw_timestamp = (
        record.get("timestamp_sec")
        if record.get("timestamp_sec") is not None
        else record.get("timestamp")
    )
    timestamp = _finite_real(raw_timestamp, f"candidates[{position}].timestamp")
    frame_index = _non_negative_int(
        record.get("frame_index"),
        f"candidates[{position}].frame_index",
    )
    shot_index = _non_negative_int(
        record.get("shot_index"),
        f"candidates[{position}].shot_index",
    )
    raw_reasons = (
        record.get("candidate_reasons")
        or record.get("source_reasons")
        or record.get("reasons")
        or record.get("selection_reason")
        or ()
    )
    if isinstance(raw_reasons, str):
        reasons = (raw_reasons,)
    elif isinstance(raw_reasons, Sequence):
        reasons = tuple(raw_reasons)
    else:
        raise TypeError(f"candidates[{position}].candidate_reasons must be a sequence")
    if any(not isinstance(reason, str) for reason in reasons):
        raise TypeError(f"candidates[{position}].candidate_reasons must contain strings")
    raw_start = (
        record.get("shot_start_sec")
        if record.get("shot_start_sec") is not None
        else record.get("shot_start")
    )
    raw_end = (
        record.get("shot_end_sec")
        if record.get("shot_end_sec") is not None
        else record.get("shot_end")
    )
    if (raw_start is None) != (raw_end is None):
        raise ValueError("shot_start and shot_end must be supplied together")
    shot_start = _finite_real(raw_start, "shot_start") if raw_start is not None else None
    shot_end = _finite_real(raw_end, "shot_end") if raw_end is not None else None
    duplicate_group = record.get("duplicate_group")
    if duplicate_group is None and record.get("phash"):
        duplicate_group = f"phash:{record['phash']}"
    if duplicate_group is not None and not isinstance(duplicate_group, str):
        raise TypeError("duplicate_group must be a string or None")
    legacy_frame_id = record.get("frame_id")
    if legacy_frame_id is not None and not isinstance(legacy_frame_id, str):
        raise TypeError("frame_id must be a string when supplied")
    return _BaseCandidate(
        candidate_id=raw_candidate_id,
        video_id=video_id,
        timestamp=timestamp,
        frame_index=frame_index,
        shot_index=shot_index,
        source_reasons=tuple(dict.fromkeys(reasons)),
        shot_start_sec=shot_start,
        shot_end_sec=shot_end,
        duplicate_group=duplicate_group,
        legacy_frame_id=legacy_frame_id,
    )


def _candidate_aliases(base: Sequence[_BaseCandidate]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for candidate in base:
        for alias in (candidate.candidate_id, candidate.legacy_frame_id):
            if not alias:
                continue
            existing = aliases.get(alias)
            if existing is not None and existing != candidate.candidate_id:
                raise ValueError(f"candidate alias is ambiguous: {alias}")
            aliases[alias] = candidate.candidate_id
    return aliases


def _index_frame_records(
    records: Iterable[Mapping[str, Any]],
    *,
    modality: str,
    video_id: str,
    alias_to_id: Mapping[str, str],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"{modality}_records[{position}] must be a mapping")
        _validate_record_video(record, video_id, f"{modality}_records[{position}]")
        candidate_id = _resolve_record_candidate_id(
            record,
            alias_to_id,
            f"{modality}_records[{position}]",
        )
        if candidate_id in indexed:
            raise ValueError(f"duplicate {modality} record for candidate: {candidate_id}")
        _record_status(record, f"{modality}_records[{position}]")
        indexed[candidate_id] = record
    return indexed


def _validate_record_video(
    record: Mapping[str, Any],
    expected: str,
    name: str,
) -> None:
    raw = record.get("video_id")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{name}.video_id must be a non-empty string")
    if not expected:
        raise ValueError(f"{name} was supplied without any base candidates")
    if raw != expected:
        raise ValueError(f"{name}.video_id {raw!r} does not match {expected!r}")


def _record_status(record: Mapping[str, Any], name: str) -> str:
    raw = record.get("status", "success")
    if raw not in {"success", "error", "skipped"}:
        raise ValueError(f"{name}.status must be success, error, or skipped")
    return str(raw)


def _resolve_record_candidate_id(
    record: Mapping[str, Any],
    alias_to_id: Mapping[str, str],
    name: str,
) -> str:
    raw_candidate_id = record.get("candidate_id")
    raw_frame_id = record.get("frame_id")
    if raw_candidate_id is None and raw_frame_id is None:
        raise ValueError(f"{name} requires candidate_id or frame_id")
    resolved_candidate: str | None = None
    resolved_frame: str | None = None
    if raw_candidate_id is not None:
        if not isinstance(raw_candidate_id, str) or not raw_candidate_id.strip():
            raise ValueError(f"{name}.candidate_id must be a non-empty string")
        canonical_ids = set(alias_to_id.values())
        if raw_candidate_id not in canonical_ids:
            raise ValueError(f"{name} references unknown candidate: {raw_candidate_id}")
        resolved_candidate = raw_candidate_id
    if raw_frame_id is not None:
        if not isinstance(raw_frame_id, str) or not raw_frame_id.strip():
            raise ValueError(f"{name}.frame_id must be a non-empty string")
        resolved_frame = alias_to_id.get(raw_frame_id)
        if resolved_frame is None:
            raise ValueError(f"{name} references unknown candidate: {raw_frame_id}")
    if (
        resolved_candidate is not None
        and resolved_frame is not None
        and resolved_candidate != resolved_frame
    ):
        raise ValueError(f"{name} candidate_id and frame_id refer to different candidates")
    result = resolved_candidate or resolved_frame
    if result is None:
        raise ValueError(f"{name} requires a known candidate_id or frame_id")
    return result


def _map_embeddings(
    embeddings: np.ndarray | None,
    records: Sequence[Mapping[str, Any]],
    *,
    video_id: str,
    alias_to_id: Mapping[str, str],
) -> dict[str, np.ndarray]:
    if embeddings is None:
        if records:
            raise ValueError("embedding_records require an embeddings matrix")
        return {}
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a numpy.ndarray")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if embeddings.dtype != np.float32:
        raise ValueError("embeddings must use float32 dtype")
    if len(records) != embeddings.shape[0]:
        raise ValueError("embedding metadata count must equal embeddings rows")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    mapped: dict[str, np.ndarray] = {}
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"embedding_records[{row_index}] must be a mapping")
        _validate_record_video(record, video_id, f"embedding_records[{row_index}]")
        embedding_index = record.get("embedding_index")
        if isinstance(embedding_index, bool) or not isinstance(embedding_index, Integral):
            raise TypeError("embedding_index must be an integer")
        if int(embedding_index) != row_index:
            raise ValueError(
                f"embedding_index mismatch at row {row_index}: {embedding_index}"
            )
        candidate_id = _resolve_record_candidate_id(
            record,
            alias_to_id,
            f"embedding_records[{row_index}]",
        )
        vector_dim = record.get("vector_dim")
        if vector_dim is not None and (
            isinstance(vector_dim, bool)
            or not isinstance(vector_dim, Integral)
            or int(vector_dim) != embeddings.shape[1]
        ):
            raise ValueError(
                f"embedding vector_dim mismatch at row {row_index}: {vector_dim}"
            )
        if candidate_id in mapped:
            raise ValueError(f"duplicate embedding for candidate: {candidate_id}")
        vector = embeddings[row_index].astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"embedding for {candidate_id} has invalid norm")
        if record.get("normalized") is True and not math.isclose(
            norm,
            1.0,
            rel_tol=1e-3,
            abs_tol=1e-3,
        ):
            raise ValueError(f"embedding for {candidate_id} is marked normalized but is not")
        mapped[candidate_id] = vector / norm
    return mapped


def _adapt_ocr(
    base: Sequence[_BaseCandidate],
    records: Mapping[str, Mapping[str, Any]],
    object_records: Mapping[str, Mapping[str, Any]],
    scores: dict[str, dict[str, float]],
    availability: dict[str, set[str]],
    config: FeatureAdapterConfig,
) -> tuple[tuple[ProtectedEvent, ...], dict[str, int]]:
    observations: list[_OcrObservation] = []
    by_id = {candidate.candidate_id: candidate for candidate in base}
    for candidate in base:
        record = records.get(candidate.candidate_id)
        if record is None or _record_status(record, "ocr record") != "success":
            continue
        availability[candidate.candidate_id].add("ocr")
        regions = record.get("text_regions")
        if regions is None:
            text = record.get("ocr_text")
            regions = [{"text": text, "confidence": record.get("confidence", 0.0)}] if text else []
        if not isinstance(regions, list):
            raise ValueError("successful OCR record text_regions must be a list")
        image_size = _record_image_size(record)
        if image_size is None:
            object_record = object_records.get(candidate.candidate_id)
            if (
                object_record is not None
                and _record_status(object_record, "object record") == "success"
            ):
                image_size = _record_image_size(object_record)
        for region_index, region in enumerate(regions):
            if not isinstance(region, Mapping):
                raise TypeError(f"OCR region {region_index} must be a mapping")
            text = _display_text(region.get("text"))
            comparison = _comparison_text(text)
            if not comparison:
                continue
            confidence = _unit_interval(
                region.get("confidence", record.get("confidence", 0.0)),
                "OCR confidence",
            )
            area_ratio, center = _polygon_geometry(region.get("polygon"), image_size)
            length_score = min(1.0, _alnum_count(comparison) / 12.0)
            area_score = min(1.0, area_ratio / 0.03) if area_ratio is not None else 0.0
            quality = _clamp01(0.55 * confidence + 0.25 * length_score + 0.20 * area_score)
            subtitle_like = bool(
                center is not None
                and area_ratio is not None
                and center[1] >= config.ocr_subtitle_band_start
                and area_ratio <= config.ocr_subtitle_max_area_ratio
            )
            observations.append(
                _OcrObservation(
                    candidate_id=candidate.candidate_id,
                    timestamp=candidate.timestamp,
                    shot_index=candidate.shot_index,
                    text=text,
                    comparison=comparison,
                    confidence=confidence,
                    area_ratio=area_ratio,
                    center=center,
                    quality=quality,
                    subtitle_like=subtitle_like,
                )
            )

    cluster_for_text = _fuzzy_text_clusters(
        {observation.comparison for observation in observations},
        config.ocr_similarity_threshold,
    )
    clustered: dict[str, list[_OcrObservation]] = defaultdict(list)
    for observation in observations:
        clustered[cluster_for_text[observation.comparison]].append(observation)

    success_frame_count = max(
        1,
        sum("ocr" in availability[candidate.candidate_id] for candidate in base),
    )
    total_shot_count = max(
        1,
        len(
            {
                candidate.shot_index
                for candidate in base
                if "ocr" in availability[candidate.candidate_id]
            }
        ),
    )
    events: list[ProtectedEvent] = []
    common_tracks = subtitle_count = weak_episodes = 0
    for cluster_key in sorted(clustered):
        values = clustered[cluster_key]
        candidate_ids = {value.candidate_id for value in values}
        shot_ids = {value.shot_index for value in values}
        frame_fraction = len(candidate_ids) / success_frame_count
        shot_fraction = len(shot_ids) / total_shot_count
        centers = [value.center for value in values if value.center is not None]
        spatially_stable = not centers or (
            max(value[0] for value in centers) - min(value[0] for value in centers)
            <= 2 * config.ocr_spatial_tolerance
            and max(value[1] for value in centers) - min(value[1] for value in centers)
            <= 2 * config.ocr_spatial_tolerance
        )
        common = bool(
            len(candidate_ids) >= config.ocr_common_min_frames
            and frame_fraction >= config.ocr_common_frame_fraction
            and shot_fraction >= config.ocr_common_shot_fraction
            and spatially_stable
        )
        if common:
            common_tracks += 1
        rarity = _clamp01(1.0 - frame_fraction)
        for value in values:
            multiplier = 0.10 if common else (0.35 if value.subtitle_like else 0.5 + 0.5 * rarity)
            scores[value.candidate_id]["ocr"] = max(
                scores[value.candidate_id]["ocr"],
                _clamp01(value.quality * multiplier),
            )
            if value.subtitle_like:
                subtitle_count += 1
        if common:
            continue

        eligible = [
            value
            for value in values
            if not value.subtitle_like
            and value.confidence >= config.ocr_min_confidence
            and _alnum_count(value.comparison) >= config.ocr_min_alnum_chars
            and (
                value.area_ratio is None
                or value.area_ratio >= config.ocr_min_area_ratio
            )
        ]
        episodes = _split_ocr_episodes(
            eligible,
            config.ocr_episode_gap_seconds,
            config.ocr_spatial_tolerance,
        )
        previous_end: float | None = None
        for episode_index, episode in enumerate(episodes):
            unique_ids = _ordered_candidate_ids(episode, by_id)
            persistent = len(unique_ids) >= config.ocr_persistence_candidates
            extreme = any(
                value.confidence >= config.ocr_extreme_confidence
                and (
                    value.area_ratio is None
                    or value.area_ratio >= config.ocr_min_area_ratio
                )
                for value in episode
            )
            reappeared = bool(
                previous_end is not None
                and episode[0].timestamp - previous_end
                >= config.ocr_reappearance_cooldown_seconds
            )
            is_rare = frame_fraction <= config.ocr_rare_frame_fraction
            is_new = episode_index == 0 or (is_rare and reappeared)
            previous_end = episode[-1].timestamp
            if not (is_new and (persistent or extreme)):
                weak_episodes += 1
                continue
            event_id = _event_id(
                "OCR",
                base[0].video_id,
                cluster_key,
                by_id[unique_ids[0]].frame_index,
            )
            events.append(
                ProtectedEvent(
                    event_id=event_id,
                    event_type="ocr_new",
                    candidate_ids=unique_ids,
                    priority=config.ocr_event_priority,
                )
            )
    return tuple(events), {
        "common_tracks": common_tracks,
        "subtitle_observations": subtitle_count,
        "weak_episodes": weak_episodes,
    }


def _adapt_objects(
    base: Sequence[_BaseCandidate],
    records: Mapping[str, Mapping[str, Any]],
    scores: dict[str, dict[str, float]],
    availability: dict[str, set[str]],
    config: FeatureAdapterConfig,
) -> tuple[tuple[ProtectedEvent, ...], dict[str, int]]:
    observations: list[_ObjectObservation] = []
    by_id = {candidate.candidate_id: candidate for candidate in base}
    for candidate in base:
        record = records.get(candidate.candidate_id)
        if record is None or _record_status(record, "object record") != "success":
            continue
        availability[candidate.candidate_id].add("objects")
        detections = record.get("objects", [])
        if not isinstance(detections, list):
            raise ValueError("successful object record objects must be a list")
        image_size = _record_image_size(record)
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, Mapping):
                raise TypeError(f"object detection {detection_index} must be a mapping")
            class_name = _comparison_text(
                detection.get("class_name")
                or detection.get("label")
                or detection.get("class")
            )
            if not class_name:
                raise ValueError("object detection requires a class_name")
            class_id = detection.get("class_id")
            if class_id is not None:
                class_id = _non_negative_int(class_id, "object class_id")
                class_key = f"{class_id}:{class_name}"
            else:
                class_key = f"name:{class_name}"
            confidence = _unit_interval(detection.get("confidence"), "object confidence")
            area_ratio = _bbox_area_ratio(
                detection.get("bbox_xyxy") or detection.get("bbox"),
                image_size,
            )
            area_score = min(1.0, area_ratio / 0.20) if area_ratio is not None else 0.0
            quality = _clamp01(0.70 * confidence + 0.30 * area_score)
            observations.append(
                _ObjectObservation(
                    candidate_id=candidate.candidate_id,
                    timestamp=candidate.timestamp,
                    class_key=class_key,
                    class_name=class_name,
                    confidence=confidence,
                    area_ratio=area_ratio,
                    quality=quality,
                )
            )

    grouped: dict[str, list[_ObjectObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.class_key].append(observation)
    success_frame_count = max(
        1,
        sum("objects" in availability[candidate.candidate_id] for candidate in base),
    )
    events: list[ProtectedEvent] = []
    weak_episodes = 0
    for class_key in sorted(grouped):
        values = sorted(
            grouped[class_key],
            key=lambda item: (item.timestamp, item.candidate_id),
        )
        frame_fraction = len({value.candidate_id for value in values}) / success_frame_count
        rarity = _clamp01(1.0 - frame_fraction)
        for value in values:
            scores[value.candidate_id]["objects"] = max(
                scores[value.candidate_id]["objects"],
                _clamp01(value.quality * (0.5 + 0.5 * rarity)),
            )
        eligible = [
            value
            for value in values
            if value.confidence >= config.object_min_confidence
            and value.area_ratio is not None
            and value.area_ratio >= config.object_min_area_ratio
        ]
        episodes = _split_by_time(eligible, config.object_episode_gap_seconds)
        previous_end: float | None = None
        for episode_index, episode in enumerate(episodes):
            unique_ids = _ordered_candidate_ids(episode, by_id)
            persistent = len(unique_ids) >= config.object_persistence_candidates
            extreme = any(
                value.confidence >= config.object_extreme_confidence
                and value.area_ratio is not None
                and value.area_ratio >= config.object_extreme_min_area_ratio
                for value in episode
            )
            reappeared = bool(
                previous_end is not None
                and episode[0].timestamp - previous_end
                >= config.object_reappearance_cooldown_seconds
            )
            is_rare = frame_fraction <= config.object_rare_frame_fraction
            is_new = episode_index == 0 or (is_rare and reappeared)
            previous_end = episode[-1].timestamp
            if not (is_new and (persistent or extreme)):
                weak_episodes += 1
                continue
            event_id = _event_id(
                "OBJECT",
                base[0].video_id,
                class_key,
                by_id[unique_ids[0]].frame_index,
            )
            events.append(
                ProtectedEvent(
                    event_id=event_id,
                    event_type="object_new",
                    candidate_ids=unique_ids,
                    priority=config.object_event_priority,
                )
            )
    return tuple(events), {"weak_episodes": weak_episodes}


def _adapt_captions(
    base: Sequence[_BaseCandidate],
    records: Mapping[str, Mapping[str, Any]],
    scores: dict[str, dict[str, float]],
    availability: dict[str, set[str]],
) -> None:
    previous_tokens: set[str] | None = None
    for candidate in base:
        record = records.get(candidate.candidate_id)
        if record is None or _record_status(record, "caption record") != "success":
            continue
        availability[candidate.candidate_id].add("caption")
        text = _comparison_text(record.get("caption") or record.get("segment_caption"))
        tokens = set(text.split())
        if not tokens:
            score = 0.0
        elif previous_tokens is None:
            score = 0.5
        else:
            union = tokens | previous_tokens
            score = 1.0 - (len(tokens & previous_tokens) / len(union)) if union else 0.0
        scores[candidate.candidate_id]["caption"] = _clamp01(score)
        if tokens:
            previous_tokens = tokens


def _adapt_transitions(
    base: Sequence[_BaseCandidate],
    embeddings: Mapping[str, np.ndarray],
    scores: dict[str, dict[str, float]],
    availability: dict[str, set[str]],
    config: FeatureAdapterConfig,
) -> tuple[tuple[ProtectedEvent, ...], dict[str, Any]]:
    for candidate_id in embeddings:
        availability[candidate_id].add("transition")
    deltas: list[tuple[_BaseCandidate, _BaseCandidate, float]] = []
    for left, right in zip(base, base[1:]):
        left_vector = embeddings.get(left.candidate_id)
        right_vector = embeddings.get(right.candidate_id)
        if left_vector is None or right_vector is None:
            continue
        delta_time = right.timestamp - left.timestamp
        if delta_time <= 0 or delta_time > config.transition_max_pair_gap_seconds:
            continue
        similarity = float(np.clip(np.dot(left_vector, right_vector), -1.0, 1.0))
        delta = _clamp01((1.0 - similarity) / 2.0)
        deltas.append((left, right, delta))
    if not deltas:
        return (), {"boundary_count": 0, "threshold": None}

    values = np.asarray([item[2] for item in deltas], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(
        config.transition_absolute_floor,
        median + config.transition_mad_multiplier * mad,
    )
    threshold = _clamp01(threshold)
    for left, right, delta in deltas:
        strength = _clamp01(delta / max(threshold, 1e-12))
        scores[left.candidate_id]["transition"] = max(
            scores[left.candidate_id]["transition"],
            strength,
        )
        scores[right.candidate_id]["transition"] = max(
            scores[right.candidate_id]["transition"],
            strength,
        )

    qualifying = sorted(
        (item for item in deltas if item[2] >= threshold - 1e-12),
        key=lambda item: (-item[2], item[1].timestamp, item[0].candidate_id, item[1].candidate_id),
    )
    selected: list[tuple[_BaseCandidate, _BaseCandidate, float]] = []
    for item in qualifying:
        if any(
            abs(item[1].timestamp - kept[1].timestamp)
            <= config.transition_nms_seconds + 1e-12
            for kept in selected
        ):
            continue
        selected.append(item)
    selected.sort(key=lambda item: (item[1].timestamp, item[1].candidate_id))

    events: list[ProtectedEvent] = []
    for left, right, _ in selected:
        key = f"{left.candidate_id}->{right.candidate_id}"
        pre_id = _event_id("TRANSITION_PRE", left.video_id, key, left.frame_index)
        post_id = _event_id("TRANSITION_POST", right.video_id, key, right.frame_index)
        events.extend(
            (
                ProtectedEvent(
                    event_id=pre_id,
                    event_type="semantic_transition_pre",
                    candidate_ids=(left.candidate_id,),
                    priority=config.transition_event_priority,
                ),
                ProtectedEvent(
                    event_id=post_id,
                    event_type="semantic_transition_post",
                    candidate_ids=(right.candidate_id,),
                    priority=config.transition_event_priority,
                ),
            )
        )
    return tuple(events), {"boundary_count": len(selected), "threshold": threshold}


def _record_image_size(record: Mapping[str, Any]) -> tuple[float, float] | None:
    raw = record.get("image_size")
    if raw is None or (isinstance(raw, str) and raw == ""):
        return None
    if (
        not isinstance(raw, (str, bytes))
        and isinstance(raw, Sequence)
        and len(raw) == 0
    ):
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 2:
        raise ValueError("image_size must be [width, height]")
    width = _finite_real(raw[0], "image width")
    height = _finite_real(raw[1], "image height")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be > 0")
    return width, height


def _polygon_geometry(
    polygon: object,
    image_size: tuple[float, float] | None,
) -> tuple[float | None, tuple[float, float] | None]:
    if polygon is None:
        return None, None
    if isinstance(polygon, (str, bytes)) or not isinstance(polygon, Sequence) or len(polygon) < 3:
        raise ValueError("OCR polygon must contain at least three points")
    points: list[tuple[float, float]] = []
    for point in polygon:
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError("OCR polygon points must contain x and y")
        points.append(
            (
                _finite_real(point[0], "OCR polygon x"),
                _finite_real(point[1], "OCR polygon y"),
            )
        )
    if image_size is None:
        return None, None
    width, height = image_size
    xs = [min(width, max(0.0, point[0])) for point in points]
    ys = [min(height, max(0.0, point[1])) for point in points]
    area_ratio = _clamp01((max(xs) - min(xs)) * (max(ys) - min(ys)) / (width * height))
    center = (((max(xs) + min(xs)) / 2.0) / width, ((max(ys) + min(ys)) / 2.0) / height)
    return area_ratio, center


def _bbox_area_ratio(
    bbox: object,
    image_size: tuple[float, float] | None,
) -> float | None:
    if bbox is None:
        raise ValueError("object detection requires bbox_xyxy")
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence) or len(bbox) != 4:
        raise ValueError("object bbox must contain four xyxy values")
    values = [_finite_real(value, "object bbox coordinate") for value in bbox]
    if image_size is None:
        return None
    width, height = image_size
    left = min(width, max(0.0, values[0]))
    top = min(height, max(0.0, values[1]))
    right = min(width, max(0.0, values[2]))
    bottom = min(height, max(0.0, values[3]))
    return _clamp01(max(0.0, right - left) * max(0.0, bottom - top) / (width * height))


def _fuzzy_text_clusters(texts: set[str], threshold: float) -> dict[str, str]:
    ordered = sorted(texts)
    parent = {text: text for text in ordered}

    def find(text: str) -> str:
        while parent[text] != text:
            parent[text] = parent[parent[text]]
            text = parent[text]
        return text

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        parent[max(left_root, right_root)] = min(left_root, right_root)

    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if SequenceMatcher(None, left, right, autojunk=False).ratio() >= threshold:
                union(left, right)
    members: dict[str, list[str]] = defaultdict(list)
    for text in ordered:
        members[find(text)].append(text)
    canonical: dict[str, str] = {}
    for values in members.values():
        label = min(values)
        for value in values:
            canonical[value] = label
    return canonical


def _split_ocr_episodes(
    values: Sequence[_OcrObservation],
    max_gap: float,
    spatial_tolerance: float,
) -> list[list[_OcrObservation]]:
    ordered = sorted(values, key=lambda item: (item.timestamp, item.candidate_id, item.comparison))
    episodes: list[list[_OcrObservation]] = []
    for value in ordered:
        if not episodes:
            episodes.append([value])
            continue
        previous = episodes[-1][-1]
        spatial_break = bool(
            previous.center is not None
            and value.center is not None
            and (
                abs(previous.center[0] - value.center[0]) > spatial_tolerance
                or abs(previous.center[1] - value.center[1]) > spatial_tolerance
            )
        )
        if value.timestamp - previous.timestamp > max_gap or spatial_break:
            episodes.append([value])
        else:
            episodes[-1].append(value)
    return episodes


def _split_by_time(
    values: Sequence[_ObjectObservation],
    max_gap: float,
) -> list[list[_ObjectObservation]]:
    ordered = sorted(values, key=lambda item: (item.timestamp, item.candidate_id))
    episodes: list[list[_ObjectObservation]] = []
    for value in ordered:
        if not episodes or value.timestamp - episodes[-1][-1].timestamp > max_gap:
            episodes.append([value])
        else:
            episodes[-1].append(value)
    return episodes


def _ordered_candidate_ids(
    observations: Sequence[_OcrObservation | _ObjectObservation],
    by_id: Mapping[str, _BaseCandidate],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {observation.candidate_id for observation in observations},
            key=lambda candidate_id: (
                by_id[candidate_id].timestamp,
                by_id[candidate_id].frame_index,
                candidate_id,
            ),
        )
    )


def _display_text(value: object) -> str:
    return _WHITESPACE.sub(
        " ",
        unicodedata.normalize("NFKC", str(value or "")).strip(),
    )


def _comparison_text(value: object) -> str:
    return _WHITESPACE.sub(
        " ",
        _NON_WORD.sub(" ", _display_text(value).casefold()),
    ).strip()


def _alnum_count(value: str) -> int:
    return sum(character.isalnum() for character in value)


def _event_id(prefix: str, video_id: str, key: str, frame_index: int) -> str:
    digest = hashlib.sha1(f"{video_id}\0{key}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{video_id}_{digest}_{frame_index:09d}"


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


__all__ = [
    "COMPONENT_ORDER",
    "CandidateComponentScore",
    "FeatureAdapterConfig",
    "FeatureAdapterReport",
    "FeatureAdapterResult",
    "adapt_feature_records",
]
