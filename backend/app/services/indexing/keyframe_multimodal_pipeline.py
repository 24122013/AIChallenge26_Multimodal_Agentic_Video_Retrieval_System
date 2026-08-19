"""Pure Phase 3 orchestration for multimodal keyframe selection.

This module joins already-produced dense-candidate artifacts in memory.  It
deliberately performs no model inference and no filesystem I/O: callers remain
responsible for atomically publishing the returned final artifacts.

The pipeline is fail-closed by default.  Every dense candidate must have an
aligned SigLIP embedding and a successful OCR and object record before hard
events are derived. Caption data is an optional soft signal.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from .keyframe_candidates import (
    REASON_DENSE_INTERVAL,
    REASON_SHOT_BOUNDARY_END,
    REASON_SHOT_BOUNDARY_START,
    REASON_TINY_SHOT_MIDPOINT,
    REASON_VIDEO_END,
    REASON_VIDEO_START,
)
from .keyframe_feature_adapter import (
    CandidateComponentScore,
    FeatureAdapterConfig,
    FeatureAdapterReport,
    FeatureAdapterResult,
    adapt_feature_records,
)
from .keyframe_selection import (
    PHASE_COVERAGE,
    ProtectedEvent,
    SelectionConfig,
    SelectionResult,
    TemporalGap,
    build_shot_protection_events,
    build_endpoint_protection_events,
    select_keyframes,
)


KEYFRAME_STRATEGY_MULTIMODAL_COVERAGE = "multimodal_coverage"
KEYFRAME_STRATEGY_VISUAL_TEMPORAL = "visual_temporal"
_DENSE_REASONS = {
    REASON_DENSE_INTERVAL,
    REASON_SHOT_BOUNDARY_START,
    REASON_SHOT_BOUNDARY_END,
    REASON_TINY_SHOT_MIDPOINT,
    REASON_VIDEO_START,
    REASON_VIDEO_END,
}


class MultimodalKeyframePipelineError(ValueError):
    """Raised when Phase 3 cannot prove all configured hard guarantees."""


@dataclass(frozen=True)
class HardGuaranteeReport:
    """Independent audit of selector output, including head and tail gaps."""

    constraints_satisfied: bool
    event_recall_satisfied: bool
    temporal_coverage_satisfied: bool
    shot_coverage_satisfied: bool
    missing_protected_event_ids: tuple[str, ...]
    missing_shot_indices: tuple[int, ...]
    temporal_gaps: tuple[TemporalGap, ...]
    violating_gaps: tuple[TemporalGap, ...]
    observed_max_gap_seconds: float
    configured_max_gap_seconds: float
    gap_tolerance_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "constraints_satisfied": self.constraints_satisfied,
            "event_recall_satisfied": self.event_recall_satisfied,
            "temporal_coverage_satisfied": self.temporal_coverage_satisfied,
            "shot_coverage_satisfied": self.shot_coverage_satisfied,
            "missing_protected_event_ids": list(self.missing_protected_event_ids),
            "missing_shot_indices": list(self.missing_shot_indices),
            "temporal_gaps": [gap.to_dict() for gap in self.temporal_gaps],
            "violating_gaps": [gap.to_dict() for gap in self.violating_gaps],
            "observed_max_gap_seconds": self.observed_max_gap_seconds,
            "configured_max_gap_seconds": self.configured_max_gap_seconds,
            "gap_tolerance_seconds": self.gap_tolerance_seconds,
        }


@dataclass(frozen=True)
class MultimodalKeyframePipelineResult:
    """In-memory final artifacts and auditable Phase 3 ledgers."""

    video_id: str
    final_records: tuple[dict[str, Any], ...]
    final_embeddings: np.ndarray
    final_embedding_records: tuple[dict[str, Any], ...]
    final_ocr_records: tuple[dict[str, Any], ...]
    final_object_records: tuple[dict[str, Any], ...]
    final_caption_records: tuple[dict[str, Any], ...]
    candidate_ledger: tuple[dict[str, Any], ...]
    event_ledger: tuple[dict[str, Any], ...]
    feature_adapter_result: FeatureAdapterResult
    selection_result: SelectionResult
    guarantee_report: HardGuaranteeReport
    allow_partial_features: bool

    @property
    def adapter_report(self) -> FeatureAdapterReport:
        return self.feature_adapter_result.report

    @property
    def selection_report(self) -> SelectionResult:
        return self.selection_result

    def to_report(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "candidate_count": len(self.candidate_ledger),
            "selected_count": len(self.final_records),
            "allow_partial_features": self.allow_partial_features,
            "adapter": self.adapter_report.to_dict(),
            "selection": self.selection_report.to_report(),
            "guarantees": self.guarantee_report.to_dict(),
            "candidate_ledger": list(self.candidate_ledger),
            "event_ledger": list(self.event_ledger),
        }


@dataclass(frozen=True)
class _CandidateIdentity:
    candidate_id: str
    frame_id: str | None
    video_id: str
    timestamp: float
    frame_index: int
    shot_index: int
    reasons: tuple[str, ...]
    record: dict[str, Any]


def run_multimodal_keyframe_pipeline(
    candidates: Iterable[Mapping[str, Any]],
    *,
    embeddings: np.ndarray,
    embedding_records: Iterable[Mapping[str, Any]],
    ocr_records: Iterable[Mapping[str, Any]],
    object_records: Iterable[Mapping[str, Any]],
    caption_records: Iterable[Mapping[str, Any]] = (),
    video_duration: float,
    selection_config: SelectionConfig,
    adapter_config: FeatureAdapterConfig | None = None,
    allow_partial_features: bool = False,
    keyframe_strategy: str = KEYFRAME_STRATEGY_MULTIMODAL_COVERAGE,
) -> MultimodalKeyframePipelineResult:
    """Select and materialize one video's final Phase 3 in-memory artifacts.

    ``target_keyframes=None`` is passed through unchanged, so selection stops
    as soon as protected-event, shot, and temporal constraints are satisfied.
    A soft target never weakens a hard guarantee; a hard-cap failure raises
    :class:`MultimodalKeyframePipelineError` and produces no publishable result.
    """

    if not isinstance(selection_config, SelectionConfig):
        raise TypeError("selection_config must be a SelectionConfig")
    if adapter_config is not None and not isinstance(adapter_config, FeatureAdapterConfig):
        raise TypeError("adapter_config must be a FeatureAdapterConfig or None")
    if not isinstance(allow_partial_features, bool):
        raise TypeError("allow_partial_features must be a boolean")
    if not isinstance(keyframe_strategy, str) or not keyframe_strategy.strip():
        raise TypeError("keyframe_strategy must be a non-empty string")

    duration = _finite_non_negative(video_duration, "video_duration")
    identities = _normalize_candidate_records(candidates)
    candidate_records = tuple(identity.record for identity in identities)
    video_id = identities[0].video_id
    aliases = _candidate_aliases(identities)

    matrix, embedding_metadata, embedding_row_by_id = _validate_embeddings(
        embeddings,
        embedding_records,
        identities=identities,
        aliases=aliases,
    )
    ocr_by_id = _index_frame_records(
        ocr_records,
        modality="ocr",
        video_id=video_id,
        aliases=aliases,
    )
    object_by_id = _index_frame_records(
        object_records,
        modality="object",
        video_id=video_id,
        aliases=aliases,
    )
    caption_by_id = _index_frame_records(
        caption_records,
        modality="caption",
        video_id=video_id,
        aliases=aliases,
    )

    if not allow_partial_features:
        _require_complete_success("OCR", identities, ocr_by_id)
        _require_complete_success("object", identities, object_by_id)

    adapter_result = adapt_feature_records(
        candidate_records,
        ocr_records=tuple(ocr_by_id.values()),
        object_records=tuple(object_by_id.values()),
        caption_records=tuple(caption_by_id.values()),
        embeddings=matrix,
        embedding_records=embedding_metadata,
        config=adapter_config,
    )
    if adapter_result.report.missing_embedding_count:
        # Kept as a defense in depth if the adapter's join contract changes.
        raise MultimodalKeyframePipelineError(
            "complete SigLIP alignment was lost during feature adaptation"
        )

    selection_result = select_keyframes(
        adapter_result.selection_candidates,
        adapter_result.protected_events,
        video_duration=duration,
        config=selection_config,
    )
    audit_events = adapter_result.protected_events
    if selection_config.protect_video_endpoints:
        audit_events = (
            *audit_events,
            *build_endpoint_protection_events(adapter_result.selection_candidates),
        )
    guarantee_report = _audit_hard_guarantees(
        identities,
        audit_events,
        selection_result,
        video_duration=duration,
        config=selection_config,
    )
    if not selection_result.constraints_satisfied:
        raise MultimodalKeyframePipelineError(
            "selector hard constraints are unsatisfied: "
            f"{selection_result.stop_reason}; "
            f"events={list(selection_result.unsatisfied_event_ids)}; "
            f"violating_gaps={len(selection_result.violating_gaps)}"
        )
    if not guarantee_report.constraints_satisfied:
        raise MultimodalKeyframePipelineError(
            "independent hard-guarantee audit failed: "
            f"events={list(guarantee_report.missing_protected_event_ids)}; "
            f"shots={list(guarantee_report.missing_shot_indices)}; "
            f"violating_gaps={len(guarantee_report.violating_gaps)}"
        )

    scores_by_id = {
        score.candidate_id: score for score in adapter_result.candidate_scores
    }
    selected_by_id = {
        item.candidate.candidate_id: item for item in selection_result.selected
    }
    candidate_by_id = {identity.candidate_id: identity for identity in identities}

    selected_ids = tuple(
        item.candidate.candidate_id for item in selection_result.selected
    )
    semantic_novelty_by_id = _selected_semantic_novelty(
        selected_ids,
        embeddings=matrix,
        embedding_row_by_id=embedding_row_by_id,
    )
    final_records = tuple(
        _build_final_record(
            candidate_by_id[item.candidate.candidate_id].record,
            item,
            scores_by_id[item.candidate.candidate_id],
            semantic_novelty=semantic_novelty_by_id[item.candidate.candidate_id],
            keyframe_strategy=keyframe_strategy,
        )
        for item in selection_result.selected
    )
    final_embeddings = np.ascontiguousarray(
        matrix[[embedding_row_by_id[candidate_id] for candidate_id in selected_ids]],
        dtype=np.float32,
    )
    final_embedding_records = tuple(
        _build_final_embedding_record(
            embedding_metadata[embedding_row_by_id[candidate_id]],
            final_records[index],
            embedding_index=index,
        )
        for index, candidate_id in enumerate(selected_ids)
    )

    final_ocr = _filter_frame_records(ocr_by_id, selected_ids)
    final_objects = _filter_frame_records(object_by_id, selected_ids)
    final_captions = _filter_frame_records(caption_by_id, selected_ids)
    candidate_ledger = _build_candidate_ledger(
        identities,
        scores_by_id,
        selected_by_id,
        semantic_novelty_by_id,
        selection_result,
    )
    event_ledger = _build_event_ledger(
        adapter_result,
        selection_result,
        selection_config,
    )

    return MultimodalKeyframePipelineResult(
        video_id=video_id,
        final_records=final_records,
        final_embeddings=final_embeddings,
        final_embedding_records=final_embedding_records,
        final_ocr_records=final_ocr,
        final_object_records=final_objects,
        final_caption_records=final_captions,
        candidate_ledger=candidate_ledger,
        event_ledger=event_ledger,
        feature_adapter_result=adapter_result,
        selection_result=selection_result,
        guarantee_report=guarantee_report,
        allow_partial_features=allow_partial_features,
    )


def run_visual_temporal_keyframe_pipeline(
    candidates: Iterable[Mapping[str, Any]],
    *,
    embeddings: np.ndarray,
    embedding_records: Iterable[Mapping[str, Any]],
    video_duration: float,
    selection_config: SelectionConfig,
    adapter_config: FeatureAdapterConfig | None = None,
) -> MultimodalKeyframePipelineResult:
    """Select before semantic enrichment using only visual-temporal evidence.

    The shared deterministic selector still enforces shot and temporal coverage,
    uses SigLIP transition/novelty signals, and subsets the already-computed
    dense embedding matrix.  Caption, OCR, and object records are deliberately
    absent here so they cannot influence pre-selection behavior.
    """

    return run_multimodal_keyframe_pipeline(
        candidates,
        embeddings=embeddings,
        embedding_records=embedding_records,
        ocr_records=(),
        object_records=(),
        caption_records=(),
        video_duration=video_duration,
        selection_config=selection_config,
        adapter_config=adapter_config,
        allow_partial_features=True,
        keyframe_strategy=KEYFRAME_STRATEGY_VISUAL_TEMPORAL,
    )


def _finite_non_negative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalize_candidate_records(
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[_CandidateIdentity, ...]:
    values: list[_CandidateIdentity] = []
    for position, raw_record in enumerate(candidates):
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"candidates[{position}] must be a mapping")
        record = dict(raw_record)
        candidate_id = _non_empty_string(
            record.get("candidate_id"),
            f"candidates[{position}].candidate_id",
        )
        frame_id_value = record.get("frame_id")
        frame_id = (
            _non_empty_string(frame_id_value, f"candidates[{position}].frame_id")
            if frame_id_value is not None
            else None
        )
        video_id = _non_empty_string(
            record.get("video_id"),
            f"candidates[{position}].video_id",
        )
        raw_timestamp = (
            record.get("timestamp_sec")
            if record.get("timestamp_sec") is not None
            else record.get("timestamp")
        )
        timestamp = _finite_non_negative(
            raw_timestamp,
            f"candidates[{position}].timestamp",
        )
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
        )
        if isinstance(raw_reasons, str):
            reasons = (raw_reasons,)
        elif isinstance(raw_reasons, Sequence) and not isinstance(raw_reasons, bytes):
            reasons = tuple(raw_reasons)
        else:
            raise ValueError(
                f"candidates[{position}] requires dense candidate reasons"
            )
        if not reasons or any(not isinstance(reason, str) for reason in reasons):
            raise ValueError(
                f"candidates[{position}].candidate_reasons must contain strings"
            )
        if not _DENSE_REASONS.intersection(reasons):
            raise ValueError(
                f"candidates[{position}] is not a recognized dense candidate"
            )
        record["candidate_id"] = candidate_id
        record["video_id"] = video_id
        record["timestamp"] = timestamp
        record["frame_index"] = frame_index
        record["shot_index"] = shot_index
        record["candidate_reasons"] = list(dict.fromkeys(reasons))
        values.append(
            _CandidateIdentity(
                candidate_id=candidate_id,
                frame_id=frame_id,
                video_id=video_id,
                timestamp=timestamp,
                frame_index=frame_index,
                shot_index=shot_index,
                reasons=tuple(dict.fromkeys(reasons)),
                record=record,
            )
        )

    if not values:
        raise ValueError("candidates must contain at least one dense candidate")
    candidate_ids = [value.candidate_id for value in values]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id values must be unique")
    video_ids = {value.video_id for value in values}
    if len(video_ids) != 1:
        raise ValueError(f"one pipeline call requires one video_id, got {sorted(video_ids)}")

    ordered = tuple(
        sorted(
            values,
            key=lambda value: (
                value.timestamp,
                value.frame_index,
                value.candidate_id,
            ),
        )
    )
    _candidate_aliases(ordered)
    return ordered


def _candidate_aliases(
    identities: Sequence[_CandidateIdentity],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for identity in identities:
        for alias in (identity.candidate_id, identity.frame_id):
            if alias is None:
                continue
            existing = aliases.get(alias)
            if existing is not None and existing != identity.candidate_id:
                raise ValueError(f"candidate alias is ambiguous: {alias}")
            aliases[alias] = identity.candidate_id
    return aliases


def _record_candidate_id(
    record: Mapping[str, Any],
    *,
    aliases: Mapping[str, str],
    name: str,
) -> str:
    candidate_value = record.get("candidate_id")
    frame_value = record.get("frame_id")
    if candidate_value is None and frame_value is None:
        raise ValueError(f"{name} requires candidate_id or frame_id")
    candidate_id: str | None = None
    frame_candidate_id: str | None = None
    if candidate_value is not None:
        candidate_key = _non_empty_string(candidate_value, f"{name}.candidate_id")
        canonical_ids = set(aliases.values())
        if candidate_key not in canonical_ids:
            raise ValueError(f"{name} references unknown candidate: {candidate_key}")
        candidate_id = candidate_key
    if frame_value is not None:
        frame_key = _non_empty_string(frame_value, f"{name}.frame_id")
        frame_candidate_id = aliases.get(frame_key)
        if frame_candidate_id is None:
            raise ValueError(f"{name} references unknown candidate: {frame_key}")
    if (
        candidate_id is not None
        and frame_candidate_id is not None
        and candidate_id != frame_candidate_id
    ):
        raise ValueError(f"{name} candidate_id and frame_id refer to different candidates")
    resolved = candidate_id or frame_candidate_id
    if resolved is None:
        raise ValueError(f"{name} requires candidate_id or frame_id")
    return resolved


def _validate_record_video(record: Mapping[str, Any], video_id: str, name: str) -> None:
    record_video_id = _non_empty_string(record.get("video_id"), f"{name}.video_id")
    if record_video_id != video_id:
        raise ValueError(
            f"{name}.video_id {record_video_id!r} does not match {video_id!r}"
        )


def _record_status(record: Mapping[str, Any], name: str) -> str:
    status = record.get("status", "success")
    if status not in {"success", "error", "skipped"}:
        raise ValueError(f"{name}.status must be success, error, or skipped")
    return str(status)


def _validate_embeddings(
    embeddings: np.ndarray,
    records: Iterable[Mapping[str, Any]],
    *,
    identities: Sequence[_CandidateIdentity],
    aliases: Mapping[str, str],
) -> tuple[np.ndarray, tuple[dict[str, Any], ...], dict[str, int]]:
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a numpy.ndarray")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if embeddings.dtype != np.float32:
        raise ValueError("embeddings must use float32 dtype")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")

    metadata: list[dict[str, Any]] = []
    row_by_id: dict[str, int] = {}
    video_id = identities[0].video_id
    for row_index, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"embedding_records[{row_index}] must be a mapping")
        record = dict(raw_record)
        _validate_record_video(record, video_id, f"embedding_records[{row_index}]")
        embedding_index = record.get("embedding_index")
        if isinstance(embedding_index, bool) or not isinstance(embedding_index, Integral):
            raise TypeError("embedding_index must be an integer")
        if int(embedding_index) != row_index:
            raise ValueError(
                f"embedding_index mismatch at row {row_index}: {embedding_index}"
            )
        candidate_id = _record_candidate_id(
            record,
            aliases=aliases,
            name=f"embedding_records[{row_index}]",
        )
        if candidate_id in row_by_id:
            raise ValueError(f"duplicate embedding for candidate: {candidate_id}")
        vector_dim = record.get("vector_dim")
        if vector_dim is not None and (
            isinstance(vector_dim, bool)
            or not isinstance(vector_dim, Integral)
            or int(vector_dim) != embeddings.shape[1]
        ):
            raise ValueError(
                f"embedding vector_dim mismatch at row {row_index}: {vector_dim}"
            )
        norm = float(np.linalg.norm(embeddings[row_index]))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"embedding for {candidate_id} has invalid norm")
        if record.get("normalized") is True and not math.isclose(
            norm,
            1.0,
            rel_tol=1e-3,
            abs_tol=1e-3,
        ):
            raise ValueError(
                f"embedding for {candidate_id} is marked normalized but is not"
            )
        record["candidate_id"] = candidate_id
        row_by_id[candidate_id] = row_index
        metadata.append(record)

    if len(metadata) != embeddings.shape[0]:
        raise ValueError("embedding metadata count must equal embeddings rows")
    expected_ids = {identity.candidate_id for identity in identities}
    actual_ids = set(row_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unknown = sorted(actual_ids - expected_ids)
        raise ValueError(
            "SigLIP embedding alignment must cover every candidate exactly once: "
            f"missing={missing}; unknown={unknown}"
        )
    if len(metadata) != len(identities):
        raise ValueError(
            "SigLIP embedding alignment must contain one row per candidate"
        )
    return embeddings, tuple(metadata), row_by_id


def _index_frame_records(
    records: Iterable[Mapping[str, Any]],
    *,
    modality: str,
    video_id: str,
    aliases: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"{modality}_records[{position}] must be a mapping")
        record = dict(raw_record)
        name = f"{modality}_records[{position}]"
        _validate_record_video(record, video_id, name)
        candidate_id = _record_candidate_id(record, aliases=aliases, name=name)
        if candidate_id in indexed:
            raise ValueError(f"duplicate {modality} record for candidate: {candidate_id}")
        _record_status(record, name)
        record["candidate_id"] = candidate_id
        indexed[candidate_id] = record
    return {
        candidate_id: indexed[candidate_id]
        for candidate_id in sorted(indexed)
    }


def _require_complete_success(
    label: str,
    identities: Sequence[_CandidateIdentity],
    indexed: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_ids = {identity.candidate_id for identity in identities}
    missing = sorted(expected_ids - set(indexed))
    unsuccessful = sorted(
        f"{candidate_id}:{_record_status(record, f'{label} record')}"
        for candidate_id, record in indexed.items()
        if _record_status(record, f"{label} record") != "success"
    )
    if missing or unsuccessful:
        raise MultimodalKeyframePipelineError(
            f"{label} features are incomplete: "
            f"missing={missing}; unsuccessful={unsuccessful}"
        )


def _independent_temporal_gaps(
    selected_ids: set[str],
    identities: Sequence[_CandidateIdentity],
    video_duration: float,
) -> tuple[TemporalGap, ...]:
    selected = sorted(
        (identity for identity in identities if identity.candidate_id in selected_ids),
        key=lambda identity: (
            identity.timestamp,
            identity.frame_index,
            identity.candidate_id,
        ),
    )
    if not selected:
        return (TemporalGap(0.0, video_duration),) if video_duration > 0 else ()
    gaps: list[TemporalGap] = [
        TemporalGap(0.0, selected[0].timestamp, None, selected[0].candidate_id)
    ]
    gaps.extend(
        TemporalGap(
            left.timestamp,
            right.timestamp,
            left.candidate_id,
            right.candidate_id,
        )
        for left, right in zip(selected, selected[1:])
    )
    gaps.append(
        TemporalGap(
            selected[-1].timestamp,
            video_duration,
            selected[-1].candidate_id,
            None,
        )
    )
    return tuple(gaps)


def _audit_hard_guarantees(
    identities: Sequence[_CandidateIdentity],
    events: Sequence[ProtectedEvent],
    selection_result: SelectionResult,
    *,
    video_duration: float,
    config: SelectionConfig,
) -> HardGuaranteeReport:
    selected_ids = {
        item.candidate.candidate_id for item in selection_result.selected
    }
    missing_events = tuple(
        sorted(
            event.event_id
            for event in events
            if selected_ids.isdisjoint(event.candidate_ids)
        )
    )
    shots = {identity.shot_index for identity in identities}
    selected_shots = {
        identity.shot_index
        for identity in identities
        if identity.candidate_id in selected_ids
    }
    missing_shots = tuple(sorted(shots - selected_shots))
    gaps = _independent_temporal_gaps(selected_ids, identities, video_duration)
    limit = config.max_gap_seconds + config.gap_tolerance_seconds
    violating = tuple(gap for gap in gaps if gap.duration > limit + 1e-12)
    event_ok = not missing_events
    temporal_ok = not violating
    shot_ok = not missing_shots
    return HardGuaranteeReport(
        constraints_satisfied=event_ok and temporal_ok and shot_ok,
        event_recall_satisfied=event_ok,
        temporal_coverage_satisfied=temporal_ok,
        shot_coverage_satisfied=shot_ok,
        missing_protected_event_ids=missing_events,
        missing_shot_indices=missing_shots,
        temporal_gaps=gaps,
        violating_gaps=violating,
        observed_max_gap_seconds=max((gap.duration for gap in gaps), default=0.0),
        configured_max_gap_seconds=config.max_gap_seconds,
        gap_tolerance_seconds=config.gap_tolerance_seconds,
    )


def _selected_semantic_novelty(
    selected_ids: Sequence[str],
    *,
    embeddings: np.ndarray,
    embedding_row_by_id: Mapping[str, int],
) -> dict[str, float]:
    """Measure final-set novelty using the same cosine scale as selector MMR.

    This is annotate-only: it never changes which candidates were selected.
    A singleton selected set is maximally novel; otherwise each frame receives
    its distance from the most similar *other* selected frame.
    """

    if not selected_ids:
        return {}
    rows = np.asarray(
        [embedding_row_by_id[candidate_id] for candidate_id in selected_ids],
        dtype=np.int64,
    )
    selected = np.asarray(embeddings[rows], dtype=np.float64)
    scales = np.max(np.abs(selected), axis=1, keepdims=True)
    scales[scales == 0.0] = 1.0
    selected = selected / scales
    norms = np.linalg.norm(selected, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    selected = selected / norms
    if len(selected_ids) == 1:
        novelty = np.ones(1, dtype=np.float64)
    else:
        similarities = np.clip(selected @ selected.T, -1.0, 1.0)
        np.fill_diagonal(similarities, -np.inf)
        novelty = np.clip((1.0 - np.max(similarities, axis=1)) / 2.0, 0.0, 1.0)
    return {
        candidate_id: round(float(value), 6)
        for candidate_id, value in zip(selected_ids, novelty)
    }


def _build_final_record(
    candidate_record: Mapping[str, Any],
    selected: Any,
    score: CandidateComponentScore,
    *,
    semantic_novelty: float,
    keyframe_strategy: str,
) -> dict[str, Any]:
    record = dict(candidate_record)
    component_scores = dict(score.component_scores)
    protected_event_ids = list(selected.covered_event_ids)
    dedup_cluster_id = next(
        (
            reason.split(":", 1)[1]
            for reason in selected.selection_reasons
            if reason.startswith("dedup_cluster:")
        ),
        None,
    )
    duplicate_override_reason = (
        "protected_override"
        if "protected_override" in selected.selection_reasons
        else (
            "temporal_repair"
            if "temporal_repair" in selected.selection_reasons
            else None
        )
    )
    provenance = {
        "strategy": keyframe_strategy,
        "selection_rank": selected.selection_rank,
        "selection_phase": selected.selection_phase,
        "selection_reasons": list(selected.selection_reasons),
        "selection_score": selected.selection_score,
        "covered_event_ids": list(selected.covered_event_ids),
        "feature_protected_event_ids": list(score.protected_event_ids),
        "importance_score": score.importance_score,
        "semantic_novelty": semantic_novelty,
        "component_scores": component_scores,
        "available_modalities": list(score.available_modalities),
    }
    record.update(
        {
            "keyframe_strategy": keyframe_strategy,
            "selection_phase": selected.selection_phase,
            "selection_rank": selected.selection_rank,
            "selection_reasons": list(selected.selection_reasons),
            "covered_event_ids": list(selected.covered_event_ids),
            "selection_score": selected.selection_score,
            "protected": bool(protected_event_ids),
            "coverage_added": selected.selection_phase == PHASE_COVERAGE,
            "importance_score": score.importance_score,
            "semantic_novelty": semantic_novelty,
            "component_scores": component_scores,
            "available_modalities": list(score.available_modalities),
            "protected_event_ids": protected_event_ids,
            "dedup_cluster_id": dedup_cluster_id,
            "duplicate_override_reason": duplicate_override_reason,
            "selection_provenance": provenance,
        }
    )
    return record


def _build_final_embedding_record(
    embedding_record: Mapping[str, Any],
    final_record: Mapping[str, Any],
    *,
    embedding_index: int,
) -> dict[str, Any]:
    record = dict(embedding_record)
    record["embedding_index"] = embedding_index
    record["candidate_id"] = final_record["candidate_id"]
    record["video_id"] = final_record["video_id"]
    if final_record.get("frame_id") is not None:
        record["frame_id"] = final_record["frame_id"]
    for field in (
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
        "dedup_cluster_id",
        "duplicate_override_reason",
        "selection_provenance",
    ):
        record[field] = final_record[field]
    return record


def _filter_frame_records(
    indexed: Mapping[str, Mapping[str, Any]],
    selected_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(indexed[candidate_id])
        for candidate_id in selected_ids
        if candidate_id in indexed
    )


def _build_candidate_ledger(
    identities: Sequence[_CandidateIdentity],
    scores_by_id: Mapping[str, CandidateComponentScore],
    selected_by_id: Mapping[str, Any],
    semantic_novelty_by_id: Mapping[str, float],
    selection_result: SelectionResult,
) -> tuple[dict[str, Any], ...]:
    removed_by_id = {
        str(record["candidate_id"]): record
        for record in selection_result.dedup_removed
    }
    values: list[dict[str, Any]] = []
    for identity in identities:
        score = scores_by_id[identity.candidate_id]
        selected = selected_by_id.get(identity.candidate_id)
        removed = removed_by_id.get(identity.candidate_id, {})
        values.append(
            {
                "candidate_id": identity.candidate_id,
                "frame_id": identity.frame_id,
                "video_id": identity.video_id,
                "timestamp": identity.timestamp,
                "frame_index": identity.frame_index,
                "shot_index": identity.shot_index,
                "candidate_reasons": list(identity.reasons),
                "selected": selected is not None,
                "importance_score": score.importance_score,
                "semantic_novelty": semantic_novelty_by_id.get(identity.candidate_id),
                "component_scores": dict(score.component_scores),
                "available_modalities": list(score.available_modalities),
                "feature_protected_event_ids": list(score.protected_event_ids),
                "selection_rank": selected.selection_rank if selected else None,
                "selection_phase": selected.selection_phase if selected else None,
                "selection_reasons": list(selected.selection_reasons) if selected else [],
                "covered_event_ids": list(selected.covered_event_ids) if selected else [],
                "dedup_cluster_id": removed.get("dedup_cluster_id"),
                "dedup_removed_reason": (
                    removed.get("reason")
                    if not removed.get("retained_after_repair")
                    else None
                ),
                "dedup_override_reason": removed.get("override_reason"),
            }
        )
    return tuple(values)


def _build_event_ledger(
    adapter_result: FeatureAdapterResult,
    selection_result: SelectionResult,
    selection_config: SelectionConfig,
) -> tuple[dict[str, Any], ...]:
    selected_ids = {
        item.candidate.candidate_id for item in selection_result.selected
    }
    events: list[tuple[str, ProtectedEvent]] = [
        ("feature_adapter", event) for event in adapter_result.protected_events
    ]
    events.extend(
        ("shot_coverage", event)
        for event in build_shot_protection_events(
            adapter_result.selection_candidates
        )
    )
    if selection_config.protect_video_endpoints:
        events.extend(
            ("endpoint_coverage", event)
            for event in build_endpoint_protection_events(
                adapter_result.selection_candidates
            )
        )
    return tuple(
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "priority": event.priority,
            "source": source,
            "candidate_ids": list(event.candidate_ids),
            "selected_candidate_ids": sorted(selected_ids.intersection(event.candidate_ids)),
            "satisfied": not selected_ids.isdisjoint(event.candidate_ids),
        }
        for source, event in sorted(
            events,
            key=lambda value: (-value[1].priority, value[1].event_id, value[0]),
        )
    )


__all__ = [
    "HardGuaranteeReport",
    "KEYFRAME_STRATEGY_MULTIMODAL_COVERAGE",
    "KEYFRAME_STRATEGY_VISUAL_TEMPORAL",
    "MultimodalKeyframePipelineError",
    "MultimodalKeyframePipelineResult",
    "run_multimodal_keyframe_pipeline",
    "run_visual_temporal_keyframe_pipeline",
]
