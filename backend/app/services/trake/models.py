"""Typed data contracts for Temporal Retrieval and Alignment of Key Events.

The models in this module deliberately keep retrieval-frame identity separate
from the original-video ``frame_index`` used by TRAKE submissions.  Public
objects expose explicit ``to_dict`` methods rather than relying on dataclass
implementation details, so their response schema remains deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from backend.app.models.retrieval import RetrievalResult


class BoundaryType(str, Enum):
    """Conservative semantic-boundary categories understood by TRAKE."""

    FIRST_CONTACT = "first_contact"
    FIRST_LEAVE = "first_leave"
    FIRST_TRANSITION = "first_transition"
    PEAK = "peak"
    STATE = "state"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


def _boundary_value(value: BoundaryType | str) -> str:
    if isinstance(value, BoundaryType):
        return value.value
    return BoundaryType(str(value)).value


def _stable_value(value: Any) -> Any:
    """Return a JSON-ready copy with deterministic mapping order."""

    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _stable_value(value.to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    return value


@dataclass(frozen=True)
class TemporalEvent:
    """One ordered event criterion, preserved independently of its query."""

    index: int
    original_text: str
    retrieval_query: str
    name: str = ""
    boundary_type: BoundaryType | str = BoundaryType.UNKNOWN
    protected_terms: tuple[str, ...] = ()
    # Additive semantic fields.  Defaults keep older constructors and clients
    # source-compatible while the parser exposes the richer TRAKE contract.
    event_context: str = ""
    target_text: str = ""
    refinement_query: str = ""
    normalized_text: str = ""
    parser_warnings: tuple[str, ...] = ()
    semantic_confidence: float = 0.0
    parser_trace: Mapping[str, Any] = field(default_factory=dict)
    source_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "protected_terms", tuple(self.protected_terms))
        object.__setattr__(self, "parser_warnings", tuple(self.parser_warnings))
        # Validate string inputs while retaining the string-compatible enum.
        object.__setattr__(
            self,
            "boundary_type",
            BoundaryType(_boundary_value(self.boundary_type)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "original_text": self.original_text,
            "retrieval_query": self.retrieval_query,
            "boundary_type": _boundary_value(self.boundary_type),
            "protected_terms": list(self.protected_terms),
            "event_context": self.event_context,
            "target_text": self.target_text,
            "refinement_query": self.refinement_query,
            "normalized_text": self.normalized_text,
            "parser_warnings": list(self.parser_warnings),
            "semantic_confidence": self.semantic_confidence,
            "parser_trace": _stable_value(self.parser_trace),
            "source_label": self.source_label,
        }


@dataclass(frozen=True)
class TemporalEventPlan:
    """A context plus the exact ordered event sequence parsed from a query."""

    original_query: str
    context: str
    events: tuple[TemporalEvent, ...]
    parser_source: str = "deterministic_fallback"
    confidence: float = 0.0
    warnings: tuple[str, ...] = ()
    structural_confidence: float = 0.0
    semantic_confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def ordered_events(self) -> tuple[TemporalEvent, ...]:
        """Readable alias used by alignment callers."""

        return self.events

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "context": self.context,
            "events": [event.to_dict() for event in self.events],
            "parser_source": self.parser_source,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "structural_confidence": self.structural_confidence,
            "semantic_confidence": self.semantic_confidence,
        }


@dataclass(frozen=True)
class EventCandidate:
    """A retrieval result scored on the scale of one specific event."""

    event_index: int
    result: RetrievalResult
    normalized_score: float
    rank: int = 0
    retrieval_query: str = ""
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def retrieval_result(self) -> RetrievalResult:
        """Explicit alias for callers that prefer the full type name."""

        return self.result

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "normalized_score": self.normalized_score,
            "rank": self.rank,
            "retrieval_query": self.retrieval_query,
            "warnings": list(self.warnings),
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class VideoCandidate:
    """Video-level gating score and its per-event supporting candidates."""

    video_id: str
    coverage: float
    event_support: float
    context_score: float = 0.0
    total_score: float = 0.0
    event_candidates: Mapping[int, tuple[EventCandidate, ...]] = field(
        default_factory=dict
    )
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        candidates = {
            int(index): tuple(values)
            for index, values in self.event_candidates.items()
        }
        object.__setattr__(self, "event_candidates", candidates)
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "coverage": self.coverage,
            "event_support": self.event_support,
            "context_score": self.context_score,
            "total_score": self.total_score,
            "event_candidates": {
                str(index): [candidate.to_dict() for candidate in candidates]
                for index, candidates in sorted(self.event_candidates.items())
            },
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class TemporalPath:
    """One coarse, same-video ordered path through event candidates."""

    video_id: str
    event_candidates: tuple[EventCandidate, ...] = ()
    score: float = 0.0
    score_breakdown: Mapping[str, Any] = field(default_factory=dict)
    path_id: str = ""
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_candidates", tuple(self.event_candidates))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def candidates(self) -> tuple[EventCandidate, ...]:
        return self.event_candidates

    @property
    def frame_ids(self) -> tuple[int | None, ...]:
        """Original frame indexes, retaining ``None`` when lineage is absent."""

        return tuple(candidate.result.frame_index for candidate in self.event_candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_ids": list(self.frame_ids),
            "score": self.score,
            "score_breakdown": _stable_value(self.score_breakdown),
            "path_id": self.path_id,
            "event_candidates": [
                candidate.to_dict() for candidate in self.event_candidates
            ],
            "warnings": list(self.warnings),
        }


def _lineage_entry(
    value: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    data = dict(value) if isinstance(value, Mapping) else {}
    original_frame_index = data.get(
        "original_frame_index",
        data.get("frame_index"),
    )
    return {
        # Do not fill missing explicit provenance from sequence position.  A
        # downstream submission validator must be able to reject it fail-closed.
        "event_index": data.get("event_index"),
        "video_id": data.get("video_id", ""),
        "original_frame_index": original_frame_index,
        "internal_frame_id": data.get("internal_frame_id", data.get("frame_id", "")),
        "source": data.get("source", ""),
    }


@dataclass(frozen=True)
class TrakeHypothesis:
    """Rankable TRAKE answer containing exactly one frame per event.

    ``frame_ids`` are original zero-based video frame indexes.  They are not
    retrieval ``frame_id`` strings.  ``lineage`` is intentionally separate and
    machine-verifiable so submission code can fail closed when the canonical
    mapping is missing.
    """

    video_id: str
    frame_ids: tuple[int, ...]
    score: float = 0.0
    score_breakdown: Mapping[str, Any] = field(default_factory=dict)
    rank: int = 0
    coarse_candidates: tuple[EventCandidate, ...] = ()
    lineage: tuple[Mapping[str, Any], ...] = ()
    path_id: str = ""
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ids", tuple(self.frame_ids))
        object.__setattr__(self, "coarse_candidates", tuple(self.coarse_candidates))
        object.__setattr__(self, "lineage", tuple(self.lineage))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def candidates(self) -> tuple[EventCandidate, ...]:
        return self.coarse_candidates

    def _serialized_lineage(self) -> list[dict[str, Any]]:
        if self.lineage:
            return [
                _lineage_entry(value)
                for value in self.lineage
            ]
        # Derive lineage only from canonical RetrievalResult fields.  Never
        # manufacture it from the submitted frame_ids themselves.
        return [
            {
                "event_index": candidate.event_index,
                "video_id": candidate.result.video_id,
                "original_frame_index": candidate.result.frame_index,
                "internal_frame_id": candidate.result.frame_id,
                "source": "retrieval_result.frame_index",
            }
            for candidate in self.coarse_candidates
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "video_id": self.video_id,
            "frame_ids": list(self.frame_ids),
            "score": self.score,
            "score_breakdown": _stable_value(self.score_breakdown),
            "path_id": self.path_id,
            "events": [candidate.to_dict() for candidate in self.coarse_candidates],
            "lineage": self._serialized_lineage(),
            "warnings": list(self.warnings),
        }


__all__ = [
    "BoundaryType",
    "EventCandidate",
    "TemporalEvent",
    "TemporalEventPlan",
    "TemporalPath",
    "TrakeHypothesis",
    "VideoCandidate",
]
