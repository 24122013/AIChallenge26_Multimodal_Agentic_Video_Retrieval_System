"""Pure constraint-based keyframe selection for the Phase 1 pipeline.

This module deliberately has no model or filesystem dependencies.  Candidate
features and protected events are supplied by later pipeline stages; Phase 1
only guarantees that detected events and temporal coverage are preserved when
the candidate pool makes that possible.
"""

from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np

from .keyframe_candidates import (
    REASON_SHOT_BOUNDARY_END,
    REASON_SHOT_BOUNDARY_START,
    REASON_TINY_SHOT_MIDPOINT,
    KeyframeCandidate,
)


PHASE_PROTECTED = "protected"
PHASE_COVERAGE = "coverage_fill"
PHASE_MMR = "mmr"


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_optional_count(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class SelectionCandidate:
    """A model-independent candidate consumed by the hard-constraint selector."""

    candidate_id: str
    timestamp: float
    frame_index: int
    shot_index: int
    importance_score: float = 0.0
    semantic_embedding: tuple[float, ...] = ()
    duplicate_group: str | None = None
    source_reasons: tuple[str, ...] = ()
    shot_start_sec: float | None = None
    shot_end_sec: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str):
            raise TypeError("candidate_id must be a string")
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        timestamp = _finite_real(self.timestamp, "candidate timestamp")
        if timestamp < 0:
            raise ValueError("candidate timestamp must be >= 0")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, Integral):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if isinstance(self.shot_index, bool) or not isinstance(self.shot_index, Integral):
            raise TypeError("shot_index must be an integer")
        if self.shot_index < 0:
            raise ValueError("shot_index must be >= 0")
        if (self.shot_start_sec is None) != (self.shot_end_sec is None):
            raise ValueError("shot_start_sec and shot_end_sec must be supplied together")
        shot_start_sec: float | None = None
        shot_end_sec: float | None = None
        if self.shot_start_sec is not None and self.shot_end_sec is not None:
            shot_start_sec = _finite_real(self.shot_start_sec, "shot_start_sec")
            shot_end_sec = _finite_real(self.shot_end_sec, "shot_end_sec")
            if shot_start_sec < 0:
                raise ValueError("shot_start_sec must be >= 0")
            if shot_end_sec < shot_start_sec:
                raise ValueError("shot_end_sec must be >= shot_start_sec")
            if not shot_start_sec - 1e-12 <= timestamp <= shot_end_sec + 1e-12:
                raise ValueError("candidate timestamp must lie inside its shot bounds")
        importance_score = _finite_real(self.importance_score, "importance_score")
        if not 0.0 <= importance_score <= 1.0:
            raise ValueError("importance_score must be between 0 and 1")
        if self.duplicate_group is not None and not isinstance(self.duplicate_group, str):
            raise TypeError("duplicate_group must be a string or None")
        if isinstance(self.source_reasons, (str, bytes)):
            raise TypeError("source_reasons must be a sequence, not a string")
        source_reasons = tuple(self.source_reasons)
        if any(not isinstance(value, str) for value in source_reasons):
            raise TypeError("source_reasons must contain only strings")

        embedding = tuple(float(value) for value in self.semantic_embedding)
        if any(not math.isfinite(value) for value in embedding):
            raise ValueError("semantic_embedding must contain only finite values")
        if embedding:
            embedding_norm = math.hypot(*embedding)
            if not math.isfinite(embedding_norm) or embedding_norm <= 0:
                raise ValueError(
                    "semantic_embedding must not be a zero vector and must have a finite norm"
                )
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "shot_start_sec", shot_start_sec)
        object.__setattr__(self, "shot_end_sec", shot_end_sec)
        object.__setattr__(self, "importance_score", importance_score)
        object.__setattr__(self, "semantic_embedding", embedding)
        object.__setattr__(
            self,
            "source_reasons",
            tuple(dict.fromkeys(source_reasons)),
        )

    @classmethod
    def from_generated_candidate(
        cls,
        candidate: KeyframeCandidate,
        *,
        importance_score: float = 0.0,
        semantic_embedding: Sequence[float] = (),
        duplicate_group: str | None = None,
    ) -> SelectionCandidate:
        """Adapt a model-free dense candidate after features become available."""

        return cls(
            candidate_id=candidate.candidate_id,
            timestamp=candidate.timestamp_sec,
            frame_index=candidate.frame_index,
            shot_index=candidate.shot_index,
            shot_start_sec=candidate.shot_start_sec,
            shot_end_sec=candidate.shot_end_sec,
            importance_score=importance_score,
            semantic_embedding=tuple(semantic_embedding),
            duplicate_group=duplicate_group,
            source_reasons=candidate.reasons,
        )


@dataclass(frozen=True)
class ProtectedEvent:
    """One detected event for which at least one representative is mandatory.

    A two-sided transition should be represented as two events (pre and post),
    because ``candidate_ids`` has OR semantics.
    """

    event_id: str
    event_type: str
    candidate_ids: tuple[str, ...]
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str):
            raise TypeError("event_id must be a string")
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if not isinstance(self.event_type, str):
            raise TypeError("event_type must be a string")
        if not self.event_type.strip():
            raise ValueError("event_type must not be empty")
        if isinstance(self.candidate_ids, (str, bytes)):
            raise TypeError("candidate_ids must be a sequence, not a string")
        candidate_ids = tuple(self.candidate_ids)
        if any(not isinstance(value, str) for value in candidate_ids):
            raise TypeError("candidate_ids must contain only strings")
        candidate_ids = tuple(dict.fromkeys(candidate_ids))
        if not candidate_ids or any(not value.strip() for value in candidate_ids):
            raise ValueError("protected events require at least one non-empty candidate_id")
        if isinstance(self.priority, bool) or not isinstance(self.priority, Integral):
            raise TypeError("event priority must be an integer")
        if self.priority < 0:
            raise ValueError("event priority must be >= 0")
        object.__setattr__(self, "candidate_ids", candidate_ids)


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for hard constraints followed by optional MMR filling."""

    max_gap_seconds: float
    target_keyframes: int | None = None
    hard_max_keyframes: int | None = None
    gap_tolerance_seconds: float = 0.0
    importance_weight: float = 0.65
    novelty_weight: float = 0.35
    exact_search_candidate_limit: int = 18
    protect_each_shot: bool = True

    def __post_init__(self) -> None:
        max_gap_seconds = _finite_real(self.max_gap_seconds, "max_gap_seconds")
        gap_tolerance_seconds = _finite_real(
            self.gap_tolerance_seconds,
            "gap_tolerance_seconds",
        )
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be > 0")
        if gap_tolerance_seconds < 0:
            raise ValueError("gap_tolerance_seconds must be >= 0")
        _validate_optional_count(self.target_keyframes, "target_keyframes")
        _validate_optional_count(self.hard_max_keyframes, "hard_max_keyframes")
        importance_weight = _finite_real(self.importance_weight, "importance_weight")
        novelty_weight = _finite_real(self.novelty_weight, "novelty_weight")
        if importance_weight < 0:
            raise ValueError("importance_weight must be >= 0")
        if novelty_weight < 0:
            raise ValueError("novelty_weight must be >= 0")
        if importance_weight + novelty_weight <= 0:
            raise ValueError("at least one MMR weight must be positive")
        if (
            isinstance(self.exact_search_candidate_limit, bool)
            or not isinstance(self.exact_search_candidate_limit, Integral)
        ):
            raise TypeError("exact_search_candidate_limit must be an integer")
        if not 0 <= self.exact_search_candidate_limit <= 18:
            raise ValueError("exact_search_candidate_limit must be between 0 and 18")
        if not isinstance(self.protect_each_shot, bool):
            raise TypeError("protect_each_shot must be a boolean")
        object.__setattr__(self, "max_gap_seconds", max_gap_seconds)
        object.__setattr__(self, "gap_tolerance_seconds", gap_tolerance_seconds)
        object.__setattr__(self, "importance_weight", importance_weight)
        object.__setattr__(self, "novelty_weight", novelty_weight)


@dataclass(frozen=True)
class TemporalGap:
    start: float
    end: float
    start_candidate_id: str | None = None
    end_candidate_id: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "start_candidate_id": self.start_candidate_id,
            "end_candidate_id": self.end_candidate_id,
        }


@dataclass(frozen=True)
class SelectedCandidate:
    candidate: SelectionCandidate
    selection_rank: int
    selection_phase: str
    selection_reasons: tuple[str, ...]
    covered_event_ids: tuple[str, ...]
    selection_score: float | None = None


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[SelectedCandidate, ...]
    status: str
    stop_reason: str
    constraints_satisfied: bool
    coverage_satisfied: bool
    infeasibility_proven: bool
    max_gap_before: float
    max_gap_after: float
    violating_gaps: tuple[TemporalGap, ...]
    unsatisfied_event_ids: tuple[str, ...]
    soft_target_reached: bool
    soft_budget_exceeded: bool
    soft_stop_reason: str
    selection_method: str
    target_keyframes: int | None
    hard_max_keyframes: int | None

    def to_report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "constraints_satisfied": self.constraints_satisfied,
            "coverage_satisfied": self.coverage_satisfied,
            "infeasibility_proven": self.infeasibility_proven,
            "selected_count": len(self.selected),
            "selected_candidate_ids": [item.candidate.candidate_id for item in self.selected],
            "max_gap_before": self.max_gap_before,
            "max_gap_after": self.max_gap_after,
            "violating_gaps": [gap.to_dict() for gap in self.violating_gaps],
            "unsatisfied_event_ids": list(self.unsatisfied_event_ids),
            "soft_target_reached": self.soft_target_reached,
            "soft_budget_exceeded": self.soft_budget_exceeded,
            "soft_stop_reason": self.soft_stop_reason,
            "selection_method": self.selection_method,
            "target_keyframes": self.target_keyframes,
            "hard_max_keyframes": self.hard_max_keyframes,
        }


@dataclass
class _SelectionState:
    selected: dict[str, SelectionCandidate] = field(default_factory=dict)
    rank: dict[str, int] = field(default_factory=dict)
    phase: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, set[str]] = field(default_factory=dict)
    score: dict[str, float | None] = field(default_factory=dict)

    def add(
        self,
        candidate: SelectionCandidate,
        *,
        phase: str,
        reason: str,
        score: float | None = None,
    ) -> bool:
        candidate_id = candidate.candidate_id
        if candidate_id in self.selected:
            self.reasons[candidate_id].add(reason)
            return False
        self.selected[candidate_id] = candidate
        self.rank[candidate_id] = len(self.rank) + 1
        self.phase[candidate_id] = phase
        self.reasons[candidate_id] = {reason}
        self.score[candidate_id] = score
        return True


@dataclass(frozen=True)
class _CoverageIndex:
    by_id: dict[str, SelectionCandidate]
    timestamps: tuple[float, ...]
    best_candidate_ids: tuple[str, ...]


def _build_coverage_index(
    candidates: Sequence[SelectionCandidate],
) -> _CoverageIndex:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    best_by_timestamp: dict[float, SelectionCandidate] = {}
    for candidate in candidates:
        current = best_by_timestamp.get(candidate.timestamp)
        if current is None or (
            -candidate.importance_score,
            candidate.frame_index,
            candidate.candidate_id,
        ) < (
            -current.importance_score,
            current.frame_index,
            current.candidate_id,
        ):
            best_by_timestamp[candidate.timestamp] = candidate
    timestamps = tuple(sorted(best_by_timestamp))
    return _CoverageIndex(
        by_id=by_id,
        timestamps=timestamps,
        best_candidate_ids=tuple(
            best_by_timestamp[timestamp].candidate_id for timestamp in timestamps
        ),
    )


def select_keyframes(
    candidates: Iterable[SelectionCandidate],
    protected_events: Iterable[ProtectedEvent],
    *,
    video_duration: float,
    config: SelectionConfig,
) -> SelectionResult:
    """Select keyframes while preserving detected events and temporal coverage.

    Hard constraints are handled before the soft MMR target.  When a hard cap
    blocks the deterministic heuristic, a bounded exact search considers the
    event-relevant candidate subset and completes temporal coverage greedily.
    Unsearched cap failures are reported as partial and are never mislabeled as
    mathematically infeasible.
    """

    video_duration = _finite_real(video_duration, "video_duration")
    if video_duration < 0:
        raise ValueError("video_duration must be >= 0")
    ordered = _validate_and_order_candidates(candidates, video_duration)
    supplied_events = tuple(protected_events)
    if config.protect_each_shot and video_duration > 0 and not ordered:
        supplied_event_ids = [event.event_id for event in supplied_events]
        if len(supplied_event_ids) != len(set(supplied_event_ids)):
            raise ValueError("protected event_id values must be unique")
        violations = _violating_gaps(
            (),
            video_duration,
            config.max_gap_seconds,
            config.gap_tolerance_seconds,
        )
        return SelectionResult(
            selected=(),
            status="partial",
            stop_reason="candidate_pool_empty",
            constraints_satisfied=False,
            coverage_satisfied=not violations,
            infeasibility_proven=True,
            max_gap_before=video_duration,
            max_gap_after=video_duration,
            violating_gaps=violations,
            unsatisfied_event_ids=tuple(
                sorted((*supplied_event_ids, "__video__:representative"))
            ),
            soft_target_reached=config.target_keyframes in (None, 0),
            soft_budget_exceeded=False,
            soft_stop_reason="candidate_pool_empty",
            selection_method="no_candidates",
            target_keyframes=config.target_keyframes,
            hard_max_keyframes=config.hard_max_keyframes,
        )

    automatic_shot_events = (
        build_shot_protection_events(ordered) if config.protect_each_shot else ()
    )
    events = _validate_and_order_events(
        (*supplied_events, *automatic_shot_events),
        ordered,
    )
    by_id = {candidate.candidate_id: candidate for candidate in ordered}
    state = _SelectionState()

    _select_protected_events(state, ordered, events, video_duration, config)
    max_gap_before = _max_gap(state.selected.values(), video_duration)
    coverage_exhausted = not _repair_temporal_coverage(
        state,
        ordered,
        video_duration,
        config,
    )

    unsatisfied = _unsatisfied_event_ids(state.selected, events)
    violations = _violating_gaps(
        state.selected.values(),
        video_duration,
        config.max_gap_seconds,
        config.gap_tolerance_seconds,
    )
    selection_method = "greedy_event_cover+per_gap_minimal_fill"
    exact_infeasible = False

    hard_cap_blocked = (
        config.hard_max_keyframes is not None
        and len(state.selected) >= config.hard_max_keyframes
        and bool(unsatisfied or violations)
    )
    if hard_cap_blocked:
        exact_ids, exact_attempted = _find_exact_hard_constraint_subset(
            ordered,
            events,
            video_duration,
            config,
        )
        if exact_attempted:
            selection_method = "exact_hard_constraint_fallback"
        if exact_attempted and exact_ids is None:
            exact_infeasible = True
        elif exact_ids is not None:
            state = _state_from_exact_subset(exact_ids, by_id, events)
            max_gap_before = _max_gap(
                (
                    candidate
                    for candidate_id, candidate in state.selected.items()
                    if state.phase[candidate_id] == PHASE_PROTECTED
                ),
                video_duration,
            )
            coverage_exhausted = False
            unsatisfied = _unsatisfied_event_ids(state.selected, events)
            violations = _violating_gaps(
                state.selected.values(),
                video_duration,
                config.max_gap_seconds,
                config.gap_tolerance_seconds,
            )

    hard_constraints_satisfied = not unsatisfied and not violations
    soft_stop_reason = "hard_constraints_unsatisfied"
    if hard_constraints_satisfied:
        soft_stop_reason = _fill_mmr(state, ordered, config)

    unsatisfied = _unsatisfied_event_ids(state.selected, events)
    violations = _violating_gaps(
        state.selected.values(),
        video_duration,
        config.max_gap_seconds,
        config.gap_tolerance_seconds,
    )
    constraints_satisfied = not unsatisfied and not violations
    hard_cap_reached = (
        config.hard_max_keyframes is not None
        and len(state.selected) >= config.hard_max_keyframes
    )

    if constraints_satisfied:
        status = "satisfied"
        stop_reason = "constraints_satisfied"
        infeasibility_proven = False
    else:
        status = "partial"
        if exact_infeasible:
            stop_reason = "hard_constraints_infeasible_within_cap"
            infeasibility_proven = True
        elif hard_cap_reached:
            stop_reason = "hard_cap_reached"
            infeasibility_proven = False
        elif coverage_exhausted and violations:
            stop_reason = "coverage_candidates_unavailable"
            infeasibility_proven = True
        else:
            stop_reason = "constraints_unsatisfied"
            infeasibility_proven = False

    selected_output = _selected_output(state, events)
    selected_count = len(selected_output)
    target = config.target_keyframes
    return SelectionResult(
        selected=selected_output,
        status=status,
        stop_reason=stop_reason,
        constraints_satisfied=constraints_satisfied,
        coverage_satisfied=not violations,
        infeasibility_proven=infeasibility_proven,
        max_gap_before=max_gap_before,
        max_gap_after=_max_gap(state.selected.values(), video_duration),
        violating_gaps=violations,
        unsatisfied_event_ids=unsatisfied,
        soft_target_reached=target is None or selected_count >= target,
        soft_budget_exceeded=target is not None and selected_count > target,
        soft_stop_reason=soft_stop_reason,
        selection_method=selection_method,
        target_keyframes=config.target_keyframes,
        hard_max_keyframes=config.hard_max_keyframes,
    )


def build_shot_protection_events(
    candidates: Iterable[SelectionCandidate],
    *,
    priority: int = 100,
) -> tuple[ProtectedEvent, ...]:
    """Create one hard representative event for every candidate shot.

    Phase 2 may disable ``protect_each_shot`` and supply events for merged
    effective shots instead.  Raw-shot protection is the safe Phase 1 default.
    """

    if isinstance(priority, bool) or not isinstance(priority, Integral):
        raise TypeError("shot event priority must be an integer")
    if priority < 0:
        raise ValueError("shot event priority must be >= 0")
    by_shot: dict[int, list[SelectionCandidate]] = {}
    for candidate in candidates:
        by_shot.setdefault(candidate.shot_index, []).append(candidate)
    boundary_reasons = {
        REASON_SHOT_BOUNDARY_START,
        REASON_SHOT_BOUNDARY_END,
        REASON_TINY_SHOT_MIDPOINT,
    }
    events: list[ProtectedEvent] = []
    for shot_index, shot_candidates in sorted(by_shot.items()):
        shot_candidates.sort(
            key=lambda candidate: (
                candidate.timestamp,
                candidate.frame_index,
                candidate.candidate_id,
            )
        )
        boundary_candidates = [
            candidate
            for candidate in shot_candidates
            if boundary_reasons.intersection(candidate.source_reasons)
        ]
        eligible = boundary_candidates or shot_candidates
        events.append(
            ProtectedEvent(
                event_id=f"__shot__:{shot_index}",
                event_type="shot_boundary" if boundary_candidates else "shot",
                candidate_ids=tuple(candidate.candidate_id for candidate in eligible),
                priority=priority,
            )
        )
    return tuple(events)


def _validate_and_order_candidates(
    candidates: Iterable[SelectionCandidate],
    video_duration: float,
) -> tuple[SelectionCandidate, ...]:
    values = tuple(candidates)
    candidate_ids = [candidate.candidate_id for candidate in values]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id values must be unique")
    outside = [
        candidate.candidate_id
        for candidate in values
        if candidate.timestamp > video_duration
    ]
    if outside:
        raise ValueError(f"candidate timestamps exceed video_duration: {outside}")
    dimensions = {
        len(candidate.semantic_embedding)
        for candidate in values
        if candidate.semantic_embedding
    }
    if len(dimensions) > 1:
        raise ValueError("non-empty semantic embeddings must have one shared dimension")
    return tuple(
        sorted(
            values,
            key=lambda candidate: (
                candidate.timestamp,
                candidate.frame_index,
                candidate.candidate_id,
            ),
        )
    )


def _validate_and_order_events(
    protected_events: Iterable[ProtectedEvent],
    candidates: Sequence[SelectionCandidate],
) -> tuple[ProtectedEvent, ...]:
    events = tuple(protected_events)
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("protected event_id values must be unique")
    known_ids = {candidate.candidate_id for candidate in candidates}
    unknown = sorted(
        {
            candidate_id
            for event in events
            for candidate_id in event.candidate_ids
            if candidate_id not in known_ids
        }
    )
    if unknown:
        raise ValueError(f"protected events reference unknown candidates: {unknown}")
    return tuple(sorted(events, key=lambda event: (-event.priority, event.event_id)))


def _cap_reached(state: _SelectionState, config: SelectionConfig) -> bool:
    return (
        config.hard_max_keyframes is not None
        and len(state.selected) >= config.hard_max_keyframes
    )


def _select_protected_events(
    state: _SelectionState,
    candidates: Sequence[SelectionCandidate],
    events: Sequence[ProtectedEvent],
    video_duration: float,
    config: SelectionConfig,
) -> None:
    uncovered = {event.event_id for event in events}
    events_by_candidate: dict[str, list[ProtectedEvent]] = {}
    for event in events:
        for candidate_id in event.candidate_ids:
            events_by_candidate.setdefault(candidate_id, []).append(event)

    while uncovered and not _cap_reached(state, config):
        gap_context = _gap_lookahead_context(state.selected.values(), video_duration)
        best_candidate: SelectionCandidate | None = None
        best_covered: tuple[ProtectedEvent, ...] = ()
        best_key: tuple[float, ...] | None = None
        for candidate in candidates:
            if candidate.candidate_id in state.selected:
                continue
            covered = tuple(
                event
                for event in events_by_candidate.get(candidate.candidate_id, ())
                if event.event_id in uncovered
            )
            if not covered:
                continue
            key = (
                float(sum(event.priority + 1 for event in covered)),
                float(len(covered)),
                -_max_gap_after_insertion(candidate.timestamp, gap_context),
                candidate.importance_score,
                -candidate.timestamp,
                -float(candidate.frame_index),
            )
            if best_key is None or key > best_key or (
                key == best_key and candidate.candidate_id < best_candidate.candidate_id
            ):
                best_key = key
                best_candidate = candidate
                best_covered = covered
        if best_candidate is None:
            break
        state.add(
            best_candidate,
            phase=PHASE_PROTECTED,
            reason="protected_event_cover",
            score=float(best_key[0]) if best_key else None,
        )
        for event in best_covered:
            state.reasons[best_candidate.candidate_id].add(f"protected:{event.event_type}")
            uncovered.discard(event.event_id)

    # A selected candidate may cover more events than the ones counted at the
    # moment it was inserted.  Recompute rather than depending on set order.
    for event in events:
        if any(candidate_id in state.selected for candidate_id in event.candidate_ids):
            uncovered.discard(event.event_id)


def _repair_temporal_coverage(
    state: _SelectionState,
    candidates: Sequence[SelectionCandidate],
    video_duration: float,
    config: SelectionConfig,
) -> bool:
    """Fill every mandatory-anchor interval with a minimum-cardinality chain.

    The farthest-reachable greedy rule is optimal for a one-dimensional gap.
    Importance only breaks ties between candidates at the same timestamp.
    """

    additions = _minimal_coverage_additions(
        tuple(state.selected),
        candidates,
        video_duration,
        config,
        allow_partial=True,
        coverage_index=_build_coverage_index(candidates),
    )
    assert additions is not None
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for candidate_id in additions:
        if _cap_reached(state, config):
            return False
        state.add(
            by_id[candidate_id],
            phase=PHASE_COVERAGE,
            reason="temporal_coverage",
        )
    return not _violating_gaps(
        state.selected.values(),
        video_duration,
        config.max_gap_seconds,
        config.gap_tolerance_seconds,
    )


def _minimal_coverage_additions(
    initial_candidate_ids: Sequence[str],
    candidates: Sequence[SelectionCandidate],
    video_duration: float,
    config: SelectionConfig,
    *,
    allow_partial: bool = False,
    coverage_index: _CoverageIndex | None = None,
) -> tuple[str, ...] | None:
    """Return the deterministic minimum chain for fixed mandatory anchors."""

    index = coverage_index or _build_coverage_index(candidates)
    selected_ids = set(initial_candidate_ids)
    anchors = sorted(
        (index.by_id[candidate_id] for candidate_id in selected_ids),
        key=lambda candidate: (
            candidate.timestamp,
            candidate.frame_index,
            candidate.candidate_id,
        ),
    )
    boundaries = [0.0, *(candidate.timestamp for candidate in anchors), video_duration]
    limit = config.max_gap_seconds + config.gap_tolerance_seconds
    epsilon = 1e-12
    additions: list[str] = []

    for start, end in zip(boundaries, boundaries[1:]):
        current = start
        while end - current > limit + epsilon:
            reachable_index = (
                bisect.bisect_right(
                    index.timestamps,
                    current + limit + epsilon,
                )
                - 1
            )
            if (
                reachable_index < 0
                or index.timestamps[reachable_index] <= current + epsilon
            ):
                return tuple(additions) if allow_partial else None
            chosen_id = index.best_candidate_ids[reachable_index]
            additions.append(chosen_id)
            current = index.timestamps[reachable_index]
    return tuple(additions)


def _find_exact_hard_constraint_subset(
    candidates: Sequence[SelectionCandidate],
    events: Sequence[ProtectedEvent],
    video_duration: float,
    config: SelectionConfig,
) -> tuple[tuple[str, ...] | None, bool]:
    cap = config.hard_max_keyframes
    if cap is None:
        return None, False
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    event_sets = [set(event.candidate_ids) for event in events]
    coverage_index = _build_coverage_index(candidates)
    relevant_ids = tuple(
        candidate.candidate_id
        for candidate in candidates
        if any(candidate.candidate_id in event_set for event_set in event_sets)
    )
    if len(relevant_ids) > config.exact_search_candidate_limit:
        return None, False

    best_ids: tuple[str, ...] | None = None
    best_key: tuple[object, ...] | None = None
    max_size = min(cap, len(relevant_ids))
    for size in range(max_size + 1):
        if best_ids is not None and size > len(best_ids):
            break
        for subset in itertools.combinations(relevant_ids, size):
            selected_ids = set(subset)
            if any(not selected_ids.intersection(candidate_ids) for candidate_ids in event_sets):
                continue
            additions = _minimal_coverage_additions(
                subset,
                candidates,
                video_duration,
                config,
                coverage_index=coverage_index,
            )
            if additions is None:
                continue
            combined = tuple(dict.fromkeys((*subset, *additions)))
            if len(combined) > cap:
                continue
            canonical_ids = tuple(sorted(combined))
            key: tuple[object, ...] = (
                len(canonical_ids),
                -sum(by_id[candidate_id].importance_score for candidate_id in canonical_ids),
                canonical_ids,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_ids = canonical_ids
    return best_ids, True


def _state_from_exact_subset(
    candidate_ids: Sequence[str],
    by_id: dict[str, SelectionCandidate],
    events: Sequence[ProtectedEvent],
) -> _SelectionState:
    state = _SelectionState()
    event_ids_by_candidate: dict[str, list[ProtectedEvent]] = {}
    for event in events:
        for candidate_id in event.candidate_ids:
            event_ids_by_candidate.setdefault(candidate_id, []).append(event)

    ordered_ids = sorted(
        candidate_ids,
        key=lambda candidate_id: (
            not bool(event_ids_by_candidate.get(candidate_id)),
            by_id[candidate_id].timestamp,
            by_id[candidate_id].frame_index,
            candidate_id,
        ),
    )
    for candidate_id in ordered_ids:
        candidate = by_id[candidate_id]
        covered = event_ids_by_candidate.get(candidate_id, [])
        phase = PHASE_PROTECTED if covered else PHASE_COVERAGE
        reason = "exact_protected_event_cover" if covered else "exact_temporal_coverage"
        state.add(candidate, phase=phase, reason=reason)
        for event in covered:
            state.reasons[candidate_id].add(f"protected:{event.event_type}")
    return state


def _fill_mmr(
    state: _SelectionState,
    candidates: Sequence[SelectionCandidate],
    config: SelectionConfig,
) -> str:
    target = config.target_keyframes
    if target is None:
        return "target_not_configured"
    if len(state.selected) >= target:
        return "target_reached"

    embedded_candidates = [
        candidate for candidate in candidates if candidate.semantic_embedding
    ]
    embedding_row_by_id = {
        candidate.candidate_id: row
        for row, candidate in enumerate(embedded_candidates)
    }
    embedding_matrix: np.ndarray | None = None
    max_similarity: np.ndarray | None = None
    has_selected_embedding = False
    if embedded_candidates:
        embedding_matrix = np.asarray(
            [candidate.semantic_embedding for candidate in embedded_candidates],
            dtype=np.float64,
        )
        scales = np.max(np.abs(embedding_matrix), axis=1, keepdims=True)
        scaled_embeddings = embedding_matrix / scales
        scaled_norms = np.linalg.norm(scaled_embeddings, axis=1, keepdims=True)
        embedding_matrix = scaled_embeddings / scaled_norms
        max_similarity = np.full(len(embedded_candidates), -np.inf, dtype=np.float64)
        for selected in state.selected.values():
            row = embedding_row_by_id.get(selected.candidate_id)
            if row is None:
                continue
            similarities = np.clip(embedding_matrix @ embedding_matrix[row], -1.0, 1.0)
            if has_selected_embedding:
                np.maximum(max_similarity, similarities, out=max_similarity)
            else:
                max_similarity[:] = similarities
                has_selected_embedding = True

    selected_duplicate_groups = {
        candidate.duplicate_group
        for candidate in state.selected.values()
        if candidate.duplicate_group
    }
    weight_sum = config.importance_weight + config.novelty_weight
    while len(state.selected) < target and not _cap_reached(state, config):
        remaining = [
            candidate
            for candidate in candidates
            if candidate.candidate_id not in state.selected
        ]
        if not remaining:
            return "candidate_pool_exhausted"
        scored: list[tuple[float, SelectionCandidate]] = []
        for candidate in remaining:
            row = embedding_row_by_id.get(candidate.candidate_id)
            if candidate.duplicate_group in selected_duplicate_groups:
                novelty = 0.0
            elif row is None:
                novelty = 0.0
            elif not has_selected_embedding:
                novelty = 1.0
            else:
                assert max_similarity is not None
                novelty = float(np.clip((1.0 - max_similarity[row]) / 2.0, 0.0, 1.0))
            score = (
                config.importance_weight * candidate.importance_score
                + config.novelty_weight * novelty
            ) / weight_sum
            scored.append((score, candidate))
        score, chosen = min(
            scored,
            key=lambda item: (
                -item[0],
                item[1].timestamp,
                item[1].frame_index,
                item[1].candidate_id,
            ),
        )
        state.add(chosen, phase=PHASE_MMR, reason="diversity_mmr", score=score)
        if chosen.duplicate_group:
            selected_duplicate_groups.add(chosen.duplicate_group)
        row = embedding_row_by_id.get(chosen.candidate_id)
        if row is not None:
            assert embedding_matrix is not None
            assert max_similarity is not None
            similarities = np.clip(embedding_matrix @ embedding_matrix[row], -1.0, 1.0)
            if has_selected_embedding:
                np.maximum(max_similarity, similarities, out=max_similarity)
            else:
                max_similarity[:] = similarities
                has_selected_embedding = True

    if len(state.selected) >= target:
        return "target_reached"
    if _cap_reached(state, config):
        return "hard_cap_reached"
    return "candidate_pool_exhausted"


def _gap_lookahead_context(
    selected: Iterable[SelectionCandidate],
    video_duration: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
    times = tuple(
        sorted({0.0, video_duration, *(candidate.timestamp for candidate in selected)})
    )
    gaps = tuple(right - left for left, right in zip(times, times[1:]))
    prefix: list[float] = [0.0]
    for gap in gaps:
        prefix.append(max(prefix[-1], gap))
    suffix = [0.0] * len(times)
    for index in range(len(gaps) - 1, -1, -1):
        suffix[index] = max(suffix[index + 1], gaps[index])
    return times, tuple(prefix), tuple(suffix), max(gaps, default=0.0)


def _max_gap_after_insertion(
    timestamp: float,
    context: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float],
) -> float:
    times, prefix, suffix, current_max = context
    position = bisect.bisect_left(times, timestamp)
    if position < len(times) and times[position] == timestamp:
        return current_max
    if position == 0 or position == len(times):
        return current_max
    gap_index = position - 1
    outside_max = max(prefix[gap_index], suffix[gap_index + 1])
    split_max = max(timestamp - times[gap_index], times[position] - timestamp)
    return max(outside_max, split_max)


def _temporal_gaps(
    selected: Iterable[SelectionCandidate],
    video_duration: float,
) -> tuple[TemporalGap, ...]:
    ordered = sorted(
        selected,
        key=lambda candidate: (candidate.timestamp, candidate.frame_index, candidate.candidate_id),
    )
    if not ordered:
        return (TemporalGap(0.0, video_duration),) if video_duration > 0 else ()
    gaps: list[TemporalGap] = [
        TemporalGap(0.0, ordered[0].timestamp, None, ordered[0].candidate_id)
    ]
    gaps.extend(
        TemporalGap(
            left.timestamp,
            right.timestamp,
            left.candidate_id,
            right.candidate_id,
        )
        for left, right in zip(ordered, ordered[1:])
    )
    gaps.append(
        TemporalGap(
            ordered[-1].timestamp,
            video_duration,
            ordered[-1].candidate_id,
            None,
        )
    )
    return tuple(gaps)


def _violating_gaps(
    selected: Iterable[SelectionCandidate],
    video_duration: float,
    max_gap_seconds: float,
    tolerance: float,
) -> tuple[TemporalGap, ...]:
    limit = max_gap_seconds + tolerance
    return tuple(
        gap
        for gap in _temporal_gaps(selected, video_duration)
        if gap.duration > limit + 1e-12
    )


def _max_gap(selected: Iterable[SelectionCandidate], video_duration: float) -> float:
    return max(
        (gap.duration for gap in _temporal_gaps(selected, video_duration)),
        default=0.0,
    )


def _unsatisfied_event_ids(
    selected: dict[str, SelectionCandidate],
    events: Sequence[ProtectedEvent],
) -> tuple[str, ...]:
    selected_ids = set(selected)
    return tuple(
        event.event_id
        for event in events
        if not selected_ids.intersection(event.candidate_ids)
    )


def _selected_output(
    state: _SelectionState,
    events: Sequence[ProtectedEvent],
) -> tuple[SelectedCandidate, ...]:
    covered_by_candidate: dict[str, list[str]] = {
        candidate_id: [] for candidate_id in state.selected
    }
    for event in events:
        for candidate_id in event.candidate_ids:
            if candidate_id in covered_by_candidate:
                covered_by_candidate[candidate_id].append(event.event_id)
    return tuple(
        SelectedCandidate(
            candidate=candidate,
            selection_rank=state.rank[candidate.candidate_id],
            selection_phase=state.phase[candidate.candidate_id],
            selection_reasons=tuple(sorted(state.reasons[candidate.candidate_id])),
            covered_event_ids=tuple(sorted(covered_by_candidate[candidate.candidate_id])),
            selection_score=state.score[candidate.candidate_id],
        )
        for candidate in sorted(
            state.selected.values(),
            key=lambda value: (value.timestamp, value.frame_index, value.candidate_id),
        )
    )


__all__ = [
    "PHASE_COVERAGE",
    "PHASE_MMR",
    "PHASE_PROTECTED",
    "ProtectedEvent",
    "SelectedCandidate",
    "SelectionCandidate",
    "SelectionConfig",
    "SelectionResult",
    "TemporalGap",
    "build_shot_protection_events",
    "select_keyframes",
]
