"""Deterministic K-best chronological alignment for sparse TRAKE candidates."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from backend.app.services.retrieval.retrieval_config import TrakeConfig
from backend.app.services.trake.models import EventCandidate, TemporalPath, VideoCandidate


@dataclass(frozen=True)
class _BeamState:
    candidates: tuple[EventCandidate, ...]
    partial_score: float


def align_candidate_video(
    video: VideoCandidate,
    *,
    event_count: int | None = None,
    config: TrakeConfig | None = None,
) -> list[TemporalPath]:
    """Generate K-best complete paths for one gated video.

    Original ``frame_index`` is the sole ordering identity.  Timestamp is never
    converted into a frame number and there is no maximum-gap rejection.
    """

    runtime = config or TrakeConfig()
    if event_count is None:
        event_count = max(video.event_candidates, default=-1) + 1
    if event_count <= 0:
        return []

    ordered: list[tuple[EventCandidate, ...]] = []
    for event_index in range(event_count):
        values = [
            item
            for item in video.event_candidates.get(event_index, ())
            if item.event_index == event_index
            and item.result.video_id == video.video_id
            and isinstance(item.result.frame_index, int)
            and not isinstance(item.result.frame_index, bool)
            and item.result.frame_index >= 0
        ]
        values.sort(
            key=lambda item: (
                int(item.result.frame_index),
                -float(item.normalized_score),
                item.result.frame_id,
            )
        )
        if not values:
            return []
        ordered.append(tuple(values))

    states = _build_beam(
        ordered,
        config=runtime,
        allow_equal_frame=False,
    )
    fallback = False
    if not states:
        states = _build_beam(
            ordered,
            config=runtime,
            allow_equal_frame=True,
        )
        fallback = bool(states)

    paths = [
        _to_temporal_path(
            video,
            state.candidates,
            config=runtime,
            warnings=("alignment_equal_frame_fallback",) if fallback else (),
        )
        for state in states
        if len(state.candidates) == event_count
    ]
    paths.sort(key=_path_sort_key)
    return paths[: runtime.k_best_paths_per_video]


def align_candidate_videos(
    videos: Iterable[VideoCandidate],
    *,
    event_count: int,
    config: TrakeConfig | None = None,
) -> list[TemporalPath]:
    runtime = config or TrakeConfig()
    paths: list[TemporalPath] = []
    for video in videos:
        paths.extend(
            align_candidate_video(
                video,
                event_count=event_count,
                config=runtime,
            )
        )
    paths.sort(key=_path_sort_key)
    return paths


def align_ordered_events(
    event_candidates: Mapping[int, Sequence[EventCandidate]],
    video_candidate: VideoCandidate,
    *,
    config: TrakeConfig | None = None,
) -> list[TemporalPath]:
    """Compatibility adapter accepting a separate candidate mapping."""

    enriched = VideoCandidate(
        video_id=video_candidate.video_id,
        coverage=video_candidate.coverage,
        event_support=video_candidate.event_support,
        context_score=video_candidate.context_score,
        total_score=video_candidate.total_score,
        event_candidates={
            index: tuple(
                item
                for item in values
                if item.result.video_id == video_candidate.video_id
            )
            for index, values in event_candidates.items()
        },
        warnings=video_candidate.warnings,
    )
    return align_candidate_video(
        enriched,
        event_count=max(event_candidates, default=-1) + 1,
        config=config,
    )


class TemporalAligner:
    def __init__(self, config: TrakeConfig | None = None) -> None:
        self.config = config or TrakeConfig()

    def align(
        self,
        videos: Iterable[VideoCandidate],
        *,
        event_count: int,
    ) -> list[TemporalPath]:
        return align_candidate_videos(
            videos,
            event_count=event_count,
            config=self.config,
        )


def score_path(
    candidates: Sequence[EventCandidate],
    video: VideoCandidate,
    *,
    config: TrakeConfig | None = None,
) -> tuple[float, dict[str, object]]:
    runtime = config or TrakeConfig()
    if not candidates:
        return 0.0, {
            "event_scores": [],
            "mean_event_score": 0.0,
            "video_score": float(video.total_score),
            "coverage": float(video.coverage),
            "gap_penalty": 0.0,
            "duplicate_location_penalty": 0.0,
        }
    event_scores = [float(item.normalized_score) for item in candidates]
    mean_event = sum(event_scores) / len(event_scores)
    gap_penalty = _gap_penalty(candidates, runtime)
    duplicate_penalty = _duplicate_location_penalty(candidates)
    base_score = 0.70 * mean_event + 0.30 * float(video.total_score)
    final = max(0.0, base_score - gap_penalty - duplicate_penalty)
    return round(final, 6), {
        "event_scores": [round(value, 6) for value in event_scores],
        "mean_event_score": round(mean_event, 6),
        "video_score": round(float(video.total_score), 6),
        "coverage": round(float(video.coverage), 6),
        "context_score": round(float(video.context_score), 6),
        "base_score": round(base_score, 6),
        "gap_penalty": round(gap_penalty, 6),
        "gap_penalty_method": runtime.gap_penalty,
        "gap_units": "original_frames",
        "duplicate_location_penalty": round(duplicate_penalty, 6),
    }


def _build_beam(
    by_event: Sequence[Sequence[EventCandidate]],
    *,
    config: TrakeConfig,
    allow_equal_frame: bool,
) -> list[_BeamState]:
    states = [
        _BeamState(candidates=(candidate,), partial_score=candidate.normalized_score)
        for candidate in by_event[0]
        if _can_complete(
            int(candidate.result.frame_index),
            by_event[1:],
            allow_equal_frame=allow_equal_frame,
        )
    ]
    states = _prune(states, config.beam_width, config)
    for event_offset, candidates in enumerate(by_event[1:], start=1):
        expanded: list[_BeamState] = []
        for state in states:
            previous_frame = int(state.candidates[-1].result.frame_index)
            for candidate in candidates:
                current_frame = int(candidate.result.frame_index)
                ordered = (
                    current_frame >= previous_frame
                    if allow_equal_frame
                    else current_frame > previous_frame
                )
                if not ordered:
                    continue
                if not _can_complete(
                    current_frame,
                    by_event[event_offset + 1 :],
                    allow_equal_frame=allow_equal_frame,
                ):
                    continue
                chain = (*state.candidates, candidate)
                expanded.append(
                    _BeamState(
                        candidates=chain,
                        partial_score=_partial_score(chain, config),
                    )
                )
        if not expanded:
            return []
        states = _prune(expanded, config.beam_width, config)
    states.sort(key=_state_sort_key)
    return states[: config.k_best_paths_per_video]


def _can_complete(
    current_frame: int,
    remaining_events: Sequence[Sequence[EventCandidate]],
    *,
    allow_equal_frame: bool,
) -> bool:
    """Greedily prove chronological reachability through all future events.

    Candidate lists are frame-sorted, so choosing the earliest compatible frame
    leaves at least as much room as every alternative.  Filtering unreachable
    states before beam pruning prevents high-scoring dead ends from evicting all
    valid partial paths.
    """

    previous = current_frame
    for candidates in remaining_events:
        next_frame: int | None = None
        for candidate in candidates:
            frame = int(candidate.result.frame_index)
            compatible = (
                frame >= previous if allow_equal_frame else frame > previous
            )
            if compatible:
                next_frame = frame
                break
        if next_frame is None:
            return False
        previous = next_frame
    return True


def _partial_score(
    candidates: Sequence[EventCandidate],
    config: TrakeConfig,
) -> float:
    mean_score = sum(item.normalized_score for item in candidates) / len(candidates)
    return mean_score - _gap_penalty(candidates, config) - _duplicate_location_penalty(candidates)


def _prune(
    states: list[_BeamState],
    width: int,
    config: TrakeConfig,
) -> list[_BeamState]:
    # ``dp`` and ``beam`` share the same bounded K-best recurrence.  DP retains
    # the best state per complete frame tuple before the global width cap.
    if config.alignment_method == "dp":
        best: dict[tuple[int, ...], _BeamState] = {}
        for state in states:
            key = tuple(int(item.result.frame_index) for item in state.candidates)
            previous = best.get(key)
            if previous is None or _state_sort_key(state) < _state_sort_key(previous):
                best[key] = state
        states = list(best.values())
    states.sort(key=_state_sort_key)
    return states[: max(1, int(width))]


def _gap_penalty(
    candidates: Sequence[EventCandidate],
    config: TrakeConfig,
) -> float:
    if len(candidates) < 2 or config.gap_penalty == "none" or config.gap_lambda == 0:
        return 0.0
    frames = [int(item.result.frame_index) for item in candidates]
    gaps = [max(0, right - left) for left, right in zip(frames, frames[1:])]
    if config.gap_penalty == "linear":
        value = config.gap_lambda * (sum(gaps) / len(gaps)) / 1000.0
    else:
        value = config.gap_lambda * (
            sum(math.log1p(gap) for gap in gaps) / len(gaps)
        )
    return min(0.35, max(0.0, float(value)))


def _duplicate_location_penalty(candidates: Sequence[EventCandidate]) -> float:
    penalty = 0.0
    for left, right in zip(candidates, candidates[1:]):
        left_frame = int(left.result.frame_index)
        right_frame = int(right.result.frame_index)
        if left_frame == right_frame:
            penalty += 0.20
        elif abs(right_frame - left_frame) <= 2:
            penalty += 0.02
        if (
            left.result.shot_id
            and left.result.shot_id == right.result.shot_id
        ):
            penalty += 0.01
    return min(0.40, penalty)


def _to_temporal_path(
    video: VideoCandidate,
    candidates: tuple[EventCandidate, ...],
    *,
    config: TrakeConfig,
    warnings: tuple[str, ...],
) -> TemporalPath:
    score, breakdown = score_path(candidates, video, config=config)
    frames = tuple(int(item.result.frame_index) for item in candidates)
    digest = hashlib.sha256(
        (video.video_id + "\0" + "\0".join(str(frame) for frame in frames)).encode("utf-8")
    ).hexdigest()[:16]
    return TemporalPath(
        video_id=video.video_id,
        event_candidates=candidates,
        score=score,
        score_breakdown=breakdown,
        path_id=f"TRP-{digest}",
        warnings=tuple(dict.fromkeys((*video.warnings, *warnings))),
    )


def _state_sort_key(state: _BeamState) -> tuple[object, ...]:
    frames = tuple(int(item.result.frame_index) for item in state.candidates)
    internal_ids = tuple(item.result.frame_id for item in state.candidates)
    return (-float(state.partial_score), frames, internal_ids)


def _path_sort_key(path: TemporalPath) -> tuple[object, ...]:
    return (-float(path.score), tuple(path.frame_ids), path.video_id, path.path_id)


__all__ = [
    "TemporalAligner",
    "align_candidate_video",
    "align_candidate_videos",
    "align_ordered_events",
    "score_path",
]
