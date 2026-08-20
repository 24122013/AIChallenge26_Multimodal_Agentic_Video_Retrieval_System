"""Deterministic, model-free keyframe candidate generation.

Shot frame ranges are inclusive.  This module deliberately performs no image
decoding or feature inference so candidate generation can be tested in
isolation and reused by later selection stages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable, Protocol


DEFAULT_INTERVAL_SEC = 0.5
DEFAULT_BOUNDARY_GUARD_SEC = 0.2
DEFAULT_TINY_SHOT_MAX_SEC = 0.5

REASON_DENSE_INTERVAL = "dense_interval"
REASON_SHOT_BOUNDARY_START = "shot_boundary_start"
REASON_SHOT_BOUNDARY_END = "shot_boundary_end"
REASON_TINY_SHOT_MIDPOINT = "tiny_shot_midpoint"
REASON_VIDEO_START = "video_start"
REASON_VIDEO_END = "video_end"

_REASON_ORDER = {
    REASON_TINY_SHOT_MIDPOINT: 0,
    REASON_DENSE_INTERVAL: 1,
    REASON_SHOT_BOUNDARY_START: 2,
    REASON_SHOT_BOUNDARY_END: 3,
    REASON_VIDEO_START: 4,
    REASON_VIDEO_END: 5,
}


class ShotLike(Protocol):
    """Minimum shot contract required by :func:`generate_keyframe_candidates`."""

    shot_index: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class KeyframeCandidate:
    """A unique frame candidate and all deterministic reasons for keeping it."""

    candidate_id: str
    video_id: str
    shot_index: int
    frame_index: int
    timestamp_sec: float
    shot_start_sec: float
    shot_end_sec: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _ValidatedShot:
    shot_index: int
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class _CandidateAccumulator:
    shot: _ValidatedShot
    reasons: set[str]


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_shots(
    shots: Iterable[ShotLike],
    *,
    frame_count: int | None,
) -> list[_ValidatedShot]:
    if frame_count is not None:
        frame_count = _integer(frame_count, "frame_count")
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than zero")

    validated: list[_ValidatedShot] = []
    for position, shot in enumerate(shots):
        try:
            shot_index_value = shot.shot_index
            start_frame_value = shot.start_frame
            end_frame_value = shot.end_frame
        except AttributeError as exc:
            raise TypeError(
                "each shot must expose shot_index, start_frame, and end_frame"
            ) from exc

        shot_index = _integer(shot_index_value, f"shots[{position}].shot_index")
        start_frame = _integer(start_frame_value, f"shots[{position}].start_frame")
        end_frame = _integer(end_frame_value, f"shots[{position}].end_frame")
        if shot_index < 0:
            raise ValueError(f"shots[{position}].shot_index must be non-negative")
        if start_frame < 0:
            raise ValueError(f"shots[{position}].start_frame must be non-negative")
        if end_frame < start_frame:
            raise ValueError(
                f"shots[{position}].end_frame must be greater than or equal to start_frame"
            )
        if frame_count is not None and end_frame >= frame_count:
            raise ValueError(
                f"shots[{position}].end_frame must be smaller than frame_count"
            )
        validated.append(
            _ValidatedShot(
                shot_index=shot_index,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )

    validated.sort(key=lambda shot: (shot.start_frame, shot.end_frame, shot.shot_index))
    seen_indices: set[int] = set()
    previous: _ValidatedShot | None = None
    for shot in validated:
        if shot.shot_index in seen_indices:
            raise ValueError(f"duplicate shot_index: {shot.shot_index}")
        if previous is not None and shot.start_frame <= previous.end_frame:
            raise ValueError(
                "shot frame ranges must not overlap: "
                f"shot {previous.shot_index} ends at {previous.end_frame}, "
                f"shot {shot.shot_index} starts at {shot.start_frame}"
            )
        seen_indices.add(shot.shot_index)
        previous = shot
    return validated


def _dense_frame_indices(
    shot: _ValidatedShot,
    *,
    fps: float,
    interval_sec: float,
) -> list[int]:
    step_frames = interval_sec * fps
    if step_frames <= 1.0:
        # Sampling at least once per frame cannot produce more unique indices.
        return list(range(shot.start_frame, shot.end_frame + 1))

    span_frames = shot.end_frame - shot.start_frame
    sample_count = int(math.floor((span_frames / step_frames) + 1e-12)) + 1
    return [
        shot.start_frame + int(round(sample_number * step_frames))
        for sample_number in range(sample_count)
    ]


def _candidate_id(video_id: str, frame_index: int) -> str:
    return f"CANDIDATE_{video_id}_{frame_index:09d}"


def generate_keyframe_candidates(
    video_id: str,
    shots: Iterable[ShotLike],
    fps: float,
    *,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    boundary_guard_sec: float = DEFAULT_BOUNDARY_GUARD_SEC,
    tiny_shot_max_sec: float = DEFAULT_TINY_SHOT_MAX_SEC,
    frame_count: int | None = None,
    include_video_endpoints: bool = False,
) -> list[KeyframeCandidate]:
    """Generate dense candidates plus boundary anchors for validated shots.

    Shots no longer than ``tiny_shot_max_sec`` (or too short to fit distinct
    guarded anchors) contribute one midpoint candidate.  When dense samples and
    anchors round to the same frame, the frame appears once and carries every
    applicable reason in a canonical order.
    """

    if not isinstance(video_id, str):
        raise TypeError("video_id must be a string")
    if not video_id.strip():
        raise ValueError("video_id must not be empty")

    fps = _finite_real(fps, "fps")
    interval_sec = _finite_real(interval_sec, "interval_sec")
    boundary_guard_sec = _finite_real(boundary_guard_sec, "boundary_guard_sec")
    tiny_shot_max_sec = _finite_real(tiny_shot_max_sec, "tiny_shot_max_sec")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be greater than zero")
    if boundary_guard_sec < 0:
        raise ValueError("boundary_guard_sec must be non-negative")
    if tiny_shot_max_sec < 0:
        raise ValueError("tiny_shot_max_sec must be non-negative")
    if not isinstance(include_video_endpoints, bool):
        raise TypeError("include_video_endpoints must be a boolean")

    validated_shots = _validate_shots(shots, frame_count=frame_count)
    by_frame: dict[int, _CandidateAccumulator] = {}

    def add_candidate(shot: _ValidatedShot, frame_index: int, reason: str) -> None:
        frame_index = max(shot.start_frame, min(frame_index, shot.end_frame))
        existing = by_frame.get(frame_index)
        if existing is None:
            by_frame[frame_index] = _CandidateAccumulator(shot=shot, reasons={reason})
            return
        # Overlapping shots are rejected above, so a duplicate can only be a
        # rounded sampling/anchor collision inside the same shot.
        existing.reasons.add(reason)

    guard_frames = int(round(boundary_guard_sec * fps))
    for shot in validated_shots:
        duration_sec = shot.frame_count / fps
        start_anchor = min(shot.end_frame, shot.start_frame + guard_frames)
        end_anchor = max(shot.start_frame, shot.end_frame - guard_frames)
        is_tiny = (
            duration_sec <= tiny_shot_max_sec + 1e-12
            or start_anchor >= end_anchor
        )
        if is_tiny:
            midpoint = shot.start_frame + ((shot.end_frame - shot.start_frame) // 2)
            add_candidate(shot, midpoint, REASON_TINY_SHOT_MIDPOINT)
            continue

        for frame_index in _dense_frame_indices(
            shot,
            fps=fps,
            interval_sec=interval_sec,
        ):
            add_candidate(shot, frame_index, REASON_DENSE_INTERVAL)
        add_candidate(shot, start_anchor, REASON_SHOT_BOUNDARY_START)
        add_candidate(shot, end_anchor, REASON_SHOT_BOUNDARY_END)

    if include_video_endpoints and validated_shots:
        add_candidate(
            validated_shots[0],
            validated_shots[0].start_frame,
            REASON_VIDEO_START,
        )
        add_candidate(
            validated_shots[-1],
            validated_shots[-1].end_frame,
            REASON_VIDEO_END,
        )

    candidates: list[KeyframeCandidate] = []
    for frame_index in sorted(by_frame):
        accumulator = by_frame[frame_index]
        shot = accumulator.shot
        candidates.append(
            KeyframeCandidate(
                candidate_id=_candidate_id(video_id, frame_index),
                video_id=video_id,
                shot_index=shot.shot_index,
                frame_index=frame_index,
                timestamp_sec=frame_index / fps,
                shot_start_sec=shot.start_frame / fps,
                shot_end_sec=(shot.end_frame + 1) / fps,
                reasons=tuple(
                    sorted(
                        accumulator.reasons,
                        key=lambda reason: (_REASON_ORDER.get(reason, len(_REASON_ORDER)), reason),
                    )
                ),
            )
        )
    return candidates


__all__ = [
    "DEFAULT_BOUNDARY_GUARD_SEC",
    "DEFAULT_INTERVAL_SEC",
    "DEFAULT_TINY_SHOT_MAX_SEC",
    "KeyframeCandidate",
    "REASON_DENSE_INTERVAL",
    "REASON_SHOT_BOUNDARY_END",
    "REASON_SHOT_BOUNDARY_START",
    "REASON_TINY_SHOT_MIDPOINT",
    "REASON_VIDEO_END",
    "REASON_VIDEO_START",
    "ShotLike",
    "generate_keyframe_candidates",
]
