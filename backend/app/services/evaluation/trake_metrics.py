"""Pure, dependency-free metrics for ranked TRAKE sequence hypotheses.

The scorer mirrors the preliminary-round contract: a wrong video receives
zero, while a correct video receives one hit per event whose original frame
index falls inside its inclusive ground-truth interval.  Validation is kept in
this module so malformed benchmark data cannot silently improve a score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence


TRAKE_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)
TRAKE_VIDEO_CUTOFFS: tuple[int, ...] = (1, 5, 20)
MAX_TRAKE_HYPOTHESES = 100


@dataclass(frozen=True)
class TrakePrediction:
    """Validated ranked-answer unit used by the pure scorer."""

    video_id: str
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrakeGroundTruth:
    """One correct video and one inclusive original-frame interval per event."""

    video_id: str
    intervals: tuple[tuple[int, int], ...]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    if is_dataclass(value):
        converted = asdict(value)
        if isinstance(converted, Mapping):
            return converted
    raise TypeError(f"{name} must be a mapping or expose to_dict()")


def _video_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return value


def validate_trake_prediction(
    prediction: Any,
    *,
    event_count: int | None = None,
    name: str = "prediction",
) -> TrakePrediction:
    """Validate one sequence without accepting timestamps or internal IDs."""

    if isinstance(prediction, TrakePrediction):
        validated = prediction
        # Revalidate constructor input because type hints are not runtime guards.
        video_id = _video_id(validated.video_id, name=f"{name}.video_id")
        raw_frame_ids: Sequence[Any] = validated.frame_ids
    else:
        payload = _mapping(prediction, name=name)
        video_id = _video_id(payload.get("video_id"), name=f"{name}.video_id")
        raw_frame_ids = _sequence(payload.get("frame_ids"), name=f"{name}.frame_ids")
    if not raw_frame_ids:
        raise ValueError(f"{name}.frame_ids must contain at least one event")
    frame_ids = tuple(
        _non_negative_int(value, name=f"{name}.frame_ids[{position}]")
        for position, value in enumerate(raw_frame_ids)
    )
    if event_count is not None and len(frame_ids) != event_count:
        raise ValueError(
            f"{name} has {len(frame_ids)} events; ground truth has {event_count}"
        )
    return TrakePrediction(video_id=video_id, frame_ids=frame_ids)


def _interval(value: Any, *, name: str) -> tuple[int, int]:
    if isinstance(value, Mapping):
        if "start_frame" in value or "end_frame" in value:
            raw_start = value.get("start_frame")
            raw_end = value.get("end_frame")
        else:
            raw_start = value.get("start")
            raw_end = value.get("end")
    else:
        pair = _sequence(value, name=name)
        if len(pair) != 2:
            raise ValueError(f"{name} must contain exactly [start, end]")
        raw_start, raw_end = pair
    start = _non_negative_int(raw_start, name=f"{name}.start")
    end = _non_negative_int(raw_end, name=f"{name}.end")
    if start > end:
        raise ValueError(f"{name} is reversed: start must be <= end")
    return start, end


def validate_trake_ground_truth(
    ground_truth: Any,
    *,
    name: str = "ground_truth",
) -> TrakeGroundTruth:
    """Validate the official one-video, N-inclusive-interval target."""

    if isinstance(ground_truth, TrakeGroundTruth):
        video_id = _video_id(ground_truth.video_id, name=f"{name}.video_id")
        raw_intervals: Sequence[Any] = ground_truth.intervals
    else:
        payload = _mapping(ground_truth, name=name)
        video_id = _video_id(payload.get("video_id"), name=f"{name}.video_id")
        raw_intervals_value = payload.get("intervals")
        if raw_intervals_value is None:
            # These aliases make the pure scorer convenient for annotation
            # ledgers while keeping one canonical normalized representation.
            raw_intervals_value = payload.get(
                "frame_intervals",
                payload.get("event_intervals"),
            )
        raw_intervals = _sequence(
            raw_intervals_value,
            name=f"{name}.intervals",
        )
    if not raw_intervals:
        raise ValueError(f"{name}.intervals must contain at least one event")
    intervals = tuple(
        _interval(value, name=f"{name}.intervals[{position}]")
        for position, value in enumerate(raw_intervals)
    )
    return TrakeGroundTruth(video_id=video_id, intervals=intervals)


def _ranked_values(predictions: Any) -> Sequence[Any]:
    if isinstance(predictions, Mapping):
        return _sequence(predictions.get("hypotheses"), name="predictions.hypotheses")
    return _sequence(predictions, name="predictions")


def validate_trake_predictions(
    predictions: Any,
    ground_truth: Any,
    *,
    max_hypotheses: int = MAX_TRAKE_HYPOTHESES,
) -> tuple[tuple[TrakePrediction, ...], TrakeGroundTruth]:
    """Validate a ranked list and reject duplicate whole-sequence hypotheses."""

    if (
        isinstance(max_hypotheses, bool)
        or not isinstance(max_hypotheses, Integral)
        or int(max_hypotheses) < 1
    ):
        raise ValueError("max_hypotheses must be a positive integer")
    ranked = _ranked_values(predictions)
    if len(ranked) > int(max_hypotheses):
        raise ValueError(
            f"predictions contains {len(ranked)} hypotheses; maximum is {max_hypotheses}"
        )
    target = validate_trake_ground_truth(ground_truth)
    normalized: list[TrakePrediction] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for position, value in enumerate(ranked):
        prediction = validate_trake_prediction(
            value,
            event_count=len(target.intervals),
            name=f"predictions[{position}]",
        )
        identity = (prediction.video_id, prediction.frame_ids)
        if identity in seen:
            raise ValueError(f"duplicate TRAKE hypothesis at rank {position + 1}")
        seen.add(identity)
        normalized.append(prediction)
    return tuple(normalized), target


def _validated_r_score(
    prediction: TrakePrediction,
    ground_truth: TrakeGroundTruth,
) -> float:
    if prediction.video_id != ground_truth.video_id:
        return 0.0
    hits = sum(
        start <= frame_id <= end
        for frame_id, (start, end) in zip(prediction.frame_ids, ground_truth.intervals)
    )
    return hits / len(ground_truth.intervals)


def trake_r_score(prediction: Any, ground_truth: Any) -> float:
    """Return the official inclusive-interval R-Score for one hypothesis."""

    target = validate_trake_ground_truth(ground_truth)
    normalized = validate_trake_prediction(
        prediction,
        event_count=len(target.intervals),
    )
    return _validated_r_score(normalized, target)


def _cutoff(value: Any, *, name: str = "k") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 1 <= result <= MAX_TRAKE_HYPOTHESES:
        raise ValueError(f"{name} must be between 1 and {MAX_TRAKE_HYPOTHESES}")
    return result


def best_r_at_k(predictions: Any, ground_truth: Any, k: int) -> float:
    """Return the best valid R-Score in the first ``k`` ranked hypotheses."""

    cutoff = _cutoff(k)
    ranked, target = validate_trake_predictions(predictions, ground_truth)
    return max(
        (_validated_r_score(value, target) for value in ranked[:cutoff]),
        default=0.0,
    )


def trake_final_score(
    predictions: Any,
    ground_truth: Any,
    *,
    cutoffs: Sequence[int] = TRAKE_CUTOFFS,
) -> float:
    """Average best R-Score over the five official ranking cutoffs."""

    normalized_cutoffs = tuple(
        _cutoff(value, name=f"cutoffs[{position}]")
        for position, value in enumerate(cutoffs)
    )
    if not normalized_cutoffs:
        raise ValueError("cutoffs must not be empty")
    if len(normalized_cutoffs) != len(set(normalized_cutoffs)):
        raise ValueError("cutoffs must not contain duplicates")
    ranked, target = validate_trake_predictions(predictions, ground_truth)
    scores = [
        max(
            (_validated_r_score(value, target) for value in ranked[:cutoff]),
            default=0.0,
        )
        for cutoff in normalized_cutoffs
    ]
    return sum(scores) / len(scores)


def final_trake_score(
    predictions: Any,
    ground_truth: Any,
    *,
    cutoffs: Sequence[int] = TRAKE_CUTOFFS,
) -> float:
    """Readable alias for :func:`trake_final_score`."""

    return trake_final_score(predictions, ground_truth, cutoffs=cutoffs)


def _event_hits(
    prediction: TrakePrediction | None,
    target: TrakeGroundTruth,
) -> list[float]:
    if prediction is None or prediction.video_id != target.video_id:
        return [0.0 for _ in target.intervals]
    return [
        float(start <= frame_id <= end)
        for frame_id, (start, end) in zip(prediction.frame_ids, target.intervals)
    ]


def trake_metrics_report(
    predictions: Any,
    ground_truth: Any,
    *,
    cutoffs: Sequence[int] = TRAKE_CUTOFFS,
) -> dict[str, Any]:
    """Build one-query diagnostics alongside the official TRAKE metrics.

    ``per_event_hit_rate`` describes the first best-scoring hypothesis within
    the largest requested cutoff.  Consequently its mean equals
    ``matched_event_ratio``; it never combines different events from unrelated
    hypotheses into an artificially perfect sequence.
    """

    normalized_cutoffs = tuple(
        _cutoff(value, name=f"cutoffs[{position}]")
        for position, value in enumerate(cutoffs)
    )
    if not normalized_cutoffs:
        raise ValueError("cutoffs must not be empty")
    if len(normalized_cutoffs) != len(set(normalized_cutoffs)):
        raise ValueError("cutoffs must not contain duplicates")
    ranked, target = validate_trake_predictions(predictions, ground_truth)
    scores = [_validated_r_score(value, target) for value in ranked]

    report: dict[str, Any] = {
        "hypothesis_count": len(ranked),
        "event_count": len(target.intervals),
    }
    r_scores: list[float] = []
    for cutoff in normalized_cutoffs:
        value = max(scores[:cutoff], default=0.0)
        report[f"r_at_{cutoff}"] = value
        r_scores.append(value)
    report["final_score"] = sum(r_scores) / len(r_scores)

    for cutoff in TRAKE_VIDEO_CUTOFFS:
        report[f"video_at_{cutoff}"] = float(
            any(value.video_id == target.video_id for value in ranked[:cutoff])
        )

    largest_cutoff = max(normalized_cutoffs)
    scoped = ranked[:largest_cutoff]
    scoped_scores = scores[:largest_cutoff]
    best_prediction: TrakePrediction | None = None
    if scoped_scores:
        # list.index deliberately selects the earliest rank on a tie.
        best_prediction = scoped[scoped_scores.index(max(scoped_scores))]
    per_event = _event_hits(best_prediction, target)
    report["per_event_hit_rate"] = per_event
    report["matched_event_ratio"] = sum(per_event) / len(per_event)
    return report


# Short, task-local alias for callers that already import this dedicated module.
final_score = trake_final_score


__all__ = [
    "MAX_TRAKE_HYPOTHESES",
    "TRAKE_CUTOFFS",
    "TRAKE_VIDEO_CUTOFFS",
    "TrakeGroundTruth",
    "TrakePrediction",
    "best_r_at_k",
    "final_score",
    "final_trake_score",
    "trake_final_score",
    "trake_metrics_report",
    "trake_r_score",
    "validate_trake_ground_truth",
    "validate_trake_prediction",
    "validate_trake_predictions",
]
