"""Pure metrics for keyframe Phase 5 validation.

These helpers deliberately know nothing about model implementations or files.
They make head/tail temporal coverage and empty-denominator semantics explicit
so evaluation cannot silently turn missing evidence into a perfect score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TemporalCoverageMetrics:
    gaps_seconds: tuple[float, ...]
    coverage_violation_count: int
    max_gap_seconds: float
    p95_gap_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "gaps_seconds": list(self.gaps_seconds),
            "coverage_violation_count": self.coverage_violation_count,
            "max_gap_seconds": self.max_gap_seconds,
            "p95_gap_seconds": self.p95_gap_seconds,
        }


def finite_non_negative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def ratio(numerator: int, denominator: int) -> float | None:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("invalid ratio counts")
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return the deterministic linear percentile used by NumPy defaults."""

    if not values:
        return 0.0
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile_value must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return round(ordered[lower], 6)
    fraction = rank - lower
    return round(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction,
        6,
    )


def temporal_coverage_metrics(
    timestamps: Iterable[object],
    *,
    video_duration: object,
    max_gap_seconds: object,
    tolerance_seconds: object = 0.0,
) -> TemporalCoverageMetrics:
    """Measure intro, internal, and outro gaps independently of selector flags."""

    duration = finite_non_negative(video_duration, "video_duration")
    max_gap = finite_non_negative(max_gap_seconds, "max_gap_seconds")
    tolerance = finite_non_negative(tolerance_seconds, "tolerance_seconds")
    ordered = sorted(
        {
            finite_non_negative(value, f"timestamps[{position}]")
            for position, value in enumerate(timestamps)
        }
    )
    if any(value > duration + 1e-9 for value in ordered):
        raise ValueError("keyframe timestamp exceeds video duration")
    anchors = [0.0, *ordered, duration]
    gaps = tuple(
        round(right - left, 6)
        for left, right in zip(anchors, anchors[1:])
    )
    limit = max_gap + tolerance
    violations = sum(gap > limit + 1e-12 for gap in gaps)
    return TemporalCoverageMetrics(
        gaps_seconds=gaps,
        coverage_violation_count=violations,
        max_gap_seconds=round(max(gaps, default=0.0), 6),
        p95_gap_seconds=percentile(gaps, 95.0),
    )


def retrieval_hit_at_k(
    ranked_hits: Iterable[Sequence[bool]],
    *,
    k_values: Sequence[int] = (1, 5, 10, 100),
) -> dict[str, float | None]:
    """Aggregate binary relevance vectors without inventing missing queries."""

    rows = [tuple(bool(value) for value in row) for row in ranked_hits]
    values: dict[str, float | None] = {}
    for raw_k in k_values:
        if isinstance(raw_k, bool) or int(raw_k) != raw_k or raw_k <= 0:
            raise ValueError("k_values must contain positive integers")
        k = int(raw_k)
        hits = sum(any(row[:k]) for row in rows)
        values[f"hit_at_{k}"] = ratio(hits, len(rows))
    return values


__all__ = [
    "TemporalCoverageMetrics",
    "finite_non_negative",
    "percentile",
    "ratio",
    "retrieval_hit_at_k",
    "temporal_coverage_metrics",
]
