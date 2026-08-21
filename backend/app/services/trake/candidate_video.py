"""Video-level gating for TRAKE sequence alignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from backend.app.services.retrieval.retrieval_config import TrakeConfig
from backend.app.services.trake.models import EventCandidate, VideoCandidate


@dataclass(frozen=True)
class VideoGatingResult:
    videos: tuple[VideoCandidate, ...]
    trace: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "videos": [video.to_dict() for video in self.videos],
            "trace": dict(self.trace),
            "warnings": list(self.warnings),
        }


def gate_candidate_videos(
    event_candidates: Mapping[int, Sequence[EventCandidate]],
    *,
    event_count: int | None = None,
    context_scores: Mapping[str, float] | None = None,
    config: TrakeConfig | None = None,
) -> VideoGatingResult:
    """Rank videos with coverage as the primary correctness signal.

    When at least one video covers every event, incomplete videos are excluded
    from alignment.  If none does, the best incomplete videos remain visible in
    the trace, but alignment still fails closed instead of inventing events.
    """

    runtime = config or TrakeConfig()
    if event_count is None:
        event_count = max(event_candidates, default=-1) + 1
    if event_count <= 0:
        return VideoGatingResult(
            videos=(),
            trace={"event_count": 0, "video_count": 0, "full_coverage_count": 0},
            warnings=("no_events_for_video_gating",),
        )

    grouped: dict[str, dict[int, list[EventCandidate]]] = {}
    for event_index in range(event_count):
        for candidate in event_candidates.get(event_index, ()):
            if candidate.event_index != event_index:
                continue
            video_id = str(candidate.result.video_id).strip()
            if not video_id:
                continue
            grouped.setdefault(video_id, {}).setdefault(event_index, []).append(candidate)

    context = dict(context_scores or {})
    weight_total = (
        runtime.coverage_weight
        + runtime.event_support_weight
        + runtime.context_weight
    )
    ranked: list[VideoCandidate] = []
    for video_id, by_event in grouped.items():
        supported = sorted(
            index
            for index, values in by_event.items()
            if any(
                item.normalized_score >= runtime.minimum_per_event_support
                and float(item.result.score) >= runtime.minimum_semantic_support
                and _has_absolute_semantic_support(item, runtime)
                for item in values
            )
        )
        coverage = len(supported) / event_count
        per_event_best = [
            max(
                (
                    item.normalized_score
                    for item in by_event.get(index, ())
                    if float(item.result.score) >= runtime.minimum_semantic_support
                    and _has_absolute_semantic_support(item, runtime)
                ),
                default=0.0,
            )
            for index in range(event_count)
        ]
        event_support = sum(per_event_best) / event_count
        context_score = max(0.0, min(1.0, float(context.get(video_id, 0.0))))
        total_score = (
            runtime.coverage_weight * coverage
            + runtime.event_support_weight * event_support
            + runtime.context_weight * context_score
        ) / weight_total
        ranked.append(
            VideoCandidate(
                video_id=video_id,
                coverage=round(coverage, 6),
                event_support=round(event_support, 6),
                context_score=round(context_score, 6),
                total_score=round(total_score, 6),
                event_candidates={
                    index: tuple(
                        sorted(
                            values,
                            key=lambda item: (
                                -item.normalized_score,
                                int(item.result.frame_index),
                                item.result.frame_id,
                            ),
                        )
                    )
                    for index, values in sorted(by_event.items())
                },
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.coverage,
            -item.total_score,
            -item.event_support,
            -item.context_score,
            item.video_id,
        )
    )
    full = [
        item for item in ranked
        if item.coverage >= runtime.minimum_sequence_coverage
        and (not runtime.complete_event_required or item.coverage >= 1.0)
    ]
    warnings: list[str] = []
    alignment_pool = full
    if not full and ranked:
        warnings.extend(("no_video_supports_all_events", "insufficient_event_support"))
    selected = tuple(alignment_pool[: runtime.top_videos])
    return VideoGatingResult(
        videos=selected,
        trace={
            "event_count": event_count,
            "video_count": len(ranked),
            "full_coverage_count": len(full),
            "selected_video_count": len(selected),
            "fallback_used": bool(ranked and not full),
            "support_policy": {
                "minimum_per_event_support": runtime.minimum_per_event_support,
                "minimum_sequence_coverage": runtime.minimum_sequence_coverage,
                "minimum_semantic_support": runtime.minimum_semantic_support,
                "minimum_visual_support": runtime.minimum_visual_support,
                "minimum_dense_support": runtime.minimum_dense_support,
                "complete_event_required": runtime.complete_event_required,
            },
            "video_scores": [
                {
                    "video_id": item.video_id,
                    "coverage": item.coverage,
                    "event_support": item.event_support,
                    "context_score": item.context_score,
                    "total_score": item.total_score,
                }
                for item in ranked[: runtime.top_videos]
            ],
        },
        warnings=tuple(warnings),
    )


class CandidateVideoRanker:
    """Small OO adapter around :func:`gate_candidate_videos`."""

    def __init__(self, config: TrakeConfig | None = None) -> None:
        self.config = config or TrakeConfig()

    def rank(
        self,
        event_candidates: Mapping[int, Sequence[EventCandidate]],
        *,
        event_count: int | None = None,
        context_scores: Mapping[str, float] | None = None,
    ) -> VideoGatingResult:
        return gate_candidate_videos(
            event_candidates,
            event_count=event_count,
            context_scores=context_scores,
            config=self.config,
        )


rank_candidate_videos = gate_candidate_videos


def _has_absolute_semantic_support(
    candidate: EventCandidate,
    config: TrakeConfig,
) -> bool:
    """Reject rank-only support using pre-RRF modality similarities.

    RRF and normalized alignment scores are intentionally relative and always
    produce a winner.  At least one absolute visual or BGE dense signal must
    cross its independently configurable floor before that winner is evidence.
    """

    scores = candidate.result.modality_scores
    visual = scores.get("visual")
    dense = scores.get(
        "dense_text",
        scores.get("bge_dense", scores.get("trake_bge_dense_raw")),
    )
    visual_ok = visual is not None and float(visual) >= config.minimum_visual_support
    dense_ok = dense is not None and float(dense) >= config.minimum_dense_support
    if visual is None and dense is None:
        # Lightweight/custom engines expose only the canonical result score.
        return float(candidate.result.score) >= config.minimum_semantic_support
    return visual_ok or dense_ok


__all__ = [
    "CandidateVideoRanker",
    "VideoGatingResult",
    "gate_candidate_videos",
    "rank_candidate_videos",
]
