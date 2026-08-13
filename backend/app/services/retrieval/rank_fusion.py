"""Weighted reciprocal-rank fusion and clip aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.candidate_merger import candidate_identity
from backend.app.services.retrieval.query_plan import QueryPlan


DEFAULT_RRF_WEIGHTS: dict[str, float] = {
    "visual": 0.55,
    "caption": 0.20,
    "ocr": 0.10,
    "objects": 0.10,
}


@dataclass(frozen=True)
class FusedCandidate:
    result: RetrievalResult
    rrf_score: float
    modality_ranks: dict[str, int]
    modality_contributions: dict[str, float]

    @property
    def clip_key(self) -> tuple[str, str]:
        clip_id = self.result.segment_id or self.result.shot_id or self.result.frame_id
        return (self.result.video_id, clip_id)


@dataclass(frozen=True)
class RankedClip:
    video_id: str
    clip_id: str
    score: float
    frames: tuple[FusedCandidate, ...]


def weighted_rrf(
    groups: Mapping[str, Sequence[RetrievalResult]],
    *,
    plan: QueryPlan,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    hint_boost: float = 1.5,
) -> list[FusedCandidate]:
    if int(k) <= 0:
        raise ValueError("RRF k must be positive")
    base_weights = dict(DEFAULT_RRF_WEIGHTS if weights is None else weights)
    if any(float(value) < 0 for value in base_weights.values()):
        raise ValueError("RRF weights must be non-negative")

    merged: dict[tuple[str, str], dict[str, object]] = {}
    for modality in sorted(groups):
        weight = float(base_weights.get(modality, 1.0))
        if modality in plan.modality_hints:
            weight *= float(hint_boost)
        if weight <= 0:
            continue
        seen_in_modality: set[tuple[str, str]] = set()
        for rank, result in enumerate(groups[modality], start=1):
            identity = candidate_identity(result)
            if identity in seen_in_modality:
                continue
            seen_in_modality.add(identity)
            contribution = weight / (int(k) + rank)
            state = merged.setdefault(
                identity,
                {
                    "result": result,
                    "score": 0.0,
                    "ranks": {},
                    "contributions": {},
                    "best_rank": rank,
                },
            )
            state["score"] = float(state["score"]) + contribution
            ranks = state["ranks"]
            contributions = state["contributions"]
            assert isinstance(ranks, dict) and isinstance(contributions, dict)
            ranks[modality] = rank
            contributions[modality] = contribution
            if rank < int(state["best_rank"]):
                state["result"] = result
                state["best_rank"] = rank

    fused = [
        FusedCandidate(
            result=state["result"],  # type: ignore[arg-type]
            rrf_score=float(state["score"]),
            modality_ranks=dict(state["ranks"]),  # type: ignore[arg-type]
            modality_contributions=dict(state["contributions"]),  # type: ignore[arg-type]
        )
        for state in merged.values()
    ]
    fused.sort(
        key=lambda item: (
            -item.rrf_score,
            item.result.video_id,
            item.result.timestamp,
            item.result.frame_id,
        )
    )
    return fused


def aggregate_clips(
    candidates: Sequence[FusedCandidate],
    *,
    top_n: int,
) -> list[RankedClip]:
    grouped: dict[tuple[str, str], list[FusedCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.clip_key, []).append(candidate)
    clips: list[RankedClip] = []
    for (video_id, clip_id), frames in grouped.items():
        ranked_frames = tuple(
            sorted(
                frames,
                key=lambda item: (
                    -item.rrf_score,
                    item.result.timestamp,
                    item.result.frame_id,
                ),
            )
        )
        # A strong first hit dominates, while corroborating modalities/frames
        # can still lift the clip without rewarding very long segments.
        top_scores = [item.rrf_score for item in ranked_frames[:3]]
        score = top_scores[0] + sum(top_scores[1:]) * 0.15
        clips.append(
            RankedClip(
                video_id=video_id,
                clip_id=clip_id,
                score=score,
                frames=ranked_frames,
            )
        )
    clips.sort(key=lambda item: (-item.score, item.video_id, item.clip_id))
    return clips[: max(0, int(top_n))]
