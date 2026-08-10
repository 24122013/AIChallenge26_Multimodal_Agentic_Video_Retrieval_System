"""Rank-only reciprocal-rank fusion over pre-aggregated segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.query_plan import QueryPlan
from backend.app.services.retrieval.query_modality_weights import (
    DEFAULT_WEIGHTED_RRF_WEIGHTS,
)
from backend.app.services.retrieval.segment_aggregation import (
    SegmentEvidence,
    SegmentResolver,
    aggregate_segments,
)


DEFAULT_RRF_WEIGHTS = DEFAULT_WEIGHTED_RRF_WEIGHTS


@dataclass(frozen=True)
class FusedCandidate:
    result: RetrievalResult
    rrf_score: float
    modality_ranks: dict[str, int]
    modality_contributions: dict[str, float]
    segment_identity: tuple[str, str] = ("", "")
    modality_evidence: dict[str, dict[str, object]] | None = None

    @property
    def clip_key(self) -> tuple[str, str]:
        if self.segment_identity[0] and self.segment_identity[1]:
            return self.segment_identity
        clip_id = self.result.segment_id or self.result.shot_id or self.result.frame_id
        return (self.result.video_id, clip_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": f"{self.clip_key[0]}:{self.clip_key[1]}",
            "video_id": self.clip_key[0],
            "segment_id": self.clip_key[1],
            "representative_frame_id": self.result.frame_id,
            "timestamp": self.result.timestamp,
            "rrf_score": self.rrf_score,
            "modality_ranks": dict(self.modality_ranks),
            "modality_contributions": dict(self.modality_contributions),
            "modality_evidence": dict(self.modality_evidence or {}),
        }


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
    segment_resolver: SegmentResolver | None = None,
) -> list[FusedCandidate]:
    if int(k) <= 0:
        raise ValueError("RRF k must be positive")
    base_weights = dict(DEFAULT_RRF_WEIGHTS if weights is None else weights)
    if any(float(value) < 0 for value in base_weights.values()):
        raise ValueError("RRF weights must be non-negative")

    segment_groups = aggregate_segments(groups, resolver=segment_resolver)
    return fuse_segment_ranks(
        segment_groups,
        k=k,
        weights=base_weights,
        plan=plan,
        hint_boost=hint_boost,
    )


def fuse_segment_ranks(
    groups: Mapping[str, Sequence[SegmentEvidence]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    plan: QueryPlan | None = None,
    hint_boost: float = 1.0,
) -> list[FusedCandidate]:
    """Fuse segment ranks without comparing raw scores across modalities."""
    if int(k) <= 0:
        raise ValueError("RRF k must be positive")
    base_weights = dict(DEFAULT_RRF_WEIGHTS if weights is None else weights)
    if any(float(value) < 0 for value in base_weights.values()):
        raise ValueError("RRF weights must be non-negative")
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for modality in sorted(groups):
        weight = float(base_weights.get(modality, 1.0))
        if plan is not None and modality in plan.modality_hints:
            weight *= float(hint_boost)
        if weight <= 0:
            continue
        seen_in_modality: set[tuple[str, str]] = set()
        for evidence in groups[modality]:
            rank = int(evidence.rank)
            result = evidence.representative
            identity = evidence.clip_key
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
                    "evidence": {},
                    "best_rank": rank,
                },
            )
            state["score"] = float(state["score"]) + contribution
            ranks = state["ranks"]
            contributions = state["contributions"]
            evidence_map = state["evidence"]
            assert isinstance(ranks, dict) and isinstance(contributions, dict)
            assert isinstance(evidence_map, dict)
            ranks[modality] = rank
            contributions[modality] = contribution
            evidence_map[modality] = evidence.to_dict()
            if rank < int(state["best_rank"]):
                state["result"] = result
                state["best_rank"] = rank

    fused = [
        FusedCandidate(
            result=state["result"],  # type: ignore[arg-type]
            rrf_score=float(state["score"]),
            modality_ranks=dict(state["ranks"]),  # type: ignore[arg-type]
            modality_contributions=dict(state["contributions"]),  # type: ignore[arg-type]
            segment_identity=identity,
            modality_evidence=dict(state["evidence"]),  # type: ignore[arg-type]
        )
        for identity, state in merged.items()
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
