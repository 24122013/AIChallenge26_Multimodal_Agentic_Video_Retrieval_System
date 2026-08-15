"""Weighted reciprocal-rank fusion and clip aggregation."""
from __future__ import annotations

from dataclasses import dataclass, replace
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

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": candidate_identity(self.result),
            "rrf_score": self.rrf_score,
            "modality_ranks": dict(self.modality_ranks),
            "modality_contributions": dict(self.modality_contributions),
        }


@dataclass(frozen=True)
class IntraModalityCandidate:
    result: RetrievalResult
    intra_score: float
    original_contribution: float
    raw_expansion_contribution: float
    max_expansion_budget: float
    expansion_contribution: float
    variant_ranks: dict[str, int]
    variant_contributions: dict[str, float]

    def as_retrieval_result(self) -> RetrievalResult:
        return replace(self.result, score=self.intra_score)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": candidate_identity(self.result),
            "original_contribution": self.original_contribution,
            "raw_expansion_contribution": self.raw_expansion_contribution,
            "max_expansion_budget": self.max_expansion_budget,
            "expansion_contribution": self.expansion_contribution,
            "final_intra_score": self.intra_score,
            "variant_ranks": dict(self.variant_ranks),
            "variant_contributions": dict(self.variant_contributions),
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


def fuse_query_variants(
    groups: Mapping[str, Sequence[RetrievalResult]],
    *,
    weights: Mapping[str, float],
    original_key: str = "original",
    k: int = 60,
    max_expansion_contribution: float = 1.0,
) -> list[IntraModalityCandidate]:
    """Fuse semantic variants inside one modality using capped weighted RRF.

    ``max_expansion_contribution`` is a ratio against the maximum rank-1
    contribution of the original query, never a raw-score cap.
    """
    if int(k) <= 0:
        raise ValueError("RRF k must be positive")
    if max_expansion_contribution <= 0:
        raise ValueError("max_expansion_contribution must be positive")
    if original_key not in groups or original_key not in weights:
        raise ValueError("intra-modality fusion requires the original ranked list")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("variant RRF weights must be non-negative")
    original_weight = float(weights[original_key])
    if any(float(value) > original_weight for value in weights.values()):
        raise ValueError("original query must have the highest variant weight")
    max_budget = (
        float(max_expansion_contribution) * original_weight / (int(k) + 1)
    )

    merged: dict[tuple[str, str], dict[str, object]] = {}
    for variant, results in groups.items():
        weight = float(weights.get(variant, 0.0))
        if weight <= 0:
            continue
        seen: set[tuple[str, str]] = set()
        for rank, result in enumerate(results, start=1):
            identity = candidate_identity(result)
            if identity in seen:
                continue
            seen.add(identity)
            contribution = weight / (int(k) + rank)
            state = merged.setdefault(
                identity,
                {
                    "result": result,
                    "best_rank": rank,
                    "original": 0.0,
                    "expansion": 0.0,
                    "ranks": {},
                    "contributions": {},
                },
            )
            ranks = state["ranks"]
            contributions = state["contributions"]
            assert isinstance(ranks, dict) and isinstance(contributions, dict)
            ranks[variant] = rank
            contributions[variant] = contribution
            if variant == original_key:
                state["original"] = contribution
                state["result"] = result
                state["best_rank"] = rank
            else:
                state["expansion"] = float(state["expansion"]) + contribution
                if float(state["original"]) == 0.0 and rank < int(state["best_rank"]):
                    state["result"] = result
                    state["best_rank"] = rank

    fused: list[IntraModalityCandidate] = []
    for state in merged.values():
        original = float(state["original"])
        raw_expansion = float(state["expansion"])
        capped = min(raw_expansion, max_budget)
        fused.append(
            IntraModalityCandidate(
                result=state["result"],  # type: ignore[arg-type]
                intra_score=original + capped,
                original_contribution=original,
                raw_expansion_contribution=raw_expansion,
                max_expansion_budget=max_budget,
                expansion_contribution=capped,
                variant_ranks=dict(state["ranks"]),  # type: ignore[arg-type]
                variant_contributions=dict(state["contributions"]),  # type: ignore[arg-type]
            )
        )
    fused.sort(
        key=lambda item: (
            -item.intra_score,
            -item.original_contribution,
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
