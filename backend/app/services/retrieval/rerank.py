"""Phase 3 multimodal reranking for merged retrieval candidates."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.candidate_merger import dedupe_by_same_shot
from backend.app.services.retrieval.query_terms import (
    content_tokens,
    weighted_term_coverage,
)


@dataclass(frozen=True)
class RerankWeights:
    visual: float = 0.55
    caption: float = 0.20
    ocr: float = 0.10
    objects: float = 0.10
    temporal: float = 0.05


@dataclass(frozen=True)
class RerankConfig:
    weights: RerankWeights = RerankWeights()
    dedupe_same_shot: bool = True


class HybridReranker:
    """Combine visual and lexical scores, with metadata overlap as fallback."""

    def __init__(self, config: RerankConfig | None = None) -> None:
        self.config = config or RerankConfig()
        values = self.config.weights.__dict__.values()
        if any(float(value) < 0 for value in values):
            raise ValueError("Retrieval rerank weights must be non-negative")
        if sum(float(value) for value in values) <= 0:
            raise ValueError("At least one retrieval rerank weight must be positive")

    def rerank(
        self,
        query: str,
        candidates: Iterable[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        query_tokens = _tokens(query)
        scored = [
            replace(candidate, score=round(self.score(query_tokens, candidate), 6))
            for candidate in candidates
        ]
        scored.sort(
            key=lambda item: (item.score, item.timestamp_confidence),
            reverse=True,
        )
        if self.config.dedupe_same_shot:
            scored = dedupe_by_same_shot(scored)
        if top_k is None:
            return scored
        return scored[: max(0, int(top_k))]

    def score(
        self,
        query_tokens: set[str],
        candidate: RetrievalResult,
    ) -> float:
        weights = self.config.weights
        modality_scores = candidate.modality_scores
        if "visual" in modality_scores:
            visual = _cosine_to_unit(modality_scores["visual"])
        elif not modality_scores:
            visual = _cosine_to_unit(candidate.score)
        else:
            visual = 0.0

        caption = max(
            _unit(modality_scores.get("caption", 0.0)),
            _metadata_match_score(query_tokens, _tokens(candidate.caption)),
        )
        ocr = max(
            _unit(modality_scores.get("ocr", 0.0)),
            _metadata_match_score(query_tokens, _tokens(candidate.ocr_text)),
        )
        objects = max(
            _unit(modality_scores.get("objects", 0.0)),
            _metadata_match_score(
                query_tokens,
                _tokens(" ".join(candidate.objects)),
            ),
        )
        temporal = _unit(candidate.timestamp_confidence)
        weighted = (
            weights.visual * visual
            + weights.caption * caption
            + weights.ocr * ocr
            + weights.objects * objects
            + weights.temporal * temporal
        )
        total_weight = sum(float(value) for value in weights.__dict__.values())
        return weighted / total_weight


def _tokens(text: str) -> set[str]:
    return set(content_tokens(text, fallback_to_all=True))


def _metadata_match_score(left: set[str], right: set[str]) -> float:
    """Favor coverage of important query terms over incidental token overlap."""
    if not left or not right:
        return 0.0
    coverage = weighted_term_coverage(left, right)
    return 0.80 * coverage + 0.20 * _jaccard(left, right)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cosine_to_unit(value: float) -> float:
    return _unit((float(value) + 1.0) / 2.0)
