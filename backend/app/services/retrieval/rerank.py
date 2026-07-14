"""Phase 3 hybrid reranking for retrieval candidates.

The reranker is intentionally dependency-light: it consumes the metadata already
attached to RetrievalResult objects and can run before OCR/caption/object
pipelines are fully populated. Missing modalities simply contribute zero.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.candidate_merger import dedupe_by_same_shot


_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RerankWeights:
    visual: float = 0.45
    caption: float = 0.25
    ocr: float = 0.15
    objects: float = 0.10
    temporal: float = 0.05


@dataclass(frozen=True)
class RerankConfig:
    weights: RerankWeights = RerankWeights()
    dedupe_same_shot: bool = True


class HybridReranker:
    """Score top visual candidates with lightweight metadata signals."""

    def __init__(self, config: RerankConfig | None = None) -> None:
        self.config = config or RerankConfig()

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
        scored.sort(key=lambda item: (item.score, item.timestamp_confidence), reverse=True)

        if self.config.dedupe_same_shot:
            scored = dedupe_by_same_shot(scored)

        if top_k is None:
            return scored
        return scored[: max(0, int(top_k))]

    def score(self, query_tokens: set[str], candidate: RetrievalResult) -> float:
        weights = self.config.weights
        return (
            weights.visual * _clamp01(candidate.score)
            + weights.caption * _jaccard(query_tokens, _tokens(candidate.caption))
            + weights.ocr * _jaccard(query_tokens, _tokens(candidate.ocr_text))
            + weights.objects * _jaccard(query_tokens, _tokens(" ".join(candidate.objects)))
            + weights.temporal * _clamp01(candidate.timestamp_confidence)
        )


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
