"""Phase 3 hybrid retrieval orchestration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.candidate_merger import merge_candidates
from backend.app.services.retrieval.rerank import HybridReranker
from backend.app.services.retrieval.temporal_search import (
    TemporalMatch,
    decompose_temporal_query,
    match_ordered_events,
)


class VisualSearchEngineLike(Protocol):
    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> VisualSearchResponse:
        ...


class TextSearchEngineLike(Protocol):
    def search_results(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        ...


@dataclass(frozen=True)
class HybridSearchConfig:
    stage1_top_k: int = 200
    text_stage1_top_k: int = 100
    rerank_pool_size: int = 300
    default_top_k: int = 20
    max_top_k: int = 200
    max_gap_seconds: float = 180.0


class HybridSearchEngine:
    """Merge visual and available text candidates before reranking."""

    def __init__(
        self,
        visual_engine: VisualSearchEngineLike,
        text_engines: dict[str, TextSearchEngineLike] | None = None,
        reranker: HybridReranker | None = None,
        config: HybridSearchConfig | None = None,
    ) -> None:
        self.visual_engine = visual_engine
        self.text_engines = dict(text_engines or {})
        self.reranker = reranker or HybridReranker()
        self.config = config or HybridSearchConfig()

    @property
    def available_modalities(self) -> tuple[str, ...]:
        return ("visual", *sorted(self.text_engines))

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> VisualSearchResponse:
        started_at = time.perf_counter()
        bounded_top_k = self._top_k(top_k)
        visual_response = self.visual_engine.search(
            query,
            top_k=self.config.stage1_top_k,
        )
        candidate_groups: list[list[RetrievalResult]] = [
            visual_response.results
        ]
        for engine in self.text_engines.values():
            candidate_groups.append(
                engine.search_results(
                    query,
                    top_k=self.config.text_stage1_top_k,
                )
            )

        merged_pool = merge_candidates(
            candidate_groups,
            top_k=self.config.rerank_pool_size,
            dedupe_same_shot=False,
        )
        results = self.reranker.rerank(
            query=query,
            candidates=merged_pool,
            top_k=bounded_top_k,
        )
        return VisualSearchResponse(
            query=query,
            top_k=bounded_top_k,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
            results=results,
        )

    def temporal_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[TemporalMatch]:
        bounded_top_k = self._top_k(top_k)
        events = decompose_temporal_query(query)
        if not events:
            return []
        event_results = [
            self.search(
                event.text,
                top_k=self.config.rerank_pool_size,
            ).results
            for event in events
        ]
        return match_ordered_events(
            event_results=event_results,
            max_gap_seconds=self.config.max_gap_seconds,
            top_k=bounded_top_k,
        )

    def _top_k(self, requested: int | None) -> int:
        value = self.config.default_top_k if requested is None else int(requested)
        return max(1, min(value, self.config.max_top_k))
