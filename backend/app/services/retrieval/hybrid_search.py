"""Phase 3 hybrid retrieval orchestration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.app.models.retrieval import VisualSearchResponse
from backend.app.services.retrieval.rerank import HybridReranker
from backend.app.services.retrieval.temporal_search import (
    TemporalMatch,
    decompose_temporal_query,
    match_ordered_events,
)

if TYPE_CHECKING:
    from backend.app.services.retrieval.search_visual import VisualSearchEngine


@dataclass(frozen=True)
class HybridSearchConfig:
    stage1_top_k: int = 500
    rerank_pool_size: int = 100
    default_top_k: int = 20
    max_gap_seconds: float = 180.0


class HybridSearchEngine:
    """Runs fast visual retrieval, metadata rerank, and temporal search."""

    def __init__(
        self,
        visual_engine: "VisualSearchEngine",
        reranker: HybridReranker | None = None,
        config: HybridSearchConfig | None = None,
    ) -> None:
        self.visual_engine = visual_engine
        self.reranker = reranker or HybridReranker()
        self.config = config or HybridSearchConfig()

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        started_at = time.perf_counter()
        requested_top_k = top_k if top_k is not None else self.config.default_top_k
        stage1 = self.visual_engine.search(query, top_k=self.config.stage1_top_k)
        pool = stage1.results[: self.config.rerank_pool_size]
        results = self.reranker.rerank(query=query, candidates=pool, top_k=requested_top_k)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
        return VisualSearchResponse(
            query=query,
            top_k=max(1, int(requested_top_k)),
            latency_ms=latency_ms,
            results=results,
        )

    def temporal_search(self, query: str, top_k: int | None = None) -> list[TemporalMatch]:
        requested_top_k = top_k if top_k is not None else self.config.default_top_k
        events = decompose_temporal_query(query)
        event_results = [
            self.search(event.text, top_k=self.config.rerank_pool_size).results
            for event in events
        ]
        return match_ordered_events(
            event_results=event_results,
            max_gap_seconds=self.config.max_gap_seconds,
            top_k=requested_top_k,
        )
