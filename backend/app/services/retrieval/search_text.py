"""Shared search wrapper for Retrieval Phase 2 text modalities."""
from __future__ import annotations

from pathlib import Path

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.text_index import TextIndexSearcher


class ModalitySearchEngine:
    def __init__(
        self,
        index_path: str | Path,
        modality: str,
        *,
        default_top_k: int = 20,
        max_top_k: int = 200,
        expected_sha256: str | None = None,
        searcher: TextIndexSearcher | None = None,
    ) -> None:
        self.modality = modality
        self.default_top_k = max(1, int(default_top_k))
        self.max_top_k = max(1, int(max_top_k))
        self.searcher = searcher or TextIndexSearcher(
            index_path,
            expected_sha256=expected_sha256,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> VisualSearchResponse:
        bounded_top_k = self._top_k(top_k)
        return self.searcher.search(
            query=query,
            modality=self.modality,
            top_k=bounded_top_k,
        )

    def search_results(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return self.searcher.search_results(
            query=query,
            modality=self.modality,
            top_k=self._top_k(top_k),
        )

    def _top_k(self, requested: int | None) -> int:
        value = self.default_top_k if requested is None else int(requested)
        return max(1, min(value, self.max_top_k))
