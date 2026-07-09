"""Caption text search over the Phase 2 lexical index."""
from __future__ import annotations

from pathlib import Path

from backend.app.models.retrieval import VisualSearchResponse
from backend.app.services.retrieval.text_index import TextIndexSearcher


class CaptionSearchEngine:
    def __init__(self, index_path: str | Path) -> None:
        self.searcher = TextIndexSearcher(index_path)

    def search(self, query: str, top_k: int = 20) -> VisualSearchResponse:
        return self.searcher.search(query=query, modality="caption", top_k=top_k)
