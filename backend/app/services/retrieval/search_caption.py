"""Caption text search over the Retrieval Phase 2 lexical index."""
from __future__ import annotations

from backend.app.services.retrieval.search_text import ModalitySearchEngine


class CaptionSearchEngine(ModalitySearchEngine):
    def __init__(self, index_path, **kwargs) -> None:
        super().__init__(index_path, modality="caption", **kwargs)
