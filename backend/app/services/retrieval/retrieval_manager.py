"""Entry points for retrieval services."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from backend.app.models.retrieval import VisualSearchResponse
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine, TemporalMatch
from backend.app.services.retrieval.search_visual import (
    VisualSearchConfig,
    VisualSearchEngine,
)


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def load_visual_search_config() -> VisualSearchConfig:
    """Load Phase 1 visual retrieval settings from environment variables."""
    defaults = VisualSearchConfig()
    return VisualSearchConfig(
        index_path=_path_from_env("RETRIEVAL_INDEX_PATH", defaults.index_path),
        frame_map_path=_path_from_env("RETRIEVAL_FRAME_MAP_PATH", defaults.frame_map_path),
        model_name=os.getenv("RETRIEVAL_MODEL_NAME", defaults.model_name),
        pretrained=os.getenv("RETRIEVAL_PRETRAINED", defaults.pretrained),
        device=os.getenv("RETRIEVAL_DEVICE", defaults.device),
        model_cache_dir=_path_from_env("RETRIEVAL_MODEL_CACHE_DIR", defaults.model_cache_dir),
        default_top_k=int(os.getenv("RETRIEVAL_DEFAULT_TOP_K", defaults.default_top_k)),
        max_top_k=int(os.getenv("RETRIEVAL_MAX_TOP_K", defaults.max_top_k)),
    )


@lru_cache(maxsize=1)
def get_visual_search_engine() -> VisualSearchEngine:
    return VisualSearchEngine(load_visual_search_config())


@lru_cache(maxsize=1)
def get_hybrid_search_engine() -> HybridSearchEngine:
    return HybridSearchEngine(get_visual_search_engine())


def search_visual(query: str, top_k: int | None = None) -> VisualSearchResponse:
    return get_visual_search_engine().search(query=query, top_k=top_k)


def search_hybrid(query: str, top_k: int | None = None) -> VisualSearchResponse:
    return get_hybrid_search_engine().search(query=query, top_k=top_k)


def search_temporal(query: str, top_k: int | None = None) -> list[TemporalMatch]:
    return get_hybrid_search_engine().temporal_search(query=query, top_k=top_k)
