"""Entry points for retrieval services."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from backend.app.models.retrieval import VisualSearchResponse
from backend.app.services.retrieval.search_visual import (
    VisualSearchConfig,
    VisualSearchEngine,
)


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def load_visual_search_config() -> VisualSearchConfig:
    """Load SigLIP2 visual retrieval settings from environment variables."""
    defaults = VisualSearchConfig()
    return VisualSearchConfig(
        index_path=_path_from_env("RETRIEVAL_INDEX_PATH", defaults.index_path),
        frame_map_path=_path_from_env("RETRIEVAL_FRAME_MAP_PATH", defaults.frame_map_path),
        manifest_path=_path_from_env("RETRIEVAL_MANIFEST_PATH", defaults.manifest_path),
        device=os.getenv("RETRIEVAL_DEVICE", defaults.device),
        model_cache_dir=_path_from_env("RETRIEVAL_MODEL_CACHE_DIR", defaults.model_cache_dir),
        no_autocast=_bool_from_env("RETRIEVAL_NO_AUTOCAST", defaults.no_autocast),
        default_top_k=int(os.getenv("RETRIEVAL_DEFAULT_TOP_K", defaults.default_top_k)),
        max_top_k=int(os.getenv("RETRIEVAL_MAX_TOP_K", defaults.max_top_k)),
    )


@lru_cache(maxsize=1)
def get_visual_search_engine() -> VisualSearchEngine:
    return VisualSearchEngine(load_visual_search_config())


def search_visual(query: str, top_k: int | None = None) -> VisualSearchResponse:
    return get_visual_search_engine().search(query=query, top_k=top_k)
