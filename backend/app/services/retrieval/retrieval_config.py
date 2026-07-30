"""Config loader for Retrieval Phase 2/3 settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.services.retrieval.hybrid_search import HybridSearchConfig
from backend.app.services.retrieval.rerank import RerankConfig, RerankWeights


DEFAULT_RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")


@dataclass(frozen=True)
class TextIndexConfig:
    path: Path = Path("data/indexes/retrieval_text_index.json")
    default_top_k: int = 20
    max_top_k: int = 200


@dataclass(frozen=True)
class RetrievalRuntimeConfig:
    hybrid: HybridSearchConfig = HybridSearchConfig()
    rerank: RerankConfig = RerankConfig()
    text_index: TextIndexConfig = TextIndexConfig()


def load_retrieval_runtime_config(
    config_path: str | Path | None = None,
) -> RetrievalRuntimeConfig:
    path = Path(
        config_path
        or os.getenv("RETRIEVAL_CONFIG_PATH")
        or DEFAULT_RETRIEVAL_CONFIG_PATH
    )
    raw = _read_simple_yaml(path) if path.exists() else {}
    hybrid_raw = _section(raw, "hybrid")
    weights_raw = _section(raw, "weights")
    dedupe_raw = _section(raw, "dedupe")
    text_raw = _section(raw, "text_index")

    hybrid = HybridSearchConfig(
        stage1_top_k=_int_env(
            "RETRIEVAL_HYBRID_STAGE1_TOP_K",
            hybrid_raw.get("stage1_top_k"),
            HybridSearchConfig.stage1_top_k,
        ),
        text_stage1_top_k=_int_env(
            "RETRIEVAL_HYBRID_TEXT_STAGE1_TOP_K",
            hybrid_raw.get("text_stage1_top_k"),
            HybridSearchConfig.text_stage1_top_k,
        ),
        rerank_pool_size=_int_env(
            "RETRIEVAL_HYBRID_RERANK_POOL_SIZE",
            hybrid_raw.get("rerank_pool_size"),
            HybridSearchConfig.rerank_pool_size,
        ),
        default_top_k=_int_env(
            "RETRIEVAL_DEFAULT_TOP_K",
            hybrid_raw.get("default_top_k"),
            HybridSearchConfig.default_top_k,
        ),
        max_top_k=_int_env(
            "RETRIEVAL_MAX_TOP_K",
            hybrid_raw.get("max_top_k"),
            HybridSearchConfig.max_top_k,
        ),
        max_gap_seconds=_float_env(
            "RETRIEVAL_TEMPORAL_MAX_GAP_SECONDS",
            hybrid_raw.get("max_gap_seconds"),
            HybridSearchConfig.max_gap_seconds,
        ),
    )
    weights = RerankWeights(
        visual=_float_env(
            "RETRIEVAL_WEIGHT_VISUAL",
            weights_raw.get("visual"),
            RerankWeights.visual,
        ),
        caption=_float_env(
            "RETRIEVAL_WEIGHT_CAPTION",
            weights_raw.get("caption"),
            RerankWeights.caption,
        ),
        ocr=_float_env(
            "RETRIEVAL_WEIGHT_OCR",
            weights_raw.get("ocr"),
            RerankWeights.ocr,
        ),
        asr=_float_env(
            "RETRIEVAL_WEIGHT_ASR",
            weights_raw.get("asr"),
            RerankWeights.asr,
        ),
        objects=_float_env(
            "RETRIEVAL_WEIGHT_OBJECTS",
            weights_raw.get("objects"),
            RerankWeights.objects,
        ),
        temporal=_float_env(
            "RETRIEVAL_WEIGHT_TEMPORAL",
            weights_raw.get("temporal"),
            RerankWeights.temporal,
        ),
    )
    text_index = TextIndexConfig(
        path=Path(
            os.getenv("RETRIEVAL_TEXT_INDEX_PATH")
            or text_raw.get("path")
            or TextIndexConfig.path
        ),
        default_top_k=_int_env(
            "RETRIEVAL_TEXT_DEFAULT_TOP_K",
            text_raw.get("default_top_k"),
            TextIndexConfig.default_top_k,
        ),
        max_top_k=_int_env(
            "RETRIEVAL_TEXT_MAX_TOP_K",
            text_raw.get("max_top_k"),
            TextIndexConfig.max_top_k,
        ),
    )
    return RetrievalRuntimeConfig(
        hybrid=hybrid,
        rerank=RerankConfig(
            weights=weights,
            dedupe_same_shot=_bool_env(
                "RETRIEVAL_DEDUPE_SAME_SHOT",
                dedupe_raw.get("same_shot"),
                RerankConfig.dedupe_same_shot,
            ),
        ),
        text_index=text_index,
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    return value if isinstance(value, dict) else {}


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Read the small, two-level YAML subset used by retrieval.yaml."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current = {}
            root[line[:-1].strip()] = current
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        current[key.strip()] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    try:
        if any(character in value for character in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _int_env(name: str, value: Any, default: int) -> int:
    raw = os.getenv(name)
    result = int(raw if raw is not None else value if value is not None else default)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _float_env(name: str, value: Any, default: float) -> float:
    raw = os.getenv(name)
    return float(raw if raw is not None else value if value is not None else default)


def _bool_env(name: str, value: Any, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(value) if value is not None else default
    normalized = raw.casefold().strip()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")
