"""Cached entry points for visual, lexical, hybrid, and temporal retrieval."""
from __future__ import annotations

import os
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from backend.app.models.retrieval import VisualSearchResponse
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine
from backend.app.services.retrieval.qa_evidence import (
    BgeCandidateReranker,
    QaEvidenceSearchEngine,
    QaRoutingConfig,
)
from backend.app.services.retrieval.qa_pipeline import QaPipelineConfig, QaSearchPipeline
from backend.app.services.retrieval.qa_answerer import (
    DEFAULT_QA_MODEL,
    DEFAULT_QA_MODEL_REVISION,
    QA_PROMPT_REVISION,
)
from backend.app.services.retrieval.rerank import HybridReranker
from backend.app.services.retrieval.retrieval_config import (
    RetrievalRuntimeConfig,
    load_retrieval_runtime_config,
)
from backend.app.services.retrieval.search_caption import CaptionSearchEngine
from backend.app.services.retrieval.search_object import ObjectSearchEngine
from backend.app.services.retrieval.search_ocr import OcrSearchEngine
from backend.app.services.retrieval.search_visual import (
    VisualSearchConfig,
    VisualSearchEngine,
)
from backend.app.services.retrieval.temporal_search import TemporalMatch


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _optional_float_from_env(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _choice_from_env(
    name: str,
    default: str,
    choices: tuple[str, ...],
) -> str:
    value = os.getenv(name, default).strip().casefold()
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


def _qa_routing_config() -> QaRoutingConfig:
    """Build the QA config while tolerating mixed-version router deployments."""

    requested: dict[str, object] = {
        "typed_parser_enabled": _bool_from_env("QA_TYPED_PARSER_ENABLED", True),
        "router_enabled": _bool_from_env("QA_ROUTER_ENABLED", True),
        "evidence_bundle_enabled": _bool_from_env("QA_EVIDENCE_BUNDLE_ENABLED", True),
        "constraint_rerank_enabled": _bool_from_env(
            "QA_CONSTRAINT_RERANK_ENABLED",
            True,
        ),
        "constraint_weight": float(os.getenv("QA_CONSTRAINT_WEIGHT", "0.15")),
        "constraint_min_signal": float(
            os.getenv("QA_CONSTRAINT_MIN_SIGNAL", "0.20")
        ),
        "temporal_routing_enabled": _bool_from_env(
            "QA_TEMPORAL_ROUTING_ENABLED",
            True,
        ),
        "temporal_max_events": int(os.getenv("QA_TEMPORAL_MAX_EVENTS", "5")),
        "temporal_max_gap_seconds": float(
            os.getenv("QA_TEMPORAL_MAX_GAP_SECONDS", "180")
        ),
    }
    supported = {field.name for field in fields(QaRoutingConfig)}
    return QaRoutingConfig(
        **{name: value for name, value in requested.items() if name in supported}
    )


def load_visual_search_config() -> VisualSearchConfig:
    """Load SigLIP2 visual retrieval settings from environment variables."""
    defaults = VisualSearchConfig()
    return VisualSearchConfig(
        index_path=_path_from_env("RETRIEVAL_INDEX_PATH", defaults.index_path),
        frame_map_path=_path_from_env(
            "RETRIEVAL_FRAME_MAP_PATH",
            defaults.frame_map_path,
        ),
        manifest_path=_path_from_env(
            "RETRIEVAL_MANIFEST_PATH",
            defaults.manifest_path,
        ),
        device=os.getenv("RETRIEVAL_DEVICE", defaults.device),
        model_cache_dir=_path_from_env(
            "RETRIEVAL_MODEL_CACHE_DIR",
            defaults.model_cache_dir,
        ),
        no_autocast=_bool_from_env(
            "RETRIEVAL_NO_AUTOCAST",
            defaults.no_autocast,
        ),
        default_top_k=int(
            os.getenv("RETRIEVAL_DEFAULT_TOP_K", defaults.default_top_k)
        ),
        max_top_k=int(
            os.getenv("RETRIEVAL_MAX_TOP_K", defaults.max_top_k)
        ),
        min_score=_optional_float_from_env(
            "RETRIEVAL_MIN_SCORE",
            defaults.min_score,
        ),
    )


@lru_cache(maxsize=1)
def get_runtime_config() -> RetrievalRuntimeConfig:
    return load_retrieval_runtime_config()


@lru_cache(maxsize=1)
def get_visual_search_engine() -> VisualSearchEngine:
    return VisualSearchEngine(load_visual_search_config())


def _text_engine_kwargs() -> dict:
    config = get_runtime_config().text_index
    return {
        "index_path": config.path,
        "default_top_k": config.default_top_k,
        "max_top_k": config.max_top_k,
    }


@lru_cache(maxsize=1)
def get_caption_search_engine() -> CaptionSearchEngine:
    return CaptionSearchEngine(**_text_engine_kwargs())


@lru_cache(maxsize=1)
def get_ocr_search_engine() -> OcrSearchEngine:
    return OcrSearchEngine(**_text_engine_kwargs())


@lru_cache(maxsize=1)
def get_object_search_engine() -> ObjectSearchEngine:
    return ObjectSearchEngine(**_text_engine_kwargs())


@lru_cache(maxsize=1)
def get_hybrid_search_engine() -> HybridSearchEngine:
    runtime = get_runtime_config()
    text_engines = {}
    if runtime.text_index.path.exists():
        text_engines = {
            "caption": get_caption_search_engine(),
            "ocr": get_ocr_search_engine(),
            "objects": get_object_search_engine(),
        }
    return HybridSearchEngine(
        visual_engine=get_visual_search_engine(),
        text_engines=text_engines,
        reranker=HybridReranker(runtime.rerank),
        config=runtime.hybrid,
    )


@lru_cache(maxsize=1)
def get_qa_evidence_search_engine() -> QaEvidenceSearchEngine:
    dense_text_engine = None
    dense_enabled = _bool_from_env("QA_BGE_DENSE_ENABLED", False)
    dense_root = _path_from_env("QA_BGE_INDEX_ROOT", Path("data/indexes/bge_m3"))
    if dense_enabled:
        from backend.app.services.retrieval.bge_dense import BgeM3DenseSearchEngine

        dense_text_engine = BgeM3DenseSearchEngine(
            dense_root,
            model_revision=os.getenv("QA_BGE_MODEL_REVISION") or None,
            device=os.getenv("QA_BGE_DEVICE", "auto"),
            cache_dir=_path_from_env(
                "QA_BGE_MODEL_CACHE_DIR",
                Path("data/model_cache/bge_m3"),
            ),
            local_files_only=_bool_from_env("QA_MODELS_LOCAL_ONLY", False),
        )
    reranker = None
    if _bool_from_env("QA_BGE_RERANKER_ENABLED", False):
        reranker = BgeCandidateReranker(
            model_name=os.getenv(
                "QA_BGE_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            ),
            model_revision=os.getenv("QA_BGE_RERANKER_REVISION", "main"),
            retrieval_alpha=float(os.getenv("QA_BGE_RERANKER_ALPHA", "0.5")),
            batch_size=int(os.getenv("QA_BGE_BATCH_SIZE", "16")),
            device=os.getenv("QA_BGE_DEVICE", "auto"),
            cache_dir=str(
                _path_from_env(
                    "QA_BGE_MODEL_CACHE_DIR",
                    Path("data/model_cache/bge_m3"),
                )
            ),
            local_files_only=_bool_from_env("QA_MODELS_LOCAL_ONLY", False),
        )
    return QaEvidenceSearchEngine(
        get_hybrid_search_engine(),
        dense_text_engine=dense_text_engine,
        candidate_reranker=reranker,
        config=_qa_routing_config(),
    )


@lru_cache(maxsize=1)
def get_qa_search_pipeline() -> QaSearchPipeline:
    answer_mode = _choice_from_env(
        "QA_ANSWER_MODE",
        "off",
        ("off", "optional", "required"),
    )
    answer_runner = None
    if answer_mode != "off":
        from backend.app.services.retrieval.qa_answerer import build_local_qwen_runner

        answer_runner = build_local_qwen_runner(
            model_name=os.getenv("QA_ANSWER_MODEL", DEFAULT_QA_MODEL),
            model_revision=os.getenv(
                "QA_ANSWER_MODEL_REVISION",
                DEFAULT_QA_MODEL_REVISION,
            ),
            device=os.getenv("QA_ANSWER_DEVICE") or None,
            quantization=os.getenv("QA_ANSWER_QUANTIZATION", "auto"),
            cache_dir=_path_from_env(
                "QA_ANSWER_MODEL_CACHE_DIR",
                Path("data/model_cache/qa_answer"),
            ),
        )
    return QaSearchPipeline(
        get_qa_evidence_search_engine(),
        config=QaPipelineConfig(
            answer_mode=answer_mode,
            model_name=os.getenv("QA_ANSWER_MODEL", DEFAULT_QA_MODEL),
            model_revision=os.getenv(
                "QA_ANSWER_MODEL_REVISION",
                DEFAULT_QA_MODEL_REVISION,
            ),
            answer_cache_root=_path_from_env(
                "QA_ANSWER_CACHE_DIR",
                Path("data/cache/qa_answers"),
            ),
            answer_timeout_seconds=float(
                os.getenv("QA_ANSWER_TIMEOUT_SECONDS", "120")
            ),
            experiment_id=os.getenv(
                "QA_EXPERIMENT_ID",
                "qa-parser-router-evidence-v1",
            ),
        ),
        answer_runner=answer_runner,
    )


def get_qa_runtime_lineage() -> dict[str, Any]:
    """Return the exact QA model/index contracts used by the current engines."""

    engine = get_qa_evidence_search_engine()
    dense = getattr(engine, "dense_text_engine", None)
    reranker = getattr(engine, "candidate_reranker", None)

    dense_manifest: Mapping[str, object] = {}
    artifacts = getattr(dense, "artifacts", None)
    manifest = getattr(artifacts, "manifest", None)
    if isinstance(manifest, Mapping):
        dense_manifest = manifest
    model_contract = dense_manifest.get("model", {})
    if not isinstance(model_contract, Mapping):
        model_contract = {}
    artifact_contract = dense_manifest.get("artifacts", {})
    if not isinstance(artifact_contract, Mapping):
        artifact_contract = {}
    source_hashes = dense_manifest.get("source_hashes", {})
    if not isinstance(source_hashes, Mapping):
        source_hashes = {}
    source_contract = dense_manifest.get("source_contract", {})
    if not isinstance(source_contract, Mapping):
        source_contract = {}

    rerank_report = getattr(reranker, "last_report", None)
    answer_mode = _choice_from_env(
        "QA_ANSWER_MODE",
        "off",
        ("off", "optional", "required"),
    )
    return {
        "answer_model": {
            "enabled": answer_mode != "off",
            "mode": answer_mode,
            "name": os.getenv("QA_ANSWER_MODEL", DEFAULT_QA_MODEL),
            "revision": os.getenv(
                "QA_ANSWER_MODEL_REVISION",
                DEFAULT_QA_MODEL_REVISION,
            ),
            "prompt_revision": QA_PROMPT_REVISION,
        },
        "dense_text": {
            "enabled": dense is not None,
            "model_name": str(model_contract.get("name") or getattr(dense, "model_name", "")),
            "model_revision": str(
                model_contract.get("revision")
                or getattr(dense, "model_revision", "")
            ),
            "index_schema_version": dense_manifest.get("schema_version"),
            "vector_count": dense_manifest.get("vector_count"),
            "artifact_checksums": dict(artifact_contract),
            "source_hashes": dict(source_hashes),
            "source_contract": dict(source_contract),
        },
        "reranker": {
            "enabled": reranker is not None,
            "model_name": str(getattr(reranker, "model_name", "")),
            "model_revision": str(getattr(reranker, "model_revision", "")),
            "last_report": dict(rerank_report) if isinstance(rerank_report, Mapping) else None,
        },
    }


def clear_retrieval_caches() -> None:
    """Clear cached engines after changing environment paths in tests or tools."""
    for cached in (
        get_qa_search_pipeline,
        get_qa_evidence_search_engine,
        get_hybrid_search_engine,
        get_object_search_engine,
        get_ocr_search_engine,
        get_caption_search_engine,
        get_visual_search_engine,
        get_runtime_config,
    ):
        cached.cache_clear()


def search_visual(
    query: str,
    top_k: int | None = None,
) -> VisualSearchResponse:
    return get_visual_search_engine().search(query=query, top_k=top_k)


def search_caption(
    query: str,
    top_k: int | None = None,
) -> VisualSearchResponse:
    return get_caption_search_engine().search(query=query, top_k=top_k)


def search_ocr(
    query: str,
    top_k: int | None = None,
) -> VisualSearchResponse:
    return get_ocr_search_engine().search(query=query, top_k=top_k)


def search_object(
    query: str,
    top_k: int | None = None,
) -> VisualSearchResponse:
    return get_object_search_engine().search(query=query, top_k=top_k)


def search_hybrid(
    query: str,
    top_k: int | None = None,
) -> VisualSearchResponse:
    return get_hybrid_search_engine().search(query=query, top_k=top_k)


def search_temporal(
    query: str,
    top_k: int | None = None,
) -> list[TemporalMatch]:
    return get_hybrid_search_engine().temporal_search(query=query, top_k=top_k)


def search_qa_evidence(
    question: str,
    top_k: int | None = None,
) -> dict:
    requested_top_k = 10 if top_k is None else int(top_k)
    return get_qa_evidence_search_engine().search(
        question=question,
        top_k=requested_top_k,
    )


def search_qa(
    query: str,
    top_k: int | None = None,
    *,
    task_mode: str = "qa",
    expanded_queries: tuple[str, ...] | list[str] | None = None,
) -> dict:
    requested_top_k = 5 if top_k is None else int(top_k)
    return get_qa_search_pipeline().search(
        query=query,
        top_k=requested_top_k,
        task_mode=task_mode,
        expanded_queries=expanded_queries or (),
    )
