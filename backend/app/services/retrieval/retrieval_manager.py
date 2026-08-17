"""Cached entry points for visual, lexical, hybrid, and temporal retrieval."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar, cast

from backend.app.models.retrieval import VisualSearchResponse
from backend.app.services.agent.query_expansion import (
    QueryExpansionProvider,
    build_production_query_expansion_provider,
)
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine
from backend.app.services.retrieval.planned_hybrid import planned_hybrid_search
from backend.app.services.retrieval.query_plan import build_query_plan
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


_query_expansion_provider_instance: QueryExpansionProvider | None = None
DEFAULT_CORPUS_MANIFEST_PATH = Path("data/metadata/offline_corpus_manifest.json")
_CachedEngine = TypeVar("_CachedEngine")


@dataclass(frozen=True)
class _CorpusCacheKey:
    """Immutable identity of the corpus a cached runtime object belongs to."""

    manifest_path: str
    bundle_generation: str | None
    manifest_contract_sha256: str | None


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _generation(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(
        hashes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_runtime_corpus_bundle(
    *,
    required_roles: tuple[str, ...] = (),
    artifact_overrides: Mapping[str, Path] | None = None,
    require_bge: bool = False,
) -> Mapping[str, Any] | None:
    """Gate runtime engines on one fully committed offline corpus generation."""

    manifest_path = _path_from_env(
        "RETRIEVAL_CORPUS_MANIFEST_PATH",
        DEFAULT_CORPUS_MANIFEST_PATH,
    )
    if not manifest_path.is_file():
        return None  # Backward-compatible legacy deployments.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "passed":
        raise ValueError("Offline corpus bundle is not fully published")
    if require_bge and manifest.get("bge_enabled") is not True:
        raise ValueError("Runtime requested BGE-M3 but the committed corpus omitted it")
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("Offline corpus manifest has no artifact declarations")
    hashes: dict[str, str] = {}
    for role, item in declared.items():
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"Offline corpus artifact declaration is invalid: {role}")
        hashes[str(role)] = str(item["sha256"])
    if manifest.get("bundle_generation") != _generation(hashes):
        raise ValueError("Offline corpus generation does not match declared hashes")

    overrides = dict(artifact_overrides or {})
    root = manifest_path.parent.parent
    for role in required_roles:
        item = declared.get(role)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"Committed corpus does not contain required role: {role}")
        path = overrides.get(role, root / str(item["path"]))
        if not path.is_file() or _sha256_file(path) != hashes[role]:
            raise ValueError(f"Runtime artifact is outside the committed corpus: {role}")
    return manifest


def _manifest_contract_sha256(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _corpus_cache_key(
    manifest: Mapping[str, Any] | None,
) -> _CorpusCacheKey:
    manifest_path = _path_from_env(
        "RETRIEVAL_CORPUS_MANIFEST_PATH",
        DEFAULT_CORPUS_MANIFEST_PATH,
    ).resolve(strict=False)
    if manifest is None:
        return _CorpusCacheKey(
            manifest_path=str(manifest_path),
            bundle_generation=None,
            manifest_contract_sha256=None,
        )
    return _CorpusCacheKey(
        manifest_path=str(manifest_path),
        bundle_generation=str(manifest["bundle_generation"]),
        manifest_contract_sha256=_manifest_contract_sha256(manifest),
    )


def _current_corpus_cache_key() -> _CorpusCacheKey:
    """Read the publication gate even when an engine is already cached."""

    return _corpus_cache_key(validate_runtime_corpus_bundle())


def _validate_expected_corpus(
    expected: _CorpusCacheKey,
    *,
    required_roles: tuple[str, ...] = (),
    artifact_overrides: Mapping[str, Path] | None = None,
    require_bge: bool = False,
) -> Mapping[str, Any] | None:
    manifest = validate_runtime_corpus_bundle(
        required_roles=required_roles,
        artifact_overrides=artifact_overrides,
        require_bge=require_bge,
    )
    if _corpus_cache_key(manifest) != expected:
        raise ValueError("Offline corpus changed while retrieval was loading")
    return manifest


def _corpus_generation_cached(
    factory: Callable[[_CorpusCacheKey], _CachedEngine],
) -> Callable[[], _CachedEngine]:
    """Cache one engine per current committed corpus, with a hot-publish gate.

    Legacy deployments without a canonical manifest keep the previous one-entry
    cache behaviour.  Once a manifest exists, every zero-argument factory call
    reads its committed identity before consulting the cache.  A new generation
    therefore evicts and rebuilds the old engine; a publishing/invalid manifest
    raises before any stale cached object can escape.
    """

    cached = lru_cache(maxsize=1)(factory)

    @wraps(factory)
    def wrapper() -> _CachedEngine:
        before = _current_corpus_cache_key()
        value = cached(before)
        after = _current_corpus_cache_key()
        if before != after:
            raise ValueError("Offline corpus changed while retrieval was loading")
        return value

    # Preserve the public cache control API used by tests and operational tools.
    setattr(wrapper, "cache_clear", cached.cache_clear)
    setattr(wrapper, "cache_info", cached.cache_info)
    setattr(wrapper, "cache_parameters", cached.cache_parameters)
    return cast(Callable[[], _CachedEngine], wrapper)


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
def get_query_expansion_provider() -> QueryExpansionProvider | None:
    """Return one lazy production provider for the current runtime config."""
    global _query_expansion_provider_instance
    config = get_runtime_config().query_expansion
    if not config.enabled:
        return None
    provider = build_production_query_expansion_provider(
        config=config,
        device=os.getenv("QUERY_EXPANSION_DEVICE", "cpu"),
        cache_dir=_path_from_env(
            "QUERY_EXPANSION_CACHE_DIR",
            Path("data/cache/query_expansion"),
        ),
        model_cache_dir=_path_from_env(
            "QUERY_EXPANSION_MODEL_CACHE_DIR",
            Path("data/model_cache/query_expansion"),
        ),
        local_files_only=_bool_from_env(
            "QUERY_EXPANSION_LOCAL_FILES_ONLY",
            False,
        ),
    )
    _query_expansion_provider_instance = provider
    return provider


@_corpus_generation_cached
def get_visual_search_engine(
    corpus_key: _CorpusCacheKey,
) -> VisualSearchEngine:
    config = load_visual_search_config()
    before = _validate_expected_corpus(
        corpus_key,
        required_roles=("visual_index", "visual_frame_map", "visual_manifest"),
        artifact_overrides={
            "visual_index": config.index_path,
            "visual_frame_map": config.frame_map_path,
            "visual_manifest": config.manifest_path,
        },
    )
    engine = VisualSearchEngine(config)
    after = _validate_expected_corpus(
        corpus_key,
        required_roles=("visual_index", "visual_frame_map", "visual_manifest"),
        artifact_overrides={
            "visual_index": config.index_path,
            "visual_frame_map": config.frame_map_path,
            "visual_manifest": config.manifest_path,
        },
    )
    if before != after:
        raise ValueError("Offline corpus changed while visual retrieval was loading")
    engine.corpus_generation = (
        str(before.get("bundle_generation")) if before is not None else None
    )
    return engine


def _text_engine_kwargs(corpus_key: _CorpusCacheKey) -> dict:
    config = get_runtime_config().text_index
    manifest = _validate_expected_corpus(
        corpus_key,
        required_roles=("text_index",),
        artifact_overrides={"text_index": config.path},
    )
    expected_sha256 = None
    if manifest is not None:
        expected_sha256 = manifest["artifacts"]["text_index"]["sha256"]
    return {
        "index_path": config.path,
        "default_top_k": config.default_top_k,
        "max_top_k": config.max_top_k,
        "expected_sha256": expected_sha256,
    }


def _build_text_search_engine(
    engine_type: type[_CachedEngine],
    corpus_key: _CorpusCacheKey,
) -> _CachedEngine:
    runtime = get_runtime_config()
    engine = engine_type(**_text_engine_kwargs(corpus_key))
    _validate_expected_corpus(
        corpus_key,
        required_roles=("text_index",),
        artifact_overrides={"text_index": runtime.text_index.path},
    )
    setattr(engine, "corpus_generation", corpus_key.bundle_generation)
    return engine


@_corpus_generation_cached
def get_caption_search_engine(
    corpus_key: _CorpusCacheKey,
) -> CaptionSearchEngine:
    return _build_text_search_engine(CaptionSearchEngine, corpus_key)


@_corpus_generation_cached
def get_ocr_search_engine(
    corpus_key: _CorpusCacheKey,
) -> OcrSearchEngine:
    return _build_text_search_engine(OcrSearchEngine, corpus_key)


@_corpus_generation_cached
def get_object_search_engine(
    corpus_key: _CorpusCacheKey,
) -> ObjectSearchEngine:
    return _build_text_search_engine(ObjectSearchEngine, corpus_key)


@_corpus_generation_cached
def get_hybrid_search_engine(
    corpus_key: _CorpusCacheKey,
) -> HybridSearchEngine:
    runtime = get_runtime_config()
    visual_config = load_visual_search_config()
    required_roles = (
        "visual_index",
        "visual_frame_map",
        "visual_manifest",
        "text_index",
    )
    overrides = {
        "visual_index": visual_config.index_path,
        "visual_frame_map": visual_config.frame_map_path,
        "visual_manifest": visual_config.manifest_path,
        "text_index": runtime.text_index.path,
    }
    before = _validate_expected_corpus(
        corpus_key,
        required_roles=required_roles,
        artifact_overrides=overrides,
    )
    text_engines = {}
    if runtime.text_index.path.exists():
        text_engines = {
            "caption": get_caption_search_engine(),
            "ocr": get_ocr_search_engine(),
            "objects": get_object_search_engine(),
        }
    visual_engine = get_visual_search_engine()
    if before is not None:
        generation = str(before["bundle_generation"])
        if getattr(visual_engine, "corpus_generation", None) != generation:
            raise ValueError("Cached visual engine belongs to another corpus generation")
        expected_text_sha = str(before["artifacts"]["text_index"]["sha256"])
        if any(
            getattr(engine.searcher, "expected_sha256", None) != expected_text_sha
            for engine in text_engines.values()
        ):
            raise ValueError("Cached text engine belongs to another corpus generation")
    engine = HybridSearchEngine(
        visual_engine=visual_engine,
        text_engines=text_engines,
        reranker=HybridReranker(runtime.rerank),
        config=runtime.hybrid,
    )
    after = _validate_expected_corpus(
        corpus_key,
        required_roles=required_roles,
        artifact_overrides=overrides,
    )
    if before != after:
        raise ValueError("Offline corpus changed while hybrid retrieval was loading")
    engine.corpus_generation = (
        str(before.get("bundle_generation")) if before is not None else None
    )
    return engine


@_corpus_generation_cached
def get_qa_evidence_search_engine(
    corpus_key: _CorpusCacheKey,
) -> QaEvidenceSearchEngine:
    dense_text_engine = None
    dense_enabled = _bool_from_env("QA_BGE_DENSE_ENABLED", False)
    dense_root = _path_from_env("QA_BGE_INDEX_ROOT", Path("data/indexes/bge_m3"))
    bge_paths: dict[str, Path] = {}
    if dense_enabled:
        from backend.app.services.retrieval.bge_dense import BgeM3ArtifactPaths

        resolved_bge_paths = BgeM3ArtifactPaths.from_root(dense_root)
        bge_paths = {
            "bge_index": resolved_bge_paths.index,
            "bge_frame_map": resolved_bge_paths.frame_map,
            "bge_manifest": resolved_bge_paths.manifest,
        }
    before = _validate_expected_corpus(
        corpus_key,
        required_roles=tuple(bge_paths) if dense_enabled else (),
        artifact_overrides=bge_paths,
        require_bge=dense_enabled,
    )
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
    hybrid_engine = get_hybrid_search_engine()
    if before is not None and getattr(
        hybrid_engine,
        "corpus_generation",
        None,
    ) != str(before["bundle_generation"]):
        raise ValueError("Cached hybrid engine belongs to another corpus generation")
    engine = QaEvidenceSearchEngine(
        hybrid_engine,
        dense_text_engine=dense_text_engine,
        candidate_reranker=reranker,
        config=_qa_routing_config(),
    )
    after = _validate_expected_corpus(
        corpus_key,
        required_roles=tuple(bge_paths) if dense_enabled else (),
        artifact_overrides=bge_paths,
        require_bge=dense_enabled,
    )
    if before != after:
        raise ValueError("Offline corpus changed while QA retrieval was loading")
    engine.corpus_generation = corpus_key.bundle_generation
    return engine


@_corpus_generation_cached
def get_qa_search_pipeline(
    corpus_key: _CorpusCacheKey,
) -> QaSearchPipeline:
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
    evidence_engine = get_qa_evidence_search_engine()
    if (
        corpus_key.bundle_generation is not None
        and getattr(evidence_engine, "corpus_generation", None)
        != corpus_key.bundle_generation
    ):
        raise ValueError("Cached QA evidence belongs to another corpus generation")
    pipeline = QaSearchPipeline(
        evidence_engine,
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
    _validate_expected_corpus(corpus_key)
    pipeline.corpus_generation = corpus_key.bundle_generation
    return pipeline


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
    global _query_expansion_provider_instance
    if _query_expansion_provider_instance is not None:
        _query_expansion_provider_instance.close()
        _query_expansion_provider_instance = None
    for cached in (
        get_qa_search_pipeline,
        get_qa_evidence_search_engine,
        get_hybrid_search_engine,
        get_object_search_engine,
        get_ocr_search_engine,
        get_caption_search_engine,
        get_visual_search_engine,
        get_query_expansion_provider,
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
    runtime = get_runtime_config()
    provider = (
        get_query_expansion_provider()
        if runtime.query_expansion.enabled
        else None
    )
    plan = build_query_plan(
        query,
        profile="auto",
        expansion_provider=provider,
        expansion_config=runtime.query_expansion,
    )
    return planned_hybrid_search(
        get_hybrid_search_engine(),
        plan,
        top_k=top_k,
        max_expansion_contribution=(
            runtime.query_expansion.max_expansion_contribution
        ),
    )


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
