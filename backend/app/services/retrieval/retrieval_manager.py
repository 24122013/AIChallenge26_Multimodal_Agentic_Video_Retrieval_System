"""Cached entry points for visual, hybrid, temporal, TRAKE, and QA retrieval."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import weakref
from dataclasses import asdict, dataclass, fields
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar, cast

from backend.app.models.retrieval import VisualSearchResponse
from backend.app.pipelines.online_pipeline import OnlinePipeline, OnlinePipelineConfig
from backend.app.services.agent.query_expansion import (
    QueryExpansionProvider,
    build_production_query_expansion_provider,
)
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine
from backend.app.services.retrieval.dense_candidate_index import (
    DenseCandidateIndexConfig,
    FaissDenseCandidateIndex,
)
from backend.app.services.retrieval.online_context import (
    DEFAULT_NEIGHBOR_PATH,
    DEFAULT_SEGMENT_PATH,
    OnlineContextIndex,
)
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
_LOGGER = logging.getLogger(__name__)
_BGE_DENSE_ENGINE_LOCK = threading.RLock()
_BGE_DENSE_ENGINES: weakref.WeakValueDictionary[tuple[object, ...], Any] = (
    weakref.WeakValueDictionary()
)


@dataclass(frozen=True)
class _CorpusCacheKey:
    """Immutable identity of the corpus a cached runtime object belongs to."""

    manifest_path: str
    bundle_generation: str | None
    manifest_contract_sha256: str | None


class _LazyTrakeSearchPipeline:
    """Resolve the heavy task pipeline only when routing an actual TRAKE call."""

    def __init__(self, expected_generation: str | None) -> None:
        self.expected_generation = expected_generation

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        pipeline = get_trake_pipeline()
        if (
            self.expected_generation is not None
            and getattr(pipeline, "corpus_generation", None)
            != self.expected_generation
        ):
            raise ValueError("Lazy TRAKE pipeline belongs to another corpus generation")
        return pipeline.search(query=query, top_k=top_k)


def _validate_lazy_generation(
    component: Any,
    *,
    expected_generation: str | None,
    component_name: str,
) -> None:
    if (
        expected_generation is not None
        and getattr(component, "corpus_generation", None) != expected_generation
    ):
        raise ValueError(
            f"Lazy {component_name} belongs to another corpus generation"
        )


class _LazyQaSearchPipeline:
    """Resolve answer-capable QA only when the selected route is QA."""

    def __init__(self, expected_generation: str | None) -> None:
        self.expected_generation = expected_generation

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        task_mode: str = "qa",
        expanded_queries: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        pipeline = get_qa_search_pipeline()
        _validate_lazy_generation(
            pipeline,
            expected_generation=self.expected_generation,
            component_name="QA pipeline",
        )
        return pipeline.search(
            query=query,
            top_k=top_k,
            task_mode=task_mode,
            expanded_queries=expanded_queries or (),
        )


class _LazyQaEvidenceSearchEngine:
    """Resolve temporal/QA evidence only when that branch is selected."""

    def __init__(self, expected_generation: str | None) -> None:
        self.expected_generation = expected_generation

    def search(
        self,
        question: str,
        top_k: int = 5,
        *,
        task_mode: str = "qa",
        expanded_queries: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        engine = get_qa_evidence_search_engine()
        _validate_lazy_generation(
            engine,
            expected_generation=self.expected_generation,
            component_name="QA evidence engine",
        )
        return engine.search(
            question=question,
            top_k=top_k,
            task_mode=task_mode,
            expanded_queries=expanded_queries or (),
        )


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
        if require_bge:
            raise FileNotFoundError(
                "BGE runtime requires a committed offline corpus manifest"
            )
        return None  # Backward-compatible non-BGE legacy deployments.
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


def _positive_int_from_env(
    name: str,
    default: int,
    *,
    maximum: int | None = None,
) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0 or (maximum is not None and value > maximum):
        suffix = f" between 1 and {maximum}" if maximum is not None else " positive"
        raise ValueError(f"{name} must be{suffix}")
    return value


def _unit_float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number between 0 and 1") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1")
    return value


def _bge_artifact_overrides(artifact_root: Path) -> dict[str, Path]:
    """Resolve the three corpus-governed BGE-M3 artifacts from one root."""

    from backend.app.services.retrieval.bge_dense import BgeM3ArtifactPaths

    paths = BgeM3ArtifactPaths.from_root(artifact_root)
    return {
        "bge_index": paths.index,
        "bge_frame_map": paths.frame_map,
        "bge_manifest": paths.manifest,
    }


def build_bge_m3_dense_search_engine(
    *,
    artifact_root: Path,
    model_name: str = "BAAI/bge-m3",
    model_revision: str | None = None,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: Path = Path("data/model_cache/bge_m3"),
    local_files_only: bool = False,
) -> Any:
    """Build the canonical validated dense engine for QA or TRAKE."""

    from backend.app.services.retrieval.bge_dense import BgeM3DenseSearchEngine

    return BgeM3DenseSearchEngine(
        artifact_root,
        model_name=model_name,
        model_revision=model_revision,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def _shared_bge_m3_dense_search_engine(
    *,
    corpus_key: _CorpusCacheKey,
    artifact_root: Path,
    model_name: str = "BAAI/bge-m3",
    model_revision: str | None = None,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: Path = Path("data/model_cache/bge_m3"),
    local_files_only: bool = False,
) -> Any:
    """Single-flight and share an identical validated BGE dense contract."""

    key: tuple[object, ...] = (
        corpus_key,
        str(artifact_root.resolve()),
        str(model_name),
        model_revision,
        int(batch_size),
        str(device),
        str(cache_dir.resolve()),
        bool(local_files_only),
    )
    with _BGE_DENSE_ENGINE_LOCK:
        engine = _BGE_DENSE_ENGINES.get(key)
        if engine is None:
            engine = build_bge_m3_dense_search_engine(
                artifact_root=artifact_root,
                model_name=model_name,
                model_revision=model_revision,
                batch_size=batch_size,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            engine.corpus_generation = corpus_key.bundle_generation
            _BGE_DENSE_ENGINES[key] = engine
        return engine


def build_bge_candidate_reranker(
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    model_revision: str = "main",
    candidate_limit: int = 100,
    retrieval_alpha: float = 0.5,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: Path = Path("data/model_cache/bge_m3"),
    local_files_only: bool = False,
) -> BgeCandidateReranker:
    """Build the shared lazy candidate-reranker adapter."""

    return BgeCandidateReranker(
        model_name=model_name,
        model_revision=model_revision,
        candidate_limit=candidate_limit,
        retrieval_alpha=retrieval_alpha,
        batch_size=batch_size,
        device=device,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
    )


def _required_trake_dependency_error(
    message: str,
    failure_code: str,
) -> RuntimeError:
    from backend.app.services.trake.event_retrieval import (
        RequiredTrakePipelineError,
    )

    return RequiredTrakePipelineError(message, failure_code=failure_code)


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


def load_dense_candidate_index_config() -> DenseCandidateIndexConfig:
    """Resolve the full dense-candidate bundle independently of selected FAISS."""

    defaults = DenseCandidateIndexConfig()
    return DenseCandidateIndexConfig(
        index_path=_path_from_env(
            "RETRIEVAL_DENSE_INDEX_PATH",
            defaults.index_path,
        ),
        metadata_path=_path_from_env(
            "RETRIEVAL_DENSE_METADATA_PATH",
            defaults.metadata_path,
        ),
        frame_map_path=_path_from_env(
            "RETRIEVAL_DENSE_FRAME_MAP_PATH",
            defaults.frame_map_path,
        ),
        manifest_path=_path_from_env(
            "RETRIEVAL_DENSE_MANIFEST_PATH",
            defaults.manifest_path,
        ),
        report_path=_path_from_env(
            "RETRIEVAL_DENSE_REPORT_PATH",
            defaults.report_path,
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


@_corpus_generation_cached
def get_dense_candidate_index(
    corpus_key: _CorpusCacheKey,
) -> FaissDenseCandidateIndex:
    """Load and cache the production corpus-wide dense SigLIP2 index."""

    config = load_dense_candidate_index_config()
    roles = (
        "dense_index",
        "dense_metadata",
        "dense_frame_map",
        "dense_manifest",
        "dense_report",
    )
    overrides = {
        "dense_index": config.index_path,
        "dense_metadata": config.metadata_path,
        "dense_frame_map": config.frame_map_path,
        "dense_manifest": config.manifest_path,
        "dense_report": config.report_path,
    }
    # A committed pre-dense corpus is a supported "missing bundle" case for
    # ``online.dense_missing_behavior=fallback_sparse``.  A partially declared
    # dense bundle is instead an invalid committed generation and must fail
    # closed rather than disguise corruption as a legacy fallback.
    base_manifest = _validate_expected_corpus(corpus_key)
    if base_manifest is not None:
        declared = base_manifest.get("artifacts")
        if not isinstance(declared, Mapping):
            raise ValueError("Offline corpus manifest has no artifact declarations")
        declared_dense_roles = tuple(role for role in roles if role in declared)
        if not declared_dense_roles:
            raise FileNotFoundError(
                "Committed corpus does not contain a dense-candidate bundle"
            )
        if len(declared_dense_roles) != len(roles):
            raise ValueError("Committed corpus dense-candidate bundle is incomplete")
    before = _validate_expected_corpus(
        corpus_key,
        required_roles=roles,
        artifact_overrides=overrides,
    )
    dense = FaissDenseCandidateIndex(config)
    visual = get_visual_search_engine()
    selected_contract = getattr(visual, "encoder_contract", None)
    if selected_contract is not None:
        selected = asdict(selected_contract)
        dense_contract = dict(dense.encoder_contract)
        comparable = (
            "model_family",
            "model_name",
            "model_revision",
            "processor_name",
            "vector_dim",
            "normalized",
            "similarity",
            "output_dtype",
        )
        if any(selected.get(name) != dense_contract.get(name) for name in comparable):
            raise ValueError(
                "Dense and selected-keyframe indexes use different encoder contracts"
            )
    after = _validate_expected_corpus(
        corpus_key,
        required_roles=roles,
        artifact_overrides=overrides,
    )
    if before != after:
        raise ValueError("Offline corpus changed while dense retrieval was loading")
    dense.corpus_generation = (
        str(before.get("bundle_generation")) if before is not None else None
    )
    return dense


def _get_online_dense_index_for_generation(
    corpus_key: _CorpusCacheKey,
) -> FaissDenseCandidateIndex:
    """Resolve lazily without ever mixing selected and dense generations."""

    dense = get_dense_candidate_index()
    if getattr(dense, "corpus_generation", None) != corpus_key.bundle_generation:
        raise ValueError("Cached dense retrieval belongs to another corpus generation")
    return dense


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
def get_online_context_index(
    corpus_key: _CorpusCacheKey,
) -> OnlineContextIndex | None:
    """Load enabled canonical context artifacts for the current generation."""

    neighbors_enabled = _bool_from_env(
        "ONLINE_NEIGHBOR_CONTEXT_ENABLED",
        False,
    )
    segments_enabled = _bool_from_env(
        "ONLINE_SEGMENT_CONTEXT_ENABLED",
        False,
    )
    if not neighbors_enabled and not segments_enabled:
        return None

    neighbor_path = _path_from_env(
        "ONLINE_NEIGHBOR_PATH",
        DEFAULT_NEIGHBOR_PATH,
    )
    segment_path = _path_from_env(
        "ONLINE_SEGMENT_PATH",
        DEFAULT_SEGMENT_PATH,
    )
    frame_map_path = load_visual_search_config().frame_map_path
    required_roles: list[str] = []
    overrides: dict[str, Path] = {}
    if neighbors_enabled and neighbor_path.is_file():
        required_roles.append("neighbor_metadata")
        overrides["neighbor_metadata"] = neighbor_path
    if segments_enabled and segment_path.is_file():
        required_roles.append("segment_metadata")
        overrides["segment_metadata"] = segment_path
    before = _validate_expected_corpus(
        corpus_key,
        required_roles=tuple(required_roles),
        artifact_overrides=overrides,
    )
    context_index = OnlineContextIndex.from_artifacts(
        neighbor_path=neighbor_path,
        segment_path=segment_path,
        frame_map_path=frame_map_path,
        load_neighbors=neighbors_enabled,
        load_segments=segments_enabled,
        # Context is optional for KIS/AVS. A missing enabled artifact disables
        # only that evidence source; a present but corrupt/uncommitted artifact
        # still fails validation above or while parsing here.
        require_neighbors=False,
        require_segments=False,
    )
    after = _validate_expected_corpus(
        corpus_key,
        required_roles=tuple(required_roles),
        artifact_overrides=overrides,
    )
    if before != after:
        raise ValueError("Offline corpus changed while online context was loading")
    return context_index


@_corpus_generation_cached
def get_qa_evidence_search_engine(
    corpus_key: _CorpusCacheKey,
) -> QaEvidenceSearchEngine:
    dense_text_engine = None
    dense_enabled = _bool_from_env("QA_BGE_DENSE_ENABLED", False)
    dense_root = _path_from_env("QA_BGE_INDEX_ROOT", Path("data/indexes/bge_m3"))
    bge_paths: dict[str, Path] = {}
    if dense_enabled:
        bge_paths = _bge_artifact_overrides(dense_root)
    before = _validate_expected_corpus(
        corpus_key,
        required_roles=tuple(bge_paths) if dense_enabled else (),
        artifact_overrides=bge_paths,
        require_bge=dense_enabled,
    )
    if dense_enabled:
        dense_text_engine = _shared_bge_m3_dense_search_engine(
            corpus_key=corpus_key,
            artifact_root=dense_root,
            model_revision=os.getenv("QA_BGE_MODEL_REVISION") or None,
            batch_size=_positive_int_from_env(
                "QA_BGE_BATCH_SIZE",
                16,
                maximum=10_000,
            ),
            device=os.getenv("QA_BGE_DEVICE", "auto"),
            cache_dir=_path_from_env(
                "QA_BGE_MODEL_CACHE_DIR",
                Path("data/model_cache/bge_m3"),
            ),
            local_files_only=_bool_from_env("QA_MODELS_LOCAL_ONLY", False),
        )
    reranker = None
    if _bool_from_env("QA_BGE_RERANKER_ENABLED", False):
        reranker = build_bge_candidate_reranker(
            model_name=os.getenv(
                "QA_BGE_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            ),
            model_revision=os.getenv("QA_BGE_RERANKER_REVISION", "main"),
            retrieval_alpha=float(os.getenv("QA_BGE_RERANKER_ALPHA", "0.5")),
            batch_size=int(os.getenv("QA_BGE_BATCH_SIZE", "16")),
            device=os.getenv("QA_BGE_DEVICE", "auto"),
            cache_dir=_path_from_env(
                "QA_BGE_MODEL_CACHE_DIR",
                Path("data/model_cache/bge_m3"),
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
        context_index=get_online_context_index(),
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


@_corpus_generation_cached
def get_trake_bge_dense_search_engine(
    corpus_key: _CorpusCacheKey,
) -> Any | None:
    """Load TRAKE's optional BGE-M3 index under the corpus publication gate."""

    config = get_runtime_config().trake
    if not config.bge_dense_enabled:
        return None

    artifact_root = _path_from_env(
        "RETRIEVAL_TRAKE_BGE_INDEX_ROOT",
        Path("data/indexes/bge_m3"),
    )
    artifact_overrides = _bge_artifact_overrides(artifact_root)
    try:
        before = _validate_expected_corpus(
            corpus_key,
            required_roles=tuple(artifact_overrides),
            artifact_overrides=artifact_overrides,
            require_bge=True,
        )
        engine = _shared_bge_m3_dense_search_engine(
            corpus_key=corpus_key,
            artifact_root=artifact_root,
            model_name=os.getenv(
                "RETRIEVAL_TRAKE_BGE_MODEL_NAME",
                "BAAI/bge-m3",
            ),
            model_revision=(
                os.getenv("RETRIEVAL_TRAKE_BGE_MODEL_REVISION") or None
            ),
            batch_size=_positive_int_from_env(
                "RETRIEVAL_TRAKE_BGE_BATCH_SIZE",
                16,
                maximum=10_000,
            ),
            device=os.getenv("RETRIEVAL_TRAKE_BGE_DEVICE", "auto"),
            cache_dir=_path_from_env(
                "RETRIEVAL_TRAKE_BGE_MODEL_CACHE_DIR",
                Path("data/model_cache/bge_m3"),
            ),
            local_files_only=_bool_from_env(
                "RETRIEVAL_TRAKE_BGE_LOCAL_FILES_ONLY",
                False,
            ),
        )
        after = _validate_expected_corpus(
            corpus_key,
            required_roles=tuple(artifact_overrides),
            artifact_overrides=artifact_overrides,
            require_bge=True,
        )
        if before != after:
            raise ValueError(
                "Offline corpus changed while TRAKE BGE-M3 retrieval was loading"
            )
    except Exception as exc:
        # A corpus generation flip is never an optional model failure. Re-read
        # the base gate so that a concurrent publication still fails closed.
        _validate_expected_corpus(corpus_key)
        if config.bge_required:
            raise _required_trake_dependency_error(
                "Required TRAKE BGE-M3 dense retrieval failed to initialize",
                "required_bge_dense_initialization_failed",
            ) from None
        _LOGGER.warning(
            "Optional TRAKE BGE-M3 dense retrieval is unavailable; "
            "continuing with canonical hybrid retrieval "
            "(reason=initialization_failed, failure_type=%s)",
            type(exc).__name__,
        )
        return None

    dense_model = _dense_model_contract(engine)
    if config.bge_required:
        if not _is_public_hub_model_id(dense_model.get("name")):
            raise _required_trake_dependency_error(
                "Required TRAKE BGE-M3 needs a public hub model id",
                "required_bge_dense_model_unverifiable",
            )
        if not _is_immutable_model_revision(dense_model.get("revision")):
            raise _required_trake_dependency_error(
                "Required TRAKE BGE-M3 manifest revision must be an immutable commit",
                "required_bge_dense_revision_unpinned",
            )
    engine.corpus_generation = corpus_key.bundle_generation
    return engine


def build_trake_bge_candidate_reranker() -> BgeCandidateReranker | None:
    """Build TRAKE's independently configured optional BGE reranker."""

    config = get_runtime_config().trake
    if not config.bge_reranker_enabled:
        return None
    model_name = os.getenv(
        "RETRIEVAL_TRAKE_BGE_RERANKER_MODEL",
        "BAAI/bge-reranker-v2-m3",
    )
    model_revision = os.getenv(
        "RETRIEVAL_TRAKE_BGE_RERANKER_REVISION",
        "main",
    )
    if config.bge_required:
        if not _is_public_hub_model_id(model_name):
            raise _required_trake_dependency_error(
                "Required TRAKE BGE reranker needs a public hub model id",
                "required_bge_reranker_model_unverifiable",
            )
        if not _is_immutable_model_revision(model_revision):
            raise _required_trake_dependency_error(
                "Required TRAKE BGE reranker revision must be an immutable commit",
                "required_bge_reranker_revision_unpinned",
            )
    try:
        return build_bge_candidate_reranker(
            model_name=model_name,
            model_revision=model_revision,
            candidate_limit=config.bge_reranker_top_k,
            retrieval_alpha=_unit_float_from_env(
                "RETRIEVAL_TRAKE_BGE_RERANKER_ALPHA",
                0.5,
            ),
            batch_size=_positive_int_from_env(
                "RETRIEVAL_TRAKE_BGE_RERANKER_BATCH_SIZE",
                16,
                maximum=10_000,
            ),
            device=os.getenv("RETRIEVAL_TRAKE_BGE_DEVICE", "auto"),
            cache_dir=_path_from_env(
                "RETRIEVAL_TRAKE_BGE_MODEL_CACHE_DIR",
                Path("data/model_cache/bge_m3"),
            ),
            local_files_only=_bool_from_env(
                "RETRIEVAL_TRAKE_BGE_LOCAL_FILES_ONLY",
                False,
            ),
        )
    except Exception as exc:
        if config.bge_required:
            raise _required_trake_dependency_error(
                "Required TRAKE BGE candidate reranker failed to initialize",
                "required_bge_reranker_initialization_failed",
            ) from None
        _LOGGER.warning(
            "Optional TRAKE BGE candidate reranker is unavailable; "
            "continuing without cross-encoder reranking "
            "(reason=initialization_failed, failure_type=%s)",
            type(exc).__name__,
        )
        return None


def _trake_bge_contract(
    *,
    corpus_key: _CorpusCacheKey,
    dense_event_engine: Any | None,
    event_reranker: Any | None,
) -> dict[str, Any]:
    """Return a path-free immutable-style model/artifact contract for trace."""

    config = get_runtime_config().trake
    dense_manifest: Mapping[str, Any] = {}
    artifacts = getattr(dense_event_engine, "artifacts", None)
    manifest = getattr(artifacts, "manifest", None)
    if isinstance(manifest, Mapping):
        dense_manifest = manifest
    model = _dense_model_contract(dense_event_engine)
    artifact_contract = dense_manifest.get("artifacts", {})
    if not isinstance(artifact_contract, Mapping):
        artifact_contract = {}

    def checksum(role: str) -> str | None:
        value = artifact_contract.get(role, {})
        if not isinstance(value, Mapping):
            return None
        digest = value.get("sha256")
        return str(digest) if isinstance(digest, str) and digest else None

    return {
        "corpus_generation": corpus_key.bundle_generation,
        "dense": {
            "enabled": bool(config.bge_dense_enabled),
            "available": dense_event_engine is not None,
            "model_name": _public_model_identifier(model.get("name")),
            "model_revision": _public_model_revision(model.get("revision")),
            "revision_source": (
                "manifest_resolved" if model.get("revision") is not None else None
            ),
            "revision_pinned": _is_immutable_model_revision(
                model.get("revision")
            ),
            "index_schema_version": dense_manifest.get("schema_version"),
            "vector_count": dense_manifest.get("vector_count"),
            "index_sha256": checksum("index"),
            "frame_map_sha256": checksum("frame_map"),
        },
        "reranker": {
            "enabled": bool(config.bge_reranker_enabled),
            "available": event_reranker is not None,
            "model_name": _public_model_identifier(
                getattr(event_reranker, "model_name", None)
            ),
            "model_revision": _public_model_revision(
                getattr(event_reranker, "model_revision", None)
            ),
            "revision_source": "requested",
            "revision_pinned": _is_immutable_model_revision(
                getattr(event_reranker, "model_revision", None)
            ),
            "candidate_limit": getattr(event_reranker, "candidate_limit", None),
        },
        "fusion": {
            "method": config.retrieval_fusion,
            "rrf_k": config.rrf_k,
            "hybrid_weight": config.hybrid_rrf_weight,
            "bge_weight": config.bge_rrf_weight,
            "required": config.bge_required,
        },
    }


def _dense_model_contract(engine: Any | None) -> Mapping[str, Any]:
    artifacts = getattr(engine, "artifacts", None)
    manifest = getattr(artifacts, "manifest", None)
    if not isinstance(manifest, Mapping):
        return {}
    model = manifest.get("model", {})
    return model if isinstance(model, Mapping) else {}


_PUBLIC_HUB_MODEL_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)
_PUBLIC_MODEL_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SENSITIVE_IDENTIFIER_TERMS = ("token", "secret", "password", "credential")


def _public_model_identifier(value: Any) -> str | None:
    """Expose a hub-style id, never a local path/URL/opaque identifier."""

    if value is None:
        return None
    identifier = str(value).strip()
    folded = identifier.casefold()
    if any(term in folded for term in _SENSITIVE_IDENTIFIER_TERMS):
        return "local_or_redacted"
    if not _PUBLIC_HUB_MODEL_ID.fullmatch(identifier):
        return "local_or_redacted"
    return identifier


def _public_model_revision(value: Any) -> str | None:
    """Expose a simple tag/commit while rejecting paths and secret-shaped text."""

    if value is None:
        return None
    revision = str(value).strip()
    folded = revision.casefold()
    if any(term in folded for term in _SENSITIVE_IDENTIFIER_TERMS):
        return "redacted"
    if not _PUBLIC_MODEL_REVISION.fullmatch(revision):
        return "redacted"
    return revision


def _is_public_hub_model_id(value: Any) -> bool:
    if value is None:
        return False
    identifier = str(value).strip()
    folded = identifier.casefold()
    return (
        not any(term in folded for term in _SENSITIVE_IDENTIFIER_TERMS)
        and _PUBLIC_HUB_MODEL_ID.fullmatch(identifier) is not None
    )


def _is_immutable_model_revision(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    # Hugging Face resolved revisions are currently Git SHA-1 commits.  Accept
    # longer hexadecimal object ids as well so the policy remains future-safe.
    return re.fullmatch(r"[0-9a-fA-F]{40,64}", str(value).strip()) is not None


@_corpus_generation_cached
def get_trake_pipeline(
    corpus_key: _CorpusCacheKey,
) -> Any:
    """Build the public TRAKE pipeline for one committed corpus generation."""

    # Lazy import keeps retrieval configuration and the online orchestrator free
    # from a cycle while preserving the expensive pipeline's on-demand loading.
    from backend.app.services.trake.pipeline import TrakePipeline

    runtime = get_runtime_config()
    if runtime.trake.bge_required and corpus_key.bundle_generation is None:
        raise _required_trake_dependency_error(
            "Required TRAKE BGE needs a committed offline corpus manifest",
            "required_corpus_manifest_unavailable",
        )
    dense_event_engine = (
        get_trake_bge_dense_search_engine()
        if runtime.trake.bge_dense_enabled
        else None
    )
    event_reranker = (
        build_trake_bge_candidate_reranker()
        if runtime.trake.bge_reranker_enabled
        else None
    )
    if runtime.trake.bge_required:
        if runtime.trake.bge_dense_enabled and dense_event_engine is None:
            raise _required_trake_dependency_error(
                "Required TRAKE BGE-M3 dense retrieval is unavailable",
                "required_bge_dense_unavailable",
            )
        if runtime.trake.bge_reranker_enabled and event_reranker is None:
            raise _required_trake_dependency_error(
                "Required TRAKE BGE candidate reranker is unavailable",
                "required_bge_reranker_unavailable",
            )
    retrieval_engine = get_hybrid_search_engine()
    if (
        corpus_key.bundle_generation is not None
        and getattr(retrieval_engine, "corpus_generation", None)
        != corpus_key.bundle_generation
    ):
        raise ValueError("Cached TRAKE retrieval belongs to another corpus generation")
    local_scorer = None
    if runtime.trake.refinement_enabled:
        # Reuse the already cached SigLIP2 encoder/model.  This is bounded
        # local raw-frame scoring, not the corpus-wide dense rescue index and
        # not a VLM/cross-encoder reranker.
        from backend.app.services.trake.temporal_refinement import (
            Siglip2LocalFrameScorer,
        )

        local_scorer = Siglip2LocalFrameScorer(retrieval_engine.visual_engine.encoder)
    pipeline = TrakePipeline(
        retrieval_engine=retrieval_engine,
        dense_event_engine=dense_event_engine,
        event_reranker=event_reranker,
        bge_contract=_trake_bge_contract(
            corpus_key=corpus_key,
            dense_event_engine=dense_event_engine,
            event_reranker=event_reranker,
        ),
        local_scorer=local_scorer,
        config=runtime.trake,
    )
    if dense_event_engine is not None:
        if (
            corpus_key.bundle_generation is not None
            and getattr(dense_event_engine, "corpus_generation", None)
            != corpus_key.bundle_generation
        ):
            raise ValueError(
                "Cached TRAKE BGE retrieval belongs to another corpus generation"
            )
        bge_root = _path_from_env(
            "RETRIEVAL_TRAKE_BGE_INDEX_ROOT",
            Path("data/indexes/bge_m3"),
        )
        bge_overrides = _bge_artifact_overrides(bge_root)
        _validate_expected_corpus(
            corpus_key,
            required_roles=tuple(bge_overrides),
            artifact_overrides=bge_overrides,
            require_bge=True,
        )
    else:
        _validate_expected_corpus(corpus_key)
    pipeline.corpus_generation = corpus_key.bundle_generation
    return pipeline


@_corpus_generation_cached
def get_online_pipeline(
    corpus_key: _CorpusCacheKey,
) -> OnlinePipeline:
    """Build the canonical online-only orchestrator for one corpus generation."""

    runtime = get_runtime_config()
    neighbors_enabled = _bool_from_env(
        "ONLINE_NEIGHBOR_CONTEXT_ENABLED",
        False,
    )
    segments_enabled = _bool_from_env(
        "ONLINE_SEGMENT_CONTEXT_ENABLED",
        False,
    )
    context_index = get_online_context_index()

    pipeline = OnlinePipeline(
        hybrid_engine=get_hybrid_search_engine(),
        runtime_config=runtime,
        query_expansion_provider=(
            get_query_expansion_provider()
            if runtime.query_expansion.enabled
            else None
        ),
        qa_pipeline=_LazyQaSearchPipeline(corpus_key.bundle_generation),
        qa_evidence_engine=_LazyQaEvidenceSearchEngine(
            corpus_key.bundle_generation
        ),
        trake_pipeline=_LazyTrakeSearchPipeline(corpus_key.bundle_generation),
        context_index=context_index,
        # Resolution is lazy so missing dense artifacts cannot block QA or
        # TRAKE.  The loader itself is generation-aware and globally cached;
        # normal KIS/AVS requests never rebuild or reload it per request.
        dense_index_loader=lambda: _get_online_dense_index_for_generation(corpus_key),
        config=OnlinePipelineConfig(
            include_neighbors=neighbors_enabled,
            include_segments=segments_enabled,
            max_top_k=runtime.hybrid.max_top_k,
        ),
    )
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
        get_online_pipeline,
        get_trake_pipeline,
        get_trake_bge_dense_search_engine,
        get_qa_search_pipeline,
        get_qa_evidence_search_engine,
        get_online_context_index,
        get_dense_candidate_index,
        get_hybrid_search_engine,
        get_object_search_engine,
        get_ocr_search_engine,
        get_caption_search_engine,
        get_visual_search_engine,
        get_query_expansion_provider,
        get_runtime_config,
    ):
        cached.cache_clear()
    with _BGE_DENSE_ENGINE_LOCK:
        _BGE_DENSE_ENGINES.clear()
    from backend.app.services.retrieval.bge_reranker import (
        clear_shared_bge_reranker_runners,
    )

    clear_shared_bge_reranker_runners()


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


def search_trake(
    query: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Return ranked same-video TRAKE frame sequences, never flattened frames."""

    config = get_runtime_config().trake
    requested_top_k = config.max_answers if top_k is None else int(top_k)
    requested_top_k = max(1, min(requested_top_k, config.max_answers, 100))
    return get_trake_pipeline().search(query=query, top_k=requested_top_k)


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


def search_online(
    query: str,
    task: str = "kis",
    top_k: int | None = None,
    *,
    expanded_queries: tuple[str, ...] | list[str] | None = None,
    include_context: bool | None = None,
    debug: bool | None = None,
) -> dict[str, Any]:
    """Run one query through the canonical online orchestration layer."""

    return get_online_pipeline().run(
        query=query,
        task=task,
        top_k=top_k,
        expanded_queries=expanded_queries or (),
        include_context=include_context,
        debug=debug,
    )
