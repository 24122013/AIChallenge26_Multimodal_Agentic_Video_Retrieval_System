"""Config loader for Retrieval Phase 2/3 settings."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode

from backend.app.core.environment import PROJECT_ENV_PATH, load_project_env
from backend.app.services.agent.query_expansion import QueryExpansionConfig
from backend.app.services.retrieval.advanced_rerank import (
    AdvancedRerankWeights,
    ContextRerankConfig,
)
from backend.app.services.retrieval.hybrid_search import HybridSearchConfig
from backend.app.services.retrieval.rerank import RerankConfig, RerankWeights


DEFAULT_RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")
DEFAULT_ENV_PATH = PROJECT_ENV_PATH
_LOGGER = logging.getLogger(__name__)
_DEPRECATED_ONLINE_SETTINGS = {
    "learned_rerank_enabled": "boolean",
    "vlm_rerank_enabled": "boolean",
}
_DEPRECATED_ONLINE_ENV = (
    "RETRIEVAL_ONLINE_LEARNED_RERANK_ENABLED",
    "RETRIEVAL_ONLINE_VLM_RERANK_ENABLED",
)


_CONFIG_SCHEMA = {
    "hybrid": {
        "stage1_top_k": "positive integer",
        "text_stage1_top_k": "positive integer",
        "rerank_pool_size": "positive integer",
        "default_top_k": "positive integer",
        "max_top_k": "positive integer",
        "max_gap_seconds": "positive number",
    },
    "weights": {
        "visual": "non-negative number",
        "caption": "non-negative number",
        "ocr": "non-negative number",
        "objects": "non-negative number",
        "temporal": "non-negative number",
    },
    "dedupe": {
        "same_shot": "boolean",
    },
    "text_index": {
        "path": "non-empty string",
        "default_top_k": "positive integer",
        "max_top_k": "positive integer",
    },
    "query_expansion": {
        "enabled": "boolean",
        "max_paraphrases": "non-negative integer",
        "original_weight": "non-negative number",
        "paraphrase_weight": "non-negative number",
        "max_expansion_contribution": "positive number",
        "max_query_chars": "positive integer",
        "hard_paraphrase_limit": "positive integer",
        "model_name": "non-empty string",
        "model_revision": "non-empty string",
        "timeout_seconds": "positive number",
        "max_new_tokens": "positive integer",
        "dtype": "non-empty string",
        "quantization": "non-empty string",
    },
    "online": {
        "coarse_to_dense_enabled": "boolean",
        "dense_enabled": "boolean",
        "dense_missing_behavior": "dense missing behavior string",
        "coarse_top_n": "positive integer",
        "dense_global_top_k": "positive integer",
        "dense_rescue_clips": "non-negative integer",
        "max_total_clips": "positive integer",
        "dense_frames_per_clip": "positive integer",
        "rrf_k": "positive integer",
        "modality_hint_boost": "positive number",
        "similarity_threshold": "unit interval number",
        "temporal_window_seconds": "non-negative number",
        "max_event_gap_seconds": "positive number",
        "rrf_enabled": "boolean",
        "cses_enabled": "boolean",
        "deterministic_rerank_enabled": "boolean",
        "debug_enabled": "boolean",
        "neighbor_scoring_enabled": "boolean",
        "segment_scoring_enabled": "boolean",
        "max_neighbors_each_side": "positive integer",
        "segment_context_candidate_limit": "positive integer",
        "segment_context_top_k": "positive integer",
        "context_max_bonus": "unit interval number",
        "rerank_coarse_rrf_weight": "non-negative number",
        "rerank_dense_visual_weight": "non-negative number",
        "rerank_caption_weight": "non-negative number",
        "rerank_ocr_weight": "non-negative number",
        "rerank_objects_weight": "non-negative number",
        "rerank_cses_gain_weight": "non-negative number",
        "rerank_temporal_consistency_weight": "non-negative number",
        "rerank_modality_alignment_weight": "non-negative number",
        "rerank_neighbor_support_weight": "non-negative number",
        "rerank_segment_support_weight": "non-negative number",
    },
    "trake": {
        "bge_dense_enabled": "boolean",
        "bge_dense_top_k": "positive integer",
        "bge_reranker_enabled": "boolean",
        "bge_reranker_top_k": "positive integer",
        "retrieval_fusion": "rrf string",
        "rrf_k": "positive integer",
        "hybrid_rrf_weight": "non-negative number",
        "bge_rrf_weight": "non-negative number",
        "bge_required": "boolean",
        "event_top_k": "positive integer",
        "top_videos": "positive integer",
        "max_candidates_per_event_per_video": "positive integer",
        "max_candidates_per_shot": "positive integer",
        "score_normalization": "rank or percentile string",
        "context_weight": "non-negative number",
        "coverage_weight": "non-negative number",
        "event_support_weight": "non-negative number",
        "alignment_method": "beam or dp string",
        "beam_width": "positive integer",
        "k_best_paths_per_video": "positive integer",
        "gap_penalty": "none, linear, or log string",
        "gap_lambda": "non-negative number",
        "refinement_enabled": "boolean",
        "refinement_top_paths": "positive integer",
        "window_before_frames": "non-negative integer",
        "window_after_frames": "non-negative integer",
        "dense_stride_frames": "positive integer",
        "local_hypotheses_per_event": "positive integer",
        "max_answers": "integer from 1 to 100",
        "ranking_cutoffs": "strictly increasing cutoff list up to 100",
    },
}


class RetrievalConfigError(ValueError):
    """Raised for invalid YAML syntax or an invalid Retrieval config schema."""


def _validate_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"trake.{name} must be a positive integer")


def _validate_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"trake.{name} must be a non-negative integer")


def _validate_non_negative_number(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"trake.{name} must be a non-negative finite number")


@dataclass(frozen=True)
class TextIndexConfig:
    path: Path = Path("data/indexes/retrieval_text_index.json")
    default_top_k: int = 20
    max_top_k: int = 200


@dataclass(frozen=True)
class OnlineRetrievalConfig:
    """Validated controls for the canonical KIS/AVS coarse-to-dense route."""

    coarse_to_dense_enabled: bool = True
    dense_enabled: bool = True
    dense_missing_behavior: str = "fallback_sparse"
    coarse_top_n: int = 50
    dense_global_top_k: int = 300
    dense_rescue_clips: int = 10
    max_total_clips: int = 60
    dense_frames_per_clip: int = 12
    rrf_k: int = 60
    modality_hint_boost: float = 1.5
    similarity_threshold: float = 0.92
    temporal_window_seconds: float = 2.0
    max_event_gap_seconds: float = 180.0
    rrf_enabled: bool = True
    cses_enabled: bool = True
    deterministic_rerank_enabled: bool = True
    debug_enabled: bool = False
    rerank_weights: AdvancedRerankWeights = AdvancedRerankWeights()
    context_config: ContextRerankConfig = ContextRerankConfig(
        neighbor_enabled=True,
        segment_enabled=True,
    )

    def __post_init__(self) -> None:
        for name in (
            "coarse_top_n",
            "dense_global_top_k",
            "max_total_clips",
            "dense_frames_per_clip",
            "rrf_k",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"online.{name} must be a positive integer")
        if (
            isinstance(self.dense_rescue_clips, bool)
            or not isinstance(self.dense_rescue_clips, int)
            or self.dense_rescue_clips < 0
        ):
            raise ValueError("online.dense_rescue_clips must be a non-negative integer")
        if self.max_total_clips < self.coarse_top_n:
            raise ValueError("online.max_total_clips must be >= online.coarse_top_n")
        for name in ("modality_hint_boost", "max_event_gap_seconds"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"online.{name} must be a positive finite number")
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"online.{name} must be a positive finite number")
        if isinstance(self.temporal_window_seconds, bool) or not isinstance(
            self.temporal_window_seconds,
            (int, float),
        ):
            raise ValueError(
                "online.temporal_window_seconds must be a non-negative finite number"
            )
        if (
            not math.isfinite(float(self.temporal_window_seconds))
            or float(self.temporal_window_seconds) < 0
        ):
            raise ValueError(
                "online.temporal_window_seconds must be a non-negative finite number"
            )
        if isinstance(self.similarity_threshold, bool) or not isinstance(
            self.similarity_threshold,
            (int, float),
        ):
            raise ValueError("online.similarity_threshold must be between 0 and 1")
        if (
            not math.isfinite(float(self.similarity_threshold))
            or not 0 <= float(self.similarity_threshold) <= 1
        ):
            raise ValueError("online.similarity_threshold must be between 0 and 1")
        behavior = str(self.dense_missing_behavior).casefold().strip()
        if behavior not in {"error", "fallback_sparse"}:
            raise ValueError(
                "online.dense_missing_behavior must be error or fallback_sparse"
            )
        object.__setattr__(self, "dense_missing_behavior", behavior)
        for name in (
            "coarse_to_dense_enabled",
            "dense_enabled",
            "rrf_enabled",
            "cses_enabled",
            "deterministic_rerank_enabled",
            "debug_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"online.{name} must be a boolean")
        if not isinstance(self.rerank_weights, AdvancedRerankWeights):
            raise TypeError("online.rerank_weights must be AdvancedRerankWeights")
        if not isinstance(self.context_config, ContextRerankConfig):
            raise TypeError("online.context_config must be ContextRerankConfig")
        values = tuple(float(value) for value in self.rerank_weights.__dict__.values())
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("online deterministic rerank weights must be non-negative")
        if sum(values) <= 0:
            raise ValueError(
                "online deterministic rerank weights must include a positive value"
            )


@dataclass(frozen=True)
class TrakeConfig:
    """Validated runtime controls for public TRAKE sequence retrieval."""

    bge_dense_enabled: bool = False
    bge_dense_top_k: int = 300
    bge_reranker_enabled: bool = False
    bge_reranker_top_k: int = 150
    retrieval_fusion: str = "rrf"
    rrf_k: int = 60
    hybrid_rrf_weight: float = 1.0
    bge_rrf_weight: float = 1.0
    bge_required: bool = False
    event_top_k: int = 300
    top_videos: int = 30
    max_candidates_per_event_per_video: int = 20
    max_candidates_per_shot: int = 2
    score_normalization: str = "rank"
    context_weight: float = 0.10
    coverage_weight: float = 0.45
    event_support_weight: float = 0.45
    alignment_method: str = "beam"
    beam_width: int = 200
    k_best_paths_per_video: int = 10
    gap_penalty: str = "log"
    gap_lambda: float = 0.02
    refinement_enabled: bool = True
    refinement_top_paths: int = 20
    window_before_frames: int = 60
    window_after_frames: int = 60
    dense_stride_frames: int = 1
    local_hypotheses_per_event: int = 3
    max_answers: int = 100
    ranking_cutoffs: tuple[int, ...] = (1, 5, 20, 50, 100)

    def __post_init__(self) -> None:
        positive_fields = (
            "bge_dense_top_k",
            "bge_reranker_top_k",
            "rrf_k",
            "event_top_k",
            "top_videos",
            "max_candidates_per_event_per_video",
            "max_candidates_per_shot",
            "beam_width",
            "k_best_paths_per_video",
            "refinement_top_paths",
            "dense_stride_frames",
            "local_hypotheses_per_event",
            "max_answers",
        )
        for name in positive_fields:
            _validate_positive_int(name, getattr(self, name))
        for name in ("bge_dense_top_k", "bge_reranker_top_k", "event_top_k"):
            if getattr(self, name) > 10_000:
                raise ValueError(f"trake.{name} must not exceed 10000")
        for name in ("window_before_frames", "window_after_frames"):
            _validate_non_negative_int(name, getattr(self, name))
        if self.max_answers > 100:
            raise ValueError("trake.max_answers must be between 1 and 100")

        for name in (
            "context_weight",
            "coverage_weight",
            "event_support_weight",
            "gap_lambda",
            "hybrid_rrf_weight",
            "bge_rrf_weight",
        ):
            _validate_non_negative_number(name, getattr(self, name))
        if self.context_weight + self.coverage_weight + self.event_support_weight <= 0:
            raise ValueError("trake video scoring weights must include a positive value")
        if self.hybrid_rrf_weight + self.bge_rrf_weight <= 0:
            raise ValueError("trake RRF weights must include a positive value")
        if self.hybrid_rrf_weight <= 0 and not self.bge_dense_enabled:
            raise ValueError(
                "trake.hybrid_rrf_weight must be positive when BGE dense is disabled"
            )
        if self.bge_dense_enabled and self.bge_rrf_weight <= 0:
            raise ValueError(
                "trake.bge_rrf_weight must be positive when BGE dense is enabled"
            )

        retrieval_fusion = str(self.retrieval_fusion).casefold().strip()
        if retrieval_fusion != "rrf":
            raise ValueError("trake.retrieval_fusion must be rrf")
        object.__setattr__(self, "retrieval_fusion", retrieval_fusion)

        score_normalization = str(self.score_normalization).casefold().strip()
        if score_normalization not in {"rank", "percentile"}:
            raise ValueError(
                "trake.score_normalization must be one of: rank, percentile"
            )
        object.__setattr__(self, "score_normalization", score_normalization)

        alignment_method = str(self.alignment_method).casefold().strip()
        if alignment_method not in {"beam", "dp"}:
            raise ValueError("trake.alignment_method must be one of: beam, dp")
        object.__setattr__(self, "alignment_method", alignment_method)

        gap_penalty = str(self.gap_penalty).casefold().strip()
        if gap_penalty not in {"none", "linear", "log"}:
            raise ValueError("trake.gap_penalty must be one of: none, linear, log")
        object.__setattr__(self, "gap_penalty", gap_penalty)

        for name in (
            "bge_dense_enabled",
            "bge_reranker_enabled",
            "bge_required",
            "refinement_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"trake.{name} must be a boolean")
        if self.bge_required and not (
            self.bge_dense_enabled or self.bge_reranker_enabled
        ):
            raise ValueError(
                "trake.bge_required requires bge_dense_enabled or "
                "bge_reranker_enabled"
            )
        if isinstance(self.ranking_cutoffs, (str, bytes)):
            raise ValueError("trake.ranking_cutoffs must be a positive integer list")
        try:
            cutoffs = tuple(self.ranking_cutoffs)
        except TypeError as exc:
            raise ValueError(
                "trake.ranking_cutoffs must be a positive integer list"
            ) from exc
        if not cutoffs:
            raise ValueError("trake.ranking_cutoffs must not be empty")
        for cutoff in cutoffs:
            _validate_positive_int("ranking_cutoffs item", cutoff)
            if cutoff > 100:
                raise ValueError("trake.ranking_cutoffs values must not exceed 100")
        if tuple(sorted(set(cutoffs))) != cutoffs:
            raise ValueError(
                "trake.ranking_cutoffs must be unique and strictly increasing"
            )
        object.__setattr__(self, "ranking_cutoffs", cutoffs)


@dataclass(frozen=True)
class RetrievalRuntimeConfig:
    hybrid: HybridSearchConfig = HybridSearchConfig()
    rerank: RerankConfig = RerankConfig()
    text_index: TextIndexConfig = TextIndexConfig()
    query_expansion: QueryExpansionConfig = QueryExpansionConfig()
    online: OnlineRetrievalConfig = OnlineRetrievalConfig()
    trake: TrakeConfig = TrakeConfig()


def load_retrieval_runtime_config(
    config_path: str | Path | None = None,
) -> RetrievalRuntimeConfig:
    path = Path(
        config_path
        or os.getenv("RETRIEVAL_CONFIG_PATH")
        or DEFAULT_RETRIEVAL_CONFIG_PATH
    )
    raw = _load_yaml_config(path) if path.exists() else {}
    _warn_deprecated_online_env()
    hybrid_raw = _section(raw, "hybrid")
    weights_raw = _section(raw, "weights")
    dedupe_raw = _section(raw, "dedupe")
    text_raw = _section(raw, "text_index")
    expansion_raw = _section(raw, "query_expansion")
    online_raw = _section(raw, "online")
    trake_raw = _section(raw, "trake")

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
    query_expansion = QueryExpansionConfig(
        enabled=_bool_env(
            "RETRIEVAL_QUERY_EXPANSION_ENABLED",
            expansion_raw.get("enabled"),
            QueryExpansionConfig.enabled,
        ),
        max_paraphrases=_non_negative_int_env(
            "RETRIEVAL_QUERY_EXPANSION_MAX_PARAPHRASES",
            expansion_raw.get("max_paraphrases"),
            QueryExpansionConfig.max_paraphrases,
        ),
        original_weight=_float_env(
            "RETRIEVAL_QUERY_EXPANSION_ORIGINAL_WEIGHT",
            expansion_raw.get("original_weight"),
            QueryExpansionConfig.original_weight,
        ),
        paraphrase_weight=_float_env(
            "RETRIEVAL_QUERY_EXPANSION_PARAPHRASE_WEIGHT",
            expansion_raw.get("paraphrase_weight"),
            QueryExpansionConfig.paraphrase_weight,
        ),
        max_expansion_contribution=_float_env(
            "RETRIEVAL_QUERY_EXPANSION_MAX_CONTRIBUTION",
            expansion_raw.get("max_expansion_contribution"),
            QueryExpansionConfig.max_expansion_contribution,
        ),
        max_query_chars=_int_env(
            "RETRIEVAL_QUERY_EXPANSION_MAX_QUERY_CHARS",
            expansion_raw.get("max_query_chars"),
            QueryExpansionConfig.max_query_chars,
        ),
        hard_paraphrase_limit=_int_env(
            "RETRIEVAL_QUERY_EXPANSION_HARD_LIMIT",
            expansion_raw.get("hard_paraphrase_limit"),
            QueryExpansionConfig.hard_paraphrase_limit,
        ),
        model_name=str(
            os.getenv("RETRIEVAL_QUERY_EXPANSION_MODEL_NAME")
            or expansion_raw.get("model_name")
            or QueryExpansionConfig.model_name
        ),
        model_revision=str(
            os.getenv("RETRIEVAL_QUERY_EXPANSION_MODEL_REVISION")
            or expansion_raw.get("model_revision")
            or QueryExpansionConfig.model_revision
        ),
        timeout_seconds=_float_env(
            "RETRIEVAL_QUERY_EXPANSION_TIMEOUT_SECONDS",
            expansion_raw.get("timeout_seconds"),
            QueryExpansionConfig.timeout_seconds,
        ),
        max_new_tokens=_int_env(
            "RETRIEVAL_QUERY_EXPANSION_MAX_NEW_TOKENS",
            expansion_raw.get("max_new_tokens"),
            QueryExpansionConfig.max_new_tokens,
        ),
        dtype=str(
            os.getenv("RETRIEVAL_QUERY_EXPANSION_DTYPE")
            or expansion_raw.get("dtype")
            or QueryExpansionConfig.dtype
        ),
        quantization=str(
            os.getenv("RETRIEVAL_QUERY_EXPANSION_QUANTIZATION")
            or expansion_raw.get("quantization")
            or QueryExpansionConfig.quantization
        ),
    )
    online = OnlineRetrievalConfig(
        coarse_to_dense_enabled=_bool_env(
            "RETRIEVAL_ONLINE_COARSE_TO_DENSE_ENABLED",
            online_raw.get("coarse_to_dense_enabled"),
            OnlineRetrievalConfig.coarse_to_dense_enabled,
        ),
        dense_enabled=_bool_env(
            "RETRIEVAL_ONLINE_DENSE_ENABLED",
            online_raw.get("dense_enabled"),
            OnlineRetrievalConfig.dense_enabled,
        ),
        dense_missing_behavior=_string_env(
            "RETRIEVAL_ONLINE_DENSE_MISSING_BEHAVIOR",
            online_raw.get("dense_missing_behavior"),
            OnlineRetrievalConfig.dense_missing_behavior,
        ),
        coarse_top_n=_int_env(
            "RETRIEVAL_ONLINE_COARSE_TOP_N",
            online_raw.get("coarse_top_n"),
            OnlineRetrievalConfig.coarse_top_n,
        ),
        dense_global_top_k=_int_env(
            "RETRIEVAL_ONLINE_DENSE_GLOBAL_TOP_K",
            online_raw.get("dense_global_top_k"),
            OnlineRetrievalConfig.dense_global_top_k,
        ),
        dense_rescue_clips=_non_negative_int_env(
            "RETRIEVAL_ONLINE_DENSE_RESCUE_CLIPS",
            online_raw.get("dense_rescue_clips"),
            OnlineRetrievalConfig.dense_rescue_clips,
        ),
        max_total_clips=_int_env(
            "RETRIEVAL_ONLINE_MAX_TOTAL_CLIPS",
            online_raw.get("max_total_clips"),
            OnlineRetrievalConfig.max_total_clips,
        ),
        dense_frames_per_clip=_int_env(
            "RETRIEVAL_ONLINE_DENSE_FRAMES_PER_CLIP",
            online_raw.get("dense_frames_per_clip"),
            OnlineRetrievalConfig.dense_frames_per_clip,
        ),
        rrf_k=_int_env(
            "RETRIEVAL_ONLINE_RRF_K",
            online_raw.get("rrf_k"),
            OnlineRetrievalConfig.rrf_k,
        ),
        modality_hint_boost=_float_env(
            "RETRIEVAL_ONLINE_MODALITY_HINT_BOOST",
            online_raw.get("modality_hint_boost"),
            OnlineRetrievalConfig.modality_hint_boost,
        ),
        similarity_threshold=_float_env(
            "RETRIEVAL_ONLINE_SIMILARITY_THRESHOLD",
            online_raw.get("similarity_threshold"),
            OnlineRetrievalConfig.similarity_threshold,
        ),
        temporal_window_seconds=_float_env(
            "RETRIEVAL_ONLINE_TEMPORAL_WINDOW_SECONDS",
            online_raw.get("temporal_window_seconds"),
            OnlineRetrievalConfig.temporal_window_seconds,
        ),
        max_event_gap_seconds=_float_env(
            "RETRIEVAL_ONLINE_MAX_EVENT_GAP_SECONDS",
            online_raw.get("max_event_gap_seconds"),
            OnlineRetrievalConfig.max_event_gap_seconds,
        ),
        rrf_enabled=_bool_env(
            "RETRIEVAL_ONLINE_RRF_ENABLED",
            online_raw.get("rrf_enabled"),
            OnlineRetrievalConfig.rrf_enabled,
        ),
        cses_enabled=_bool_env(
            "RETRIEVAL_ONLINE_CSES_ENABLED",
            online_raw.get("cses_enabled"),
            OnlineRetrievalConfig.cses_enabled,
        ),
        deterministic_rerank_enabled=_bool_env(
            "RETRIEVAL_ONLINE_DETERMINISTIC_RERANK_ENABLED",
            online_raw.get("deterministic_rerank_enabled"),
            OnlineRetrievalConfig.deterministic_rerank_enabled,
        ),
        debug_enabled=_bool_env(
            "RETRIEVAL_ONLINE_DEBUG_ENABLED",
            online_raw.get("debug_enabled"),
            OnlineRetrievalConfig.debug_enabled,
        ),
        rerank_weights=AdvancedRerankWeights(
            coarse_rrf=_float_env(
                "RETRIEVAL_ONLINE_RERANK_COARSE_RRF_WEIGHT",
                online_raw.get("rerank_coarse_rrf_weight"),
                AdvancedRerankWeights.coarse_rrf,
            ),
            dense_visual=_float_env(
                "RETRIEVAL_ONLINE_RERANK_DENSE_VISUAL_WEIGHT",
                online_raw.get("rerank_dense_visual_weight"),
                AdvancedRerankWeights.dense_visual,
            ),
            caption=_float_env(
                "RETRIEVAL_ONLINE_RERANK_CAPTION_WEIGHT",
                online_raw.get("rerank_caption_weight"),
                AdvancedRerankWeights.caption,
            ),
            ocr=_float_env(
                "RETRIEVAL_ONLINE_RERANK_OCR_WEIGHT",
                online_raw.get("rerank_ocr_weight"),
                AdvancedRerankWeights.ocr,
            ),
            objects=_float_env(
                "RETRIEVAL_ONLINE_RERANK_OBJECTS_WEIGHT",
                online_raw.get("rerank_objects_weight"),
                AdvancedRerankWeights.objects,
            ),
            cses_gain=_float_env(
                "RETRIEVAL_ONLINE_RERANK_CSES_GAIN_WEIGHT",
                online_raw.get("rerank_cses_gain_weight"),
                AdvancedRerankWeights.cses_gain,
            ),
            temporal_consistency=_float_env(
                "RETRIEVAL_ONLINE_RERANK_TEMPORAL_CONSISTENCY_WEIGHT",
                online_raw.get("rerank_temporal_consistency_weight"),
                AdvancedRerankWeights.temporal_consistency,
            ),
            modality_alignment=_float_env(
                "RETRIEVAL_ONLINE_RERANK_MODALITY_ALIGNMENT_WEIGHT",
                online_raw.get("rerank_modality_alignment_weight"),
                AdvancedRerankWeights.modality_alignment,
            ),
            neighbor_support=_float_env(
                "RETRIEVAL_ONLINE_RERANK_NEIGHBOR_SUPPORT_WEIGHT",
                online_raw.get("rerank_neighbor_support_weight"),
                AdvancedRerankWeights.neighbor_support,
            ),
            segment_support=_float_env(
                "RETRIEVAL_ONLINE_RERANK_SEGMENT_SUPPORT_WEIGHT",
                online_raw.get("rerank_segment_support_weight"),
                AdvancedRerankWeights.segment_support,
            ),
        ),
        context_config=ContextRerankConfig(
            neighbor_enabled=_bool_env(
                "RETRIEVAL_ONLINE_NEIGHBOR_SCORING_ENABLED",
                online_raw.get("neighbor_scoring_enabled"),
                True,
            ),
            segment_enabled=_bool_env(
                "RETRIEVAL_ONLINE_SEGMENT_SCORING_ENABLED",
                online_raw.get("segment_scoring_enabled"),
                True,
            ),
            max_neighbors_each_side=_int_env(
                "RETRIEVAL_ONLINE_MAX_NEIGHBORS_EACH_SIDE",
                online_raw.get("max_neighbors_each_side"),
                ContextRerankConfig.max_neighbors_each_side,
            ),
            segment_candidate_limit=_int_env(
                "RETRIEVAL_ONLINE_SEGMENT_CONTEXT_CANDIDATE_LIMIT",
                online_raw.get("segment_context_candidate_limit"),
                ContextRerankConfig.segment_candidate_limit,
            ),
            segment_top_k=_int_env(
                "RETRIEVAL_ONLINE_SEGMENT_CONTEXT_TOP_K",
                online_raw.get("segment_context_top_k"),
                ContextRerankConfig.segment_top_k,
            ),
            max_bonus=_float_env(
                "RETRIEVAL_ONLINE_CONTEXT_MAX_BONUS",
                online_raw.get("context_max_bonus"),
                ContextRerankConfig.max_bonus,
            ),
        ),
    )
    trake = TrakeConfig(
        bge_dense_enabled=_bool_env(
            "RETRIEVAL_TRAKE_BGE_DENSE_ENABLED",
            trake_raw.get("bge_dense_enabled"),
            TrakeConfig.bge_dense_enabled,
        ),
        bge_dense_top_k=_int_env(
            "RETRIEVAL_TRAKE_BGE_DENSE_TOP_K",
            trake_raw.get("bge_dense_top_k"),
            TrakeConfig.bge_dense_top_k,
        ),
        bge_reranker_enabled=_bool_env(
            "RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED",
            trake_raw.get("bge_reranker_enabled"),
            TrakeConfig.bge_reranker_enabled,
        ),
        bge_reranker_top_k=_int_env(
            "RETRIEVAL_TRAKE_BGE_RERANKER_TOP_K",
            trake_raw.get("bge_reranker_top_k"),
            TrakeConfig.bge_reranker_top_k,
        ),
        retrieval_fusion=_string_env(
            "RETRIEVAL_TRAKE_RETRIEVAL_FUSION",
            trake_raw.get("retrieval_fusion"),
            TrakeConfig.retrieval_fusion,
        ),
        rrf_k=_int_env(
            "RETRIEVAL_TRAKE_RRF_K",
            trake_raw.get("rrf_k"),
            TrakeConfig.rrf_k,
        ),
        hybrid_rrf_weight=_float_env(
            "RETRIEVAL_TRAKE_HYBRID_RRF_WEIGHT",
            trake_raw.get("hybrid_rrf_weight"),
            TrakeConfig.hybrid_rrf_weight,
        ),
        bge_rrf_weight=_float_env(
            "RETRIEVAL_TRAKE_BGE_RRF_WEIGHT",
            trake_raw.get("bge_rrf_weight"),
            TrakeConfig.bge_rrf_weight,
        ),
        bge_required=_bool_env(
            "RETRIEVAL_TRAKE_BGE_REQUIRED",
            trake_raw.get("bge_required"),
            TrakeConfig.bge_required,
        ),
        event_top_k=_int_env(
            "RETRIEVAL_TRAKE_EVENT_TOP_K",
            trake_raw.get("event_top_k"),
            TrakeConfig.event_top_k,
        ),
        top_videos=_int_env(
            "RETRIEVAL_TRAKE_TOP_VIDEOS",
            trake_raw.get("top_videos"),
            TrakeConfig.top_videos,
        ),
        max_candidates_per_event_per_video=_int_env(
            "RETRIEVAL_TRAKE_MAX_CANDIDATES_PER_EVENT_PER_VIDEO",
            trake_raw.get("max_candidates_per_event_per_video"),
            TrakeConfig.max_candidates_per_event_per_video,
        ),
        max_candidates_per_shot=_int_env(
            "RETRIEVAL_TRAKE_MAX_CANDIDATES_PER_SHOT",
            trake_raw.get("max_candidates_per_shot"),
            TrakeConfig.max_candidates_per_shot,
        ),
        score_normalization=_string_env(
            "RETRIEVAL_TRAKE_SCORE_NORMALIZATION",
            trake_raw.get("score_normalization"),
            TrakeConfig.score_normalization,
        ),
        context_weight=_float_env(
            "RETRIEVAL_TRAKE_CONTEXT_WEIGHT",
            trake_raw.get("context_weight"),
            TrakeConfig.context_weight,
        ),
        coverage_weight=_float_env(
            "RETRIEVAL_TRAKE_COVERAGE_WEIGHT",
            trake_raw.get("coverage_weight"),
            TrakeConfig.coverage_weight,
        ),
        event_support_weight=_float_env(
            "RETRIEVAL_TRAKE_EVENT_SUPPORT_WEIGHT",
            trake_raw.get("event_support_weight"),
            TrakeConfig.event_support_weight,
        ),
        alignment_method=_string_env(
            "RETRIEVAL_TRAKE_ALIGNMENT_METHOD",
            trake_raw.get("alignment_method"),
            TrakeConfig.alignment_method,
        ),
        beam_width=_int_env(
            "RETRIEVAL_TRAKE_BEAM_WIDTH",
            trake_raw.get("beam_width"),
            TrakeConfig.beam_width,
        ),
        k_best_paths_per_video=_int_env(
            "RETRIEVAL_TRAKE_K_BEST_PATHS_PER_VIDEO",
            trake_raw.get("k_best_paths_per_video"),
            TrakeConfig.k_best_paths_per_video,
        ),
        gap_penalty=_string_env(
            "RETRIEVAL_TRAKE_GAP_PENALTY",
            trake_raw.get("gap_penalty"),
            TrakeConfig.gap_penalty,
        ),
        gap_lambda=_float_env(
            "RETRIEVAL_TRAKE_GAP_LAMBDA",
            trake_raw.get("gap_lambda"),
            TrakeConfig.gap_lambda,
        ),
        refinement_enabled=_bool_env(
            "RETRIEVAL_TRAKE_REFINEMENT_ENABLED",
            trake_raw.get("refinement_enabled"),
            TrakeConfig.refinement_enabled,
        ),
        refinement_top_paths=_int_env(
            "RETRIEVAL_TRAKE_REFINEMENT_TOP_PATHS",
            trake_raw.get("refinement_top_paths"),
            TrakeConfig.refinement_top_paths,
        ),
        window_before_frames=_non_negative_int_env(
            "RETRIEVAL_TRAKE_WINDOW_BEFORE_FRAMES",
            trake_raw.get("window_before_frames"),
            TrakeConfig.window_before_frames,
        ),
        window_after_frames=_non_negative_int_env(
            "RETRIEVAL_TRAKE_WINDOW_AFTER_FRAMES",
            trake_raw.get("window_after_frames"),
            TrakeConfig.window_after_frames,
        ),
        dense_stride_frames=_int_env(
            "RETRIEVAL_TRAKE_DENSE_STRIDE_FRAMES",
            trake_raw.get("dense_stride_frames"),
            TrakeConfig.dense_stride_frames,
        ),
        local_hypotheses_per_event=_int_env(
            "RETRIEVAL_TRAKE_LOCAL_HYPOTHESES_PER_EVENT",
            trake_raw.get("local_hypotheses_per_event"),
            TrakeConfig.local_hypotheses_per_event,
        ),
        max_answers=_int_env(
            "RETRIEVAL_TRAKE_MAX_ANSWERS",
            trake_raw.get("max_answers"),
            TrakeConfig.max_answers,
        ),
        ranking_cutoffs=_int_tuple_env(
            "RETRIEVAL_TRAKE_RANKING_CUTOFFS",
            trake_raw.get("ranking_cutoffs"),
            TrakeConfig.ranking_cutoffs,
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
        query_expansion=query_expansion,
        online=online,
        trake=trake,
    )


def _warn_deprecated_online_env() -> None:
    for name in _DEPRECATED_ONLINE_ENV:
        if os.getenv(name) is None:
            continue
        _bool_env(name, None, False)
        _LOGGER.warning(
            "%s is deprecated and ignored; canonical KIS/AVS uses only the "
            "deterministic multimodal reranker",
            name,
        )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RetrievalConfigError(
            f"Retrieval config section {name!r} must be a mapping"
        )
    return value


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load standard YAML with PyYAML, then validate the Retrieval schema."""
    text = path.read_text(encoding="utf-8")
    try:
        root_node = yaml.compose(text, Loader=yaml.SafeLoader)
        locations = _collect_config_locations(root_node, path=path)
        loaded = yaml.safe_load(text)
    except RetrievalConfigError:
        raise
    except yaml.YAMLError as exc:
        raise _yaml_parser_error(path, exc) from exc

    if loaded is None:
        root: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        root = loaded
    else:
        line_number = root_node.start_mark.line + 1 if root_node is not None else 1
        raise _yaml_error(
            path,
            line_number,
            f"document root must be a mapping; got {type(loaded).__name__}",
        )

    _validate_config_schema(root, locations=locations, path=path)
    return root


def _collect_config_locations(
    root_node: Node | None,
    *,
    path: Path,
) -> dict[tuple[str, ...], int]:
    """Collect setting lines and reject duplicate mapping keys before loading."""
    locations: dict[tuple[str, ...], int] = {}
    if not isinstance(root_node, MappingNode):
        return locations

    section_lines: dict[str, int] = {}
    for section_key_node, section_node in root_node.value:
        section_name = _config_key_name(section_key_node, path=path)
        if section_name is None:
            continue
        section_line = section_key_node.start_mark.line + 1
        if section_name in section_lines:
            raise _yaml_error(
                path,
                section_line,
                f"duplicate section {section_name!r}; first declared on line "
                f"{section_lines[section_name]}",
            )
        section_lines[section_name] = section_line
        locations[(section_name,)] = section_line

        if not isinstance(section_node, MappingNode):
            continue
        setting_lines: dict[str, int] = {}
        for setting_key_node, _ in section_node.value:
            setting_name = _config_key_name(setting_key_node, path=path)
            if setting_name is None:
                continue
            setting_line = setting_key_node.start_mark.line + 1
            if setting_name in setting_lines:
                raise _yaml_error(
                    path,
                    setting_line,
                    f"duplicate setting {section_name}.{setting_name}; first "
                    f"declared on line {setting_lines[setting_name]}",
                )
            setting_lines[setting_name] = setting_line
            locations[(section_name, setting_name)] = setting_line
    return locations


def _config_key_name(node: Node, *, path: Path) -> str | None:
    if not isinstance(node, ScalarNode):
        raise _yaml_error(
            path,
            node.start_mark.line + 1,
            "config mapping keys must be scalar strings",
        )
    if node.tag == "tag:yaml.org,2002:merge":
        return None
    if node.tag != "tag:yaml.org,2002:str":
        raise _yaml_error(
            path,
            node.start_mark.line + 1,
            f"config mapping keys must be strings; got {node.tag.rsplit(':', 1)[-1]}",
        )
    return node.value


def _validate_config_schema(
    root: dict[str, Any],
    *,
    locations: dict[tuple[str, ...], int],
    path: Path,
) -> None:
    for section_name, section in root.items():
        if not isinstance(section_name, str):
            raise _yaml_error(
                path,
                1,
                f"section names must be strings; got {section_name!r}",
            )
        section_line = locations.get((section_name,), 1)
        section_schema = _CONFIG_SCHEMA.get(section_name)
        if section_schema is None:
            supported = ", ".join(sorted(_CONFIG_SCHEMA))
            raise _yaml_error(
                path,
                section_line,
                f"unknown section {section_name!r}; supported sections: {supported}",
            )
        if not isinstance(section, dict):
            raise _yaml_error(
                path,
                section_line,
                f"section {section_name!r} must be a mapping; got "
                f"{type(section).__name__}",
            )

        for key, value in section.items():
            if not isinstance(key, str):
                raise _yaml_error(
                    path,
                    section_line,
                    f"setting names in {section_name!r} must be strings; got {key!r}",
                )
            expected = section_schema.get(key)
            line_number = locations.get((section_name, key), section_line)
            setting = f"{section_name}.{key}"
            if expected is None:
                deprecated_expected = (
                    _DEPRECATED_ONLINE_SETTINGS.get(key)
                    if section_name == "online"
                    else None
                )
                if deprecated_expected is not None:
                    if not _matches_expected_type(value, deprecated_expected):
                        raise _yaml_error(
                            path,
                            line_number,
                            f"deprecated {setting} must be a {deprecated_expected}; "
                            f"got {value!r}",
                        )
                    _LOGGER.warning(
                        "%s is deprecated and ignored; canonical KIS/AVS uses "
                        "only the deterministic multimodal reranker",
                        setting,
                    )
                    continue
                supported = ", ".join(sorted(section_schema))
                raise _yaml_error(
                    path,
                    line_number,
                    f"unknown setting {setting}; supported settings in "
                    f"{section_name!r}: {supported}",
                )
            if not _matches_expected_type(value, expected):
                raise _yaml_error(
                    path,
                    line_number,
                    f"{setting} must be a {expected}; got {value!r}",
                )


def _matches_expected_type(value: Any, expected: str) -> bool:
    is_number = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )
    if expected == "positive integer":
        return not isinstance(value, bool) and isinstance(value, int) and value > 0
    if expected == "non-negative integer":
        return not isinstance(value, bool) and isinstance(value, int) and value >= 0
    if expected == "positive number":
        return is_number and value > 0
    if expected == "non-negative number":
        return is_number and value >= 0
    if expected == "unit interval number":
        return is_number and 0 <= value <= 1
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "non-empty string":
        return isinstance(value, str) and bool(value.strip())
    if expected == "rrf string":
        return isinstance(value, str) and value.casefold().strip() == "rrf"
    if expected == "dense missing behavior string":
        return isinstance(value, str) and value.casefold().strip() in {
            "error",
            "fallback_sparse",
        }
    if expected == "positive integer list":
        return (
            isinstance(value, list)
            and bool(value)
            and all(
                not isinstance(item, bool)
                and isinstance(item, int)
                and item > 0
                for item in value
            )
        )
    if expected == "rank or percentile string":
        return isinstance(value, str) and value.casefold().strip() in {
            "rank",
            "percentile",
        }
    if expected == "beam or dp string":
        return isinstance(value, str) and value.casefold().strip() in {"beam", "dp"}
    if expected == "none, linear, or log string":
        return isinstance(value, str) and value.casefold().strip() in {
            "none",
            "linear",
            "log",
        }
    if expected == "integer from 1 to 100":
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and 1 <= value <= 100
        )
    if expected == "strictly increasing cutoff list up to 100":
        return (
            isinstance(value, list)
            and bool(value)
            and all(
                not isinstance(item, bool)
                and isinstance(item, int)
                and 1 <= item <= 100
                for item in value
            )
            and value == sorted(set(value))
        )
    raise AssertionError(f"Unknown retrieval config schema rule: {expected}")


def _yaml_parser_error(path: Path, error: yaml.YAMLError) -> RetrievalConfigError:
    mark = getattr(error, "problem_mark", None)
    problem = getattr(error, "problem", None)
    detail = problem or str(error).splitlines()[0] or type(error).__name__
    if mark is None:
        return RetrievalConfigError(f"Invalid retrieval YAML at {path}: {detail}")
    return RetrievalConfigError(
        f"Invalid retrieval YAML at {path}:{mark.line + 1}:{mark.column + 1}: "
        f"{detail}"
    )


def _yaml_error(path: Path, line_number: int, message: str) -> RetrievalConfigError:
    return RetrievalConfigError(
        f"Invalid retrieval YAML at {path}:{line_number}: {message}"
    )


def _int_env(name: str, value: Any, default: int) -> int:
    raw = os.getenv(name)
    result = int(raw if raw is not None else value if value is not None else default)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _float_env(name: str, value: Any, default: float) -> float:
    raw = os.getenv(name)
    return float(raw if raw is not None else value if value is not None else default)


def _non_negative_int_env(name: str, value: Any, default: int) -> int:
    raw = os.getenv(name)
    result = int(raw if raw is not None else value if value is not None else default)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


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


def _string_env(name: str, value: Any, default: str) -> str:
    raw = os.getenv(name)
    result = str(raw if raw is not None else value if value is not None else default)
    if not result.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _int_tuple_env(
    name: str,
    value: Any,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    raw = os.getenv(name)
    if raw is not None:
        pieces = [piece.strip() for piece in raw.strip().strip("[]").split(",")]
        if not pieces or any(not piece for piece in pieces):
            raise ValueError(f"{name} must be a comma-separated integer list")
        try:
            return tuple(int(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a comma-separated integer list"
            ) from exc
    selected = default if value is None else value
    if isinstance(selected, (str, bytes)):
        raise ValueError(f"{name} must be an integer list")
    try:
        return tuple(selected)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer list") from exc
