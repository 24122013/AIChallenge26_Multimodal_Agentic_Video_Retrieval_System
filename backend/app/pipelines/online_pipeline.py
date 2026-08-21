"""Canonical orchestration for online KIS, AVS, temporal, TRAKE, and QA."""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from backend.app.services.agent.query_expansion import (
    QueryExpansionConfig,
    QueryExpansionProvider,
)
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine
from backend.app.services.retrieval.advanced_search import (
    AdvancedSearchConfig,
    DenseCandidateIndex,
    advanced_text_search,
)
from backend.app.services.retrieval.online_context import OnlineContextIndex
from backend.app.services.retrieval.planned_hybrid import planned_hybrid_search
from backend.app.services.retrieval.qa_evidence import QaEvidenceSearchEngine
from backend.app.services.retrieval.qa_pipeline import QaSearchPipeline
from backend.app.services.retrieval.query_plan import QueryPlan, build_query_plan
from backend.app.services.retrieval.retrieval_config import (
    RetrievalRuntimeConfig,
    load_project_env,
)


ONLINE_SCHEMA_VERSION = "1.0"
SUPPORTED_ONLINE_TASKS = (
    "auto",
    "kis",
    "kis_visual",
    "kis_temporal",
    "avs",
    "temporal",
    "trake",
    "qa",
)


class TrakeSearchPipeline(Protocol):
    """Narrow dependency contract for the separately owned TRAKE pipeline."""

    def search(self, query: str, top_k: int = 100) -> Mapping[str, Any]: ...


DenseIndexLoader = Callable[[], DenseCandidateIndex]


class _TimedExpansionProvider:
    """Measure only provider work while preserving the existing provider API."""

    def __init__(self, provider: QueryExpansionProvider) -> None:
        self.provider = provider
        self.provider_name = provider.provider_name
        self.model_name = provider.model_name
        self.model_revision = provider.model_revision
        self.elapsed_ms = 0.0
        self.calls = 0

    def expand(self, query, protected):
        started = time.perf_counter()
        self.calls += 1
        try:
            return self.provider.expand(query, protected)
        finally:
            self.elapsed_ms = round(self.elapsed_ms + _elapsed_ms(started), 3)

    def close(self) -> None:
        # The production provider is shared and owned by retrieval_manager.
        return None


@dataclass(frozen=True)
class OnlinePipelineConfig:
    """Online-only feature switches.

    Canonical neighbor/segment context stays opt-in until retrieval ablations
    demonstrate that it should be promoted to the default critical path.
    """

    include_neighbors: bool = False
    include_segments: bool = False
    max_top_k: int = 200


@dataclass(frozen=True)
class Candidate:
    """Stable candidate schema shared by all online task routes."""

    video_id: str
    keyframe_id: str
    timestamp: float
    shot_id: str = ""
    segment_id: str = ""
    visual_score: float | None = None
    caption_score: float | None = None
    ocr_score: float | None = None
    object_score: float | None = None
    bge_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float = 0.0
    neighbors: tuple[dict[str, Any], ...] = ()
    segment_context: dict[str, Any] | None = None
    frame_index: int | None = None
    faiss_index: int | None = None
    keyframe_path: str = ""
    thumbnail_path: str = ""
    caption: str = ""
    ocr_text: str = ""
    objects: tuple[str, ...] = ()
    modality_scores: dict[str, float] = field(default_factory=dict)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    score_contributions: dict[str, float] = field(default_factory=dict)
    context_scoring: dict[str, Any] = field(default_factory=dict)
    cses_selection: dict[str, Any] | None = None
    temporal: dict[str, Any] = field(default_factory=dict)
    context_sources: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        context_index: OnlineContextIndex | None = None,
        include_neighbors: bool = False,
        include_segments: bool = False,
        max_neighbors_each_side: int | None = None,
    ) -> "Candidate":
        modality_scores = _float_mapping(value.get("modality_scores"))
        video_id = str(value.get("video_id") or "")
        keyframe_id = str(
            value.get("keyframe_id")
            or value.get("frame_id")
            or value.get("start_keyframe")
            or ""
        )
        timestamp = _float(value.get("timestamp"), default=0.0)
        segment_id = str(value.get("segment_id") or "")
        neighbors = tuple(
            dict(item)
            for item in (value.get("neighbors") or ())
            if isinstance(item, Mapping)
        )
        segment_context = None
        context_sources: tuple[str, ...] = ()
        if context_index is not None and (include_neighbors or include_segments):
            context = context_index.lookup(
                video_id=video_id,
                frame_id=keyframe_id,
                timestamp=timestamp,
                segment_id=segment_id,
                existing_neighbors=neighbors,
                max_neighbors_each_side=max_neighbors_each_side,
            )
            if include_neighbors:
                neighbors = context.neighbors
            if include_segments:
                segment_id = context.segment_id
                segment_context = context.segment
            allowed_sources = {
                "neighbors_all" if include_neighbors else "",
                "segments_all" if include_segments else "",
            }
            context_sources = tuple(
                source for source in context.sources if source in allowed_sources
            )

        temporal = {
            key: value[key]
            for key in (
                "temporal_event_index",
                "temporal_match_rank",
                "temporal_match_mode",
                "temporal_chain_id",
                "temporal_event_query",
                "temporal_event_role",
                "temporal_chain_score",
            )
            if key in value and value[key] is not None
        }
        fusion_value = value.get("fusion_score")
        if fusion_value is None:
            fusion_value = modality_scores.get(
                "rrf",
                modality_scores.get("fusion"),
            )
        rerank_value = value.get("rerank_score")
        if rerank_value is None:
            rerank_value = value.get("score", value.get("retrieval_score", 0.0))
        return cls(
            video_id=video_id,
            keyframe_id=keyframe_id,
            timestamp=timestamp,
            shot_id=str(value.get("shot_id") or ""),
            segment_id=segment_id,
            visual_score=_modality_score(modality_scores, "visual"),
            caption_score=_modality_score(modality_scores, "caption"),
            ocr_score=_modality_score(modality_scores, "ocr"),
            object_score=_modality_score(modality_scores, "objects", "object"),
            bge_score=_modality_score(modality_scores, "dense_text", "bge"),
            fusion_score=_optional_float(fusion_value),
            rerank_score=_float(rerank_value, default=0.0),
            neighbors=neighbors,
            segment_context=segment_context,
            frame_index=_optional_int(value.get("frame_index")),
            faiss_index=_optional_int(value.get("faiss_index")),
            keyframe_path=str(value.get("keyframe_path") or ""),
            thumbnail_path=str(value.get("thumbnail_path") or ""),
            caption=str(value.get("caption") or ""),
            ocr_text=str(value.get("ocr_text") or ""),
            objects=tuple(str(item) for item in (value.get("objects") or ()) if item),
            modality_scores=modality_scores,
            score_breakdown=_float_mapping(
                value.get("score_breakdown") or value.get("breakdown")
            ),
            score_contributions=_float_mapping(value.get("score_contributions")),
            context_scoring=(
                dict(value["context_scoring"])
                if isinstance(value.get("context_scoring"), Mapping)
                else {}
            ),
            cses_selection=(
                dict(value["cses_selection"])
                if isinstance(value.get("cses_selection"), Mapping)
                else (
                    dict(value["cses"])
                    if isinstance(value.get("cses"), Mapping)
                    else (
                        dict(value["selection"])
                        if isinstance(value.get("selection"), Mapping)
                        else None
                    )
                )
            ),
            temporal=temporal,
            context_sources=context_sources,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return explicit scores plus compatibility aliases used by the UI."""

        return {
            "video_id": self.video_id,
            "keyframe_id": self.keyframe_id,
            "frame_id": self.keyframe_id,
            "timestamp": self.timestamp,
            "shot_id": self.shot_id,
            "segment_id": self.segment_id,
            "visual_score": self.visual_score,
            "caption_score": self.caption_score,
            "ocr_score": self.ocr_score,
            "object_score": self.object_score,
            "bge_score": self.bge_score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "score": self.rerank_score,
            "neighbors": [dict(item) for item in self.neighbors],
            "segment_context": (
                dict(self.segment_context) if self.segment_context is not None else None
            ),
            "frame_index": self.frame_index,
            "faiss_index": self.faiss_index,
            "keyframe_path": self.keyframe_path,
            "thumbnail_path": self.thumbnail_path,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "objects": list(self.objects),
            "modality_scores": dict(self.modality_scores),
            "score_breakdown": dict(self.score_breakdown),
            "score_contributions": dict(self.score_contributions),
            "context_scoring": dict(self.context_scoring),
            "cses_selection": (
                dict(self.cses_selection)
                if self.cses_selection is not None
                else None
            ),
            "temporal": dict(self.temporal),
            "context_sources": list(self.context_sources),
        }


class OnlinePipeline:
    """One public entrypoint over the existing online retrieval services."""

    def __init__(
        self,
        *,
        hybrid_engine: HybridSearchEngine,
        runtime_config: RetrievalRuntimeConfig | None = None,
        query_expansion_provider: QueryExpansionProvider | None = None,
        qa_pipeline: QaSearchPipeline | None = None,
        qa_evidence_engine: QaEvidenceSearchEngine | None = None,
        trake_pipeline: TrakeSearchPipeline | None = None,
        context_index: OnlineContextIndex | None = None,
        dense_index: DenseCandidateIndex | None = None,
        dense_index_loader: DenseIndexLoader | None = None,
        config: OnlinePipelineConfig | None = None,
    ) -> None:
        self.hybrid_engine = hybrid_engine
        self.runtime_config = runtime_config or RetrievalRuntimeConfig()
        self.query_expansion_provider = query_expansion_provider
        self.qa_pipeline = qa_pipeline
        self.qa_evidence_engine = qa_evidence_engine
        self.trake_pipeline = trake_pipeline
        self.context_index = context_index
        self._dense_index = dense_index
        self._dense_index_loader = dense_index_loader
        self._dense_index_resolved = dense_index is not None
        self._dense_index_error = ""
        self._dense_index_lock = threading.RLock()
        self.config = config or OnlinePipelineConfig()
        if int(self.config.max_top_k) <= 0:
            raise ValueError("online max_top_k must be positive")

    def run(
        self,
        query: str,
        task: str = "kis",
        top_k: int | None = None,
        *,
        expanded_queries: Sequence[str] = (),
        include_context: bool | None = None,
        debug: bool | None = None,
    ) -> dict[str, Any]:
        """Route by task, then normalize all result candidates."""

        started_at = time.perf_counter()
        original_query = " ".join(str(query).split())
        if not original_query:
            raise ValueError("query must not be empty")
        requested_task = str(task or "kis").casefold().strip()
        if requested_task not in SUPPORTED_ONLINE_TASKS:
            raise ValueError(
                "Unsupported online task; expected "
                + ", ".join(SUPPORTED_ONLINE_TASKS)
            )
        requested_top_k = self._top_k(top_k, task=requested_task)

        if requested_task == "trake":
            if self.trake_pipeline is None:
                raise RuntimeError("TRAKE pipeline is unavailable in this online runtime")
            raw_trake = self.trake_pipeline.search(
                original_query,
                top_k=requested_top_k,
            )
            return _trake_response(
                raw_trake,
                query=original_query,
                top_k=requested_top_k,
                started_at=started_at,
                expansion_requested=self.runtime_config.query_expansion.enabled,
            )

        plan: QueryPlan | None = None
        stage_latency: dict[str, float] = {
            "query_planning_ms": 0.0,
            "query_expansion_ms": 0.0,
        }
        expansion_provider_calls = 0
        debug_enabled = (
            self.runtime_config.online.debug_enabled
            if debug is None
            else bool(debug)
        )
        include_neighbors, include_segments = self._context_flags(include_context)
        resolved_task = requested_task
        if requested_task in {"auto", "kis", "kis_visual", "kis_temporal", "avs"}:
            planning_profile = {
                "kis_visual": "kis",
                "kis_temporal": "temporal",
            }.get(requested_task, requested_task)
            modality_scope = (
                ("visual",) if requested_task == "kis_visual" else None
            )
            plan, planning_ms, expansion_ms, expansion_provider_calls = self._plan(
                original_query,
                planning_profile,
                modality_scope=modality_scope,
            )
            stage_latency["query_planning_ms"] = planning_ms
            stage_latency["query_expansion_ms"] = expansion_ms
            resolved_task = (
                plan.profile
                if requested_task == "auto"
                else (
                    "kis"
                    if requested_task in {"kis_visual", "kis_temporal"}
                    else requested_task
                )
            )
            if requested_task == "auto" and top_k is None and resolved_task == "qa":
                requested_top_k = 5

        if resolved_task in {"kis", "avs"}:
            assert plan is not None
            raw = self._run_kis_avs(
                original_query,
                plan=plan,
                top_k=requested_top_k,
                debug=debug_enabled,
                include_neighbors=include_neighbors,
                include_segments=include_segments,
            )
        elif resolved_task == "qa":
            if self.qa_pipeline is None:
                raise RuntimeError("QA pipeline is unavailable in this online runtime")
            raw = self.qa_pipeline.search(
                original_query,
                top_k=min(requested_top_k, 5),
                task_mode="auto" if requested_task == "auto" else "qa",
                expanded_queries=tuple(expanded_queries),
            )
        elif resolved_task == "temporal":
            if self.qa_evidence_engine is None:
                raise RuntimeError(
                    "Temporal evidence engine is unavailable in this online runtime"
                )
            raw = self.qa_evidence_engine.search(
                original_query,
                top_k=requested_top_k,
                task_mode="auto" if requested_task == "auto" else "temporal",
                # Event decomposition owns temporal queries; whole-question
                # expansions must not alter event order or meaning.
                expanded_queries=(),
            )
        else:
            raise RuntimeError(f"Online task routing failed: {resolved_task}")

        query_plan = _query_plan_payload(raw, plan)
        _validate_original_anchor(original_query, query_plan)
        context_started = time.perf_counter()
        raw_candidates = raw.get("results") or ()
        candidates = [
            Candidate.from_mapping(
                item,
                context_index=self.context_index,
                include_neighbors=include_neighbors,
                include_segments=include_segments,
                max_neighbors_each_side=(
                    self.runtime_config.online.context_config.max_neighbors_each_side
                ),
            )
            for item in raw_candidates
            if isinstance(item, Mapping)
        ]
        stage_latency["context_attachment_ms"] = _elapsed_ms(context_started)
        context_summary = (
            self.context_index.summary() if self.context_index is not None else {}
        )
        neighbors_available = int(context_summary.get("neighbor_record_count") or 0) > 0
        segments_available = int(context_summary.get("segment_record_count") or 0) > 0

        response: dict[str, Any] = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "query": original_query,
            "requested_task": requested_task,
            "task": resolved_task,
            "top_k": requested_top_k,
            "query_plan": query_plan,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "context": {
                "enabled": include_neighbors or include_segments,
                "neighbors_enabled": include_neighbors,
                "segments_enabled": include_segments,
                "neighbors_available": neighbors_available,
                "segments_available": segments_available,
                "fallback_reason": (
                    ""
                    if (
                        (not include_neighbors or neighbors_available)
                        and (not include_segments or segments_available)
                    )
                    else "requested_context_artifact_unavailable"
                ),
                "index": context_summary if self.context_index is not None else None,
            },
            "latency_ms": _elapsed_ms(started_at),
        }
        trace = raw.get("routing_trace") or raw.get("trace")
        if isinstance(trace, Mapping):
            routing_trace = dict(trace)
        else:
            routing_trace = {}
        raw_latency = routing_trace.get("latency")
        if not isinstance(raw_latency, Mapping):
            raw_latency = routing_trace.get("stage_latency_ms")
        if isinstance(raw_latency, Mapping):
            combined_latency = _float_mapping(raw_latency)
            combined_latency.update(stage_latency)
            stage_latency = combined_latency
        stage_latency["total_ms"] = response["latency_ms"]
        routing_trace["latency"] = stage_latency
        routing_trace["debug_enabled"] = debug_enabled
        expansion_trace = _query_expansion_trace(
            requested=self.runtime_config.query_expansion.enabled,
            executed_calls=expansion_provider_calls,
            resolved_task=resolved_task,
            query_plan=query_plan,
            config=self.runtime_config.query_expansion,
            provider=self.query_expansion_provider,
        )
        if (
            requested_task == "kis_temporal"
            and expansion_trace["requested"]
            and not expansion_trace["executed"]
        ):
            expansion_trace["skip_reason"] = "temporal_profile"
        routing_trace["query_expansion"] = expansion_trace
        routing_trace["route"] = (
            requested_task
            if requested_task in {"kis_visual", "kis_temporal"}
            else resolved_task
        )
        routing_trace["retrieval_profile"] = (
            "visual"
            if requested_task == "kis_visual"
            else str(query_plan.get("profile") or resolved_task)
        )
        response["routing_trace"] = routing_trace
        for field_name in (
            "answer",
            "answer_report",
            "answer_eligible",
            "preflight_block_reason",
            "evidence",
            "temporal_matches",
            "experiment_id",
        ):
            if field_name in raw:
                response[field_name] = raw[field_name]
        return response

    def _plan(
        self,
        query: str,
        task: str,
        *,
        modality_scope: Sequence[str] | None = None,
    ) -> tuple[QueryPlan, float, float, int]:
        started = time.perf_counter()
        provider = (
            self.query_expansion_provider
            if self.runtime_config.query_expansion.enabled
            else None
        )
        timed_provider = _TimedExpansionProvider(provider) if provider is not None else None
        plan = build_query_plan(
            query,
            profile=task,
            expansion_provider=timed_provider,
            expansion_config=self.runtime_config.query_expansion,
            modality_scope=modality_scope,
        )
        total_ms = _elapsed_ms(started)
        expansion_ms = timed_provider.elapsed_ms if timed_provider is not None else 0.0
        provider_calls = timed_provider.calls if timed_provider is not None else 0
        return (
            plan,
            round(max(0.0, total_ms - expansion_ms), 3),
            expansion_ms,
            provider_calls,
        )

    def _run_kis_avs(
        self,
        query: str,
        *,
        plan: QueryPlan,
        top_k: int,
        debug: bool,
        include_neighbors: bool,
        include_segments: bool,
    ) -> dict[str, Any]:
        online = self.runtime_config.online
        context_summary = (
            self.context_index.summary() if self.context_index is not None else {}
        )
        neighbor_available = int(context_summary.get("neighbor_record_count") or 0) > 0
        segment_available = int(context_summary.get("segment_record_count") or 0) > 0
        dense_index: DenseCandidateIndex | None = None
        fallback_reason = ""
        if online.coarse_to_dense_enabled and online.dense_enabled:
            try:
                dense_index = self._resolve_dense_index()
            except FileNotFoundError as exc:
                if online.dense_missing_behavior == "error":
                    raise
                fallback_reason = f"{type(exc).__name__}: {exc}"

        if dense_index is None:
            if not fallback_reason:
                if not online.coarse_to_dense_enabled:
                    fallback_reason = "coarse_to_dense_disabled"
                elif not online.dense_enabled:
                    fallback_reason = "dense_disabled"
                else:
                    fallback_reason = self._dense_index_error or "dense_index_unavailable"
            sparse = planned_hybrid_search(
                self.hybrid_engine,
                plan,
                top_k=top_k,
                max_expansion_contribution=(
                    self.runtime_config.query_expansion.max_expansion_contribution
                ),
            ).to_dict()
            trace = dict(sparse.get("trace") or {})
            trace["coarse_to_dense"] = {
                "enabled": bool(online.coarse_to_dense_enabled),
                "dense_enabled": bool(online.dense_enabled),
                "executed": False,
                "mode": "selected_only_fallback",
                "missing_behavior": online.dense_missing_behavior,
                "fallback_reason": fallback_reason,
            }
            trace["cses"] = {
                "requested": bool(online.cses_enabled),
                "executed": False,
                "profile": plan.profile,
                "fallback_reason": fallback_reason,
            }
            neighbor_scoring_requested = bool(
                include_neighbors and online.context_config.neighbor_enabled
            )
            segment_scoring_requested = bool(
                include_segments and online.context_config.segment_enabled
            )
            trace["context_scoring"] = {
                "neighbor": {
                    "requested": neighbor_scoring_requested,
                    "artifact_available": neighbor_available,
                    "executed": False,
                    "fallback_reason": (
                        "disabled_by_request"
                        if not include_neighbors
                        else (
                            "disabled_by_config"
                            if not online.context_config.neighbor_enabled
                            else (
                                "dense_rerank_unavailable"
                                if neighbor_available
                                else "neighbor_artifact_unavailable"
                            )
                        )
                    ),
                },
                "segment": {
                    "requested": segment_scoring_requested,
                    "artifact_available": segment_available,
                    "executed": False,
                    "fallback_reason": (
                        "disabled_by_request"
                        if not include_segments
                        else (
                            "disabled_by_config"
                            if not online.context_config.segment_enabled
                            else (
                                "dense_rerank_unavailable"
                                if segment_available
                                else "segment_artifact_unavailable"
                            )
                        )
                    ),
                },
            }
            sparse["trace"] = trace
            return sparse

        encoder = getattr(self.hybrid_engine.visual_engine, "encoder", None)
        if encoder is None or not callable(getattr(encoder, "encode", None)):
            raise RuntimeError("Canonical visual query encoder is unavailable")
        response = advanced_text_search(
            query,
            hybrid_engine=self.hybrid_engine,
            text_encoder=encoder,
            dense_index=dense_index,
            profile=plan.profile,
            plan=plan,
            config=AdvancedSearchConfig(
                coarse_top_n=online.coarse_top_n,
                dense_global_top_k=online.dense_global_top_k,
                dense_rescue_clips=online.dense_rescue_clips,
                max_total_clips=online.max_total_clips,
                dense_frames_per_clip=online.dense_frames_per_clip,
                rrf_k=online.rrf_k,
                modality_hint_boost=online.modality_hint_boost,
                similarity_threshold=online.similarity_threshold,
                temporal_window_seconds=online.temporal_window_seconds,
                max_event_gap_seconds=online.max_event_gap_seconds,
                rrf_enabled=online.rrf_enabled,
                dense_rescue_enabled=online.dense_enabled,
                cses_enabled=online.cses_enabled,
                deterministic_rerank_enabled=online.deterministic_rerank_enabled,
                query_expansion=self.runtime_config.query_expansion,
                rerank_weights=online.rerank_weights,
                context_config=replace(
                    online.context_config,
                    neighbor_enabled=(
                        online.context_config.neighbor_enabled
                        and include_neighbors
                    ),
                    segment_enabled=(
                        online.context_config.segment_enabled
                        and include_segments
                    ),
                ),
            ),
            context_index=self.context_index,
        )
        raw = response.to_dict(top_k=top_k)
        trace = dict(raw.get("trace") or {})
        trace["coarse_to_dense"] = {
            "enabled": True,
            "dense_enabled": True,
            "executed": True,
            "mode": "coarse_to_dense",
            "missing_behavior": online.dense_missing_behavior,
            "fallback_reason": "",
        }
        if not debug:
            trace.pop("intra_modality_fusion", None)
            trace.pop("inter_modality_fusion", None)
        raw["trace"] = trace
        return raw

    def _resolve_dense_index(self) -> DenseCandidateIndex | None:
        with self._dense_index_lock:
            if self._dense_index_resolved:
                return self._dense_index
            if self._dense_index_loader is None:
                self._dense_index_error = "dense_index_loader_unavailable"
                raise FileNotFoundError(self._dense_index_error)
            try:
                self._dense_index = self._dense_index_loader()
            except FileNotFoundError as exc:
                self._dense_index_error = f"{type(exc).__name__}: {exc}"
                raise
            self._dense_index_resolved = True
            self._dense_index_error = ""
            if self._dense_index is None:
                raise RuntimeError("Dense index loader returned no index")
            return self._dense_index

    def _top_k(self, value: int | None, *, task: str) -> int:
        if task == "qa":
            default = 5
            maximum = int(self.config.max_top_k)
        elif task == "trake":
            default = int(self.runtime_config.trake.max_answers)
            maximum = min(
                int(self.config.max_top_k),
                int(self.runtime_config.trake.max_answers),
                100,
            )
        else:
            default = self.runtime_config.hybrid.default_top_k
            maximum = int(self.config.max_top_k)
        requested = default if value is None else int(value)
        return max(1, min(requested, maximum))

    def _context_flags(self, override: bool | None) -> tuple[bool, bool]:
        if override is None:
            return (self.config.include_neighbors, self.config.include_segments)
        enabled = bool(override)
        return (enabled, enabled)


def _query_plan_payload(
    raw: Mapping[str, Any],
    plan: QueryPlan | None,
) -> dict[str, Any]:
    value = raw.get("query_plan")
    if isinstance(value, Mapping):
        return dict(value)
    trace = raw.get("trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("query_plan"), Mapping):
        return dict(trace["query_plan"])
    if plan is not None:
        return plan.to_dict()
    raise RuntimeError("Online route did not return its query plan")


def _trake_response(
    raw: Mapping[str, Any],
    *,
    query: str,
    top_k: int,
    started_at: float,
    expansion_requested: bool,
) -> dict[str, Any]:
    """Preserve the sequence-first TRAKE contract without candidate flattening."""

    if not isinstance(raw, Mapping):
        raise RuntimeError("TRAKE pipeline returned a non-mapping response")
    response = dict(raw)
    returned_query = " ".join(str(response.get("query") or query).split())
    if returned_query != query:
        raise RuntimeError("TRAKE response did not preserve the original query anchor")
    returned_task = str(response.get("task") or "trake").casefold().strip()
    if returned_task != "trake":
        raise RuntimeError("TRAKE pipeline returned an invalid task identity")
    event_plan = response.get("event_plan")
    if isinstance(event_plan, Mapping):
        plan_query = " ".join(str(event_plan.get("original_query") or query).split())
        if plan_query != query:
            raise RuntimeError(
                "TRAKE event plan did not preserve the original query anchor"
            )

    raw_hypotheses = response.get("hypotheses", ())
    if (
        not isinstance(raw_hypotheses, Sequence)
        or isinstance(raw_hypotheses, (str, bytes))
    ):
        raise RuntimeError("TRAKE pipeline hypotheses must be a sequence")
    hypotheses = [
        dict(item)
        for item in raw_hypotheses[:top_k]
        if isinstance(item, Mapping)
    ]
    if len(hypotheses) != len(raw_hypotheses[:top_k]):
        raise RuntimeError("TRAKE pipeline hypotheses must be mappings")

    response.setdefault("schema_version", ONLINE_SCHEMA_VERSION)
    response["query"] = query
    response["requested_task"] = "trake"
    response["task"] = "trake"
    response["top_k"] = top_k
    response["hypotheses"] = hypotheses
    trace = response.get("trace")
    trace_payload = dict(trace) if isinstance(trace, Mapping) else {}
    trace_payload["query_expansion"] = {
        "requested": bool(expansion_requested),
        "executed": False,
        "provider_call_count": 0,
        "skip_reason": "trake_route",
    }
    response["trace"] = trace_payload
    if "candidates" in response:
        # The compatibility alias, when supplied by the core pipeline, remains
        # a list of complete sequences and must obey the same public limit.
        response["candidates"] = hypotheses
    response.setdefault(
        "latency_ms",
        round((time.perf_counter() - started_at) * 1000.0, 3),
    )
    return response


def _query_expansion_trace(
    *,
    requested: bool,
    executed_calls: int,
    resolved_task: str,
    query_plan: Mapping[str, Any],
    config: QueryExpansionConfig,
    provider: QueryExpansionProvider | None,
) -> dict[str, Any]:
    expansion_plan = query_plan.get("expansion_plan")
    expansion_payload = (
        dict(expansion_plan) if isinstance(expansion_plan, Mapping) else {}
    )
    executed = int(executed_calls) > 0
    if resolved_task not in {"kis", "avs"}:
        skip_reason = f"{resolved_task}_route"
    elif not requested:
        skip_reason = "disabled_by_config"
    elif not executed:
        skip_reason = str(
            expansion_payload.get("fallback_reason")
            or expansion_payload.get("status")
            or "provider_not_called"
        )
    else:
        skip_reason = ""
    return {
        "enabled": bool(config.enabled),
        "requested": bool(requested),
        "executed": executed,
        "provider_call_count": int(executed_calls),
        "cache_hit": bool(expansion_payload.get("cache_hit", False)),
        "provider": str(expansion_payload.get("provider_name") or ""),
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "quantization": config.quantization,
        "device": str(getattr(provider, "device", "not_constructed")),
        "skip_reason": skip_reason,
        "status": str(expansion_payload.get("status") or ""),
    }


def _validate_original_anchor(query: str, plan: Mapping[str, Any]) -> None:
    original = " ".join(str(plan.get("original_query") or "").split())
    if original != query:
        raise RuntimeError(
            "Online query plan did not preserve the original query anchor"
        )


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, float] = {}
    for key, raw in value.items():
        try:
            output[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return output


def _modality_score(
    values: Mapping[str, float],
    *names: str,
) -> float | None:
    for name in names:
        if name in values:
            return float(values[name])
    return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: object, *, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _optional_int(value: object) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one query through the canonical online pipeline."
    )
    parser.add_argument(
        "--task",
        choices=SUPPORTED_ONLINE_TASKS,
        default="kis",
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--expanded-query",
        action="append",
        default=[],
        help="Caller-owned QA expansion; repeat for multiple values.",
    )
    parser.add_argument(
        "--with-context",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Request canonical neighbor and segment context; unavailable optional "
            "artifacts are reported in the trace."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_const",
        const=True,
        default=None,
        help="Expose detailed fusion and per-stage latency trace.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; stdout is always printed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_project_env()
    # Lazy import avoids a module cycle: retrieval_manager owns the cached
    # production OnlinePipeline instance and imports this class definition.
    from backend.app.services.retrieval.retrieval_manager import search_online

    args = build_parser().parse_args(argv)
    response = search_online(
        query=args.query,
        task=args.task,
        top_k=args.top_k,
        expanded_queries=args.expanded_query,
        include_context=args.with_context,
        debug=args.debug,
    )
    serialized = json.dumps(response, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


__all__ = [
    "Candidate",
    "ONLINE_SCHEMA_VERSION",
    "OnlinePipeline",
    "OnlinePipelineConfig",
    "SUPPORTED_ONLINE_TASKS",
    "TrakeSearchPipeline",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
