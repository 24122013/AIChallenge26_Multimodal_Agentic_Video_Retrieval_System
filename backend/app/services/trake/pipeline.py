"""End-to-end public TRAKE orchestration."""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from backend.app.services.retrieval.retrieval_config import TrakeConfig
from backend.app.services.trake.candidate_video import gate_candidate_videos
from backend.app.services.trake.event_retrieval import (
    CandidateRerankerLike,
    EventRetriever,
    RetrievalEngineLike,
)
from backend.app.services.trake.models import TemporalEventPlan
from backend.app.services.trake.query_parser import TrakeQueryParser
from backend.app.services.trake.ranking import rank_hypotheses
from backend.app.services.trake.temporal_alignment import align_candidate_videos
from backend.app.services.trake.temporal_refinement import (
    LocalFrameDecoder,
    LocalFrameScorer,
    RefinementVariant,
    TemporalRefiner,
)


TRAKE_SCHEMA_VERSION = "1.0"


class TrakeStageDeadlineExceeded(RuntimeError):
    """Public, sanitized timeout raised between bounded TRAKE stages."""

    def __init__(self, stage: str, deadline_seconds: float) -> None:
        super().__init__(f"TRAKE stage deadline exceeded after {deadline_seconds:g}s ({stage})")
        self.response = {
            "task": "trake",
            "status": "timeout",
            "hypotheses": [],
            "warnings": ["stage_deadline_exceeded"],
            "trace": {
                "status": "timeout",
                "failure_code": "stage_deadline_exceeded",
                "stage": stage,
            },
        }


class QueryParserLike(Protocol):
    def parse(self, query: str) -> TemporalEventPlan:
        ...


class RefinerLike(Protocol):
    def refine(self, path: Any, plan: TemporalEventPlan) -> list[RefinementVariant]:
        ...


class TrakePipeline:
    """Parse, retrieve, gate, align, locally refine, and globally rank TRAKE."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngineLike | None = None,
        *,
        hybrid_engine: RetrievalEngineLike | None = None,
        config: TrakeConfig | None = None,
        parser: QueryParserLike | Callable[[str], TemporalEventPlan] | None = None,
        refiner: RefinerLike | None = None,
        local_scorer: LocalFrameScorer | None = None,
        local_decoder: LocalFrameDecoder | None = None,
        video_root: str | Path | None = None,
        dense_event_engine: RetrievalEngineLike | None = None,
        event_reranker: CandidateRerankerLike | None = None,
        bge_contract: Mapping[str, Any] | None = None,
    ) -> None:
        engine = retrieval_engine or hybrid_engine
        if engine is None:
            raise TypeError("TrakePipeline requires the canonical retrieval engine")
        if retrieval_engine is not None and hybrid_engine is not None and retrieval_engine is not hybrid_engine:
            raise ValueError("provide only one TRAKE retrieval engine")
        self.retrieval_engine = engine
        self.dense_event_engine = dense_event_engine
        self.event_reranker = event_reranker
        # Keep the runtime contract immutable from the caller's point of view.
        # It is nested, so shallow copies would let one Python response mutate
        # the trace returned by later requests.
        self.bge_contract = copy.deepcopy(dict(bge_contract or {}))
        self.config = config or TrakeConfig()
        self.parser = parser or TrakeQueryParser()
        self.event_retriever = EventRetriever(
            engine,
            self.config,
            dense_event_engine=dense_event_engine,
            event_reranker=event_reranker,
        )
        self.refiner = refiner or TemporalRefiner(
            config=self.config,
            video_root=video_root,
            decoder=local_decoder,
            scorer=local_scorer,
        )

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        requested = self.config.max_answers if top_k is None else int(top_k)
        bounded_top_k = max(1, min(requested, self.config.max_answers, 100))
        stage_latency: dict[str, float] = {}
        model_loads_before = _model_load_snapshot(self)
        deadline_at = started + float(self.config.stage_deadline_seconds)

        def ensure_deadline(stage: str) -> None:
            if time.perf_counter() > deadline_at:
                raise TrakeStageDeadlineExceeded(
                    stage,
                    float(self.config.stage_deadline_seconds),
                )

        stage_started = time.perf_counter()
        plan = self._parse(query)
        stage_latency["parse_ms"] = _elapsed_ms(stage_started)
        ensure_deadline("parse")

        stage_started = time.perf_counter()
        retrieval = self.event_retriever.retrieve(plan, include_context=True)
        stage_latency["event_retrieval_ms"] = _elapsed_ms(stage_started)
        model_loads_after = _model_load_snapshot(self)
        stage_latency["model_cold_start_ms"] = round(
            sum(
                float(model_loads_after[name]["last_model_load_ms"])
                for name in model_loads_after
                if int(model_loads_after[name]["count"])
                > int(model_loads_before.get(name, {}).get("count", 0))
            ),
            3,
        )
        stage_latency["bge_retrieval_ms"] = float(
            retrieval.trace.get("bge", {}).get("dense_latency_ms", 0.0)
        )
        ensure_deadline("event_retrieval")

        stage_started = time.perf_counter()
        gating = gate_candidate_videos(
            retrieval.event_candidates,
            event_count=len(plan.events),
            context_scores=retrieval.context_scores,
            config=self.config,
        )
        stage_latency["video_gating_ms"] = _elapsed_ms(stage_started)
        ensure_deadline("video_gating")

        stage_started = time.perf_counter()
        paths = align_candidate_videos(
            gating.videos,
            event_count=len(plan.events),
            config=self.config,
        )
        stage_latency["alignment_ms"] = _elapsed_ms(stage_started)
        ensure_deadline("alignment")

        stage_started = time.perf_counter()
        rankable: list[Any] = []
        refined_path_count = 0
        refinement_variant_count = 0
        refinement_fallback_count = 0
        refinement_begin = getattr(self.refiner, "begin_request", None)
        refinement_end = getattr(self.refiner, "end_request", None)
        if self.config.refinement_enabled:
            if callable(refinement_begin):
                refinement_begin()
            for index, path in enumerate(paths):
                if index < self.config.refinement_top_paths:
                    try:
                        variants = self.refiner.refine(path, plan)
                    except Exception:
                        variants = []
                    if variants:
                        rankable.extend(variants)
                    else:
                        rankable.append(
                            replace(
                                path,
                                warnings=tuple(
                                    dict.fromkeys(
                                        (*path.warnings, "local_refinement_failed_coarse_fallback")
                                    )
                                ),
                            )
                        )
                        refinement_fallback_count += 1
                    refined_path_count += 1
                    refinement_variant_count += len(variants)
                else:
                    rankable.append(path)
            if callable(refinement_end):
                refinement_end()
        else:
            rankable.extend(paths)
        stage_latency["refinement_ms"] = _elapsed_ms(stage_started)
        ensure_deadline("refinement")
        refinement_stats = getattr(self.refiner, "request_stats", {})
        if not isinstance(refinement_stats, Mapping):
            refinement_stats = {}
        stage_latency["frame_decode_ms"] = float(
            refinement_stats.get("frame_decode_ms", 0.0)
        )
        stage_latency["frame_embedding_ms"] = float(
            refinement_stats.get("frame_embedding_ms", 0.0)
        )

        stage_started = time.perf_counter()
        hypotheses, ranking_trace = rank_hypotheses(
            rankable,
            max_answers=bounded_top_k,
            expected_event_count=len(plan.events),
            config=self.config,
            # Exact sequence dedupe remains enabled.  A zero near-NMS radius
            # lets distinct adjacent frame tuples participate in the public
            # 100-result contract instead of silently collapsing them.
            sequence_nms_radius_frames=0,
        )
        stage_latency["ranking_ms"] = _elapsed_ms(stage_started)
        ensure_deadline("ranking")

        warnings = list(_collect_warnings(
            plan.warnings,
            retrieval.warnings,
            gating.warnings,
            *(path.warnings for path in paths),
            *(hypothesis.warnings for hypothesis in hypotheses),
        ))
        if not hypotheses:
            if any(not event.original_text.strip() for event in plan.events):
                warnings.extend(("empty_event_marker", "insufficient_event_support"))
            elif not gating.videos:
                warnings.extend(("no_video_supports_all_events", "insufficient_event_support"))
            elif not paths:
                warnings.append("insufficient_event_support")
        warnings = list(dict.fromkeys(warnings))
        status = "ok" if hypotheses else "insufficient_support"
        hypothesis_payloads = [hypothesis.to_dict() for hypothesis in hypotheses]
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        trace = {
            "parser": {
                "source": plan.parser_source,
                "confidence": plan.confidence,
                "structural_confidence": plan.structural_confidence,
                "semantic_confidence": plan.semantic_confidence,
                "fallback_used": "fallback" in plan.parser_source,
                "warning_count": len(plan.warnings),
                "events": [event.parser_trace for event in plan.events],
            },
            "candidate_counts": {
                "events": {
                    str(index): len(values)
                    for index, values in sorted(retrieval.event_candidates.items())
                },
                "context_videos": len(retrieval.context_scores),
                "gated_videos": len(gating.videos),
                "coarse_paths": len(paths),
                "refined_paths": refined_path_count,
                "refinement_variants": refinement_variant_count,
                "refinement_coarse_fallbacks": refinement_fallback_count,
                "ranked_hypotheses": len(hypotheses),
            },
            "event_retrieval": retrieval.trace,
            "bge_contract": copy.deepcopy(self.bge_contract),
            "video_gating": gating.trace,
            "alignment": {
                "method": self.config.alignment_method,
                "beam_width": self.config.beam_width,
                "hard_max_gap": None,
                "gap_penalty": self.config.gap_penalty,
                "gap_lambda": self.config.gap_lambda,
                "ordering_field": "original_frame_index",
            },
            "refinement": {
                "enabled": self.config.refinement_enabled,
                "strategy": "bounded_local_decode",
                "top_paths": self.config.refinement_top_paths,
                "window_before_frames": self.config.window_before_frames,
                "window_after_frames": self.config.window_after_frames,
                "dense_stride_frames": self.config.dense_stride_frames,
                "scorer_available": getattr(self.refiner, "scorer", None) is not None,
                "fallback_is_canonical_frame_index": True,
                "request_cache": dict(refinement_stats),
            },
            "ranking": {
                **ranking_trace.to_dict(),
                "cutoffs": list(self.config.ranking_cutoffs),
                "max_answers": self.config.max_answers,
            },
            "feature_flags": {
                "context_branch": self.config.context_weight > 0 and bool(plan.context),
                "local_refinement": self.config.refinement_enabled,
                "retrieval_modalities": list(
                    getattr(self.retrieval_engine, "available_modalities", ())
                ),
                "retrieval_engine": type(self.retrieval_engine).__name__,
                "bge_dense": bool(
                    getattr(self.config, "bge_dense_enabled", False)
                ),
                "bge_dense_engine": (
                    type(self.dense_event_engine).__name__
                    if self.dense_event_engine is not None
                    else None
                ),
                "bge_reranker": bool(
                    getattr(self.config, "bge_reranker_enabled", False)
                ),
                "bge_reranker_engine": (
                    type(self.event_reranker).__name__
                    if self.event_reranker is not None
                    else None
                ),
                "bge_required": bool(getattr(self.config, "bge_required", False)),
                "reranker": type(
                    getattr(self.retrieval_engine, "reranker", None)
                ).__name__
                if getattr(self.retrieval_engine, "reranker", None) is not None
                else None,
                "local_scorer": type(
                    getattr(self.refiner, "scorer", None)
                ).__name__
                if getattr(self.refiner, "scorer", None) is not None
                else None,
            },
            "status": status,
            "warnings": list(warnings),
            "latency": {**stage_latency, "total_ms": latency_ms},
            "preflight": {
                "corpus_video_ids": sorted(
                    {
                        candidate.result.video_id
                        for values in retrieval.event_candidates.values()
                        for candidate in values
                    }
                ),
                "event_count": len(plan.events),
                "model_index_ready": bool(retrieval.event_candidates),
                "model_loads": model_loads_after,
                "bge_branch_status": retrieval.trace.get("bge", {}),
                "local_refinement_status": (
                    "ready" if getattr(self.refiner, "scorer", None) is not None
                    else "scorer_unavailable"
                ),
                "missing_artifacts": (
                    ["local_refinement_scorer"]
                    if self.config.refinement_enabled and getattr(self.refiner, "scorer", None) is None
                    else []
                ),
                "fallback_branch": "none" if hypotheses else status,
            },
        }
        return {
            "schema_version": TRAKE_SCHEMA_VERSION,
            "query": plan.original_query,
            "task": "trake",
            "top_k": bounded_top_k,
            "status": status,
            "warnings": list(warnings),
            "event_plan": plan.to_dict(),
            "hypotheses": hypothesis_payloads,
            # Compatibility alias: each item remains a complete sequence, never
            # a flattened event/frame candidate.
            "candidates": hypothesis_payloads,
            "trace": trace,
            "latency_ms": latency_ms,
        }

    def run(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        return self.search(query, top_k=top_k)

    def _parse(self, query: str) -> TemporalEventPlan:
        if hasattr(self.parser, "parse"):
            plan = self.parser.parse(query)  # type: ignore[union-attr]
        else:
            plan = self.parser(query)  # type: ignore[operator]
        if not isinstance(plan, TemporalEventPlan):
            raise TypeError("TRAKE parser must return TemporalEventPlan")
        if " ".join(plan.original_query.split()) != " ".join(query.split()):
            raise ValueError("TRAKE parser must preserve the original query anchor")
        if not plan.events:
            raise ValueError("TRAKE query must contain at least one event")
        indexes = [event.index for event in plan.events]
        if indexes != list(range(len(plan.events))):
            raise ValueError("TRAKE event indexes must be contiguous and ordered")
        return plan


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _collect_warnings(*groups: Any) -> tuple[str, ...]:
    output: list[str] = []
    for group in groups:
        for warning in group or ():
            value = str(warning).strip()
            if value and value not in output:
                output.append(value)
    return tuple(output)


def _model_load_snapshot(pipeline: TrakePipeline) -> dict[str, dict[str, float | int]]:
    visual_engine = getattr(pipeline.retrieval_engine, "visual_engine", None)
    visual_encoder = getattr(visual_engine, "encoder", None)
    dense_encoder = getattr(pipeline.dense_event_engine, "encoder", None)
    output: dict[str, dict[str, float | int]] = {}
    for name, encoder in (("siglip2", visual_encoder), ("bge_m3", dense_encoder)):
        if encoder is None:
            continue
        output[name] = {
            "count": int(getattr(encoder, "model_load_count", 0)),
            "last_model_load_ms": float(getattr(encoder, "last_model_load_ms", 0.0)),
        }
    return output


__all__ = ["TRAKE_SCHEMA_VERSION", "TrakePipeline", "TrakeStageDeadlineExceeded"]
