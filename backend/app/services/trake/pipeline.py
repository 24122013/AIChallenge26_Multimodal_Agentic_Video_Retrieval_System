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

        stage_started = time.perf_counter()
        plan = self._parse(query)
        stage_latency["parse_ms"] = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        retrieval = self.event_retriever.retrieve(plan, include_context=True)
        stage_latency["event_retrieval_ms"] = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        gating = gate_candidate_videos(
            retrieval.event_candidates,
            event_count=len(plan.events),
            context_scores=retrieval.context_scores,
            config=self.config,
        )
        stage_latency["video_gating_ms"] = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        paths = align_candidate_videos(
            gating.videos,
            event_count=len(plan.events),
            config=self.config,
        )
        stage_latency["alignment_ms"] = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        rankable: list[Any] = []
        refined_path_count = 0
        refinement_variant_count = 0
        refinement_fallback_count = 0
        if self.config.refinement_enabled:
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
        else:
            rankable.extend(paths)
        stage_latency["refinement_ms"] = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        hypotheses, ranking_trace = rank_hypotheses(
            rankable,
            max_answers=bounded_top_k,
            expected_event_count=len(plan.events),
            config=self.config,
        )
        stage_latency["ranking_ms"] = _elapsed_ms(stage_started)

        warnings = _collect_warnings(
            plan.warnings,
            retrieval.warnings,
            gating.warnings,
            *(path.warnings for path in paths),
            *(hypothesis.warnings for hypothesis in hypotheses),
        )
        hypothesis_payloads = [hypothesis.to_dict() for hypothesis in hypotheses]
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        trace = {
            "parser": {
                "source": plan.parser_source,
                "confidence": plan.confidence,
                "fallback_used": "fallback" in plan.parser_source,
                "warning_count": len(plan.warnings),
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
            "warnings": list(warnings),
            "latency": {**stage_latency, "total_ms": latency_ms},
        }
        return {
            "schema_version": TRAKE_SCHEMA_VERSION,
            "query": plan.original_query,
            "task": "trake",
            "top_k": bounded_top_k,
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


__all__ = ["TRAKE_SCHEMA_VERSION", "TrakePipeline"]
