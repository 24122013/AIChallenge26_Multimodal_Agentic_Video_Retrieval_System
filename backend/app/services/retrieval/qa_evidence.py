"""Typed QA routing and provenance-preserving evidence bundles.

The module is intentionally deterministic.  It does not translate or answer a
question.  Caller-owned expansions are supported for non-temporal QA only;
temporal QA retrieves every parsed event through this same routing stack.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.candidate_merger import (
    candidate_identity,
    merge_candidates,
)
from backend.app.services.retrieval.online_context import OnlineContextIndex
from backend.app.services.retrieval.query_plan import QueryPlan, build_query_plan
from backend.app.services.retrieval.query_terms import weighted_query_coverage
from backend.app.services.retrieval.rank_fusion import weighted_rrf
from backend.app.services.retrieval.temporal_search import match_ordered_events


class HybridSearchEngineLike(Protocol):
    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> VisualSearchResponse:
        ...


class CandidateRerankerLike(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> Any:
        ...


class BgeCandidateReranker:
    """Adapter that gives the functional BGE reranker a router interface."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model_revision: str = "main",
        retrieval_alpha: float = 0.5,
        batch_size: int = 16,
        device: str = "auto",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.retrieval_alpha = float(retrieval_alpha)
        self.batch_size = int(batch_size)
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self.last_report: dict[str, Any] | None = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        from backend.app.services.retrieval.bge_reranker import rerank_with_bge

        results, report = rerank_with_bge(
            candidates,
            query=query,
            model_name=self.model_name,
            model_revision=self.model_revision,
            output_k=20 if top_k is None else int(top_k),
            retrieval_alpha=self.retrieval_alpha,
            batch_size=self.batch_size,
            device=self.device,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self.last_report = report.to_dict()
        return results


QaEvidencePlan = QueryPlan


@dataclass(frozen=True)
class QaRoutingConfig:
    typed_parser_enabled: bool = True
    router_enabled: bool = True
    evidence_bundle_enabled: bool = True
    per_modality_limit: int = 100
    rerank_pool_size: int = 100
    fusion_pool_size: int = 20
    evidence_limit: int = 5
    rrf_k: int = 60
    modality_hint_boost: float = 1.5
    consensus_bonus: float = 0.03
    constraint_rerank_enabled: bool = True
    constraint_weight: float = 0.15
    constraint_min_signal: float = 0.20
    temporal_routing_enabled: bool = True
    temporal_max_events: int = 5
    temporal_max_gap_seconds: float = 180.0
    weights: tuple[tuple[str, float], ...] = (
        ("visual", 0.55),
        ("caption", 0.20),
        ("ocr", 0.10),
        ("objects", 0.10),
        ("dense_text", 0.20),
        ("hybrid", 1.0),
    )

    def __post_init__(self) -> None:
        if self.per_modality_limit < 1:
            raise ValueError("per_modality_limit must be >= 1")
        if self.fusion_pool_size < 1:
            raise ValueError("fusion_pool_size must be >= 1")
        if self.rerank_pool_size < self.fusion_pool_size:
            raise ValueError("rerank_pool_size must be >= fusion_pool_size")
        if self.evidence_limit < 1:
            raise ValueError("evidence_limit must be >= 1")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        if self.modality_hint_boost < 1:
            raise ValueError("modality_hint_boost must be >= 1")
        if self.consensus_bonus < 0:
            raise ValueError("consensus_bonus must be non-negative")
        if not 0.0 <= self.constraint_weight <= 1.0:
            raise ValueError("constraint_weight must be between 0 and 1")
        if not 0.0 <= self.constraint_min_signal <= 1.0:
            raise ValueError("constraint_min_signal must be between 0 and 1")
        if self.temporal_max_events < 1:
            raise ValueError("temporal_max_events must be >= 1")
        if self.temporal_max_gap_seconds < 0:
            raise ValueError("temporal_max_gap_seconds must be non-negative")


@dataclass(frozen=True)
class QaEvidence:
    evidence_id: str
    video_id: str
    frame_id: str
    frame_index: int | None
    shot_id: str
    timestamp: float
    image_path: str
    caption: str
    ocr_text: str
    objects: tuple[str, ...]
    source_modalities: tuple[str, ...]
    retrieval_score: float
    base_retrieval_score: float
    constraint_score: float
    matched_constraints: tuple[str, ...]
    temporal_event_index: int | None = None
    temporal_match_rank: int | None = None
    temporal_match_mode: str = ""
    temporal_chain_id: str = ""
    temporal_event_query: str = ""
    temporal_event_role: str = ""
    temporal_chain_score: float | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "shot_id": self.shot_id,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "objects": list(self.objects),
            "source_modalities": list(self.source_modalities),
            "retrieval_score": self.retrieval_score,
            "base_retrieval_score": self.base_retrieval_score,
            "constraint_score": self.constraint_score,
            "matched_constraints": list(self.matched_constraints),
            "temporal_event_index": self.temporal_event_index,
            "temporal_match_rank": self.temporal_match_rank,
            "temporal_match_mode": self.temporal_match_mode,
            "temporal_chain_id": self.temporal_chain_id,
            "temporal_event_query": self.temporal_event_query,
            "temporal_event_role": self.temporal_event_role,
            "temporal_chain_score": self.temporal_chain_score,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _ConstraintEvidence:
    base_score: float
    constraint_score: float = 0.0
    matched_constraints: tuple[str, ...] = ()


@dataclass
class _RoutingTrace:
    queries: list[dict[str, Any]] = field(default_factory=list)
    modality_queries: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    reranker: str = "off"
    constraint_rerank: dict[str, Any] = field(default_factory=dict)
    temporal_route: dict[str, Any] = field(default_factory=dict)
    fallback_reasons: list[str] = field(default_factory=list)

    def to_dict(
        self,
        *,
        plan: QueryPlan,
        config: QaRoutingConfig,
    ) -> dict[str, Any]:
        return {
            "queries": list(self.queries),
            "modality_queries": list(self.modality_queries),
            "modality_hints": list(plan.modality_hints),
            "weights": dict(config.weights),
            "hint_boost": config.modality_hint_boost,
            "rrf_k": config.rrf_k,
            "per_modality_limit": config.per_modality_limit,
            "fusion_pool_size": config.fusion_pool_size,
            "rerank_pool_size": config.rerank_pool_size,
            "fallback_used": self.fallback_used,
            "fallback_reasons": list(dict.fromkeys(self.fallback_reasons)),
            "reranker": self.reranker,
            "constraint_rerank": dict(self.constraint_rerank),
            "temporal_route": dict(self.temporal_route),
            "temporal_handoff": plan.needs_temporal,
            "feature_flags": {
                "typed_parser": config.typed_parser_enabled,
                "qa_router": config.router_enabled,
                "evidence_bundle": config.evidence_bundle_enabled,
                "constraint_rerank": config.constraint_rerank_enabled,
                "temporal_routing": config.temporal_routing_enabled,
            },
        }


class QaEvidenceSearchEngine:
    """Route QA to existing retrievers, fuse candidates, and bundle evidence."""

    def __init__(
        self,
        hybrid_engine: HybridSearchEngineLike,
        *,
        dense_text_engine: object | None = None,
        candidate_reranker: CandidateRerankerLike | None = None,
        context_index: OnlineContextIndex | None = None,
        config: QaRoutingConfig | None = None,
        candidate_multiplier: int | None = None,
        consensus_bonus: float | None = None,
    ) -> None:
        # The two legacy keyword arguments remain accepted for callers/tests
        # from the old QA-evidence implementation.
        resolved = config or QaRoutingConfig()
        if candidate_multiplier is not None:
            if candidate_multiplier < 1:
                raise ValueError("candidate_multiplier must be >= 1")
            resolved = replace(
                resolved,
                per_modality_limit=min(
                    200,
                    max(resolved.per_modality_limit, int(candidate_multiplier) * 20),
                ),
            )
        if consensus_bonus is not None:
            resolved = replace(resolved, consensus_bonus=float(consensus_bonus))
        self.hybrid_engine = hybrid_engine
        self.dense_text_engine = dense_text_engine
        self.candidate_reranker = candidate_reranker
        self.context_index = context_index
        self.config = resolved

    def search(
        self,
        question: str,
        top_k: int = 5,
        *,
        task_mode: str = "qa",
        expanded_queries: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        requested_top_k = max(1, min(int(top_k), 200))
        evidence_top_k = min(requested_top_k, self.config.evidence_limit)
        plan = plan_qa_question(question, task_mode=task_mode)
        if not self.config.typed_parser_enabled:
            plan = replace(
                plan,
                answer_type="unknown",
                retrieval_statement=plan.normalized_query,
                known_constraints=(),
                constraint_roles=(),
                answer_event_index=None,
                confidence=0.0,
                reasons=(*plan.reasons, "typed parser disabled by feature flag"),
                modality_queries=tuple(
                    (modality, plan.normalized_query)
                    for modality in ("visual", "caption", "objects", "ocr")
                ),
            )
        if (
            plan.needs_temporal
            and self.config.temporal_routing_enabled
        ):
            return self._search_temporal(
                plan,
                requested_top_k=requested_top_k,
                expanded_queries=expanded_queries or (),
            )

        external_queries = _external_queries(
            expanded_queries or (),
            base_query=plan.retrieval_statement,
        )
        query_specs = _query_branches(plan, external_queries)
        queries = tuple(query for query, _source in query_specs)
        trace = _RoutingTrace(
            queries=[
                {
                    "query": query,
                    "source": source,
                }
                for query, source in query_specs
            ],
            temporal_route={
                "executed": False,
                "event_queries": [],
                "event_count": 0,
                "match_count": 0,
                "match_mode": "none",
                "warnings": (
                    ["temporal_routing_disabled"]
                    if plan.needs_temporal
                    else []
                ),
                "answer_eligible": not plan.needs_temporal,
                "reason": (
                    "temporal_routing_disabled"
                    if plan.needs_temporal
                    else None
                ),
                "external_expansions_ignored": False,
            },
        )
        if plan.needs_temporal:
            trace.fallback_used = True
            _append_reason(trace, "temporal:temporal_routing_disabled")

        query_groups: list[list[RetrievalResult]] = []
        conflict_sources: list[RetrievalResult] = []
        for query, source in query_specs:
            group, sources = self._retrieve_and_fuse(
                query,
                plan=plan,
                trace=trace,
                use_plan_queries=source in {"parser", "full_proposition"},
            )
            query_groups.append(group)
            conflict_sources.extend(sources)

        fused_pool = _fuse_evidence(
            query_groups,
            top_k=self.config.rerank_pool_size,
            consensus_bonus=self.config.consensus_bonus,
        )
        rerank_query = (
            _neutral_context_query(plan)
            if plan.answer_type == "yes_no"
            else plan.retrieval_statement
        )
        reranked = self._rerank(rerank_query, fused_pool, trace)
        constrained, constraint_details = self._constraint_rerank(
            reranked,
            plan=plan,
            trace=trace,
        )
        final_pool = constrained[: self.config.fusion_pool_size]
        selected = merge_candidates(
            [final_pool],
            top_k=requested_top_k,
            dedupe_same_shot=True,
        )
        conflict_keys = _conflicting_shots(conflict_sources)
        evidence = (
            [
                _build_evidence(
                    result,
                    index=index,
                    conflict_keys=conflict_keys,
                    constraint_detail=constraint_details.get(
                        candidate_identity(result),
                        _ConstraintEvidence(base_score=float(result.score)),
                    ),
                )
                for index, result in enumerate(
                    selected[:evidence_top_k],
                    start=1,
                )
            ]
            if self.config.evidence_bundle_enabled
            else []
        )
        answer_eligible = bool(evidence) and not plan.needs_temporal
        trace.temporal_route["answer_eligible"] = answer_eligible
        trace.temporal_route["reason"] = (
            "temporal_routing_disabled"
            if plan.needs_temporal
            else (None if answer_eligible else "no_evidence")
        )
        return {
            **_legacy_plan_fields(plan, queries),
            "query_plan": plan.to_dict(),
            "routing_trace": trace.to_dict(plan=plan, config=self.config),
            "top_k": requested_top_k,
            "answer_mode": "manual_visual_inspection",
            "answer_eligible": answer_eligible,
            "preflight_block_reason": (
                "temporal_routing_disabled"
                if plan.needs_temporal
                else (None if answer_eligible else "no_evidence")
            ),
            "temporal_matches": [],
            "evidence_count": len(evidence),
            "evidence": [item.to_dict() for item in evidence],
            # Kept for /retrieval/qa-evidence and mode=qa compatibility.
            "results": [result.to_dict() for result in selected],
        }

    def _search_temporal(
        self,
        plan: QueryPlan,
        *,
        requested_top_k: int,
        expanded_queries: Sequence[str],
    ) -> dict[str, Any]:
        events = tuple(
            " ".join(str(event).split())
            for event in getattr(plan, "temporal_events", ())
            if " ".join(str(event).split())
        )
        if len(events) > self.config.temporal_max_events:
            raise ValueError(
                "temporal_query_too_complex: "
                f"supports at most {self.config.temporal_max_events} events"
            )
        if not events:
            events = (plan.retrieval_statement,)

        ignored_expansions = bool(tuple(expanded_queries))
        trace = _RoutingTrace(
            queries=[
                {
                    "query": event,
                    "source": "temporal_event",
                    "event_index": index,
                }
                for index, event in enumerate(events)
            ],
            temporal_route={
                "executed": True,
                "event_queries": list(events),
                "event_count": len(events),
                "external_expansions_ignored": ignored_expansions,
            },
        )
        if ignored_expansions:
            trace.fallback_reasons.append(
                "external_expansions_ignored_for_temporal"
            )

        event_results: list[list[RetrievalResult]] = []
        detail_by_event: dict[
            tuple[int, tuple[str, str]],
            _ConstraintEvidence,
        ] = {}
        conflict_sources: list[RetrievalResult] = []
        constraint_reports: list[dict[str, Any]] = []
        canonical_segment_mappings = 0
        for event_index, event in enumerate(events):
            group, sources = self._retrieve_and_fuse(
                event,
                plan=plan,
                trace=trace,
                use_plan_queries=False,
            )
            conflict_sources.extend(sources)
            reranked = self._rerank(event, group, trace)
            constrained, details = self._constraint_rerank(
                reranked,
                plan=plan,
                trace=trace,
                query_scope=event,
            )
            constraint_reports.append(dict(trace.constraint_rerank))
            pool = constrained[: self.config.fusion_pool_size]
            pool, mapped_count = self._canonical_segment_candidates(pool)
            canonical_segment_mappings += mapped_count
            event_results.append(pool)
            for identity, detail in details.items():
                detail_by_event[(event_index, identity)] = detail

        matches = match_ordered_events(
            event_results,
            max_gap_seconds=self.config.temporal_max_gap_seconds,
            top_k=requested_top_k,
            event_queries=list(events),
        )
        best_match = matches[0] if matches else None
        best_events = list(best_match.events) if best_match is not None else []
        match_mode = best_match.match_mode if best_match is not None else "none"
        route_warnings = list(best_match.warnings) if best_match is not None else [
            "temporal_no_chain"
        ]
        event_roles, target_valid, target_reason = _temporal_event_roles(
            plan,
            len(events),
        )
        complete_chain = bool(best_match and len(best_events) == len(events))
        if best_match is None:
            block_reason = "temporal_no_chain"
        elif match_mode != "strict":
            block_reason = f"temporal_match_not_strict:{match_mode}"
        elif not complete_chain:
            block_reason = "temporal_chain_incomplete"
        elif not target_valid:
            block_reason = target_reason
        else:
            block_reason = None
        answer_eligible = block_reason is None
        if not answer_eligible:
            trace.fallback_used = True
            _append_reason(trace, f"temporal:{block_reason}")
        constraint_applied = any(
            bool(report.get("applied"))
            for report in constraint_reports
        )
        trace.constraint_rerank = {
            "per_event": constraint_reports,
            "applied": constraint_applied,
            "status": "applied" if constraint_applied else "not_applied",
            "weight": self.config.constraint_weight,
            "min_signal": self.config.constraint_min_signal,
            "max_signal": max(
                (
                    float(report.get("max_signal", 0.0))
                    for report in constraint_reports
                ),
                default=0.0,
            ),
        }
        trace.temporal_route.update(
            {
                "match_count": len(matches),
                "match_mode": match_mode,
                "warnings": route_warnings,
                "answer_eligible": answer_eligible,
                "reason": block_reason,
                "answer_event_index": plan.answer_event_index,
                "chain_id": best_match.chain_id if best_match else "",
                "chain_score": best_match.score if best_match else None,
                "canonical_segment_context": {
                    "enabled": self.context_index is not None,
                    "mapped_candidate_count": canonical_segment_mappings,
                },
            }
        )

        conflict_keys = _conflicting_shots(conflict_sources)
        evidence = []
        if self.config.evidence_bundle_enabled and best_match is not None:
            for evidence_index, result in enumerate(
                best_events,
                start=1,
            ):
                event_index = evidence_index - 1
                evidence.append(
                    _build_evidence(
                        result,
                        index=evidence_index,
                        conflict_keys=conflict_keys,
                        constraint_detail=detail_by_event.get(
                            (event_index, candidate_identity(result)),
                            _ConstraintEvidence(
                                base_score=float(result.score)
                            ),
                        ),
                        temporal_event_index=event_index,
                        temporal_match_rank=1,
                        temporal_match_mode=match_mode,
                        temporal_chain_id=best_match.chain_id,
                        temporal_event_query=events[event_index],
                        temporal_event_role=event_roles[event_index],
                        temporal_chain_score=best_match.score,
                        extra_warnings=best_match.warnings,
                    )
                )

        temporal_results = []
        if best_match is not None:
            for event_index, result in enumerate(best_events):
                payload = result.to_dict()
                payload.update(
                    {
                        "temporal_event_index": event_index,
                        "temporal_match_rank": 1,
                        "temporal_match_mode": match_mode,
                        "temporal_chain_id": best_match.chain_id,
                        "temporal_event_query": events[event_index],
                        "temporal_event_role": event_roles[event_index],
                        "temporal_chain_score": best_match.score,
                    }
                )
                temporal_results.append(payload)

        return {
            **_legacy_plan_fields(plan, events),
            "query_plan": plan.to_dict(),
            "routing_trace": trace.to_dict(plan=plan, config=self.config),
            "top_k": requested_top_k,
            "answer_mode": "manual_visual_inspection",
            "answer_eligible": answer_eligible,
            "preflight_block_reason": block_reason,
            "temporal_matches": [match.to_dict() for match in matches],
            "evidence_count": len(evidence),
            "evidence": [item.to_dict() for item in evidence],
            "results": temporal_results,
        }

    def _canonical_segment_candidates(
        self,
        candidates: Sequence[RetrievalResult],
    ) -> tuple[list[RetrievalResult], int]:
        """Attach segment IDs from ``segments_all`` before temporal matching."""

        if self.context_index is None:
            return (list(candidates), 0)
        output: list[RetrievalResult] = []
        mapped_count = 0
        for candidate in candidates:
            context = self.context_index.lookup(
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                timestamp=candidate.timestamp,
                segment_id=candidate.segment_id,
            )
            if context.segment_id and context.segment_id != candidate.segment_id:
                candidate = replace(candidate, segment_id=context.segment_id)
                mapped_count += 1
            elif context.segment is not None:
                mapped_count += 1
            output.append(candidate)
        return (output, mapped_count)

    def _retrieve_and_fuse(
        self,
        query: str,
        *,
        plan: QueryPlan,
        trace: _RoutingTrace,
        use_plan_queries: bool,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        modality_groups = (
            self._retrieve_modalities(
                query,
                plan,
                trace,
                use_plan_queries=use_plan_queries,
            )
            if self.config.router_enabled
            else {}
        )
        conflict_sources = [
            result
            for group in modality_groups.values()
            for result in group
        ]
        if not modality_groups:
            trace.fallback_used = True
            _append_reason(trace, "hybrid_router_fallback")
            fallback = self.hybrid_engine.search(
                query,
                top_k=self.config.per_modality_limit,
            ).results
            modality_groups = {"hybrid": fallback}
            conflict_sources.extend(fallback)
        fused = _rrf_results(
                modality_groups,
                plan=plan,
                config=self.config,
            )
        return (fused[: self.config.rerank_pool_size], conflict_sources)

    def _retrieve_modalities(
        self,
        query: str,
        plan: QueryPlan,
        trace: _RoutingTrace,
        *,
        use_plan_queries: bool,
    ) -> dict[str, list[RetrievalResult]]:
        groups: dict[str, list[RetrievalResult]] = {}
        visual_engine = getattr(self.hybrid_engine, "visual_engine", None)
        if visual_engine is not None:
            visual_query = plan.query_for("visual") if use_plan_queries else query
            results = visual_engine.search(
                visual_query,
                top_k=self.config.per_modality_limit,
            ).results
            groups["visual"] = list(results)
            trace.modality_queries.append(
                {
                    "query": visual_query,
                    "modality": "visual",
                    "candidate_count": len(results),
                }
            )

        text_engines = getattr(self.hybrid_engine, "text_engines", {})
        for modality, engine in sorted(dict(text_engines or {}).items()):
            modality_query = plan.query_for(modality) if use_plan_queries else query
            results = engine.search_results(
                modality_query,
                top_k=self.config.per_modality_limit,
            )
            groups[modality] = list(results)
            trace.modality_queries.append(
                {
                    "query": modality_query,
                    "modality": modality,
                    "candidate_count": len(results),
                }
            )

        if self.dense_text_engine is not None:
            try:
                results = _search_dense(
                    self.dense_text_engine,
                    query,
                    self.config.per_modality_limit,
                )
                groups["dense_text"] = results
                trace.modality_queries.append(
                    {
                        "query": query,
                        "modality": "dense_text",
                        "candidate_count": len(results),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - feature-flag fallback.
                trace.fallback_used = True
                _append_reason(trace, f"dense_text:{type(exc).__name__}")
                trace.modality_queries.append(
                    {
                        "query": query,
                        "modality": "dense_text",
                        "candidate_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return groups

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: _RoutingTrace,
    ) -> list[RetrievalResult]:
        if self.candidate_reranker is None:
            return candidates
        try:
            output = self.candidate_reranker.rerank(
                query,
                candidates,
                top_k=self.config.rerank_pool_size,
            )
            if hasattr(output, "results"):
                output = output.results
            if not isinstance(output, Sequence):
                raise TypeError("reranker output must be a sequence")
            report = getattr(self.candidate_reranker, "last_report", None)
            trace.reranker = (
                f"fallback:{report.get('fallback_reason', 'model_error')}"
                if isinstance(report, Mapping) and report.get("status") == "fallback"
                else "applied"
            )
            if trace.reranker.startswith("fallback:"):
                trace.fallback_used = True
                _append_reason(trace, f"reranker:{trace.reranker}")
            return list(output)
        except Exception as exc:  # noqa: BLE001 - documented Phase 5 fallback.
            trace.reranker = f"fallback:{type(exc).__name__}"
            trace.fallback_used = True
            _append_reason(trace, f"reranker:{type(exc).__name__}")
            return candidates

    def _constraint_rerank(
        self,
        candidates: list[RetrievalResult],
        *,
        plan: QueryPlan,
        trace: _RoutingTrace,
        query_scope: str | None = None,
    ) -> tuple[
        list[RetrievalResult],
        dict[tuple[str, str], _ConstraintEvidence],
    ]:
        base_details = {
            candidate_identity(candidate): _ConstraintEvidence(
                base_score=float(candidate.score)
            )
            for candidate in candidates
        }
        if not self.config.constraint_rerank_enabled:
            trace.constraint_rerank = {
                "applied": False,
                "status": "disabled",
                "weight": self.config.constraint_weight,
                "min_signal": self.config.constraint_min_signal,
            }
            return candidates, base_details

        constraints = _context_constraints(plan, query_scope=query_scope)
        if not constraints:
            trace.constraint_rerank = {
                "applied": False,
                "status": "no_context_constraints",
                "weight": self.config.constraint_weight,
                "min_signal": self.config.constraint_min_signal,
            }
            return candidates, base_details

        try:
            scored: list[
                tuple[int, RetrievalResult, float, tuple[str, ...]]
            ] = []
            for index, candidate in enumerate(candidates):
                score, matched = _constraint_score(candidate, constraints)
                scored.append((index, candidate, score, matched))
            max_signal = max((item[2] for item in scored), default=0.0)
            details = {
                candidate_identity(candidate): _ConstraintEvidence(
                    base_score=float(candidate.score),
                    constraint_score=round(score, 8),
                    matched_constraints=matched,
                )
                for _index, candidate, score, matched in scored
            }
            if max_signal < self.config.constraint_min_signal:
                trace.constraint_rerank = {
                    "applied": False,
                    "status": "below_min_signal",
                    "weight": self.config.constraint_weight,
                    "min_signal": self.config.constraint_min_signal,
                    "max_signal": round(max_signal, 8),
                    "candidate_count": len(candidates),
                }
                return candidates, details

            constraint_weight = self.config.constraint_weight
            reranked = [
                (
                    index,
                    replace(
                        candidate,
                        score=round(
                            (1.0 - constraint_weight) * float(candidate.score)
                            + constraint_weight * score,
                            8,
                        ),
                    ),
                )
                for index, candidate, score, _matched in scored
            ]
            reranked.sort(key=lambda item: (-item[1].score, item[0]))
            trace.constraint_rerank = {
                "applied": True,
                "status": "applied",
                "weight": constraint_weight,
                "min_signal": self.config.constraint_min_signal,
                "max_signal": round(max_signal, 8),
                "candidate_count": len(candidates),
                "context_constraints": {
                    category: list(values)
                    for category, values in constraints.items()
                },
            }
            return [item[1] for item in reranked], details
        except Exception as exc:  # noqa: BLE001 - fail-open is the contract.
            trace.fallback_used = True
            _append_reason(trace, f"constraint_rerank:{type(exc).__name__}")
            trace.constraint_rerank = {
                "applied": False,
                "status": "scorer_error",
                "weight": self.config.constraint_weight,
                "min_signal": self.config.constraint_min_signal,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return candidates, base_details


def plan_qa_question(
    question: str,
    *,
    task_mode: str = "qa",
) -> QueryPlan:
    """Return the shared immutable QueryPlan contract for QA."""
    return build_query_plan(question, task_mode=task_mode)


def _temporal_event_roles(
    plan: QueryPlan,
    event_count: int,
) -> tuple[tuple[str, ...], bool, str | None]:
    """Assign answer/context roles and validate the parser's target contract."""

    if plan.answer_type == "yes_no":
        if plan.answer_event_index is not None:
            return (
                tuple("whole_chain" for _ in range(event_count)),
                False,
                "temporal_yes_no_target_must_be_whole_chain",
            )
        return (
            tuple("whole_chain" for _ in range(event_count)),
            True,
            None,
        )

    answer_index = plan.answer_event_index
    if (
        isinstance(answer_index, bool)
        or not isinstance(answer_index, int)
        or not 0 <= answer_index < event_count
    ):
        return (
            tuple("context" for _ in range(event_count)),
            False,
            "temporal_answer_target_missing",
        )
    return (
        tuple(
            "answer_target" if index == answer_index else "context"
            for index in range(event_count)
        ),
        True,
        None,
    )


def _legacy_plan_fields(
    plan: QueryPlan,
    queries: Sequence[str],
) -> dict[str, Any]:
    return {
        "question": plan.original_query,
        "answer_target": plan.answer_target,
        "answer_type": plan.answer_type,
        "retrieval_queries": list(queries),
    }


def _query_branches(
    plan: QueryPlan,
    external_queries: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    base_query = plan.retrieval_statement or plan.retrieval_query
    branches: list[tuple[str, str]] = []
    if plan.answer_type == "yes_no":
        neutral_query = _neutral_context_query(plan)
        if neutral_query and neutral_query.casefold() != base_query.casefold():
            branches.append((neutral_query, "neutral_context"))
        branches.append((base_query, "full_proposition"))
    else:
        branches.append((base_query, "parser"))
    branches.extend((query, "external_expansion") for query in external_queries)
    return tuple(branches)


def _neutral_context_query(plan: QueryPlan) -> str:
    context = _context_constraints(plan)
    phrases = [
        phrase
        for values in context.values()
        for phrase in values
    ]
    if phrases:
        return " ".join(dict.fromkeys(phrases))

    query = plan.retrieval_statement or plan.retrieval_query
    for phrase in _hypothesis_phrases(plan):
        query = re.sub(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            " ",
            query,
            flags=re.IGNORECASE,
        )
    query = re.sub(
        r"^\s*(?:is|are|was|were|do|does|did|has|have|can|could|có|phải|liệu)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"\bkhông\s*$", "", query, flags=re.IGNORECASE)
    return " ".join(query.split()).strip(" ?.!;,")


def _context_constraints(
    plan: QueryPlan,
    *,
    query_scope: str | None = None,
) -> dict[str, tuple[str, ...]]:
    payload = plan.to_dict()
    raw_constraints = payload.get("known_constraints", plan.constraints)
    if not isinstance(raw_constraints, Mapping):
        raw_constraints = plan.constraints
    roles = _constraint_roles(plan)
    context: dict[str, tuple[str, ...]] = {}
    for raw_category, raw_values in raw_constraints.items():
        category = str(raw_category)
        if isinstance(raw_values, str):
            values: Sequence[object] = (raw_values,)
        elif isinstance(raw_values, Sequence):
            values = raw_values
        else:
            continue
        accepted: list[str] = []
        for raw_phrase in values:
            phrase = " ".join(str(raw_phrase).split())
            if not phrase:
                continue
            role = roles.get(category, {}).get(phrase.casefold(), "context")
            if role != "context":
                continue
            accepted.append(phrase)
        if accepted:
            context[category] = tuple(dict.fromkeys(accepted))

    # Whole-query constraints can contain later event actions.  For an event
    # retrieval, retain subjects as cross-event anchors and only use other
    # phrases that actually occur in the clean event clause.  If the parser
    # cannot align a phrase, retrieval remains fail-open rather than guessing.
    if query_scope:
        scoped: dict[str, tuple[str, ...]] = {}
        normalized_scope = _normalize_match_text(query_scope)
        for category, values in context.items():
            kept = tuple(
                phrase
                for phrase in values
                if category == "subject"
                or _normalized_phrase_present(phrase, normalized_scope)
            )
            if kept:
                scoped[category] = kept
        return scoped
    return context


def _hypothesis_phrases(plan: QueryPlan) -> tuple[str, ...]:
    roles = _constraint_roles(plan)
    return tuple(
        phrase
        for category_roles in roles.values()
        for phrase, role in category_roles.items()
        if role == "hypothesis"
    )


def _constraint_roles(plan: QueryPlan) -> dict[str, dict[str, str]]:
    payload = plan.to_dict()
    raw = payload.get("constraint_roles")
    if raw is None:
        raw = getattr(plan, "constraint_roles", ())
    normalized: dict[str, dict[str, str]] = {}
    entries = raw.items() if isinstance(raw, Mapping) else raw or ()
    for entry in entries:
        try:
            raw_category, raw_roles = entry
        except (TypeError, ValueError):
            continue
        category = str(raw_category)
        role_entries = (
            raw_roles.items()
            if isinstance(raw_roles, Mapping)
            else raw_roles or ()
        )
        category_roles: dict[str, str] = {}
        for role_entry in role_entries:
            try:
                raw_phrase, raw_role = role_entry
            except (TypeError, ValueError):
                continue
            role = str(raw_role).strip().casefold()
            if role not in {"context", "hypothesis"}:
                continue
            category_roles[str(raw_phrase).casefold()] = role
        if category_roles:
            normalized[category] = category_roles
    return normalized


def _constraint_score(
    candidate: RetrievalResult,
    constraints: Mapping[str, Sequence[str]],
) -> tuple[float, tuple[str, ...]]:
    category_scores: list[float] = []
    matched: list[str] = []
    for category, phrases in constraints.items():
        fields = _constraint_fields(candidate, category)
        if not fields:
            category_scores.append(0.0)
            continue
        phrase_scores: list[float] = []
        for phrase in phrases:
            normalized_phrase = _normalize_match_text(phrase)
            exact = any(
                _normalized_phrase_present(phrase, _normalize_match_text(text))
                for text in fields
            )
            score = 1.0 if exact else max(
                (weighted_query_coverage(phrase, text) for text in fields),
                default=0.0,
            )
            bounded = max(0.0, min(1.0, float(score)))
            phrase_scores.append(bounded)
            if bounded > 0.0 and normalized_phrase:
                matched.append(f"{category}:{phrase}")
        if phrase_scores:
            category_scores.append(sum(phrase_scores) / len(phrase_scores))
    if not category_scores:
        return 0.0, ()
    return (
        sum(category_scores) / len(category_scores),
        tuple(dict.fromkeys(matched)),
    )


def _constraint_fields(
    candidate: RetrievalResult,
    category: str,
) -> tuple[str, ...]:
    normalized = category.strip().casefold()
    if normalized in {"subject", "subjects", "object", "objects"}:
        return tuple(
            text
            for text in (candidate.caption, " ".join(candidate.objects))
            if str(text).strip()
        )
    if normalized in {"attributes", "actions", "locations"}:
        return (candidate.caption,) if candidate.caption.strip() else ()
    if normalized == "ocr_terms":
        return (candidate.ocr_text,) if candidate.ocr_text.strip() else ()
    return ()


def _normalize_match_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, re.UNICODE))


def _normalized_phrase_present(phrase: str, normalized_text: str) -> bool:
    normalized_phrase = _normalize_match_text(phrase)
    if not normalized_phrase or not normalized_text:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def _append_reason(trace: _RoutingTrace, reason: str) -> None:
    if reason not in trace.fallback_reasons:
        trace.fallback_reasons.append(reason)


def _external_queries(
    values: Sequence[str],
    *,
    base_query: str,
) -> tuple[str, ...]:
    if len(values) > 20:
        raise ValueError("expanded_queries supports at most 20 caller-provided queries")
    seen = {base_query.casefold()}
    accepted: list[str] = []
    for value in values:
        query = " ".join(str(value).split())
        if not query:
            continue
        folded = query.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        accepted.append(query)
    return tuple(accepted)


def _search_dense(
    engine: object,
    query: str,
    top_k: int,
) -> list[RetrievalResult]:
    if hasattr(engine, "search_results"):
        return list(engine.search_results(query, top_k=top_k))
    response = engine.search(query, top_k=top_k)
    if hasattr(response, "results"):
        return list(response.results)
    return list(response)


def _rrf_results(
    groups: Mapping[str, Sequence[RetrievalResult]],
    *,
    plan: QueryPlan,
    config: QaRoutingConfig,
) -> list[RetrievalResult]:
    fused = weighted_rrf(
        groups,
        plan=plan,
        k=config.rrf_k,
        weights=dict(config.weights),
        hint_boost=config.modality_hint_boost,
    )
    if not fused:
        return []
    maximum = max(item.rrf_score for item in fused) or 1.0
    results: list[RetrievalResult] = []
    for item in fused:
        scores = dict(item.result.modality_scores)
        scores["rrf"] = round(item.rrf_score, 8)
        results.append(
            replace(
                item.result,
                score=round(item.rrf_score / maximum, 8),
                modality_scores=scores,
            )
        )
    return results


def _fuse_evidence(
    candidate_groups: list[list[RetrievalResult]],
    *,
    top_k: int,
    consensus_bonus: float,
) -> list[RetrievalResult]:
    occurrences: dict[tuple[str, str], int] = {}
    for group in candidate_groups:
        seen_in_query: set[tuple[str, str]] = set()
        for candidate in group:
            identity = candidate_identity(candidate)
            if identity in seen_in_query:
                continue
            seen_in_query.add(identity)
            occurrences[identity] = occurrences.get(identity, 0) + 1

    merged = merge_candidates(candidate_groups, dedupe_same_shot=False)
    boosted = [
        replace(
            candidate,
            score=round(
                min(
                    1.0,
                    candidate.score
                    + consensus_bonus
                    * max(0, occurrences.get(candidate_identity(candidate), 1) - 1),
                ),
                8,
            ),
        )
        for candidate in merged
    ]
    boosted.sort(
        key=lambda candidate: (
            -candidate.score,
            -candidate.timestamp_confidence,
            candidate.video_id,
            candidate.timestamp,
            candidate.frame_id,
        )
    )
    return boosted[:top_k]


def _shot_key(result: RetrievalResult) -> tuple[str, str]:
    return (
        result.video_id,
        result.shot_id or result.segment_id or result.frame_id,
    )


def _conflicting_shots(
    candidates: Sequence[RetrievalResult],
) -> set[tuple[str, str]]:
    values: dict[tuple[str, str], set[tuple[str, str, tuple[str, ...]]]] = {}
    for candidate in candidates:
        signature = (
            candidate.caption.strip(),
            candidate.ocr_text.strip(),
            tuple(sorted(str(value) for value in candidate.objects)),
        )
        if any(signature):
            values.setdefault(_shot_key(candidate), set()).add(signature)
    return {key for key, signatures in values.items() if len(signatures) > 1}


def _build_evidence(
    result: RetrievalResult,
    *,
    index: int,
    conflict_keys: set[tuple[str, str]],
    constraint_detail: _ConstraintEvidence,
    temporal_event_index: int | None = None,
    temporal_match_rank: int | None = None,
    temporal_match_mode: str = "",
    temporal_chain_id: str = "",
    temporal_event_query: str = "",
    temporal_event_role: str = "",
    temporal_chain_score: float | None = None,
    extra_warnings: Sequence[str] = (),
) -> QaEvidence:
    warnings: list[str] = list(extra_warnings)
    image_path = result.keyframe_path or result.thumbnail_path
    if not image_path:
        warnings.append("missing_image_path")
    if not result.shot_id and not result.segment_id:
        warnings.append("missing_shot_metadata")
    if _shot_key(result) in conflict_keys:
        warnings.append("conflicting_metadata")
    modalities = tuple(
        sorted(
            modality
            for modality, score in result.modality_scores.items()
            if modality != "rrf" and float(score) != 0.0
        )
    )
    return QaEvidence(
        evidence_id=f"E{index:03d}",
        video_id=result.video_id,
        frame_id=result.frame_id,
        frame_index=result.frame_index,
        shot_id=result.shot_id or result.segment_id,
        timestamp=float(result.timestamp),
        image_path=image_path,
        caption=result.caption,
        ocr_text=result.ocr_text,
        objects=tuple(str(value) for value in result.objects),
        source_modalities=modalities,
        retrieval_score=round(float(result.score), 8),
        base_retrieval_score=round(float(constraint_detail.base_score), 8),
        constraint_score=round(float(constraint_detail.constraint_score), 8),
        matched_constraints=constraint_detail.matched_constraints,
        temporal_event_index=temporal_event_index,
        temporal_match_rank=temporal_match_rank,
        temporal_match_mode=temporal_match_mode,
        temporal_chain_id=temporal_chain_id,
        temporal_event_query=temporal_event_query,
        temporal_event_role=temporal_event_role,
        temporal_chain_score=temporal_chain_score,
        warnings=tuple(dict.fromkeys(warnings)),
    )
