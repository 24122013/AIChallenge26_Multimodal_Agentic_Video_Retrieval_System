"""Event-wise retrieval for TRAKE.

This module calls the canonical public hybrid search service once per ordered
event and can independently add BGE-M3 dense candidates.  Incompatible raw
score scales are combined with weighted reciprocal-rank fusion, then normalized
inside each event list so they cannot dominate temporal alignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.retrieval_config import TrakeConfig
from backend.app.services.trake.models import EventCandidate, TemporalEventPlan


DEFAULT_TEMPORAL_NMS_RADIUS_FRAMES = 2


class RequiredTrakePipelineError(RuntimeError):
    """Sanitized dependency failure for a required TRAKE BGE branch."""

    def __init__(self, message: str, *, failure_code: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.response = {
            "task": "trake",
            "hypotheses": [],
            "trace": {
                "status": "dependency_unavailable",
                "failure_code": failure_code,
            },
        }


class RetrievalEngineLike(Protocol):
    def search(self, query: str, top_k: int | None = None) -> Any:
        """Return a response carrying ``results`` or a result sequence."""


class CandidateRerankerLike(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> Any:
        """Return reranked results, optionally together with an audit report."""


@dataclass(frozen=True)
class EventRetrievalBatch:
    """Internal, typed result of event-wise and optional context retrieval."""

    event_candidates: dict[int, tuple[EventCandidate, ...]]
    context_scores: dict[str, float] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_candidates": {
                str(index): [candidate.to_dict() for candidate in candidates]
                for index, candidates in sorted(self.event_candidates.items())
            },
            "context_scores": {
                key: self.context_scores[key] for key in sorted(self.context_scores)
            },
            "trace": dict(self.trace),
            "warnings": list(self.warnings),
        }


class EventRetriever:
    """Retrieve, fuse, and optionally rerank each planned event independently."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngineLike,
        config: TrakeConfig | None = None,
        *,
        dense_event_engine: RetrievalEngineLike | None = None,
        event_reranker: CandidateRerankerLike | None = None,
    ) -> None:
        self.retrieval_engine = retrieval_engine
        self.config = config or TrakeConfig()
        self.dense_event_engine = dense_event_engine
        self.event_reranker = event_reranker

    def retrieve(
        self,
        plan: TemporalEventPlan,
        *,
        include_context: bool = True,
    ) -> EventRetrievalBatch:
        if not plan.events:
            return EventRetrievalBatch(
                event_candidates={},
                trace={"event_count": 0, "retrieval_calls": 0},
                warnings=("no_events_to_retrieve",),
            )

        by_event: dict[int, tuple[EventCandidate, ...]] = {}
        event_trace: list[dict[str, Any]] = []
        warnings: list[str] = []
        search_method, wide_pool_method = _search_method(self.retrieval_engine)
        dense_enabled = bool(_config_value(self.config, "bge_dense_enabled", False))
        reranker_enabled = bool(
            _config_value(self.config, "bge_reranker_enabled", False)
        )
        bge_required = bool(_config_value(self.config, "bge_required", False))
        dense_calls = 0
        reranker_calls = 0
        dense_circuit_open = False
        reranker_circuit_open = False
        for event in plan.events:
            response = search_method(
                event.retrieval_query,
                top_k=self.config.event_top_k,
            )
            raw_results = _response_results(response)
            valid_results, missing_lineage = _valid_event_results(raw_results)
            source_trace: dict[str, Any] = {
                "canonical_hybrid": {
                    "enabled": True,
                    "available": True,
                    "status": "success",
                    "requested_top_k": self.config.event_top_k,
                    "retrieved_count": len(raw_results),
                    "valid_lineage_count": len(valid_results),
                    "missing_lineage_count": missing_lineage,
                }
            }

            fused_results = list(valid_results)
            fusion_trace: dict[str, Any] = {
                "method": str(_config_value(self.config, "retrieval_fusion", "rrf")),
                "status": "disabled",
                "output_count": len(fused_results),
            }
            dense_trace: dict[str, Any] = {
                "enabled": dense_enabled,
                "available": self.dense_event_engine is not None,
                "status": "disabled" if not dense_enabled else "not_attempted",
                "requested_top_k": int(
                    _config_value(self.config, "bge_dense_top_k", self.config.event_top_k)
                ),
                "retrieved_count": 0,
                "valid_lineage_count": 0,
                "missing_lineage_count": 0,
            }
            if dense_enabled:
                if dense_circuit_open:
                    dense_trace.update(
                        status="skipped_circuit_open",
                        fallback="canonical_hybrid",
                        failure_code="request_circuit_open",
                    )
                    fusion_trace.update(
                        status="not_applied_request_circuit_open",
                        fallback="canonical_hybrid",
                    )
                elif self.dense_event_engine is None:
                    dense_trace.update(
                        status="failed_required" if bge_required else "failed_optional",
                        fallback="none" if bge_required else "canonical_hybrid",
                        failure_code="engine_unavailable",
                    )
                    if bge_required:
                        _raise_required_bge("dense_retrieval", event.index)
                    warnings.append(
                        f"event_{event.index}_bge_dense_engine_unavailable"
                    )
                    dense_circuit_open = True
                    fusion_trace.update(
                        status="not_applied_dense_unavailable",
                        fallback="canonical_hybrid",
                    )
                else:
                    try:
                        dense_search, dense_method = _search_method(
                            self.dense_event_engine
                        )
                        dense_calls += 1
                        dense_response = dense_search(
                            event.retrieval_query,
                            top_k=dense_trace["requested_top_k"],
                        )
                        dense_raw = _response_results(dense_response)
                        dense_valid, dense_missing_lineage = _valid_event_results(
                            dense_raw
                        )
                        dense_trace.update(
                            status="success",
                            method=dense_method,
                            retrieved_count=len(dense_raw),
                            valid_lineage_count=len(dense_valid),
                            missing_lineage_count=dense_missing_lineage,
                        )
                        if dense_missing_lineage and bge_required:
                            _raise_required_bge("dense_lineage_contract", event.index)
                        if dense_missing_lineage:
                            warnings.append(
                                f"event_{event.index}_bge_dense_missing_frame_lineage"
                            )
                        if dense_raw and not dense_valid:
                            dense_trace.update(
                                status="failed_optional",
                                fallback="canonical_hybrid",
                                failure_code="no_valid_frame_lineage",
                            )
                            dense_circuit_open = True
                            fusion_trace.update(
                                status="not_applied_invalid_dense_lineage",
                                fallback="canonical_hybrid",
                                output_count=len(fused_results),
                            )
                        else:
                            fused_results, fusion_trace = _weighted_rrf_event_results(
                                valid_results,
                                dense_valid,
                                config=self.config,
                            )
                    except Exception:
                        dense_trace.update(
                            status=(
                                "failed_required" if bge_required else "failed_optional"
                            ),
                            fallback="none" if bge_required else "canonical_hybrid",
                            failure_code="dense_search_failed",
                        )
                        if bge_required:
                            _raise_required_bge("dense_retrieval", event.index)
                        warnings.append(
                            f"event_{event.index}_bge_dense_failed_optional"
                        )
                        dense_circuit_open = True
                        fused_results = list(valid_results)
                        fusion_trace.update(
                            status="not_applied_dense_failure",
                            fallback="canonical_hybrid",
                            output_count=len(fused_results),
                        )
            source_trace["bge_dense"] = dense_trace

            reranker_trace: dict[str, Any] = {
                "enabled": reranker_enabled,
                "available": self.event_reranker is not None,
                "status": "disabled" if not reranker_enabled else "not_attempted",
                "requested_top_k": int(
                    _config_value(
                        self.config,
                        "bge_reranker_top_k",
                        self.config.event_top_k,
                    )
                ),
                "input_count": len(fused_results),
                "output_count": len(fused_results),
            }
            if reranker_enabled:
                if reranker_circuit_open:
                    reranker_trace.update(
                        status="skipped_circuit_open",
                        fallback="pre_rerank_order",
                        failure_code="request_circuit_open",
                    )
                elif self.event_reranker is None:
                    reranker_trace.update(
                        status="failed_required" if bge_required else "failed_optional",
                        fallback="none" if bge_required else "pre_rerank_order",
                        failure_code="reranker_unavailable",
                    )
                    if bge_required:
                        _raise_required_bge("reranker", event.index)
                    warnings.append(
                        f"event_{event.index}_bge_reranker_engine_unavailable"
                    )
                    reranker_circuit_open = True
                elif fused_results:
                    try:
                        reranker_calls += 1
                        fused_results, reranker_trace = _rerank_event_results(
                            event.retrieval_query,
                            fused_results,
                            reranker=self.event_reranker,
                            top_k=reranker_trace["requested_top_k"],
                        )
                        if reranker_trace["status"] == "reported_fallback":
                            if bge_required:
                                _raise_required_bge("reranker", event.index)
                            warnings.append(
                                f"event_{event.index}_bge_reranker_failed_optional"
                            )
                            reranker_circuit_open = True
                        else:
                            report = reranker_trace.get("report", {})
                            if (
                                isinstance(report, Mapping)
                                and reranker_trace.get("scored_pool_count", 0) > 0
                                and report.get("scored_count") == 0
                            ):
                                reranker_trace.update(
                                    status="no_scorable_content",
                                    fallback="pre_rerank_order",
                                    failure_code="no_scorable_content",
                                )
                                if bge_required:
                                    _raise_required_bge("reranker_content", event.index)
                                warnings.append(
                                    f"event_{event.index}_bge_reranker_no_scorable_content"
                                )
                            elif int(reranker_trace.get("rejected_count", 0)) > 0:
                                if bge_required:
                                    _raise_required_bge("reranker_contract", event.index)
                                warnings.append(
                                    f"event_{event.index}_bge_reranker_rejected_output"
                                )
                                reranker_circuit_open = True
                    except Exception:
                        reranker_trace.update(
                            status=(
                                "failed_required" if bge_required else "failed_optional"
                            ),
                            fallback="none" if bge_required else "pre_rerank_order",
                            failure_code="reranker_failed",
                            output_count=len(fused_results),
                        )
                        if bge_required:
                            _raise_required_bge("reranker", event.index)
                        warnings.append(
                            f"event_{event.index}_bge_reranker_failed_optional"
                        )
                        reranker_circuit_open = True
                else:
                    reranker_trace.update(status="skipped_empty", output_count=0)

            candidates = diversify_and_normalize(
                event_index=event.index,
                retrieval_query=event.retrieval_query,
                results=fused_results,
                config=self.config,
            )
            by_event[event.index] = tuple(candidates)
            if missing_lineage:
                warnings.append(f"event_{event.index}_missing_frame_lineage")
            event_trace.append(
                {
                    "event_index": event.index,
                    "requested_top_k": self.config.event_top_k,
                    "retrieved_count": len(raw_results),
                    "valid_lineage_count": len(valid_results),
                    "candidate_count": len(candidates),
                    "missing_lineage_count": missing_lineage,
                    "sources": source_trace,
                    "fusion": fusion_trace,
                    "reranker": reranker_trace,
                }
            )

        context_scores: dict[str, float] = {}
        context_trace: dict[str, Any] = {
            "enabled": False,
            "retrieved_count": 0,
            "video_count": 0,
        }
        context = plan.context.strip()
        if include_context and self.config.context_weight > 0 and context:
            try:
                response = search_method(
                    context,
                    top_k=self.config.event_top_k,
                )
                context_results = _response_results(response)
                context_scores = normalize_context_video_scores(context_results)
                context_trace = {
                    "enabled": True,
                    "retrieved_count": len(context_results),
                    "video_count": len(context_scores),
                    "status": "success",
                }
            except Exception:
                # Context is only a video-level prior.  Event evidence remains
                # usable when this optional branch is unavailable.
                warnings.append("context_retrieval_failed")
                context_trace = {
                    "enabled": True,
                    "retrieved_count": 0,
                    "video_count": 0,
                    "status": "failed_optional",
                }

        return EventRetrievalBatch(
            event_candidates=by_event,
            context_scores=context_scores,
            trace={
                "event_count": len(plan.events),
                "retrieval_calls": (
                    len(plan.events)
                    + dense_calls
                    + int(context_trace["enabled"])
                ),
                "reranker_calls": reranker_calls,
                "events": event_trace,
                "context": context_trace,
                "score_normalization": self.config.score_normalization,
                "lineage_policy": "original_frame_index_required",
                "available_modalities": list(
                    getattr(self.retrieval_engine, "available_modalities", ())
                ),
                "bge": {
                    "required": bge_required,
                    "dense_enabled": dense_enabled,
                    "dense_engine_available": self.dense_event_engine is not None,
                    "dense_calls": dense_calls,
                    "dense_circuit_open": dense_circuit_open,
                    "reranker_enabled": reranker_enabled,
                    "reranker_available": self.event_reranker is not None,
                    "reranker_calls": reranker_calls,
                    "reranker_circuit_open": reranker_circuit_open,
                },
                "retrieval_engine": type(self.retrieval_engine).__name__,
                "wide_pool_method": wide_pool_method,
            },
            warnings=tuple(dict.fromkeys(warnings)),
        )


def retrieve_event_candidates(
    plan: TemporalEventPlan,
    retrieval_engine: RetrievalEngineLike,
    config: TrakeConfig | None = None,
    *,
    include_context: bool = True,
    dense_event_engine: RetrievalEngineLike | None = None,
    event_reranker: CandidateRerankerLike | None = None,
) -> EventRetrievalBatch:
    """Functional adapter used by tests and lightweight callers."""

    return EventRetriever(
        retrieval_engine,
        config,
        dense_event_engine=dense_event_engine,
        event_reranker=event_reranker,
    ).retrieve(
        plan,
        include_context=include_context,
    )


def _weighted_rrf_event_results(
    hybrid_results: Sequence[RetrievalResult],
    dense_results: Sequence[RetrievalResult],
    *,
    config: TrakeConfig,
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    """Fuse event-local ranks without comparing incompatible raw scores.

    Identity deliberately uses the original-video frame lineage rather than an
    index-internal frame id.  Hybrid metadata remains canonical on overlap and
    is enriched with non-conflicting dense metadata.
    """

    method = str(_config_value(config, "retrieval_fusion", "rrf")).casefold().strip()
    if method != "rrf":
        raise ValueError("TRAKE event retrieval fusion must be 'rrf'")
    rrf_k = int(_config_value(config, "rrf_k", 60))
    hybrid_weight = float(_config_value(config, "hybrid_rrf_weight", 1.0))
    dense_weight = float(_config_value(config, "bge_rrf_weight", 1.0))
    if rrf_k <= 0 or hybrid_weight < 0 or dense_weight < 0:
        raise ValueError("invalid TRAKE weighted RRF configuration")
    if hybrid_weight + dense_weight <= 0:
        raise ValueError("TRAKE weighted RRF requires a positive source weight")

    groups = (
        ("canonical_hybrid", hybrid_results, hybrid_weight),
        ("bge_dense", dense_results, dense_weight),
    )
    states: dict[tuple[str, int], dict[str, Any]] = {}
    unique_counts: dict[str, int] = {}
    for source, results, weight in groups:
        seen: set[tuple[str, int]] = set()
        for rank, result in enumerate(results, start=1):
            identity = _frame_identity(result)
            if identity in seen:
                existing = states.get(identity)
                if existing is not None:
                    existing["result"] = _merge_canonical_result(
                        existing["result"], result
                    )
                continue
            seen.add(identity)
            contribution = weight / (rrf_k + rank) if weight > 0 else 0.0
            state = states.get(identity)
            if state is None:
                state = {
                    "result": result,
                    "score": 0.0,
                    "ranks": {},
                    "contributions": {},
                    "raw_scores": {},
                }
                states[identity] = state
            elif source == "canonical_hybrid":
                # This branch is defensive because hybrid is currently visited
                # first.  It guarantees canonical precedence if groups change.
                state["result"] = _merge_canonical_result(result, state["result"])
            else:
                state["result"] = _merge_canonical_result(state["result"], result)
            state["score"] = float(state["score"]) + contribution
            state["ranks"][source] = rank
            state["contributions"][source] = contribution
            state["raw_scores"][source] = float(result.score)
        unique_counts[source] = len(seen)

    # Normalize against sources that can actually contribute to this event.
    # Counting an enabled-but-empty dense source would artificially depress
    # every canonical-hybrid score and change the rerank blend on sparse
    # events, even though no BGE candidate participated in fusion.
    active_weight_sum = sum(
        weight
        for _source, results, weight in groups
        if results and weight > 0
    )
    max_rrf = active_weight_sum / (rrf_k + 1)
    ranked: list[tuple[RetrievalResult, float, float, int]] = []
    for identity, state in states.items():
        contributions = state["contributions"]
        total = float(state["score"])
        normalized = total / max_rrf if max_rrf > 0 else 0.0
        result = state["result"]
        modalities = dict(result.modality_scores)
        modalities.update(
            {
                "trake_rrf_hybrid": round(
                    float(contributions.get("canonical_hybrid", 0.0)), 10
                ),
                "trake_rrf_bge_dense": round(
                    float(contributions.get("bge_dense", 0.0)), 10
                ),
                "trake_rrf_fused": round(total, 10),
            }
        )
        raw_scores = state["raw_scores"]
        if "canonical_hybrid" in raw_scores:
            modalities["trake_hybrid_raw"] = float(
                raw_scores["canonical_hybrid"]
            )
        if "bge_dense" in raw_scores:
            modalities["trake_bge_dense_raw"] = float(raw_scores["bge_dense"])
        result = replace(
            result,
            score=round(normalized, 8),
            modality_scores=modalities,
        )
        ranked.append(
            (
                result,
                total,
                float(contributions.get("canonical_hybrid", 0.0)),
                min(state["ranks"].values()),
            )
        )
    ranked.sort(
        key=lambda item: (
            -item[1],
            -item[2],
            item[3],
            item[0].video_id,
            int(item[0].frame_index),
            item[0].frame_id,
        )
    )
    output = [item[0] for item in ranked]
    overlap = sum(
        1
        for state in states.values()
        if "canonical_hybrid" in state["ranks"] and "bge_dense" in state["ranks"]
    )
    return output, {
        "method": "rrf",
        "status": "applied",
        "rrf_k": rrf_k,
        "weights": {
            "canonical_hybrid": hybrid_weight,
            "bge_dense": dense_weight,
        },
        "active_weight_sum": active_weight_sum,
        "input_counts": {
            "canonical_hybrid": len(hybrid_results),
            "bge_dense": len(dense_results),
        },
        "unique_source_counts": unique_counts,
        "overlap_count": overlap,
        "output_count": len(output),
    }


def _rerank_event_results(
    query: str,
    results: Sequence[RetrievalResult],
    *,
    reranker: CandidateRerankerLike,
    top_k: int,
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    """Rerank a bounded head while retaining the remaining recall pool."""

    limit = min(max(1, int(top_k)), len(results))
    pool = list(results[:limit])
    reporting = getattr(reranker, "rerank_with_report", None)
    if callable(reporting):
        response = reporting(query, pool, top_k=limit)
        report_value: Any = None
    else:
        response = reranker.rerank(query, pool, top_k=limit)
        report_value = getattr(reranker, "last_report", None)
    if (
        isinstance(response, tuple)
        and len(response) == 2
        and not isinstance(response[0], RetrievalResult)
    ):
        response, report_value = response
    reranked = _response_results(response)
    report = _sanitize_reranker_report(report_value)
    reported_status = str(report.get("status", "")).casefold()
    if reported_status in {"fallback", "failed", "failure", "error"}:
        return list(results), {
            "enabled": True,
            "available": True,
            "status": "reported_fallback",
            "requested_top_k": int(top_k),
            "input_count": len(results),
            "scored_pool_count": len(pool),
            "returned_count": len(reranked),
            "rejected_count": 0,
            "output_count": len(results),
            "fallback": "pre_rerank_order",
            "failure_code": "reranker_reported_fallback",
            "report": report,
        }

    # A reranker may only reorder/rescore the bounded pool it was given.  This
    # prevents an adapter from injecting an unscored tail result into the head.
    canonical: dict[tuple[str, int], RetrievalResult] = {}
    for result in pool:
        canonical.setdefault(_frame_identity(result), result)
    output: list[RetrievalResult] = []
    seen: set[tuple[str, int]] = set()
    rejected_count = 0
    for reranked_result in reranked:
        try:
            identity = _frame_identity(reranked_result)
        except (TypeError, ValueError):
            rejected_count += 1
            continue
        base = canonical.get(identity)
        if base is None or identity in seen:
            rejected_count += 1
            continue
        raw_score = reranked_result.score
        if isinstance(raw_score, bool):
            rejected_count += 1
            continue
        try:
            reranked_score = float(raw_score)
        except (TypeError, ValueError):
            rejected_count += 1
            continue
        if not math.isfinite(reranked_score):
            rejected_count += 1
            continue
        enriched = _merge_canonical_result(base, reranked_result)
        output.append(
            replace(
                enriched,
                score=reranked_score,
                modality_scores={
                    **enriched.modality_scores,
                    **reranked_result.modality_scores,
                },
            )
        )
        seen.add(identity)
    if pool and not output:
        raise RuntimeError("BGE reranker returned no canonical candidates")

    # Keep candidates omitted by a top-k returning adapter.  The reranker only
    # changes priority; it must not silently reduce event/video recall.
    for result in results:
        identity = _frame_identity(result)
        if identity not in seen:
            output.append(result)
            seen.add(identity)
    trace: dict[str, Any] = {
        "enabled": True,
        "available": True,
        "status": "success",
        "requested_top_k": int(top_k),
        "input_count": len(results),
        "scored_pool_count": len(pool),
        "returned_count": len(reranked),
        "rejected_count": rejected_count,
        "output_count": len(output),
    }
    if report:
        trace["report"] = report
    return output, trace


def _merge_canonical_result(
    primary: RetrievalResult,
    secondary: RetrievalResult,
) -> RetrievalResult:
    """Enrich a canonical frame without changing its identity/provenance."""

    if _frame_identity(primary) != _frame_identity(secondary):
        raise ValueError("cannot merge results with different frame lineage")
    modalities = dict(primary.modality_scores)
    if any(not math.isfinite(float(value)) for value in modalities.values()):
        raise ValueError("cannot merge non-finite modality scores")
    for name, value in secondary.modality_scores.items():
        numeric = float(value)
        existing = float(modalities.get(name, numeric))
        if not math.isfinite(numeric) or not math.isfinite(existing):
            raise ValueError("cannot merge non-finite modality scores")
        modalities[name] = max(numeric, existing)
    neighbors = list(primary.neighbors)
    neighbor_keys = {(item.video_id, item.frame_id) for item in neighbors}
    for neighbor in secondary.neighbors:
        key = (neighbor.video_id, neighbor.frame_id)
        if key not in neighbor_keys:
            neighbors.append(neighbor)
            neighbor_keys.add(key)
    return replace(
        primary,
        frame_id=primary.frame_id or secondary.frame_id,
        segment_id=primary.segment_id or secondary.segment_id,
        shot_id=primary.shot_id or secondary.shot_id,
        faiss_index=(
            primary.faiss_index
            if primary.faiss_index is not None
            else secondary.faiss_index
        ),
        keyframe_path=primary.keyframe_path or secondary.keyframe_path,
        thumbnail_path=primary.thumbnail_path or secondary.thumbnail_path,
        timestamp_source=(
            primary.timestamp_source
            if primary.timestamp_source != "unknown"
            else secondary.timestamp_source
        ),
        timestamp_confidence=max(
            float(primary.timestamp_confidence),
            float(secondary.timestamp_confidence),
        ),
        caption=primary.caption or secondary.caption,
        ocr_text=primary.ocr_text or secondary.ocr_text,
        objects=list(dict.fromkeys([*primary.objects, *secondary.objects])),
        modality_scores=modalities,
        neighbors=neighbors,
    )


def _sanitize_reranker_report(value: Any) -> dict[str, Any]:
    """Whitelist non-sensitive aggregate fields from a third-party report."""

    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, Any] = {}
    status = value.get("status")
    if isinstance(status, str):
        normalized = status.casefold().strip()
        if normalized in {
            "passed",
            "success",
            "applied",
            "fallback",
            "failed",
            "failure",
            "error",
        }:
            output["status"] = normalized
    for name in ("candidate_count", "scored_count", "output_count"):
        raw = value.get(name)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            output[name] = raw
    alpha = value.get("retrieval_alpha")
    if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        numeric = float(alpha)
        if math.isfinite(numeric) and 0.0 <= numeric <= 1.0:
            output["retrieval_alpha"] = numeric
    if value.get("fallback_reason"):
        output["fallback_code"] = "reranker_reported_fallback"
    return output


def _frame_identity(result: RetrievalResult) -> tuple[str, int]:
    frame_index = result.frame_index
    if (
        not str(result.video_id).strip()
        or isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
    ):
        raise ValueError("result lacks canonical original frame lineage")
    return str(result.video_id), frame_index


def _search_method(engine: RetrievalEngineLike) -> tuple[Any, str]:
    method = getattr(engine, "search_pool", None)
    if callable(method):
        return method, "search_pool"
    method = getattr(engine, "search", None)
    if not callable(method):
        raise TypeError("retrieval engine must expose search or search_pool")
    return method, "search"


def _config_value(config: TrakeConfig, name: str, default: Any) -> Any:
    # The fallback keeps lightweight callers compatible while configuration is
    # rolled out atomically across independently deployed services.
    return getattr(config, name, default)


def _raise_required_bge(stage: str, event_index: int) -> None:
    raise RequiredTrakePipelineError(
        f"required TRAKE BGE {stage} unavailable for event {event_index}",
        failure_code=f"required_bge_{stage}_unavailable",
    ) from None


def diversify_and_normalize(
    *,
    event_index: int,
    retrieval_query: str,
    results: Sequence[RetrievalResult],
    config: TrakeConfig,
    temporal_nms_radius_frames: int = DEFAULT_TEMPORAL_NMS_RADIUS_FRAMES,
) -> list[EventCandidate]:
    """Apply per-shot/per-video diversity while retaining event-local ranks."""

    normalized = normalize_event_scores(results, method=config.score_normalization)
    if temporal_nms_radius_frames < 0:
        raise ValueError("temporal NMS radius must be non-negative")
    per_video_count: dict[str, int] = {}
    per_shot_count: dict[tuple[str, str], int] = {}
    selected_frames_by_video: dict[str, list[int]] = {}
    seen_locations: set[tuple[str, int]] = set()
    output: list[EventCandidate] = []
    for rank, (result, score) in enumerate(zip(results, normalized), start=1):
        video_id = result.video_id
        frame_index = int(result.frame_index)  # validated by caller
        location = (video_id, frame_index)
        if location in seen_locations:
            continue
        if per_video_count.get(video_id, 0) >= config.max_candidates_per_event_per_video:
            continue
        shot_key = result.shot_id or result.segment_id
        if shot_key:
            key = (video_id, shot_key)
            if per_shot_count.get(key, 0) >= config.max_candidates_per_shot:
                continue
            per_shot_count[key] = per_shot_count.get(key, 0) + 1
        elif any(
            abs(frame_index - selected) <= temporal_nms_radius_frames
            for selected in selected_frames_by_video.get(video_id, ())
        ):
            # Legacy/minimal frame maps may have no shot identity.  A small
            # original-frame NMS keeps adjacent technical keyframes from using
            # every alignment slot while retaining wider semantic alternatives.
            continue
        selected_frames_by_video.setdefault(video_id, []).append(frame_index)
        seen_locations.add(location)
        per_video_count[video_id] = per_video_count.get(video_id, 0) + 1
        output.append(
            EventCandidate(
                event_index=event_index,
                result=result,
                normalized_score=round(float(score), 6),
                rank=rank,
                retrieval_query=retrieval_query,
            )
        )
    return output


def normalize_event_scores(
    results: Sequence[RetrievalResult],
    *,
    method: str = "rank",
) -> list[float]:
    """Return deterministic rank/percentile scores in ``(0, 1]``.

    Raw scores are intentionally ignored: a rank from one event is comparable
    to a rank from another even when their retrievers have different scales.
    """

    count = len(results)
    if count == 0:
        return []
    normalized_method = str(method).casefold().strip()
    if normalized_method not in {"rank", "percentile"}:
        raise ValueError("score normalization must be 'rank' or 'percentile'")
    if normalized_method == "rank":
        return [1.0 / (1.0 + (rank - 1) / max(1.0, count / 10.0)) for rank in range(1, count + 1)]
    if count == 1:
        return [1.0]
    return [1.0 - (rank - 1) / count for rank in range(1, count + 1)]


def normalize_context_video_scores(
    results: Sequence[RetrievalResult],
) -> dict[str, float]:
    """Collapse context hits to one rank-normalized score per video."""

    valid = [item for item in results if str(item.video_id).strip()]
    scores = normalize_event_scores(valid, method="percentile")
    by_video: dict[str, float] = {}
    for item, score in zip(valid, scores):
        video_id = str(item.video_id)
        by_video[video_id] = max(by_video.get(video_id, 0.0), float(score))
    return by_video


def _valid_event_results(
    results: Sequence[RetrievalResult],
) -> tuple[list[RetrievalResult], int]:
    valid: list[RetrievalResult] = []
    rejected = 0
    for result in results:
        frame_index = result.frame_index
        if (
            not str(result.video_id).strip()
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            rejected += 1
            continue
        valid.append(result)
    return valid, rejected


def _response_results(response: Any) -> list[RetrievalResult]:
    if response is None:
        return []
    if isinstance(response, Mapping):
        raw = response.get("results", response.get("candidates", ()))
    elif hasattr(response, "results"):
        raw = response.results
    else:
        raw = response
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("retrieval response must contain a result sequence")
    output = list(raw)
    if any(not isinstance(item, RetrievalResult) for item in output):
        raise TypeError("event retrieval requires RetrievalResult items")
    return output


__all__ = [
    "CandidateRerankerLike",
    "DEFAULT_TEMPORAL_NMS_RADIUS_FRAMES",
    "EventRetrievalBatch",
    "EventRetriever",
    "RequiredTrakePipelineError",
    "RetrievalEngineLike",
    "diversify_and_normalize",
    "normalize_context_video_scores",
    "normalize_event_scores",
    "retrieve_event_candidates",
]
