"""Typed QA routing and provenance-preserving evidence bundles.

The module is intentionally deterministic.  It does not translate, expand, or
answer a question.  Expanded queries are accepted only as caller-owned input;
temporal intent is exposed in the query plan but is not executed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.candidate_merger import (
    candidate_identity,
    merge_candidates,
)
from backend.app.services.retrieval.query_plan import QueryPlan, build_query_plan
from backend.app.services.retrieval.rank_fusion import weighted_rrf


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
        model_revision: str = "main",
        retrieval_alpha: float = 0.5,
        device: str = "auto",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_revision = model_revision
        self.retrieval_alpha = float(retrieval_alpha)
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
            model_revision=self.model_revision,
            output_k=20 if top_k is None else int(top_k),
            retrieval_alpha=self.retrieval_alpha,
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


@dataclass(frozen=True)
class QaEvidence:
    evidence_id: str
    video_id: str
    frame_id: str
    shot_id: str
    timestamp: float
    image_path: str
    caption: str
    ocr_text: str
    objects: tuple[str, ...]
    source_modalities: tuple[str, ...]
    retrieval_score: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "shot_id": self.shot_id,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "objects": list(self.objects),
            "source_modalities": list(self.source_modalities),
            "retrieval_score": self.retrieval_score,
            "warnings": list(self.warnings),
        }


@dataclass
class _RoutingTrace:
    queries: list[dict[str, str]] = field(default_factory=list)
    modality_queries: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    reranker: str = "off"

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
            "reranker": self.reranker,
            "temporal_handoff": plan.needs_temporal,
            "feature_flags": {
                "typed_parser": config.typed_parser_enabled,
                "qa_router": config.router_enabled,
                "evidence_bundle": config.evidence_bundle_enabled,
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
                confidence=0.0,
                reasons=(*plan.reasons, "typed parser disabled by feature flag"),
                modality_queries=tuple(
                    (modality, plan.normalized_query)
                    for modality in ("visual", "caption", "objects", "ocr")
                ),
            )
        external_queries = _external_queries(
            expanded_queries or (),
            base_query=plan.retrieval_statement,
        )
        queries = (plan.retrieval_statement, *external_queries)
        trace = _RoutingTrace(
            queries=[
                {
                    "query": query,
                    "source": "parser" if index == 0 else "external_expansion",
                }
                for index, query in enumerate(queries)
            ]
        )

        query_groups: list[list[RetrievalResult]] = []
        conflict_sources: list[RetrievalResult] = []
        for query in queries:
            modality_groups = (
                self._retrieve_modalities(query, plan, trace)
                if self.config.router_enabled
                else {}
            )
            conflict_sources.extend(
                result for group in modality_groups.values() for result in group
            )
            if not modality_groups:
                trace.fallback_used = True
                fallback = self.hybrid_engine.search(
                    query,
                    top_k=self.config.per_modality_limit,
                ).results
                modality_groups = {"hybrid": fallback}
                conflict_sources.extend(fallback)
            query_groups.append(
                _rrf_results(
                    modality_groups,
                    plan=plan,
                    config=self.config,
                )
            )

        fused_pool = _fuse_evidence(
            query_groups,
            top_k=(
                self.config.rerank_pool_size
                if self.candidate_reranker is not None
                else self.config.fusion_pool_size
            ),
            consensus_bonus=self.config.consensus_bonus,
        )
        reranked = self._rerank(plan.retrieval_statement, fused_pool, trace)
        selected = merge_candidates(
            [reranked],
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
                )
                for index, result in enumerate(
                    selected[:evidence_top_k],
                    start=1,
                )
            ]
            if self.config.evidence_bundle_enabled
            else []
        )
        return {
            **_legacy_plan_fields(plan, queries),
            "query_plan": plan.to_dict(),
            "routing_trace": trace.to_dict(plan=plan, config=self.config),
            "top_k": requested_top_k,
            "answer_mode": "manual_visual_inspection",
            "evidence_count": len(evidence),
            "evidence": [item.to_dict() for item in evidence],
            # Kept for /retrieval/qa-evidence and mode=qa compatibility.
            "results": [result.to_dict() for result in selected],
        }

    def _retrieve_modalities(
        self,
        query: str,
        plan: QueryPlan,
        trace: _RoutingTrace,
    ) -> dict[str, list[RetrievalResult]]:
        groups: dict[str, list[RetrievalResult]] = {}
        visual_engine = getattr(self.hybrid_engine, "visual_engine", None)
        if visual_engine is not None:
            visual_query = plan.query_for("visual") if query == plan.retrieval_statement else query
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
            modality_query = plan.query_for(modality) if query == plan.retrieval_statement else query
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
                top_k=self.config.fusion_pool_size,
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
            return list(output)
        except Exception as exc:  # noqa: BLE001 - documented Phase 5 fallback.
            trace.reranker = f"fallback:{type(exc).__name__}"
            return candidates


def plan_qa_question(
    question: str,
    *,
    task_mode: str = "qa",
) -> QueryPlan:
    """Return the shared immutable QueryPlan contract for QA."""
    return build_query_plan(question, task_mode=task_mode)


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
) -> QaEvidence:
    warnings: list[str] = []
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
        shot_id=result.shot_id or result.segment_id,
        timestamp=float(result.timestamp),
        image_path=image_path,
        caption=result.caption,
        ocr_text=result.ocr_text,
        objects=tuple(str(value) for value in result.objects),
        source_modalities=modalities,
        retrieval_score=round(float(result.score), 8),
        warnings=tuple(warnings),
    )
