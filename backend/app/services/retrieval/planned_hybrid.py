"""Query-plan-aware adapter around the canonical hybrid search engines."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.candidate_merger import (
    candidate_identity,
    merge_candidates,
)
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine
from backend.app.services.retrieval.query_plan import QueryPlan
from backend.app.services.retrieval.rank_fusion import (
    fuse_query_variants,
    weighted_rrf,
)


_TEXT_MODALITIES = ("caption", "ocr", "objects")


def planned_hybrid_search(
    engine: HybridSearchEngine,
    plan: QueryPlan,
    *,
    top_k: int | None = None,
    max_expansion_contribution: float = 1.0,
) -> VisualSearchResponse:
    """Execute one parsed plan without reparsing or regenerating variants."""
    started_at = time.perf_counter()
    bounded_top_k = engine._top_k(top_k)
    variants = plan.expansion_plan.accepted_variants
    variant_keys = [
        "original" if variant.type == "original" else f"paraphrase_{index}"
        for index, variant in enumerate(variants)
    ]
    variant_weights = {
        key: float(variant.weight)
        for key, variant in zip(variant_keys, variants)
    }
    retrieval_calls: list[dict[str, Any]] = []
    searched_modalities: list[str] = []
    modality_groups: dict[str, list[RetrievalResult]] = {}
    intra_modality_trace: dict[str, list[dict[str, object]]] = {}
    skipped_modalities: dict[str, str] = {}

    visual_groups: dict[str, list[RetrievalResult]] = {}
    for key, variant in zip(variant_keys, variants):
        response = engine.visual_engine.search(
            variant.text,
            top_k=engine.config.stage1_top_k,
        )
        visual_groups[key] = response.results
        retrieval_calls.append(
            _call_trace("visual", variant.text, key, variant.weight)
        )
    visual_fused = fuse_query_variants(
        visual_groups,
        weights=variant_weights,
        max_expansion_contribution=max_expansion_contribution,
    )
    modality_groups["visual"] = [
        item.as_retrieval_result() for item in visual_fused
    ]
    intra_modality_trace["visual"] = [item.to_dict() for item in visual_fused]
    searched_modalities.append("visual")

    for modality in _TEXT_MODALITIES:
        text_engine = engine.text_engines.get(modality)
        if text_engine is None:
            skipped_modalities[modality] = "engine_unavailable"
            continue
        if modality == "caption":
            caption_groups: dict[str, list[RetrievalResult]] = {}
            for key, variant in zip(variant_keys, variants):
                caption_groups[key] = text_engine.search_results(
                    variant.text,
                    top_k=engine.config.text_stage1_top_k,
                )
                retrieval_calls.append(
                    _call_trace(modality, variant.text, key, variant.weight)
                )
            caption_fused = fuse_query_variants(
                caption_groups,
                weights=variant_weights,
                max_expansion_contribution=max_expansion_contribution,
            )
            modality_groups[modality] = [
                item.as_retrieval_result() for item in caption_fused
            ]
            intra_modality_trace[modality] = [
                item.to_dict() for item in caption_fused
            ]
            searched_modalities.append(modality)
            continue

        modality_query = plan.query_for(modality).strip()
        if not modality_query:
            skipped_modalities[modality] = "no_reliable_modality_terms"
            continue
        modality_groups[modality] = text_engine.search_results(
            modality_query,
            top_k=engine.config.text_stage1_top_k,
        )
        retrieval_calls.append(
            _call_trace(modality, modality_query, "decomposition", 1.0)
        )
        searched_modalities.append(modality)

    # Merge metadata/scores by stable frame identity, but rank the coarse pool
    # with proper cross-modality weighted RRF.  The final HybridReranker does
    # not consume the RRF value, avoiding a double-counted signal.
    merged_metadata = merge_candidates(
        modality_groups.values(),
        top_k=None,
        dedupe_same_shot=False,
    )
    metadata_by_identity = {
        candidate_identity(candidate): candidate for candidate in merged_metadata
    }
    inter_fused = weighted_rrf(modality_groups, plan=plan)
    merged_pool = []
    for fused in inter_fused[: engine.config.rerank_pool_size]:
        candidate = metadata_by_identity.get(
            candidate_identity(fused.result),
            fused.result,
        )
        merged_pool.append(
            replace(
                candidate,
                score=float(fused.rrf_score),
                modality_scores={
                    **candidate.modality_scores,
                    "rrf": float(fused.rrf_score),
                    "fusion": float(fused.rrf_score),
                },
            )
        )
    results = engine.reranker.rerank(
        query=plan.original_query,
        candidates=merged_pool,
        top_k=bounded_top_k,
    )
    trace = {
        "query_plan": plan.to_dict(),
        "variant_weights": variant_weights,
        "retrieval_calls": retrieval_calls,
        "searched_modalities": searched_modalities,
        "skipped_modalities": skipped_modalities,
        "intra_modality_fusion": intra_modality_trace,
        "inter_modality_fusion": [item.to_dict() for item in inter_fused],
        "rerank_canonical_query": plan.original_query,
    }
    return VisualSearchResponse(
        query=plan.original_query,
        top_k=bounded_top_k,
        latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
        results=results,
        trace=trace,
    )


def _call_trace(
    modality: str,
    query: str,
    variant: str,
    weight: float,
) -> dict[str, object]:
    return {
        "modality": modality,
        "query": query,
        "variant": variant,
        "weight": float(weight),
    }


__all__ = ["planned_hybrid_search"]
