"""Coarse-to-dense retrieval with RRF, dense rescue, CSES and reranking."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.advanced_rerank import (
    AdvancedRankedFrame,
    rerank_dense_candidates,
)
from backend.app.services.retrieval.cses import CSESConfig, CSESSelection, select_cses
from backend.app.services.retrieval.query_plan import QueryPlan, build_query_plan
from backend.app.services.retrieval.rank_fusion import aggregate_clips, weighted_rrf
from backend.app.services.retrieval.rank_fusion import FusedCandidate
from competition.dense_index import DenseCandidateIndex


@dataclass(frozen=True)
class AdvancedSearchConfig:
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
    query_plan_enabled: bool = True
    rrf_enabled: bool = True
    dense_rescue_enabled: bool = True
    cses_enabled: bool = True
    deterministic_rerank_enabled: bool = True


@dataclass(frozen=True)
class AdvancedSearchResponse:
    plan: QueryPlan
    results: tuple[AdvancedRankedFrame, ...]
    coarse_clip_count: int
    dense_rescue_clip_count: int
    candidate_clip_count: int
    candidate_row_count: int
    selected_row_count: int

    def trace(self) -> dict[str, object]:
        return {
            "query_plan": self.plan.to_dict(),
            "coarse_clip_count": self.coarse_clip_count,
            "dense_rescue_clip_count": self.dense_rescue_clip_count,
            "candidate_clip_count": self.candidate_clip_count,
            "candidate_row_count": self.candidate_row_count,
            "selected_row_count": self.selected_row_count,
            "results": [result.to_dict() for result in self.results],
        }


def advanced_text_search(
    query: str,
    *,
    hybrid_engine: object,
    text_encoder: object,
    dense_index: DenseCandidateIndex,
    dense_text_engine: object | None = None,
    profile: str = "auto",
    config: AdvancedSearchConfig | None = None,
) -> AdvancedSearchResponse:
    config = config or AdvancedSearchConfig()
    plan = build_query_plan(query, profile=profile)
    if not config.query_plan_enabled:
        plan = replace(
            plan,
            normalized_query=plan.original_query,
            retrieval_query=plan.original_query,
            modality_hints=(),
            expansions=(),
            modality_queries=tuple(
                (modality, plan.original_query)
                for modality in ("visual", "caption", "ocr", "objects")
            ),
            reasons=("query planning disabled by ablation",),
        )
    visual_engine = getattr(hybrid_engine, "visual_engine")
    text_engines = getattr(hybrid_engine, "text_engines")
    groups: dict[str, Sequence[RetrievalResult]] = {
        "visual": visual_engine.search(
            plan.query_for("visual"),
            top_k=max(config.coarse_top_n * 4, 200),
        ).results
    }
    for modality, engine in sorted(text_engines.items()):
        groups[modality] = engine.search_results(
            plan.query_for(modality),
            top_k=max(config.coarse_top_n * 2, 100),
        )
    if dense_text_engine is not None:
        search = getattr(dense_text_engine, "search")
        groups["dense_text"] = search(
            plan.retrieval_query,
            top_k=max(config.coarse_top_n * 2, 100),
        )
    if config.rrf_enabled:
        fused = weighted_rrf(
            groups,
            plan=plan,
            k=config.rrf_k,
            hint_boost=config.modality_hint_boost,
        )
    else:
        legacy = hybrid_engine.search(
            plan.retrieval_query,
            top_k=max(config.coarse_top_n * 4, 200),
        ).results
        fused = [
            FusedCandidate(
                result=result,
                rrf_score=float(result.score),
                modality_ranks={},
                modality_contributions={},
            )
            for result in legacy
        ]
    clips = aggregate_clips(fused, top_n=config.coarse_top_n)
    coarse_clip_scores: dict[tuple[str, str], float] = {}
    coarse_clip_keys: list[tuple[str, str]] = []
    for clip in clips:
        key = _resolve_coarse_clip_key(dense_index, clip.frames[0].result)
        if key is None:
            continue
        coarse_clip_scores[key] = max(coarse_clip_scores.get(key, 0.0), clip.score)
        if key not in coarse_clip_keys:
            coarse_clip_keys.append(key)

    query_vector = np.asarray(text_encoder.encode(plan.retrieval_query), dtype=np.float32).reshape(-1)
    event_vectors = [
        np.asarray(text_encoder.encode(event), dtype=np.float32).reshape(-1)
        for event in plan.temporal_events
    ] if len(plan.temporal_events) > 1 else []
    return _select_and_rerank(
        plan=plan,
        query_vector=query_vector,
        event_vectors=event_vectors,
        dense_index=dense_index,
        coarse_clip_keys=coarse_clip_keys,
        coarse_clip_scores=coarse_clip_scores,
        config=config,
    )


def advanced_vector_search(
    query_vector: np.ndarray,
    *,
    coarse_results: Sequence[RetrievalResult],
    dense_index: DenseCandidateIndex,
    config: AdvancedSearchConfig | None = None,
) -> AdvancedSearchResponse:
    config = config or AdvancedSearchConfig()
    plan = build_query_plan("visual image instance", profile="kis")
    coarse_clip_keys: list[tuple[str, str]] = []
    coarse_clip_scores: dict[tuple[str, str], float] = {}
    for result in coarse_results:
        key = _resolve_coarse_clip_key(dense_index, result)
        if key is None:
            continue
        score = float(result.score)
        coarse_clip_scores[key] = max(coarse_clip_scores.get(key, -1.0), score)
        if key not in coarse_clip_keys:
            coarse_clip_keys.append(key)
        if len(coarse_clip_keys) >= config.coarse_top_n:
            break
    return _select_and_rerank(
        plan=plan,
        query_vector=np.asarray(query_vector, dtype=np.float32).reshape(-1),
        event_vectors=[],
        dense_index=dense_index,
        coarse_clip_keys=coarse_clip_keys,
        coarse_clip_scores=coarse_clip_scores,
        config=config,
    )


def _select_and_rerank(
    *,
    plan: QueryPlan,
    query_vector: np.ndarray,
    event_vectors: Sequence[np.ndarray],
    dense_index: DenseCandidateIndex,
    coarse_clip_keys: Sequence[tuple[str, str]],
    coarse_clip_scores: dict[tuple[str, str], float],
    config: AdvancedSearchConfig,
) -> AdvancedSearchResponse:
    dense_hits = dense_index.search(query_vector, config.dense_global_top_k)
    chosen_clips = list(dict.fromkeys(coarse_clip_keys))[: config.coarse_top_n]
    chosen_set = set(chosen_clips)
    rescued = 0
    for row, score in dense_hits:
        record = dense_index.records[row]
        key = (
            str(record.get("video_id") or ""),
            str(record.get("segment_id") or record.get("shot_id") or ""),
        )
        if key in chosen_set:
            coarse_clip_scores[key] = max(coarse_clip_scores.get(key, -1.0), score)
            continue
        if (
            not config.dense_rescue_enabled
            or rescued >= config.dense_rescue_clips
            or len(chosen_clips) >= config.max_total_clips
        ):
            continue
        chosen_clips.append(key)
        chosen_set.add(key)
        coarse_clip_scores[key] = score
        rescued += 1

    selections: list[CSESSelection] = []
    candidate_row_count = 0
    cses_config = CSESConfig(
        max_frames=config.dense_frames_per_clip,
        similarity_threshold=config.similarity_threshold,
        temporal_window_seconds=config.temporal_window_seconds,
    )
    for clip in chosen_clips:
        rows = dense_index.rows_by_clip.get(clip, [])
        candidate_row_count += len(rows)
        if config.cses_enabled:
            selections.extend(
                select_cses(
                    rows=rows,
                    records=dense_index.records,
                    vectors=dense_index.vectors,
                    query_vector=query_vector,
                    profile=plan.profile,
                    config=cses_config,
                )
            )
        else:
            query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            query /= np.linalg.norm(query)
            scored_rows = sorted(
                rows,
                key=lambda row: (
                    -float(np.dot(np.asarray(dense_index.vectors[row]), query)),
                    float(dense_index.records[row].get("timestamp", 0.0)),
                    str(dense_index.records[row].get("candidate_id") or ""),
                ),
            )[: config.dense_frames_per_clip]
            selections.extend(
                CSESSelection(
                    row=row,
                    selection_rank=index + 1,
                    selection_gain=round(
                        (float(np.dot(np.asarray(dense_index.vectors[row]), query)) + 1.0)
                        / 2.0,
                        8,
                    ),
                    relevance=round(
                        (float(np.dot(np.asarray(dense_index.vectors[row]), query)) + 1.0)
                        / 2.0,
                        8,
                    ),
                    visual_coverage_gain=0.0,
                    temporal_coverage_gain=0.0,
                    preserved_event_ids=(),
                )
                for index, row in enumerate(scored_rows)
            )
    if config.deterministic_rerank_enabled:
        ranked = rerank_dense_candidates(
            plan=plan,
            selections=selections,
            records=dense_index.records,
            vectors=dense_index.vectors,
            query_vector=query_vector,
            coarse_scores=coarse_clip_scores,
            event_vectors=event_vectors,
            max_event_gap_seconds=config.max_event_gap_seconds,
        )
    else:
        ranked = [
            AdvancedRankedFrame(
                dense_row=selection.row,
                record=dense_index.records[selection.row],
                score=selection.relevance,
                breakdown={"dense_visual": selection.relevance},
                selection=selection,
            )
            for selection in selections
        ]
        ranked.sort(
            key=lambda item: (
                -item.score,
                float(item.record.get("timestamp", 0.0)),
                str(item.record.get("candidate_id") or ""),
            )
        )
    return AdvancedSearchResponse(
        plan=plan,
        results=tuple(ranked),
        coarse_clip_count=len(coarse_clip_keys),
        dense_rescue_clip_count=rescued,
        candidate_clip_count=len(chosen_clips),
        candidate_row_count=candidate_row_count,
        selected_row_count=len(selections),
    )


def _resolve_coarse_clip_key(
    dense_index: DenseCandidateIndex,
    result: RetrievalResult,
) -> tuple[str, str] | None:
    for clip_id in (result.segment_id, result.shot_id):
        key = (result.video_id, clip_id)
        if clip_id and key in dense_index.rows_by_clip:
            return key
    # Text segments may not share the dense shot id.  Resolve by timestamp while
    # retaining the documented segment-first, shot-fallback policy.
    video_rows = [
        row
        for (video_id, _), rows in dense_index.rows_by_clip.items()
        if video_id == result.video_id
        for row in rows
    ]
    if not video_rows:
        return None
    nearest = min(
        video_rows,
        key=lambda row: (
            abs(float(dense_index.records[row].get("timestamp", 0.0)) - result.timestamp),
            str(dense_index.records[row].get("candidate_id") or ""),
        ),
    )
    record = dense_index.records[nearest]
    return (
        result.video_id,
        str(record.get("segment_id") or record.get("shot_id") or ""),
    )
