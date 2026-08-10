"""Coarse-to-dense retrieval with RRF, dense rescue, CSES and reranking."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Callable, Sequence

import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.advanced_rerank import (
    AdvancedRankedFrame,
    rerank_dense_candidates,
)
from backend.app.services.retrieval.cses import CSESConfig, CSESSelection, select_cses
from backend.app.services.retrieval.dense_recovery import (
    DenseRecoveryConfig,
    DenseRecoveryResult,
    recover_dense_frames,
)
from backend.app.services.retrieval.query_plan import QueryPlan, build_query_plan
from backend.app.services.retrieval.query_modality_weights import (
    DEFAULT_WEIGHTED_RRF_WEIGHTS,
    FUSION_MODES,
    resolve_modality_weights,
)
from backend.app.services.retrieval.rank_fusion import (
    FusedCandidate,
    aggregate_clips,
    fuse_segment_ranks,
)
from backend.app.services.retrieval.segment_aggregation import aggregate_segments
from competition.dense_index import DenseCandidateIndex


@dataclass(frozen=True)
class AdvancedSearchConfig:
    coarse_top_n: int = 100
    visual_top_k: int = 300
    caption_top_k: int = 300
    ocr_top_k: int = 200
    objects_top_k: int = 200
    asr_top_k: int = 100
    enabled_modalities: tuple[str, ...] = ("visual", "caption", "ocr", "objects", "asr")
    dense_global_top_k: int = 300
    dense_rescue_clips: int = 10
    max_total_clips: int = 120
    dense_frames_per_clip: int = 12
    rerank_top_n: int = 300
    final_top_k: int = 100
    dense_expansion_before_sec: float = 1.0
    dense_expansion_after_sec: float = 1.0
    rrf_k: int = 60
    modality_hint_boost: float = 1.5
    fusion_mode: str = "adaptive_rrf"
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
    latency_ms: float = 0.0
    modality_weights: dict[str, float] | None = None
    retrieval_branches: dict[str, list[dict[str, object]]] | None = None
    fusion_candidates: tuple[dict[str, object], ...] = ()
    dense_recovery: dict[str, object] | None = None

    def trace(self) -> dict[str, object]:
        return {
            "query_plan": self.plan.to_dict(),
            "query": self.plan.original_query,
            "detected_modalities": list(self.plan.modality_hints),
            "modality_weights": dict(self.modality_weights or {}),
            "retrieval_branches": dict(self.retrieval_branches or {}),
            "fusion_candidates": list(self.fusion_candidates),
            "coarse_clip_count": self.coarse_clip_count,
            "dense_rescue_clip_count": self.dense_rescue_clip_count,
            "candidate_clip_count": self.candidate_clip_count,
            "candidate_row_count": self.candidate_row_count,
            "selected_row_count": self.selected_row_count,
            "latency_ms": self.latency_ms,
            "candidates_to_rerank": self.selected_row_count,
            "dense_recovery": dict(self.dense_recovery or {}),
            "results": [result.to_dict() for result in self.results],
        }


def advanced_text_search(
    query: str,
    *,
    hybrid_engine: object,
    text_encoder: object,
    dense_index: DenseCandidateIndex,
    profile: str = "auto",
    config: AdvancedSearchConfig | None = None,
) -> AdvancedSearchResponse:
    started = time.perf_counter()
    config = config or AdvancedSearchConfig()
    if config.fusion_mode not in FUSION_MODES:
        raise ValueError(f"Unsupported fusion mode: {config.fusion_mode}")
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
                for modality in ("visual", "caption", "ocr", "asr", "objects")
            ),
            reasons=("query planning disabled by ablation",),
        )
    visual_engine = getattr(hybrid_engine, "visual_engine")
    text_engines = getattr(hybrid_engine, "text_engines")
    clip_resolver = _build_clip_resolver(dense_index)
    groups: dict[str, Sequence[RetrievalResult]] = {}
    if "visual" in config.enabled_modalities:
        groups["visual"] = visual_engine.search(
            plan.query_for("visual"),
            top_k=config.visual_top_k,
        ).results
    for modality, engine in sorted(text_engines.items()):
        if modality not in config.enabled_modalities:
            continue
        branch_top_k = {
            "caption": config.caption_top_k,
            "ocr": config.ocr_top_k,
            "objects": config.objects_top_k,
            "asr": config.asr_top_k,
        }.get(modality, max(config.coarse_top_n, 100))
        groups[modality] = engine.search_results(
            plan.query_for(modality),
            top_k=branch_top_k,
        )
    segment_groups = aggregate_segments(
        groups,
        resolver=clip_resolver,
    )
    requested_fusion = config.fusion_mode if config.rrf_enabled else "legacy"
    weighting = resolve_modality_weights(
        plan,
        fusion_mode=requested_fusion,
        base_weights=DEFAULT_WEIGHTED_RRF_WEIGHTS,
    )
    if requested_fusion != "legacy":
        fused = fuse_segment_ranks(
            segment_groups,
            k=config.rrf_k,
            weights=weighting.weights,
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
                segment_identity=(clip_resolver(result) or ("", "")),
            )
            for result in legacy
        ]
    clips = aggregate_clips(fused, top_n=config.coarse_top_n)
    coarse_clip_scores: dict[tuple[str, str], float] = {}
    coarse_clip_keys: list[tuple[str, str]] = []
    for clip in clips:
        key = (clip.video_id, clip.clip_id)
        if key not in dense_index.rows_by_clip:
            key = clip_resolver(clip.frames[0].result)
        if key is None:
            continue
        coarse_clip_scores[key] = max(coarse_clip_scores.get(key, 0.0), clip.score)
        if key not in coarse_clip_keys:
            coarse_clip_keys.append(key)

    query_vector = np.asarray(
        text_encoder.encode(plan.retrieval_query), dtype=np.float32
    ).reshape(-1)
    event_vectors = (
        [
            np.asarray(text_encoder.encode(event), dtype=np.float32).reshape(-1)
            for event in plan.temporal_events
        ]
        if len(plan.temporal_events) > 1
        else []
    )
    return _select_and_rerank(
        plan=plan,
        query_vector=query_vector,
        event_vectors=event_vectors,
        dense_index=dense_index,
        coarse_clip_keys=coarse_clip_keys,
        coarse_clip_scores=coarse_clip_scores,
        config=config,
        started=started,
        modality_weights=weighting.weights,
        retrieval_branches={
            modality: [item.to_dict() for item in values[:20]]
            for modality, values in segment_groups.items()
        },
        fusion_candidates=tuple(item.to_dict() for item in fused[: config.coarse_top_n]),
    )


def advanced_vector_search(
    query_vector: np.ndarray,
    *,
    coarse_results: Sequence[RetrievalResult],
    dense_index: DenseCandidateIndex,
    config: AdvancedSearchConfig | None = None,
) -> AdvancedSearchResponse:
    started = time.perf_counter()
    config = config or AdvancedSearchConfig()
    plan = build_query_plan("visual image instance", profile="kis")
    clip_resolver = _build_clip_resolver(dense_index)
    coarse_clip_keys: list[tuple[str, str]] = []
    coarse_clip_scores: dict[tuple[str, str], float] = {}
    for result in coarse_results:
        key = clip_resolver(result)
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
        started=started,
        modality_weights={"visual": 1.0},
        retrieval_branches={
            "visual": [
                {
                    "candidate_id": f"{item.video_id}:{item.segment_id or item.shot_id}",
                    "video_id": item.video_id,
                    "segment_id": item.segment_id or item.shot_id,
                    "frame_id": item.frame_id,
                    "timestamp": item.timestamp,
                    "rank": rank,
                    "raw_score": item.score,
                    "modality": "visual",
                }
                for rank, item in enumerate(coarse_results[:20], start=1)
            ]
        },
        fusion_candidates=(),
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
    started: float,
    modality_weights: dict[str, float],
    retrieval_branches: dict[str, list[dict[str, object]]],
    fusion_candidates: tuple[dict[str, object], ...],
) -> AdvancedSearchResponse:
    chosen_clips = list(dict.fromkeys(coarse_clip_keys))[: config.coarse_top_n]
    recovery = recover_dense_frames(
        dense_index=dense_index,
        coarse_clip_keys=chosen_clips,
        coarse_clip_scores=coarse_clip_scores,
        config=DenseRecoveryConfig(
            enabled=config.dense_rescue_enabled,
            expansion_before_sec=config.dense_expansion_before_sec,
            expansion_after_sec=config.dense_expansion_after_sec,
            max_candidate_clips=config.max_total_clips,
        ),
    )
    chosen_clips = list(recovery.rows_by_clip)
    coarse_clip_scores = dict(recovery.recovered_clip_scores)

    selections: list[CSESSelection] = []
    candidate_row_count = recovery.candidate_row_count
    cses_config = CSESConfig(
        max_frames=config.dense_frames_per_clip,
        similarity_threshold=config.similarity_threshold,
        temporal_window_seconds=config.temporal_window_seconds,
    )
    for clip in chosen_clips:
        rows = recovery.rows_by_clip.get(clip, ())
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
    if len(selections) > config.rerank_top_n:
        selections = sorted(
            selections,
            key=lambda selection: (
                -float(selection.relevance),
                -float(
                    coarse_clip_scores.get(
                        (
                            str(dense_index.records[selection.row].get("video_id") or ""),
                            str(
                                dense_index.records[selection.row].get("segment_id")
                                or dense_index.records[selection.row].get("shot_id")
                                or ""
                            ),
                        ),
                        0.0,
                    )
                ),
                float(dense_index.records[selection.row].get("timestamp", 0.0)),
                str(dense_index.records[selection.row].get("candidate_id") or ""),
            ),
        )[: config.rerank_top_n]
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
    ranked = ranked[: max(0, int(config.final_top_k))]
    return AdvancedSearchResponse(
        plan=plan,
        results=tuple(ranked),
        coarse_clip_count=len(coarse_clip_keys),
        dense_rescue_clip_count=max(0, recovery.expanded_clip_count - recovery.source_clip_count),
        candidate_clip_count=len(chosen_clips),
        candidate_row_count=candidate_row_count,
        selected_row_count=len(selections),
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        modality_weights=modality_weights,
        retrieval_branches=retrieval_branches,
        fusion_candidates=fusion_candidates,
        dense_recovery=recovery.to_dict(),
    )


def _build_clip_resolver(
    dense_index: DenseCandidateIndex,
) -> Callable[[RetrievalResult], tuple[str, str] | None]:
    """Pre-index dense rows once so branch aggregation remains query-latency safe."""
    rows_by_video: dict[str, list[int]] = {}
    for (video_id, _), rows in dense_index.rows_by_clip.items():
        rows_by_video.setdefault(video_id, []).extend(rows)
    for video_id in rows_by_video:
        rows_by_video[video_id] = list(dict.fromkeys(rows_by_video[video_id]))

    def resolve(result: RetrievalResult) -> tuple[str, str] | None:
        for clip_id in (result.segment_id, result.shot_id):
            key = (result.video_id, clip_id)
            if clip_id and key in dense_index.rows_by_clip:
                return key
        video_rows = rows_by_video.get(result.video_id, ())
        if not video_rows:
            return None
        nearest = min(
            video_rows,
            key=lambda row: (
                abs(
                    float(dense_index.records[row].get("timestamp", 0.0))
                    - result.timestamp
                ),
                str(dense_index.records[row].get("candidate_id") or ""),
            ),
        )
        record = dense_index.records[nearest]
        return (
            result.video_id,
            str(record.get("segment_id") or record.get("shot_id") or ""),
        )

    return resolve
