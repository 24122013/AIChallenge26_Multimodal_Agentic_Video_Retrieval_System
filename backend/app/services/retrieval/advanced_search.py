"""Coarse-to-dense retrieval with RRF, dense rescue, CSES and reranking."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.agent.query_expansion import (
    QueryExpansionConfig,
    QueryExpansionProvider,
    build_query_expansion_plan,
)
from backend.app.services.retrieval.advanced_rerank import (
    AdvancedRankedFrame,
    AdvancedRerankWeights,
    ContextRerankConfig,
    rerank_dense_candidates,
)
from backend.app.services.retrieval.cses import CSESConfig, CSESSelection, select_cses
from backend.app.services.retrieval.online_context import OnlineContextIndex
from backend.app.services.retrieval.query_plan import QueryPlan, build_query_plan
from backend.app.services.retrieval.rank_fusion import (
    aggregate_clips,
    fuse_query_variants,
    weighted_rrf,
)
from backend.app.services.retrieval.rank_fusion import FusedCandidate


class DenseCandidateIndex(Protocol):
    """Backend-owned structural contract for the optional dense rescue index."""

    records: Sequence[Mapping[str, Any]]
    vectors: np.ndarray
    rows_by_clip: Mapping[tuple[str, str], Sequence[int]]
    row_by_frame: Mapping[tuple[str, str], int]

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]: ...


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
    rerank_weights: AdvancedRerankWeights = AdvancedRerankWeights()
    context_config: ContextRerankConfig = ContextRerankConfig()
    query_expansion: QueryExpansionConfig = QueryExpansionConfig()

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
                raise ValueError(f"Advanced search {name} must be a positive integer")
        if (
            isinstance(self.dense_rescue_clips, bool)
            or not isinstance(self.dense_rescue_clips, int)
            or self.dense_rescue_clips < 0
        ):
            raise ValueError(
                "Advanced search dense_rescue_clips must be a non-negative integer"
            )
        if self.max_total_clips < self.coarse_top_n:
            raise ValueError("max_total_clips must be at least coarse_top_n")
        for name in (
            "modality_hint_boost",
            "similarity_threshold",
            "temporal_window_seconds",
            "max_event_gap_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Advanced search {name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"Advanced search {name} must be a finite number")
        if self.modality_hint_boost < 0:
            raise ValueError("modality_hint_boost must be non-negative")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be within [0, 1]")
        if self.temporal_window_seconds < 0 or self.max_event_gap_seconds < 0:
            raise ValueError("Advanced search temporal windows must be non-negative")
        for name in (
            "query_plan_enabled",
            "rrf_enabled",
            "dense_rescue_enabled",
            "cses_enabled",
            "deterministic_rerank_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Advanced search {name} must be a boolean")
        if not isinstance(self.rerank_weights, AdvancedRerankWeights):
            raise TypeError("rerank_weights must be an AdvancedRerankWeights instance")
        if not isinstance(self.context_config, ContextRerankConfig):
            raise TypeError("context_config must be a ContextRerankConfig instance")
        if not isinstance(self.query_expansion, QueryExpansionConfig):
            raise TypeError("query_expansion must be a QueryExpansionConfig instance")


@dataclass(frozen=True)
class AdvancedSearchResponse:
    plan: QueryPlan
    results: tuple[AdvancedRankedFrame, ...]
    coarse_clip_count: int
    dense_rescue_clip_count: int
    candidate_clip_count: int
    candidate_row_count: int
    selected_row_count: int
    intra_modality_trace: Mapping[str, Sequence[Mapping[str, object]]]
    skipped_modalities: Mapping[str, str]
    inter_modality_trace: Sequence[Mapping[str, object]]
    stage_latency_ms: Mapping[str, float] = field(default_factory=dict)
    context_trace: Mapping[str, object] = field(default_factory=dict)
    exact_duplicate_count: int = 0

    def trace(self) -> dict[str, object]:
        return {
            "query_plan": self.plan.to_dict(),
            "coarse_clip_count": self.coarse_clip_count,
            "dense_rescue_clip_count": self.dense_rescue_clip_count,
            "candidate_clip_count": self.candidate_clip_count,
            "candidate_row_count": self.candidate_row_count,
            "selected_row_count": self.selected_row_count,
            "exact_duplicate_count": self.exact_duplicate_count,
            "latency": dict(self.stage_latency_ms),
            "intra_modality_fusion": {
                name: [dict(value) for value in values]
                for name, values in self.intra_modality_trace.items()
            },
            "skipped_modalities": dict(self.skipped_modalities),
            "inter_modality_fusion": [dict(value) for value in self.inter_modality_trace],
            "rerank_canonical_query": self.plan.original_query,
            "context_scoring": dict(self.context_trace),
        }

    def to_dict(self, top_k: int | None = None) -> dict[str, object]:
        """Return the existing ``query/top_k/results/trace`` service contract."""

        if top_k is None:
            effective_top_k = len(self.results)
        else:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("advanced search top_k must be a positive integer")
            effective_top_k = top_k
        return {
            "query": self.plan.original_query,
            "top_k": effective_top_k,
            "results": [
                result.to_result_mapping()
                for result in self.results[:effective_top_k]
            ],
            "trace": self.trace(),
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
    expansion_provider: QueryExpansionProvider | None = None,
    plan: QueryPlan | None = None,
    context_index: OnlineContextIndex | None = None,
) -> AdvancedSearchResponse:
    overall_started = time.perf_counter()
    stage_latency_ms = _empty_stage_latency()
    config = config or AdvancedSearchConfig()
    stage_started = time.perf_counter()
    plan = plan or build_query_plan(
        query,
        profile=profile,
        expansion_provider=expansion_provider,
        expansion_config=config.query_expansion,
    )
    if not config.query_plan_enabled:
        disabled_expansion = build_query_expansion_plan(
            plan.original_query,
            provider=None,
            config=replace(config.query_expansion, enabled=False),
        )
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
            expansion_plan=disabled_expansion,
        )
    stage_latency_ms["query_planning_ms"] = _elapsed_ms(stage_started)
    visual_engine = getattr(hybrid_engine, "visual_engine")
    text_engines = getattr(hybrid_engine, "text_engines")
    semantic_variants = plan.expansion_plan.accepted_variants
    stage_started = time.perf_counter()
    query_vector = np.asarray(
        text_encoder.encode(plan.original_query),
        dtype=np.float32,
    ).reshape(-1)
    event_vectors = [
        np.asarray(text_encoder.encode(event), dtype=np.float32).reshape(-1)
        for event in plan.temporal_events
    ] if len(plan.temporal_events) > 1 else []
    stage_latency_ms["query_encoding_ms"] = _elapsed_ms(stage_started)
    variant_groups: dict[str, Sequence[RetrievalResult]] = {}
    variant_weights: dict[str, float] = {}
    stage_started = time.perf_counter()
    for index, variant in enumerate(semantic_variants):
        key = "original" if variant.type == "original" else f"paraphrase_{index}"
        search_by_vector = getattr(visual_engine, "search_by_vector", None)
        if variant.type == "original" and callable(search_by_vector):
            response = search_by_vector(
                variant.text,
                query_vector,
                top_k=max(config.coarse_top_n * 4, 200),
            )
        else:
            response = visual_engine.search(
                variant.text,
                top_k=max(config.coarse_top_n * 4, 200),
            )
        variant_groups[key] = response.results
        variant_weights[key] = variant.weight
    stage_latency_ms["selected_visual_ms"] = _elapsed_ms(stage_started)
    stage_started = time.perf_counter()
    visual_intra = fuse_query_variants(
        variant_groups,
        weights=variant_weights,
        k=config.rrf_k,
        max_expansion_contribution=config.query_expansion.max_expansion_contribution,
    )
    stage_latency_ms["fusion_ms"] += _elapsed_ms(stage_started)
    groups: dict[str, Sequence[RetrievalResult]] = {
        "visual": [value.as_retrieval_result() for value in visual_intra]
    }
    intra_trace: dict[str, list[Mapping[str, object]]] = {
        "visual": [value.to_dict() for value in visual_intra]
    }
    skipped_modalities: dict[str, str] = {}
    for modality, engine in sorted(text_engines.items()):
        if modality == "caption":
            caption_groups: dict[str, Sequence[RetrievalResult]] = {}
            for index, key in enumerate(variant_groups):
                stage_started = time.perf_counter()
                caption_groups[key] = engine.search_results(
                    semantic_variants[index].text,
                    top_k=max(config.coarse_top_n * 2, 100),
                )
                stage_latency_ms["text_retrieval_ms"] += _elapsed_ms(stage_started)
            stage_started = time.perf_counter()
            caption_intra = fuse_query_variants(
                caption_groups,
                weights=variant_weights,
                k=config.rrf_k,
                max_expansion_contribution=config.query_expansion.max_expansion_contribution,
            )
            stage_latency_ms["fusion_ms"] += _elapsed_ms(stage_started)
            groups[modality] = [value.as_retrieval_result() for value in caption_intra]
            intra_trace[modality] = [value.to_dict() for value in caption_intra]
            continue
        modality_query = plan.query_for(modality).strip()
        if modality in {"ocr", "objects"} and not modality_query:
            skipped_modalities[modality] = "no_reliable_modality_terms"
            continue
        stage_started = time.perf_counter()
        groups[modality] = engine.search_results(
            modality_query,
            top_k=max(config.coarse_top_n * 2, 100),
        )
        stage_latency_ms["text_retrieval_ms"] += _elapsed_ms(stage_started)
    if dense_text_engine is not None:
        search = getattr(dense_text_engine, "search")
        stage_started = time.perf_counter()
        groups["dense_text"] = search(
            plan.retrieval_query,
            top_k=max(config.coarse_top_n * 2, 100),
        )
        stage_latency_ms["text_retrieval_ms"] += _elapsed_ms(stage_started)
    stage_started = time.perf_counter()
    if config.rrf_enabled:
        fused = weighted_rrf(
            groups,
            plan=plan,
            k=config.rrf_k,
            hint_boost=config.modality_hint_boost,
        )
        inter_trace = [value.to_dict() for value in fused]
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
        inter_trace = []
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
    stage_latency_ms["fusion_ms"] += _elapsed_ms(stage_started)

    return _select_and_rerank(
        plan=plan,
        query_vector=query_vector,
        event_vectors=event_vectors,
        dense_index=dense_index,
        coarse_clip_keys=coarse_clip_keys,
        coarse_clip_scores=coarse_clip_scores,
        config=config,
        intra_modality_trace=intra_trace,
        skipped_modalities=skipped_modalities,
        inter_modality_trace=inter_trace,
        overall_started=overall_started,
        stage_latency_ms=stage_latency_ms,
        context_index=context_index,
    )


def advanced_vector_search(
    query_vector: np.ndarray,
    *,
    coarse_results: Sequence[RetrievalResult],
    dense_index: DenseCandidateIndex,
    config: AdvancedSearchConfig | None = None,
    context_index: OnlineContextIndex | None = None,
) -> AdvancedSearchResponse:
    overall_started = time.perf_counter()
    stage_latency_ms = _empty_stage_latency()
    config = config or AdvancedSearchConfig()
    stage_started = time.perf_counter()
    plan = build_query_plan(
        "visual image instance",
        profile="kis",
        expansion_config=QueryExpansionConfig(enabled=False),
    )
    stage_latency_ms["query_planning_ms"] = _elapsed_ms(stage_started)
    stage_started = time.perf_counter()
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
    stage_latency_ms["fusion_ms"] = _elapsed_ms(stage_started)
    return _select_and_rerank(
        plan=plan,
        query_vector=np.asarray(query_vector, dtype=np.float32).reshape(-1),
        event_vectors=[],
        dense_index=dense_index,
        coarse_clip_keys=coarse_clip_keys,
        coarse_clip_scores=coarse_clip_scores,
        config=config,
        intra_modality_trace={},
        skipped_modalities={"query_expansion": "VKIS image query"},
        inter_modality_trace=[],
        overall_started=overall_started,
        stage_latency_ms=stage_latency_ms,
        context_index=context_index,
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
    intra_modality_trace: Mapping[str, Sequence[Mapping[str, object]]],
    skipped_modalities: Mapping[str, str],
    inter_modality_trace: Sequence[Mapping[str, object]],
    overall_started: float,
    stage_latency_ms: dict[str, float],
    context_index: OnlineContextIndex | None,
) -> AdvancedSearchResponse:
    _validate_dense_index_contract(dense_index, query_vector)
    stage_started = time.perf_counter()
    if config.dense_rescue_enabled and config.dense_rescue_clips > 0:
        dense_hits = _validated_dense_hits(
            dense_index,
            dense_index.search(query_vector, config.dense_global_top_k),
        )
    else:
        dense_hits = []
    stage_latency_ms["dense_global_ms"] = _elapsed_ms(stage_started)

    stage_started = time.perf_counter()
    chosen_clips = list(dict.fromkeys(coarse_clip_keys))[: config.coarse_top_n]
    chosen_set = set(chosen_clips)
    rescued = 0
    for row, score in dense_hits:
        record = dense_index.records[row]
        key = _record_clip_key(record)
        if not key[0] or not key[1] or key not in dense_index.rows_by_clip:
            continue
        if key in chosen_set:
            continue
        if (
            not config.dense_rescue_enabled
            or rescued >= config.dense_rescue_clips
            or len(chosen_clips) >= config.max_total_clips
        ):
            continue
        chosen_clips.append(key)
        chosen_set.add(key)
        # Dense similarity is scored independently downstream.  A rescued clip
        # has no coarse RRF evidence and must not masquerade as one here.
        coarse_clip_scores.setdefault(key, 0.0)
        rescued += 1
    stage_latency_ms["dense_rescue_ms"] = _elapsed_ms(stage_started)

    stage_started = time.perf_counter()
    selections: list[CSESSelection] = []
    candidate_row_count = 0
    cses_config = CSESConfig(
        max_frames=config.dense_frames_per_clip,
        similarity_threshold=config.similarity_threshold,
        temporal_window_seconds=config.temporal_window_seconds,
    )
    for clip in chosen_clips:
        rows = _validated_clip_rows(dense_index, clip)
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
    raw_selection_count = len(selections)
    selections = _dedupe_selections(selections)
    exact_duplicate_count = raw_selection_count - len(selections)
    stage_latency_ms["cses_ms"] = _elapsed_ms(stage_started)

    stage_started = time.perf_counter()
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
            weights=config.rerank_weights,
            context_index=context_index,
            row_by_frame=dense_index.row_by_frame,
            context_config=config.context_config,
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
    before_result_dedupe = len(ranked)
    ranked = _dedupe_ranked_frames(ranked)
    exact_duplicate_count += before_result_dedupe - len(ranked)
    stage_latency_ms["deterministic_rerank_ms"] = _elapsed_ms(stage_started)
    stage_latency_ms["total_ms"] = _elapsed_ms(overall_started)
    context_trace = _context_trace_summary(
        ranked,
        requested=config.context_config,
        context_index=context_index,
        rerank_enabled=config.deterministic_rerank_enabled,
    )
    return AdvancedSearchResponse(
        plan=plan,
        results=tuple(ranked),
        coarse_clip_count=len(coarse_clip_keys),
        dense_rescue_clip_count=rescued,
        candidate_clip_count=len(chosen_clips),
        candidate_row_count=candidate_row_count,
        selected_row_count=len(ranked),
        intra_modality_trace=intra_modality_trace,
        skipped_modalities=skipped_modalities,
        inter_modality_trace=inter_modality_trace,
        stage_latency_ms=stage_latency_ms,
        context_trace=context_trace,
        exact_duplicate_count=exact_duplicate_count,
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


def _empty_stage_latency() -> dict[str, float]:
    return {
        "query_planning_ms": 0.0,
        "query_encoding_ms": 0.0,
        "selected_visual_ms": 0.0,
        "text_retrieval_ms": 0.0,
        "fusion_ms": 0.0,
        "dense_global_ms": 0.0,
        "dense_rescue_ms": 0.0,
        "cses_ms": 0.0,
        "deterministic_rerank_ms": 0.0,
        "total_ms": 0.0,
    }


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _validate_dense_index_contract(
    dense_index: DenseCandidateIndex,
    query_vector: np.ndarray,
) -> None:
    vectors = np.asarray(dense_index.vectors)
    records = dense_index.records
    if vectors.ndim != 2:
        raise ValueError("Dense candidate vectors must be a two-dimensional matrix")
    if vectors.shape[0] != len(records):
        raise ValueError("Dense candidate records and vectors must have equal row counts")
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if vectors.shape[1] != query.shape[0]:
        raise ValueError("Dense candidate and query vector dimensions do not match")
    norm = float(np.linalg.norm(query))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Dense candidate query vector must have a positive finite norm")
    if not isinstance(dense_index.rows_by_clip, Mapping):
        raise TypeError("Dense candidate rows_by_clip must be a mapping")
    if not isinstance(dense_index.row_by_frame, Mapping):
        raise TypeError("Dense candidate row_by_frame must be a mapping")


def _validated_dense_hits(
    dense_index: DenseCandidateIndex,
    hits: Sequence[tuple[int, float]],
) -> list[tuple[int, float]]:
    output: list[tuple[int, float]] = []
    for position, hit in enumerate(hits):
        if not isinstance(hit, Sequence) or isinstance(hit, (str, bytes)) or len(hit) != 2:
            raise TypeError(f"Dense search hit {position} must be a (row, score) pair")
        raw_row, raw_score = hit
        if isinstance(raw_row, bool) or not isinstance(raw_row, Integral):
            raise TypeError(f"Dense search row {position} must be an integer")
        row = int(raw_row)
        # FAISS uses -1 when top_k is larger than ntotal.  It is a sentinel, not
        # Python's valid negative index for the last record.
        if row == -1:
            continue
        if row < 0 or row >= len(dense_index.records):
            raise IndexError(f"Dense search row is outside the candidate index: {row}")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"Dense search score at row {row} must be finite")
        output.append((row, score))
    output.sort(key=lambda item: (-item[1], item[0]))
    return output


def _record_clip_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("video_id") or ""),
        str(record.get("segment_id") or record.get("shot_id") or ""),
    )


def _context_trace_summary(
    frames: Sequence[AdvancedRankedFrame],
    *,
    requested: ContextRerankConfig,
    context_index: OnlineContextIndex | None,
    rerank_enabled: bool,
) -> dict[str, object]:
    summary = context_index.summary() if context_index is not None else {}

    def feature(name: str, *, record_count_key: str) -> dict[str, object]:
        was_requested = bool(getattr(requested, f"{name}_enabled"))
        available = bool(
            context_index is not None and int(summary.get(record_count_key) or 0) > 0
        )
        executed = bool(was_requested and available and rerank_enabled)
        if not was_requested:
            fallback_reason = "disabled_by_config"
        elif not rerank_enabled:
            fallback_reason = "deterministic_rerank_disabled"
        elif context_index is None:
            fallback_reason = "context_index_unavailable"
        elif not available:
            fallback_reason = f"{name}_artifact_unavailable"
        else:
            fallback_reason = ""
        evidence_key = f"{name}_evidence_count"
        return {
            "requested": was_requested,
            "artifact_available": available,
            "executed": executed,
            "fallback_reason": fallback_reason,
            "results_with_evidence": sum(
                1
                for frame in frames
                if int(frame.context_trace.get(evidence_key) or 0) > 0
            ),
        }

    return {
        "neighbor": feature("neighbor", record_count_key="neighbor_record_count"),
        "segment": feature("segment", record_count_key="segment_record_count"),
        "max_neighbors_each_side": requested.max_neighbors_each_side,
        "segment_candidate_limit": requested.segment_candidate_limit,
        "segment_top_k": requested.segment_top_k,
        "context_bonus_cap": requested.max_bonus,
    }


def _validated_clip_rows(
    dense_index: DenseCandidateIndex,
    clip: tuple[str, str],
) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for raw_row in dense_index.rows_by_clip.get(clip, ()):
        if isinstance(raw_row, bool) or not isinstance(raw_row, Integral):
            raise TypeError(f"Dense clip row for {clip!r} must be an integer")
        row = int(raw_row)
        if row < 0 or row >= len(dense_index.records):
            raise IndexError(f"Dense clip row for {clip!r} is outside the index: {row}")
        if row in seen:
            continue
        seen.add(row)
        output.append(row)
    return output


def _selection_priority(selection: CSESSelection) -> tuple[object, ...]:
    return (
        float(selection.selection_gain),
        float(selection.relevance),
        float(selection.visual_coverage_gain),
        float(selection.temporal_coverage_gain),
        len(selection.preserved_event_ids),
        tuple(selection.preserved_event_ids),
        -int(selection.selection_rank),
    )


def _dedupe_selections(
    selections: Sequence[CSESSelection],
) -> list[CSESSelection]:
    """Keep one strongest deterministic selection for each exact dense row."""

    output: list[CSESSelection] = []
    position_by_row: dict[int, int] = {}
    for selection in selections:
        position = position_by_row.get(selection.row)
        if position is None:
            position_by_row[selection.row] = len(output)
            output.append(selection)
            continue
        if _selection_priority(selection) > _selection_priority(output[position]):
            output[position] = selection
    return output


def _ranked_identity(frame: AdvancedRankedFrame) -> tuple[str, str, str]:
    record = frame.record
    video_id = str(record.get("video_id") or "")
    frame_id = str(record.get("frame_id") or record.get("keyframe_id") or "")
    if frame_id:
        return ("frame", video_id, frame_id)
    candidate_id = str(record.get("candidate_id") or "")
    if candidate_id:
        return ("candidate", video_id, candidate_id)
    return ("dense_row", "", str(frame.dense_row))


def _dedupe_ranked_frames(
    frames: Sequence[AdvancedRankedFrame],
) -> list[AdvancedRankedFrame]:
    """Remove exact duplicate candidates without same-shot/temporal collapsing."""

    output: list[AdvancedRankedFrame] = []
    seen: set[tuple[str, str, str]] = set()
    for frame in frames:
        identity = _ranked_identity(frame)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(frame)
    return output
