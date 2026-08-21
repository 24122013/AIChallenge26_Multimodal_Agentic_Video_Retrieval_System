"""Deterministic multimodal reranking for dense CSES candidates."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from backend.app.services.retrieval.cses import CSESSelection
from backend.app.services.retrieval.online_context import OnlineContextIndex
from backend.app.services.retrieval.query_plan import QueryPlan
from backend.app.services.retrieval.query_terms import content_tokens, weighted_term_coverage


@dataclass(frozen=True)
class AdvancedRerankWeights:
    coarse_rrf: float = 0.14
    dense_visual: float = 0.39
    caption: float = 0.11
    ocr: float = 0.10
    objects: float = 0.07
    cses_gain: float = 0.04
    temporal_consistency: float = 0.03
    modality_alignment: float = 0.02
    neighbor_support: float = 0.05
    segment_support: float = 0.05

    def __post_init__(self) -> None:
        values: dict[str, float] = {}
        for name, raw_value in self.__dict__.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError("Advanced rerank weights must be finite numbers")
            values[name] = float(raw_value)
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("Advanced rerank weights must be finite")
        if any(value < 0 for value in values.values()):
            raise ValueError("Advanced rerank weights must be non-negative")
        if sum(values.values()) <= 0:
            raise ValueError("At least one advanced rerank weight must be positive")


@dataclass(frozen=True)
class ContextRerankConfig:
    """Bounded metadata-only context controls for canonical KIS/AVS."""

    neighbor_enabled: bool = False
    segment_enabled: bool = False
    max_neighbors_each_side: int = 2
    segment_candidate_limit: int = 12
    segment_top_k: int = 3
    max_bonus: float = 0.08
    neighbor_max_ratio: float = 0.65
    segment_max_ratio: float = 0.60
    temporal_window_seconds: float = 5.0
    temporal_proximity_weight: float = 0.10

    def __post_init__(self) -> None:
        for name in ("neighbor_enabled", "segment_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Context rerank {name} must be a boolean")
        for name in (
            "max_neighbors_each_side",
            "segment_candidate_limit",
            "segment_top_k",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Context rerank {name} must be a positive integer")
        if self.segment_top_k > self.segment_candidate_limit:
            raise ValueError("segment_top_k must not exceed segment_candidate_limit")
        for name in (
            "max_bonus",
            "neighbor_max_ratio",
            "segment_max_ratio",
            "temporal_proximity_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"Context rerank {name} must be within [0, 1]")
        if (
            isinstance(self.temporal_window_seconds, bool)
            or not isinstance(self.temporal_window_seconds, (int, float))
            or not math.isfinite(float(self.temporal_window_seconds))
            or float(self.temporal_window_seconds) <= 0
        ):
            raise ValueError(
                "Context rerank temporal_window_seconds must be positive and finite"
            )


@dataclass(frozen=True)
class AdvancedRankedFrame:
    dense_row: int
    record: Mapping[str, object]
    score: float
    breakdown: Mapping[str, float]
    selection: CSESSelection
    contributions: Mapping[str, float] = field(default_factory=dict)
    context_trace: Mapping[str, Any] = field(default_factory=dict)
    temporal_chain_id: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return the backward-compatible mapping used by online candidates."""

        return self.to_result_mapping()

    def to_result_mapping(self) -> dict[str, object]:
        """Preserve dense lineage while overlaying final ranking diagnostics.

        Dense records carry the canonical frame/image/text fields expected by
        ``OnlinePipeline.Candidate.from_mapping``.  Starting from that record is
        therefore safer than rebuilding a narrow result and accidentally
        dropping ``frame_id``, ``shot_id``, paths, or modality evidence.
        """

        output = dict(self.record)
        breakdown = {
            str(name): float(value)
            for name, value in self.breakdown.items()
            if math.isfinite(float(value))
        }
        modality_scores = _float_mapping(output.get("modality_scores"))
        for name in (
            "coarse_rrf",
            "dense_visual",
            "caption",
            "ocr",
            "objects",
            "cses_gain",
            "visual_coverage",
            "temporal_consistency",
            "modality_alignment",
            "neighbor_support",
            "segment_support",
        ):
            if name in breakdown:
                modality_scores[name] = breakdown[name]
        # The existing Candidate contract calls its visual field ``visual``.
        # Keep the more precise dense name as well for debug/ablation consumers.
        if "dense_visual" in breakdown:
            modality_scores["visual"] = breakdown["dense_visual"]

        selection = self.selection.to_dict()
        output.update(
            {
                "dense_row": int(self.dense_row),
                "score": float(self.score),
                "rerank_score": float(self.score),
                "dense_visual_score": breakdown.get("dense_visual"),
                "modality_scores": modality_scores,
                "breakdown": breakdown,
                "score_breakdown": breakdown,
                "score_contributions": {
                    str(name): float(value)
                    for name, value in self.contributions.items()
                    if math.isfinite(float(value))
                },
                "context_scoring": dict(self.context_trace),
                "selection": selection,
                "cses": selection,
                "cses_selection": selection,
                "temporal_chain_id": self.temporal_chain_id,
            }
        )
        if output.get("fusion_score") is None:
            output["fusion_score"] = breakdown.get("coarse_rrf")
        return output


def rerank_dense_candidates(
    *,
    plan: QueryPlan,
    selections: Sequence[CSESSelection],
    records: Sequence[Mapping[str, object]],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    coarse_scores: Mapping[tuple[str, str], float] | None = None,
    event_vectors: Sequence[np.ndarray] = (),
    max_event_gap_seconds: float = 180.0,
    weights: AdvancedRerankWeights | None = None,
    context_index: OnlineContextIndex | None = None,
    row_by_frame: Mapping[tuple[str, str], int] | None = None,
    context_config: ContextRerankConfig | None = None,
) -> list[AdvancedRankedFrame]:
    if not selections:
        return []
    weights = weights or AdvancedRerankWeights()
    if not isinstance(weights, AdvancedRerankWeights):
        raise TypeError("weights must be an AdvancedRerankWeights instance")
    context_config = context_config or ContextRerankConfig()
    if not isinstance(context_config, ContextRerankConfig):
        raise TypeError("context_config must be a ContextRerankConfig instance")

    rows = [selection.row for selection in selections]
    matrix = np.asarray(vectors[rows], dtype=np.float32)
    query = _normalized(query_vector)
    dense_scores = (np.clip(matrix @ query, -1.0, 1.0) + 1.0) / 2.0
    # The canonical semantic target is always the user's Original Query.  Query
    # expansion/normalization may widen recall upstream, while candidate
    # metadata remains evidence scored against that unchanged target here.
    query_tokens = set(content_tokens(plan.original_query, fallback_to_all=True))
    max_cses = max(selection.selection_gain for selection in selections) or 1.0
    raw_coarse = dict(coarse_scores or {})
    coarse_scale = max(
        (abs(float(value)) for value in raw_coarse.values()),
        default=1.0,
    ) or 1.0
    has_negative_coarse = any(float(value) < 0 for value in raw_coarse.values())
    temporal_scores, temporal_chain_ids = temporal_chain_consistency(
        rows=rows,
        records=records,
        vectors=vectors,
        event_vectors=event_vectors,
        max_gap_seconds=max_event_gap_seconds,
    )
    if plan.profile == "temporal" and len(event_vectors) <= 1:
        boundary_scores = temporal_boundary_consistency(
            rows=rows,
            records=records,
            cues=plan.temporal_cues,
        )
        if boundary_scores:
            temporal_scores = boundary_scores

    context_summary = context_index.summary() if context_index is not None else {}
    neighbor_active = bool(
        context_config.neighbor_enabled
        and context_index is not None
        and row_by_frame is not None
        and int(context_summary.get("neighbor_record_count") or 0) > 0
    )
    segment_active = bool(
        context_config.segment_enabled
        and context_index is not None
        and row_by_frame is not None
        and int(context_summary.get("segment_record_count") or 0) > 0
    )
    visual_only = tuple(plan.modality_scope) == ("visual",)
    active_weight_names = (
        ["coarse_rrf", "dense_visual", "cses_gain", "visual_coverage"]
        if visual_only
        else [
            "coarse_rrf",
            "dense_visual",
            "caption",
            "ocr",
            "objects",
            "cses_gain",
            "temporal_consistency",
            "modality_alignment",
        ]
    )
    if neighbor_active:
        active_weight_names.append("neighbor_support")
    if segment_active:
        active_weight_names.append("segment_support")
    total_weight = sum(
        float(getattr(weights, "cses_gain" if name == "visual_coverage" else name))
        for name in active_weight_names
    )
    if total_weight <= 0:
        raise ValueError("Active advanced rerank weights must include a positive value")

    context_by_row: dict[int, Any] = {}
    if context_index is not None and (neighbor_active or segment_active):
        for selection in selections:
            record = records[selection.row]
            context_by_row[selection.row] = context_index.lookup(
                video_id=str(record.get("video_id") or ""),
                frame_id=str(record.get("frame_id") or record.get("keyframe_id") or ""),
                timestamp=float(record.get("timestamp") or 0.0),
                segment_id=str(record.get("segment_id") or record.get("shot_id") or ""),
                max_neighbors_each_side=context_config.max_neighbors_each_side,
            )

    context_relevance_cache: dict[int, float] = {}

    def context_relevance(row: int, *, center_timestamp: float) -> float:
        cached = context_relevance_cache.get(row)
        if cached is None:
            cached = _context_row_relevance(
                row=row,
                records=records,
                vectors=vectors,
                query=query,
                query_tokens=query_tokens,
                weights=weights,
            )
            context_relevance_cache[row] = cached
        delta = abs(float(records[row].get("timestamp") or 0.0) - center_timestamp)
        proximity = max(
            0.0,
            1.0 - delta / float(context_config.temporal_window_seconds),
        )
        proximity_weight = float(context_config.temporal_proximity_weight)
        return (1.0 - proximity_weight) * cached + proximity_weight * proximity

    ranked: list[AdvancedRankedFrame] = []
    for local_index, selection in enumerate(selections):
        record = records[selection.row]
        clip_key = (
            str(record.get("video_id") or ""),
            str(record.get("segment_id") or record.get("shot_id") or ""),
        )
        modality = {
            "caption": _text_score(query_tokens, str(record.get("caption") or "")),
            "ocr": _text_score(query_tokens, str(record.get("ocr_text") or "")),
            "objects": _text_score(
                query_tokens,
                " ".join(str(value) for value in record.get("objects", []) or []),
            ),
        }
        hinted = [modality[name] for name in plan.modality_hints if name in modality]
        hint_alignment = max(hinted, default=0.0)
        raw_coarse_score = float(raw_coarse.get(clip_key, 0.0))
        normalized_coarse = raw_coarse_score / coarse_scale
        if has_negative_coarse:
            # Visual cosine scores may be negative.  Mapping the signed range
            # to [0, 1] preserves ordering; dividing by a negative maximum would
            # otherwise make the weakest hit look strongest.
            normalized_coarse = (normalized_coarse + 1.0) / 2.0
        breakdown: dict[str, float] = {
            "coarse_rrf": normalized_coarse,
            "coarse_visual": normalized_coarse,
            "dense_visual": float(dense_scores[local_index]),
            **modality,
            "cses_gain": float(selection.selection_gain) / max_cses,
            "visual_coverage": float(selection.visual_coverage_gain),
            "temporal_consistency": temporal_scores.get(selection.row, 0.0),
            "modality_alignment": hint_alignment,
        }
        context = context_by_row.get(selection.row)
        center_timestamp = float(record.get("timestamp") or 0.0)
        neighbor_values: list[float] = []
        missing_neighbor_refs = 0
        if neighbor_active and context is not None and row_by_frame is not None:
            seen_neighbor_rows: set[int] = set()
            for neighbor in context.neighbors:
                identity = (
                    str(neighbor.get("video_id") or record.get("video_id") or ""),
                    str(neighbor.get("frame_id") or neighbor.get("keyframe_id") or ""),
                )
                neighbor_row = row_by_frame.get(identity)
                if neighbor_row is None:
                    missing_neighbor_refs += 1
                    continue
                if neighbor_row == selection.row or neighbor_row in seen_neighbor_rows:
                    continue
                seen_neighbor_rows.add(neighbor_row)
                neighbor_values.append(
                    context_relevance(neighbor_row, center_timestamp=center_timestamp)
                )
        neighbor_support = _max_top_mean(
            neighbor_values,
            max_ratio=context_config.neighbor_max_ratio,
        )

        segment_values: list[float] = []
        missing_segment_refs = 0
        if (
            segment_active
            and context is not None
            and context.segment is not None
            and context_index is not None
            and row_by_frame is not None
        ):
            segment_frame_ids = context_index.segment_frame_window(
                video_id=str(record.get("video_id") or ""),
                segment_id=context.segment_id,
                center_frame_id=str(
                    record.get("frame_id") or record.get("keyframe_id") or ""
                ),
                center_timestamp=center_timestamp,
                limit=context_config.segment_candidate_limit,
            )
            seen_segment_rows: set[int] = set()
            for frame_id in segment_frame_ids:
                segment_row = row_by_frame.get(
                    (str(record.get("video_id") or ""), str(frame_id))
                )
                if segment_row is None:
                    missing_segment_refs += 1
                    continue
                if segment_row == selection.row or segment_row in seen_segment_rows:
                    continue
                seen_segment_rows.add(segment_row)
                segment_values.append(
                    context_relevance(segment_row, center_timestamp=center_timestamp)
                )
        segment_support = _max_top_mean(
            segment_values,
            max_ratio=context_config.segment_max_ratio,
            top_k=context_config.segment_top_k,
        )
        breakdown["neighbor_support"] = neighbor_support if neighbor_active else 0.0
        breakdown["segment_support"] = segment_support if segment_active else 0.0

        raw_contributions = {
            name: float(
                getattr(weights, "cses_gain" if name == "visual_coverage" else name)
            )
            * float(breakdown[name])
            / total_weight
            for name in active_weight_names
        }
        direct_names = [
            name
            for name in active_weight_names
            if name not in {"neighbor_support", "segment_support"}
        ]
        raw_direct_score = sum(raw_contributions[name] for name in direct_names)
        raw_context_bonus = sum(
            raw_contributions.get(name, 0.0)
            for name in ("neighbor_support", "segment_support")
        )
        context_bonus = min(raw_context_bonus, float(context_config.max_bonus))
        context_scale = (
            context_bonus / raw_context_bonus if raw_context_bonus > 0 else 0.0
        )
        contributions = {
            name: value
            for name, value in raw_contributions.items()
            if name not in {"neighbor_support", "segment_support"}
        }
        for name in ("neighbor_support", "segment_support"):
            if name in raw_contributions:
                contributions[name] = raw_contributions[name] * context_scale
        score = raw_direct_score + context_bonus
        contributions.update(
            {
                "raw_direct_score": raw_direct_score,
                "context_bonus_before_cap": raw_context_bonus,
                "context_bonus_after_cap": context_bonus,
                "final_score": score,
            }
        )
        context_trace = {
            "neighbor_requested": bool(context_config.neighbor_enabled),
            "neighbor_used_for_scoring": neighbor_active,
            "neighbor_evidence_count": len(neighbor_values),
            "neighbor_missing_dense_count": missing_neighbor_refs,
            "segment_requested": bool(context_config.segment_enabled),
            "segment_used_for_scoring": segment_active,
            "segment_evidence_count": len(segment_values),
            "segment_missing_dense_count": missing_segment_refs,
            "context_bonus_before_cap": round(raw_context_bonus, 8),
            "context_bonus_after_cap": round(context_bonus, 8),
            "context_bonus_cap": float(context_config.max_bonus),
            "cap_applied": raw_context_bonus > context_bonus,
        }
        ranked.append(
            AdvancedRankedFrame(
                dense_row=selection.row,
                record=record,
                score=round(score, 8),
                breakdown={name: round(float(value), 8) for name, value in breakdown.items()},
                selection=selection,
                contributions={
                    name: round(float(value), 8)
                    for name, value in contributions.items()
                },
                context_trace=context_trace,
                temporal_chain_id=temporal_chain_ids.get(selection.row, ""),
            )
        )
    ranked.sort(
        key=lambda item: (
            -item.score,
            -item.breakdown["dense_visual"],
            float(item.record.get("timestamp", 0.0)),
            str(item.record.get("candidate_id") or ""),
        )
    )
    return ranked


def _context_row_relevance(
    *,
    row: int,
    records: Sequence[Mapping[str, object]],
    vectors: np.ndarray,
    query: np.ndarray,
    query_tokens: set[str],
    weights: AdvancedRerankWeights,
) -> float:
    if isinstance(row, bool) or not isinstance(row, int) or row < 0 or row >= len(records):
        raise IndexError(f"Context dense row is outside the index: {row}")
    record = records[row]
    signals = {
        "dense_visual": float(
            (np.clip(np.asarray(vectors[row], dtype=np.float32) @ query, -1.0, 1.0) + 1.0)
            / 2.0
        )
    }
    caption = str(record.get("caption") or "").strip()
    ocr_text = str(record.get("ocr_text") or "").strip()
    object_text = " ".join(
        str(value) for value in record.get("objects", []) or []
    ).strip()
    if caption:
        signals["caption"] = _text_score(query_tokens, caption)
    if ocr_text:
        signals["ocr"] = _text_score(query_tokens, ocr_text)
    if object_text:
        signals["objects"] = _text_score(query_tokens, object_text)
    denominator = sum(float(getattr(weights, name)) for name in signals)
    if denominator <= 0:
        return 0.0
    return sum(
        float(getattr(weights, name)) * float(value)
        for name, value in signals.items()
    ) / denominator


def _max_top_mean(
    values: Sequence[float],
    *,
    max_ratio: float,
    top_k: int | None = None,
) -> float:
    if not values:
        return 0.0
    ordered = sorted((float(value) for value in values), reverse=True)
    limit = len(ordered) if top_k is None else min(len(ordered), int(top_k))
    selected = ordered[:limit]
    top_mean = sum(selected) / len(selected)
    return float(max_ratio) * selected[0] + (1.0 - float(max_ratio)) * top_mean


def temporal_chain_consistency(
    *,
    rows: Sequence[int],
    records: Sequence[Mapping[str, object]],
    vectors: np.ndarray,
    event_vectors: Sequence[np.ndarray],
    max_gap_seconds: float,
) -> tuple[dict[int, float], dict[int, str]]:
    """Score complete ordered chains before frames are flattened for output."""
    if len(event_vectors) <= 1:
        return ({row: 1.0 for row in rows}, {})
    normalized_events = [_normalized(vector) for vector in event_vectors]
    by_video: dict[str, list[int]] = {}
    for row in rows:
        by_video.setdefault(str(records[row].get("video_id") or ""), []).append(row)

    scores: dict[int, float] = {}
    chain_ids: dict[int, str] = {}
    chain_number = 0
    for video_id in sorted(by_video):
        video_rows = sorted(
            by_video[video_id],
            key=lambda row: (
                float(records[row].get("timestamp", 0.0)),
                str(records[row].get("candidate_id") or ""),
            ),
        )
        matrix = np.asarray(vectors[video_rows], dtype=np.float32)
        event_relevance = [
            (np.clip(matrix @ event, -1.0, 1.0) + 1.0) / 2.0
            for event in normalized_events
        ]
        dp = np.asarray(event_relevance[0], dtype=np.float64)
        parents: list[list[int]] = []
        for event_index in range(1, len(event_relevance)):
            next_dp = np.full(len(video_rows), -np.inf, dtype=np.float64)
            next_parent = [-1] * len(video_rows)
            for current, current_row in enumerate(video_rows):
                current_time = float(records[current_row].get("timestamp", 0.0))
                for previous, previous_row in enumerate(video_rows):
                    previous_time = float(records[previous_row].get("timestamp", 0.0))
                    gap = current_time - previous_time
                    if gap < 0 or gap > max_gap_seconds:
                        continue
                    value = dp[previous] + float(event_relevance[event_index][current])
                    if value > next_dp[current]:
                        next_dp[current] = value
                        next_parent[current] = previous
            dp = next_dp
            parents.append(next_parent)
        if not np.isfinite(dp).any():
            continue
        final_index = int(np.nanargmax(dp))
        chain_score = float(dp[final_index]) / len(normalized_events)
        local_chain = [final_index]
        for parent_map in reversed(parents):
            final_index = parent_map[final_index]
            if final_index < 0:
                local_chain = []
                break
            local_chain.append(final_index)
        if not local_chain:
            continue
        chain_number += 1
        chain_id = f"CHAIN_{video_id}_{chain_number:03d}"
        for local_index in reversed(local_chain):
            row = video_rows[local_index]
            scores[row] = max(scores.get(row, 0.0), chain_score)
            chain_ids[row] = chain_id
    return scores, chain_ids


def temporal_boundary_consistency(
    *,
    rows: Sequence[int],
    records: Sequence[Mapping[str, object]],
    cues: Sequence[str],
) -> dict[int, float]:
    """Score first/last boundary intent deterministically within each clip."""

    cue_set = {str(cue).casefold() for cue in cues}
    prefer_first = "first" in cue_set and "last" not in cue_set
    prefer_last = "last" in cue_set and "first" not in cue_set
    if not prefer_first and not prefer_last:
        return {}
    by_clip: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        record = records[row]
        key = (
            str(record.get("video_id") or ""),
            str(record.get("segment_id") or record.get("shot_id") or ""),
        )
        by_clip.setdefault(key, []).append(row)
    scores: dict[int, float] = {}
    for clip_rows in by_clip.values():
        timestamps = [float(records[row].get("timestamp", 0.0)) for row in clip_rows]
        start = min(timestamps)
        end = max(timestamps)
        width = max(end - start, 1e-9)
        for row, timestamp in zip(clip_rows, timestamps):
            position = (timestamp - start) / width
            scores[row] = 1.0 - position if prefer_first else position
    return scores


def _normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Rerank vector must have a positive finite norm")
    return value / norm


def _text_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens or not text.strip():
        return 0.0
    evidence = set(content_tokens(text, fallback_to_all=True))
    return float(weighted_term_coverage(query_tokens, evidence)) if evidence else 0.0


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, float] = {}
    for name, raw in value.items():
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            output[str(name)] = parsed
    return output
