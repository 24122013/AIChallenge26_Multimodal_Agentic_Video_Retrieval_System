"""Deterministic multimodal reranking for dense CSES candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from backend.app.services.retrieval.cses import CSESSelection
from backend.app.services.retrieval.query_plan import QueryPlan
from backend.app.services.retrieval.query_terms import content_tokens, weighted_term_coverage


@dataclass(frozen=True)
class AdvancedRerankWeights:
    coarse_rrf: float = 0.16
    dense_visual: float = 0.38
    caption: float = 0.10
    ocr: float = 0.10
    asr: float = 0.10
    objects: float = 0.06
    cses_gain: float = 0.05
    temporal_consistency: float = 0.03
    modality_alignment: float = 0.02


@dataclass(frozen=True)
class AdvancedRankedFrame:
    dense_row: int
    record: Mapping[str, object]
    score: float
    breakdown: Mapping[str, float]
    selection: CSESSelection
    temporal_chain_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "dense_row": self.dense_row,
            "candidate_id": self.record.get("candidate_id"),
            "video_id": self.record.get("video_id"),
            "segment_id": self.record.get("segment_id") or self.record.get("shot_id"),
            "timestamp": self.record.get("timestamp"),
            "frame_index": self.record.get("frame_index"),
            "selected_offline": bool(self.record.get("selected_offline", False)),
            "score": self.score,
            "breakdown": dict(self.breakdown),
            "selection": self.selection.to_dict(),
            "temporal_chain_id": self.temporal_chain_id,
        }


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
) -> list[AdvancedRankedFrame]:
    if not selections:
        return []
    weights = weights or AdvancedRerankWeights()
    if any(float(value) < 0 for value in weights.__dict__.values()):
        raise ValueError("Advanced rerank weights must be non-negative")
    total_weight = sum(float(value) for value in weights.__dict__.values())
    if total_weight <= 0:
        raise ValueError("At least one advanced rerank weight must be positive")

    rows = [selection.row for selection in selections]
    matrix = np.asarray(vectors[rows], dtype=np.float32)
    query = _normalized(query_vector)
    dense_scores = (np.clip(matrix @ query, -1.0, 1.0) + 1.0) / 2.0
    query_tokens = set(content_tokens(plan.normalized_query, fallback_to_all=True))
    max_cses = max(selection.selection_gain for selection in selections) or 1.0
    raw_coarse = dict(coarse_scores or {})
    max_coarse = max(raw_coarse.values(), default=1.0) or 1.0
    temporal_scores, temporal_chain_ids = temporal_chain_consistency(
        rows=rows,
        records=records,
        vectors=vectors,
        event_vectors=event_vectors,
        max_gap_seconds=max_event_gap_seconds,
    )

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
            "asr": _text_score(query_tokens, str(record.get("asr_text") or "")),
            "objects": _text_score(
                query_tokens,
                " ".join(str(value) for value in record.get("objects", []) or []),
            ),
        }
        hinted = [modality[name] for name in plan.modality_hints if name in modality]
        hint_alignment = max(hinted, default=0.0)
        breakdown = {
            "coarse_rrf": float(raw_coarse.get(clip_key, 0.0)) / max_coarse,
            "dense_visual": float(dense_scores[local_index]),
            **modality,
            "cses_gain": float(selection.selection_gain) / max_cses,
            "temporal_consistency": temporal_scores.get(selection.row, 0.0),
            "modality_alignment": hint_alignment,
        }
        score = sum(
            float(getattr(weights, name)) * float(value)
            for name, value in breakdown.items()
        ) / total_weight
        ranked.append(
            AdvancedRankedFrame(
                dense_row=selection.row,
                record=record,
                score=round(score, 8),
                breakdown={name: round(float(value), 8) for name, value in breakdown.items()},
                selection=selection,
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
