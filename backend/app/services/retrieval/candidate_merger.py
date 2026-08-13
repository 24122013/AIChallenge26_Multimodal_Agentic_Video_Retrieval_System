"""Candidate merging and duplicate removal for multimodal retrieval pools."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from backend.app.models.retrieval import NeighborFrame, RetrievalResult


def candidate_identity(candidate: RetrievalResult) -> tuple[str, str]:
    """Return the strongest stable identity available for a result."""
    if candidate.frame_id:
        return ("frame", f"{candidate.video_id}:{candidate.frame_id}")
    if candidate.segment_id:
        return ("segment", f"{candidate.video_id}:{candidate.segment_id}")
    if candidate.faiss_index is not None:
        return ("faiss", str(candidate.faiss_index))
    return ("video_time", f"{candidate.video_id}:{candidate.timestamp:.3f}")


def same_shot_identity(candidate: RetrievalResult) -> tuple[str, str]:
    shot_or_segment = candidate.shot_id or candidate.segment_id
    if shot_or_segment:
        return ("shot", f"{candidate.video_id}:{shot_or_segment}")
    kind, value = candidate_identity(candidate)
    return (kind, value)


def merge_candidates(
    candidate_groups: Iterable[Iterable[RetrievalResult]],
    top_k: int | None = None,
    dedupe_same_shot: bool = False,
) -> list[RetrievalResult]:
    """Merge ranked candidate groups and preserve scores from every modality."""
    merged: dict[tuple[str, str], RetrievalResult] = {}
    for group in candidate_groups:
        for candidate in group:
            key = candidate_identity(candidate)
            current = merged.get(key)
            merged[key] = candidate if current is None else _combine(current, candidate)

    ranked = sorted(
        merged.values(),
        key=lambda item: (item.score, item.timestamp_confidence),
        reverse=True,
    )
    if dedupe_same_shot:
        ranked = dedupe_by_same_shot(ranked)
    if top_k is None:
        return ranked
    return ranked[: max(0, int(top_k))]


def dedupe_by_same_shot(
    candidates: Iterable[RetrievalResult],
) -> list[RetrievalResult]:
    seen: set[tuple[str, str]] = set()
    kept: list[RetrievalResult] = []
    for candidate in candidates:
        key = same_shot_identity(candidate)
        if key in seen:
            continue
        seen.add(key)
        kept.append(candidate)
    return kept


def _combine(left: RetrievalResult, right: RetrievalResult) -> RetrievalResult:
    primary, secondary = (
        (right, left) if right.score > left.score else (left, right)
    )
    modality_scores = dict(left.modality_scores)
    for modality, score in right.modality_scores.items():
        modality_scores[modality] = max(
            float(score),
            float(modality_scores.get(modality, score)),
        )
    return replace(
        primary,
        score=round(max(left.score, right.score), 6),
        caption=primary.caption or secondary.caption,
        ocr_text=primary.ocr_text or secondary.ocr_text,
        objects=_unique([*primary.objects, *secondary.objects]),
        modality_scores=modality_scores,
        neighbors=_merge_neighbors(primary.neighbors, secondary.neighbors),
        keyframe_path=primary.keyframe_path or secondary.keyframe_path,
        thumbnail_path=primary.thumbnail_path or secondary.thumbnail_path,
        frame_index=(
            primary.frame_index
            if primary.frame_index is not None
            else secondary.frame_index
        ),
        faiss_index=(
            primary.faiss_index
            if primary.faiss_index is not None
            else secondary.faiss_index
        ),
    )


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _merge_neighbors(
    left: Iterable[NeighborFrame],
    right: Iterable[NeighborFrame],
) -> list[NeighborFrame]:
    merged: dict[tuple[str, str], NeighborFrame] = {}
    for neighbor in [*left, *right]:
        merged[(neighbor.video_id, neighbor.frame_id)] = neighbor
    return sorted(merged.values(), key=lambda item: item.timestamp)
