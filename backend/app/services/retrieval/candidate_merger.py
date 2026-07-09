"""Candidate merging and duplicate removal for retrieval result pools."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from backend.app.models.retrieval import RetrievalResult


def candidate_identity(candidate: RetrievalResult) -> tuple[str, str]:
    """Return the strongest stable identity available for a result."""
    if candidate.frame_id:
        return ("frame", candidate.frame_id)
    if candidate.faiss_index is not None:
        return ("faiss", str(candidate.faiss_index))
    return ("video_time", f"{candidate.video_id}:{candidate.timestamp:.3f}")


def same_shot_identity(candidate: RetrievalResult) -> tuple[str, str]:
    shot_or_frame = candidate.shot_id or candidate.segment_id or candidate.frame_id
    return (candidate.video_id, shot_or_frame)


def merge_candidates(
    candidate_groups: Iterable[Iterable[RetrievalResult]],
    top_k: int | None = None,
    dedupe_same_shot: bool = False,
) -> list[RetrievalResult]:
    """Merge ranked candidate groups, keeping the best score per identity."""
    merged: dict[tuple[str, str], RetrievalResult] = {}
    for group in candidate_groups:
        for candidate in group:
            key = candidate_identity(candidate)
            current = merged.get(key)
            if current is None:
                merged[key] = candidate
                continue
            merged[key] = _combine(current, candidate)

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


def dedupe_by_same_shot(candidates: Iterable[RetrievalResult]) -> list[RetrievalResult]:
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
    if right.score > left.score:
        primary, secondary = right, left
    else:
        primary, secondary = left, right
    return replace(
        primary,
        score=round(max(left.score, right.score), 6),
        caption=primary.caption or secondary.caption,
        ocr_text=primary.ocr_text or secondary.ocr_text,
        objects=primary.objects or secondary.objects,
        neighbors=primary.neighbors or secondary.neighbors,
    )
