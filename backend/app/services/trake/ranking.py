"""Global ranking and controlled sequence diversity for TRAKE hypotheses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from backend.app.services.retrieval.retrieval_config import TrakeConfig
from backend.app.services.trake.models import TemporalPath, TrakeHypothesis
from backend.app.services.trake.temporal_refinement import (
    LocalFrameHypothesis,
    RefinementVariant,
)


@dataclass(frozen=True)
class RankingTrace:
    input_count: int
    valid_count: int
    exact_duplicate_count: int
    near_duplicate_count: int
    output_count: int
    sequence_nms_radius_frames: int

    def to_dict(self) -> dict[str, int]:
        return {
            "input_count": self.input_count,
            "valid_count": self.valid_count,
            "exact_duplicate_count": self.exact_duplicate_count,
            "near_duplicate_count": self.near_duplicate_count,
            "output_count": self.output_count,
            "sequence_nms_radius_frames": self.sequence_nms_radius_frames,
        }


def rank_hypotheses(
    variants: Iterable[RefinementVariant | TemporalPath],
    *,
    max_answers: int | None = None,
    expected_event_count: int | None = None,
    config: TrakeConfig | None = None,
    sequence_nms_radius_frames: int = 2,
) -> tuple[list[TrakeHypothesis], RankingTrace]:
    """Validate, deduplicate and diversify sequence-level answers.

    The best raw hypothesis is never displaced.  A first pass distributes slots
    across videos and coarse paths; a second pass fills unused slots while still
    enforcing exact dedupe and near-sequence NMS.
    """

    runtime = config or TrakeConfig()
    limit = runtime.max_answers if max_answers is None else int(max_answers)
    limit = max(1, min(limit, runtime.max_answers, 100))
    if sequence_nms_radius_frames < 0:
        raise ValueError("sequence NMS radius must be non-negative")
    materialized = [_coerce_variant(item) for item in variants]
    materialized.sort(
        key=lambda item: (
            -float(item.score),
            item.frame_indices,
            item.video_id,
            item.path_id,
        )
    )

    valid: list[RefinementVariant] = []
    exact_seen: set[tuple[str, tuple[int, ...]]] = set()
    exact_duplicates = 0
    for variant in materialized:
        frames = variant.frame_indices
        if expected_event_count is not None and len(frames) != expected_event_count:
            continue
        if not frames or any(
            isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
            for frame in frames
        ):
            continue
        if not str(variant.video_id).strip():
            continue
        if any(right < left for left, right in zip(frames, frames[1:])):
            continue
        path = variant.coarse_path
        if len(path.event_candidates) != len(frames):
            continue
        if len(variant.event_refinements) != len(frames):
            continue
        if any(
            candidate.event_index != position
            for position, candidate in enumerate(path.event_candidates)
        ):
            continue
        if any(
            refinement.frame_index != frames[position]
            or not str(refinement.source).strip()
            for position, refinement in enumerate(variant.event_refinements)
        ):
            continue
        if any(
            candidate.result.video_id != variant.video_id
            or candidate.result.frame_index is None
            for candidate in path.event_candidates
        ):
            continue
        identity = (variant.video_id, tuple(frames))
        if identity in exact_seen:
            exact_duplicates += 1
            continue
        exact_seen.add(identity)
        valid.append(variant)

    selected: list[RefinementVariant] = []
    selected_ids: set[tuple[str, tuple[int, ...]]] = set()
    video_counts: dict[str, int] = {}
    path_counts: dict[tuple[str, str], int] = {}
    near_duplicates = 0
    near_duplicate_identities: set[tuple[str, tuple[int, ...]]] = set()
    first_pass_video_cap = max(5, (limit + 2) // 3)
    first_pass_path_cap = max(1, runtime.local_hypotheses_per_event)

    def try_select(
        variant: RefinementVariant,
        *,
        enforce_distribution: bool,
    ) -> bool:
        nonlocal near_duplicates
        identity = (variant.video_id, variant.frame_indices)
        if identity in selected_ids:
            return False
        path_key = (variant.video_id, variant.path_id)
        if enforce_distribution:
            if video_counts.get(variant.video_id, 0) >= first_pass_video_cap:
                return False
            if path_counts.get(path_key, 0) >= first_pass_path_cap:
                return False
        if any(
            _near_sequence(
                variant,
                existing,
                radius=sequence_nms_radius_frames,
            )
            for existing in selected
        ):
            if identity not in near_duplicate_identities:
                near_duplicate_identities.add(identity)
                near_duplicates += 1
            return False
        selected.append(variant)
        selected_ids.add(identity)
        video_counts[variant.video_id] = video_counts.get(variant.video_id, 0) + 1
        path_counts[path_key] = path_counts.get(path_key, 0) + 1
        return True

    for variant in valid:
        if len(selected) >= limit:
            break
        try_select(variant, enforce_distribution=True)
    if len(selected) < limit:
        for variant in valid:
            if len(selected) >= limit:
                break
            try_select(variant, enforce_distribution=False)

    hypotheses = [
        _to_hypothesis(variant, rank=index)
        for index, variant in enumerate(selected, start=1)
    ]
    return hypotheses, RankingTrace(
        input_count=len(materialized),
        valid_count=len(valid),
        exact_duplicate_count=exact_duplicates,
        near_duplicate_count=near_duplicates,
        output_count=len(hypotheses),
        sequence_nms_radius_frames=sequence_nms_radius_frames,
    )


def diversify_hypotheses(
    variants: Iterable[RefinementVariant | TemporalPath],
    *,
    top_k: int = 100,
    event_count: int | None = None,
    config: TrakeConfig | None = None,
) -> list[TrakeHypothesis]:
    hypotheses, _ = rank_hypotheses(
        variants,
        max_answers=top_k,
        expected_event_count=event_count,
        config=config,
    )
    return hypotheses


class TrakeRanker:
    def __init__(self, config: TrakeConfig | None = None) -> None:
        self.config = config or TrakeConfig()

    def rank(
        self,
        variants: Iterable[RefinementVariant | TemporalPath],
        *,
        top_k: int | None = None,
        event_count: int | None = None,
    ) -> tuple[list[TrakeHypothesis], RankingTrace]:
        return rank_hypotheses(
            variants,
            max_answers=top_k,
            expected_event_count=event_count,
            config=self.config,
        )


def _coerce_variant(value: RefinementVariant | TemporalPath) -> RefinementVariant:
    if isinstance(value, RefinementVariant):
        return value
    if not isinstance(value, TemporalPath):
        raise TypeError("TRAKE ranking requires TemporalPath or RefinementVariant")
    frames = tuple(value.frame_ids)
    if any(frame is None for frame in frames):
        # Invalid values remain in-band and are rejected by the validation pass.
        normalized_frames: tuple[int, ...] = tuple(
            -1 if frame is None else int(frame) for frame in frames
        )
    else:
        normalized_frames = tuple(int(frame) for frame in frames)
    local = tuple(
        LocalFrameHypothesis(
            frame_index=frame,
            score=None,
            strategy="coarse_alignment",
            confidence=0.0,
            source="canonical_metadata",
        )
        for frame in normalized_frames
    )
    return RefinementVariant(
        coarse_path=value,
        frame_indices=normalized_frames,
        score=float(value.score),
        event_refinements=local,
        warnings=value.warnings,
    )


def _near_sequence(
    left: RefinementVariant,
    right: RefinementVariant,
    *,
    radius: int,
) -> bool:
    if left.video_id != right.video_id or len(left.frame_indices) != len(right.frame_indices):
        return False
    return all(
        abs(left_frame - right_frame) <= radius
        for left_frame, right_frame in zip(left.frame_indices, right.frame_indices)
    )


def _to_hypothesis(
    variant: RefinementVariant,
    *,
    rank: int,
) -> TrakeHypothesis:
    path = variant.coarse_path
    refinements = variant.event_refinements
    lineage = tuple(
        {
            "event_index": candidate.event_index,
            "video_id": variant.video_id,
            "original_frame_index": frame_index,
            "internal_frame_id": candidate.result.frame_id,
            "source": refinement.source,
        }
        for candidate, frame_index, refinement in zip(
            path.event_candidates,
            variant.frame_indices,
            refinements,
        )
    )
    applied = any(item.source == "local_refinement" for item in refinements)
    breakdown = {
        **dict(path.score_breakdown),
        "refinement": {
            "applied": applied,
            "coarse_score": float(path.score),
            "final_score": float(variant.score),
            "strategies": [item.strategy for item in refinements],
            "confidences": [item.confidence for item in refinements],
        },
    }
    return TrakeHypothesis(
        video_id=variant.video_id,
        frame_ids=variant.frame_indices,
        score=round(float(variant.score), 6),
        score_breakdown=breakdown,
        rank=rank,
        coarse_candidates=path.event_candidates,
        lineage=lineage,
        path_id=path.path_id,
        warnings=tuple(dict.fromkeys(variant.warnings)),
    )


__all__ = [
    "RankingTrace",
    "TrakeRanker",
    "diversify_hypotheses",
    "rank_hypotheses",
]
