"""Aggregate ranked frame evidence into one candidate per modality and segment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from backend.app.models.retrieval import RetrievalResult


SegmentResolver = Callable[[RetrievalResult], tuple[str, str] | None]


@dataclass(frozen=True)
class SegmentEvidence:
    """Best evidence retained for one modality/segment pair."""

    video_id: str
    segment_id: str
    modality: str
    rank: int
    source_rank: int
    raw_score: float
    representative: RetrievalResult
    evidence_frame_ids: tuple[str, ...]

    @property
    def candidate_id(self) -> str:
        return f"{self.video_id}:{self.segment_id}"

    @property
    def clip_key(self) -> tuple[str, str]:
        return (self.video_id, self.segment_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "video_id": self.video_id,
            "segment_id": self.segment_id,
            "frame_id": self.representative.frame_id,
            "timestamp": self.representative.timestamp,
            "rank": self.rank,
            "source_rank": self.source_rank,
            "raw_score": self.raw_score,
            "modality": self.modality,
            "evidence_frame_ids": list(self.evidence_frame_ids),
        }


def aggregate_segments(
    groups: Mapping[str, Sequence[RetrievalResult]],
    *,
    resolver: SegmentResolver | None = None,
) -> dict[str, list[SegmentEvidence]]:
    """Collapse frame lists before fusion so long shots cannot flood RRF ranks."""
    aggregated: dict[str, list[SegmentEvidence]] = {}
    for modality in sorted(groups):
        by_segment: dict[tuple[str, str], dict[str, object]] = {}
        for source_rank, result in enumerate(groups[modality], start=1):
            key = resolver(result) if resolver is not None else _default_key(result)
            if key is None or not key[0] or not key[1]:
                continue
            state = by_segment.get(key)
            if state is None:
                by_segment[key] = {
                    "representative": result,
                    "source_rank": source_rank,
                    "raw_score": float(result.score),
                    "frame_ids": [result.frame_id] if result.frame_id else [],
                }
                continue
            frame_ids = state["frame_ids"]
            assert isinstance(frame_ids, list)
            if result.frame_id and result.frame_id not in frame_ids:
                frame_ids.append(result.frame_id)
            # Input rank is authoritative. Raw scores are retained only as debug
            # evidence and never used to combine modalities.
            if source_rank < int(state["source_rank"]):
                state["representative"] = result
                state["source_rank"] = source_rank
                state["raw_score"] = float(result.score)

        ordered = sorted(
            by_segment.items(),
            key=lambda item: (
                int(item[1]["source_rank"]),
                item[0][0],
                item[0][1],
            ),
        )
        aggregated[modality] = [
            SegmentEvidence(
                video_id=key[0],
                segment_id=key[1],
                modality=modality,
                rank=rank,
                source_rank=int(state["source_rank"]),
                raw_score=float(state["raw_score"]),
                representative=state["representative"],  # type: ignore[arg-type]
                evidence_frame_ids=tuple(state["frame_ids"]),  # type: ignore[arg-type]
            )
            for rank, (key, state) in enumerate(ordered, start=1)
        ]
    return aggregated


def _default_key(result: RetrievalResult) -> tuple[str, str] | None:
    clip_id = result.segment_id or result.shot_id or result.frame_id
    return (result.video_id, clip_id) if result.video_id and clip_id else None
