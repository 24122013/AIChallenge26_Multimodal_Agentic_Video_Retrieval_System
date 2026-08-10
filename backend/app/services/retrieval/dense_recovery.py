"""Query-scoped dense frame expansion for coarse candidate segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from competition.dense_index import DenseCandidateIndex


@dataclass(frozen=True)
class DenseRecoveryConfig:
    enabled: bool = True
    expansion_before_sec: float = 1.0
    expansion_after_sec: float = 1.0
    max_candidate_clips: int = 120


@dataclass(frozen=True)
class DenseRecoveryResult:
    rows_by_clip: Mapping[tuple[str, str], tuple[int, ...]]
    recovered_clip_scores: Mapping[tuple[str, str], float]
    source_clip_count: int
    expanded_clip_count: int
    candidate_row_count: int
    recovered_frames: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_segment_count": self.source_clip_count,
            "expanded_segment_count": self.expanded_clip_count,
            "candidate_row_count": self.candidate_row_count,
            "recovered_frame_count": len(self.recovered_frames),
            "recovered_frames": list(self.recovered_frames),
        }


def recover_dense_frames(
    *,
    dense_index: DenseCandidateIndex,
    coarse_clip_keys: Sequence[tuple[str, str]],
    coarse_clip_scores: Mapping[tuple[str, str], float],
    config: DenseRecoveryConfig,
) -> DenseRecoveryResult:
    """Expand only segments admitted by coarse retrieval; never global-search dense."""
    source_keys = list(dict.fromkeys(coarse_clip_keys))
    video_rows: dict[str, list[int]] = {}
    for row, record in enumerate(dense_index.records):
        video_rows.setdefault(str(record.get("video_id") or ""), []).append(row)

    rows_by_clip: dict[tuple[str, str], list[int]] = {}
    recovered_scores: dict[tuple[str, str], float] = {}
    recovered_frames: dict[int, dict[str, object]] = {}
    for source_key in source_keys:
        source_rows = list(dense_index.rows_by_clip.get(source_key, ()))
        if not source_rows:
            continue
        candidates = source_rows
        if config.enabled:
            start, end = _segment_window(dense_index, source_rows)
            lower = start - max(0.0, float(config.expansion_before_sec))
            upper = end + max(0.0, float(config.expansion_after_sec))
            candidates = [
                row
                for row in video_rows.get(source_key[0], ())
                if lower
                <= float(dense_index.records[row].get("timestamp", 0.0))
                <= upper
            ]
        for row in candidates:
            record = dense_index.records[row]
            actual_key = (
                str(record.get("video_id") or ""),
                str(record.get("segment_id") or record.get("shot_id") or ""),
            )
            if not actual_key[1]:
                continue
            if actual_key not in rows_by_clip:
                if len(rows_by_clip) >= max(1, int(config.max_candidate_clips)):
                    continue
                rows_by_clip[actual_key] = []
            if row not in rows_by_clip[actual_key]:
                rows_by_clip[actual_key].append(row)
            recovered_scores[actual_key] = max(
                recovered_scores.get(actual_key, 0.0),
                float(coarse_clip_scores.get(source_key, 0.0)),
            )
            if not bool(record.get("selected_offline", False)):
                recovered_frames[row] = {
                    "candidate_id": record.get("candidate_id"),
                    "video_id": record.get("video_id"),
                    "segment_id": record.get("segment_id") or record.get("shot_id"),
                    "frame_index": record.get("frame_index"),
                    "timestamp": record.get("timestamp"),
                }

    frozen_rows = {
        key: tuple(
            sorted(
                rows,
                key=lambda row: (
                    float(dense_index.records[row].get("timestamp", 0.0)),
                    str(dense_index.records[row].get("candidate_id") or ""),
                ),
            )
        )
        for key, rows in rows_by_clip.items()
    }
    return DenseRecoveryResult(
        rows_by_clip=frozen_rows,
        recovered_clip_scores=recovered_scores,
        source_clip_count=len(source_keys),
        expanded_clip_count=len(frozen_rows),
        candidate_row_count=sum(len(rows) for rows in frozen_rows.values()),
        recovered_frames=tuple(recovered_frames[row] for row in sorted(recovered_frames)),
    )


def _segment_window(
    dense_index: DenseCandidateIndex,
    rows: Sequence[int],
) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for row in rows:
        record = dense_index.records[row]
        timestamp = float(record.get("timestamp", 0.0))
        starts.append(float(record.get("shot_start", timestamp)))
        ends.append(float(record.get("shot_end", timestamp)))
    return min(starts), max(ends)
