"""Query-aware coverage selection over dense candidate frames (CSES)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PROFILE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "kis": (0.70, 0.20, 0.10),
    "avs": (0.50, 0.30, 0.20),
    "temporal": (0.55, 0.15, 0.30),
    "qa": (0.60, 0.15, 0.25),
}


@dataclass(frozen=True)
class CSESConfig:
    max_frames: int = 12
    similarity_threshold: float = 0.92
    temporal_window_seconds: float = 2.0
    temporal_bins: int = 4


@dataclass(frozen=True)
class CSESSelection:
    row: int
    selection_rank: int
    selection_gain: float
    relevance: float
    visual_coverage_gain: float
    temporal_coverage_gain: float
    preserved_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "row": self.row,
            "selection_rank": self.selection_rank,
            "selection_gain": self.selection_gain,
            "relevance": self.relevance,
            "visual_coverage_gain": self.visual_coverage_gain,
            "temporal_coverage_gain": self.temporal_coverage_gain,
            "preserved_event_ids": list(self.preserved_event_ids),
        }


def select_cses(
    *,
    rows: Sequence[int],
    records: Sequence[Mapping[str, object]],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    profile: str,
    config: CSESConfig | None = None,
) -> list[CSESSelection]:
    config = config or CSESConfig()
    if profile not in PROFILE_WEIGHTS:
        raise ValueError(f"Unsupported CSES profile: {profile}")
    if config.max_frames <= 0 or config.temporal_bins <= 0:
        raise ValueError("CSES cardinality and temporal bins must be positive")
    unique_rows = list(dict.fromkeys(int(row) for row in rows))
    if not unique_rows:
        return []
    matrix = np.asarray(vectors[unique_rows], dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError("CSES vector dimensions do not match")
    query_norm = float(np.linalg.norm(query))
    if not np.isfinite(query_norm) or query_norm <= 0:
        raise ValueError("CSES query vector must have a positive finite norm")
    query = query / query_norm
    relevance = np.clip(matrix @ query, -1.0, 1.0)
    relevance_unit = (relevance + 1.0) / 2.0
    similarities = np.clip(matrix @ matrix.T, -1.0, 1.0)

    timestamps = np.asarray(
        [float(records[row].get("timestamp", 0.0)) for row in unique_rows],
        dtype=np.float64,
    )
    start = float(timestamps.min())
    end = float(timestamps.max())
    width = max(end - start, 1e-9)
    bins = np.minimum(
        config.temporal_bins - 1,
        np.floor((timestamps - start) / width * config.temporal_bins).astype(int),
    )
    event_sets = [
        tuple(str(value) for value in records[row].get("protected_event_ids", []) or [])
        for row in unique_rows
    ]
    all_events = sorted({event for values in event_sets for event in values})
    weights = PROFILE_WEIGHTS[profile]

    selected_local: list[int] = []
    selections: list[CSESSelection] = []
    current_facility = np.zeros(len(unique_rows), dtype=np.float32)
    covered_bins: set[int] = set()
    covered_events: set[str] = set()

    def suppressed(local_row: int) -> bool:
        for chosen in selected_local:
            if (
                similarities[local_row, chosen] > config.similarity_threshold
                and abs(timestamps[local_row] - timestamps[chosen])
                <= config.temporal_window_seconds
            ):
                # A duplicate may still preserve a not-yet-covered event.
                if not (set(event_sets[local_row]) - covered_events):
                    return True
        return False

    def gains(local_row: int) -> tuple[float, float, float]:
        proposed = np.maximum(current_facility, np.maximum(similarities[:, local_row], 0.0))
        facility_gain = float(proposed.mean() - current_facility.mean())
        new_bin = 1.0 if int(bins[local_row]) not in covered_bins else 0.0
        new_events = set(event_sets[local_row]) - covered_events
        event_gain = len(new_events) / max(1, len(all_events))
        temporal_gain = 0.5 * new_bin / config.temporal_bins + 0.5 * event_gain
        total = (
            weights[0] * float(relevance_unit[local_row])
            + weights[1] * facility_gain
            + weights[2] * temporal_gain
        )
        return total, facility_gain, temporal_gain

    def choose(local_row: int) -> None:
        nonlocal current_facility
        total, facility_gain, temporal_gain = gains(local_row)
        selected_local.append(local_row)
        current_facility = np.maximum(
            current_facility,
            np.maximum(similarities[:, local_row], 0.0),
        )
        covered_bins.add(int(bins[local_row]))
        new_events = tuple(sorted(set(event_sets[local_row]) - covered_events))
        covered_events.update(event_sets[local_row])
        selections.append(
            CSESSelection(
                row=unique_rows[local_row],
                selection_rank=len(selections) + 1,
                selection_gain=round(total, 8),
                relevance=round(float(relevance_unit[local_row]), 8),
                visual_coverage_gain=round(facility_gain, 8),
                temporal_coverage_gain=round(temporal_gain, 8),
                preserved_event_ids=new_events,
            )
        )

    # Guarantee one representative for every event when the clip budget permits.
    for event_id in all_events:
        if len(selections) >= min(config.max_frames, len(unique_rows)):
            break
        candidates = [
            local_row
            for local_row, values in enumerate(event_sets)
            if event_id in values and local_row not in selected_local
        ]
        candidates.sort(
            key=lambda local_row: (
                -float(relevance_unit[local_row]),
                timestamps[local_row],
                str(records[unique_rows[local_row]].get("candidate_id", "")),
            )
        )
        if candidates:
            choose(candidates[0])

    while len(selections) < min(config.max_frames, len(unique_rows)):
        ranked: list[tuple[float, float, float, str, int]] = []
        for local_row in range(len(unique_rows)):
            if local_row in selected_local or suppressed(local_row):
                continue
            total, _, _ = gains(local_row)
            ranked.append(
                (
                    -total,
                    -float(relevance_unit[local_row]),
                    timestamps[local_row],
                    str(records[unique_rows[local_row]].get("candidate_id", "")),
                    local_row,
                )
            )
        if not ranked:
            break
        ranked.sort()
        choose(ranked[0][-1])
    return selections
