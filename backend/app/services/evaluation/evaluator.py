"""Artifact-backed keyframe evaluator for Phase 5.

Hard metrics are recomputed from candidate/final ledgers rather than copied
from selector reports.  Human and retrieval metrics remain explicitly absent
until matching evidence is supplied.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from typing import Any

from .metrics import (
    finite_non_negative,
    percentile,
    ratio,
    retrieval_hit_at_k,
    temporal_coverage_metrics,
)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_measurement(
    evidence: Mapping[str, object] | None,
    name: str,
) -> float | None:
    if evidence is None or evidence.get(name) is None:
        return None
    return finite_non_negative(evidence[name], name)


def _manual_event_metrics(
    manual_events: Sequence[Mapping[str, object]],
    *,
    selected_timestamps: Sequence[float],
    detected_event_support: Sequence[Sequence[float]],
    tolerance_seconds: float,
    video_duration: float,
) -> dict[str, object]:
    selected_hits = 0
    detector_hits = 0
    seen: set[str] = set()
    for position, event in enumerate(manual_events):
        event_id = _required_text(
            event.get("event_id"),
            f"manual_events[{position}].event_id",
        )
        if event_id in seen:
            raise ValueError(f"duplicate manual event_id: {event_id}")
        seen.add(event_id)
        start = finite_non_negative(
            event.get("start_time"),
            f"manual_events[{position}].start_time",
        )
        end = finite_non_negative(
            event.get("end_time"),
            f"manual_events[{position}].end_time",
        )
        if start > end:
            raise ValueError(f"manual event {event_id} starts after it ends")
        if end > video_duration + 1e-9:
            raise ValueError(f"manual event {event_id} exceeds video duration")
        lower = max(0.0, start - tolerance_seconds)
        upper = end + tolerance_seconds
        selected_hits += any(lower <= value <= upper for value in selected_timestamps)
        detector_hits += any(
            any(lower <= value <= upper for value in support)
            for support in detected_event_support
        )
    total = len(manual_events)
    return {
        "manual_event_count": total,
        "manual_selected_event_count": selected_hits,
        "manual_detected_event_count": detector_hits,
        "manual_end_to_end_event_recall": ratio(selected_hits, total),
        "manual_detector_event_recall": ratio(detector_hits, total),
        "manual_visual_inspection": "provided" if total else "not_provided",
    }


def _false_protection_metrics(
    reviews: Sequence[Mapping[str, object]],
    *,
    detected_event_ids: set[str],
) -> dict[str, object]:
    reviewed: dict[str, bool] = {}
    for position, review in enumerate(reviews):
        event_id = _required_text(
            review.get("detected_event_id"),
            f"protection_reviews[{position}].detected_event_id",
        )
        if event_id not in detected_event_ids:
            raise ValueError(f"review references unknown detected event: {event_id}")
        if event_id in reviewed:
            raise ValueError(f"duplicate detected-event review: {event_id}")
        raw_valid = review.get("is_true_event")
        if not isinstance(raw_valid, bool):
            raise TypeError(
                f"protection_reviews[{position}].is_true_event must be boolean"
            )
        reviewed[event_id] = raw_valid
    false_count = sum(not value for value in reviewed.values())
    return {
        "reviewed_protected_event_count": len(reviewed),
        "false_protected_event_count": false_count,
        "protection_review_coverage": ratio(len(reviewed), len(detected_event_ids)),
        "false_protection_rate": ratio(false_count, len(reviewed)),
    }


def evaluate_keyframe_video(
    *,
    video_id: str,
    final_records: Sequence[Mapping[str, object]],
    candidate_records: Sequence[Mapping[str, object]],
    event_records: Sequence[Mapping[str, object]],
    video_duration: object,
    max_gap_seconds: object,
    gap_tolerance_seconds: object = 0.0,
    target_keyframes: int | None = None,
    degraded: bool = False,
    manual_events: Sequence[Mapping[str, object]] = (),
    protection_reviews: Sequence[Mapping[str, object]] = (),
    manual_tolerance_seconds: object = 0.0,
    resource_usage: Mapping[str, object] | None = None,
    disk_bytes: int = 0,
) -> dict[str, Any]:
    """Independently evaluate one Phase 3 canonical video publish."""

    canonical_video_id = _required_text(video_id, "video_id")
    duration = finite_non_negative(video_duration, "video_duration")
    max_gap = finite_non_negative(max_gap_seconds, "max_gap_seconds")
    gap_tolerance = finite_non_negative(
        gap_tolerance_seconds,
        "gap_tolerance_seconds",
    )
    manual_tolerance = finite_non_negative(
        manual_tolerance_seconds,
        "manual_tolerance_seconds",
    )
    if isinstance(disk_bytes, bool) or not isinstance(disk_bytes, Integral):
        raise TypeError("disk_bytes must be an integer")
    if disk_bytes < 0:
        raise ValueError("disk_bytes must be non-negative")
    if target_keyframes is not None:
        target_keyframes = _non_negative_int(target_keyframes, "target_keyframes")

    candidates_by_id: dict[str, Mapping[str, object]] = {}
    candidate_time_by_id: dict[str, float] = {}
    candidate_shot_by_id: dict[str, int] = {}
    for position, record in enumerate(candidate_records):
        candidate_id = _required_text(
            record.get("candidate_id"),
            f"candidate_records[{position}].candidate_id",
        )
        if candidate_id in candidates_by_id:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        if record.get("video_id") not in {None, canonical_video_id}:
            raise ValueError(f"candidate {candidate_id} belongs to another video")
        candidates_by_id[candidate_id] = record
        candidate_time_by_id[candidate_id] = finite_non_negative(
            record.get("timestamp"),
            f"candidate_records[{position}].timestamp",
        )
        if candidate_time_by_id[candidate_id] > duration + 1e-9:
            raise ValueError(f"candidate {candidate_id} exceeds video duration")
        candidate_shot_by_id[candidate_id] = _non_negative_int(
            record.get("shot_index"),
            f"candidate_records[{position}].shot_index",
        )
    if not candidates_by_id:
        raise ValueError("candidate_records must not be empty")

    selected_ids: set[str] = set()
    selected_timestamps: list[float] = []
    selected_shots: set[int] = set()
    protected_selected_count = 0
    for position, record in enumerate(final_records):
        candidate_id = _required_text(
            record.get("candidate_id"),
            f"final_records[{position}].candidate_id",
        )
        if candidate_id not in candidates_by_id:
            raise ValueError(f"selected candidate is absent from ledger: {candidate_id}")
        if candidate_id in selected_ids:
            raise ValueError(f"duplicate selected candidate_id: {candidate_id}")
        if record.get("video_id") not in {None, canonical_video_id}:
            raise ValueError(f"selected candidate {candidate_id} belongs to another video")
        selected_ids.add(candidate_id)
        selected_timestamp = finite_non_negative(
            record.get("timestamp"),
            f"final_records[{position}].timestamp",
        )
        selected_shot = _non_negative_int(
            record.get("shot_index"),
            f"final_records[{position}].shot_index",
        )
        if not math.isclose(
            selected_timestamp,
            candidate_time_by_id[candidate_id],
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or selected_shot != candidate_shot_by_id[candidate_id]:
            raise ValueError(
                f"selected candidate {candidate_id} does not match candidate ledger"
            )
        selected_timestamps.append(selected_timestamp)
        selected_shots.add(selected_shot)
        protected_selected_count += record.get("protected") is True
    if not selected_ids:
        raise ValueError("final_records must not be empty")

    candidate_shots = set(candidate_shot_by_id.values())
    missing_shots = sorted(candidate_shots - selected_shots)
    temporal = temporal_coverage_metrics(
        selected_timestamps,
        video_duration=duration,
        max_gap_seconds=max_gap,
        tolerance_seconds=gap_tolerance,
    )

    detected_events: list[Mapping[str, object]] = []
    detected_support: list[list[float]] = []
    satisfied_event_count = 0
    event_type_counts: Counter[str] = Counter()
    detected_ids: set[str] = set()
    for position, event in enumerate(event_records):
        if event.get("source") != "feature_adapter":
            continue
        event_id = _required_text(
            event.get("event_id"),
            f"event_records[{position}].event_id",
        )
        if event_id in detected_ids:
            raise ValueError(f"duplicate detected event_id: {event_id}")
        detected_ids.add(event_id)
        raw_candidate_ids = event.get("candidate_ids")
        if not isinstance(raw_candidate_ids, Sequence) or isinstance(
            raw_candidate_ids,
            (str, bytes),
        ):
            raise TypeError(f"event {event_id} candidate_ids must be a sequence")
        candidate_ids = [
            _required_text(value, f"event {event_id} candidate_id")
            for value in raw_candidate_ids
        ]
        if not candidate_ids:
            raise ValueError(f"event {event_id} has no supporting candidates")
        unknown = sorted(set(candidate_ids) - set(candidates_by_id))
        if unknown:
            raise ValueError(f"event {event_id} references unknown candidates: {unknown}")
        satisfied_event_count += not selected_ids.isdisjoint(candidate_ids)
        detected_events.append(event)
        detected_support.append([candidate_time_by_id[value] for value in candidate_ids])
        event_type_counts[str(event.get("event_type") or "unknown")] += 1

    manual = _manual_event_metrics(
        manual_events,
        selected_timestamps=selected_timestamps,
        detected_event_support=detected_support,
        tolerance_seconds=manual_tolerance,
        video_duration=duration,
    )
    protection = _false_protection_metrics(
        protection_reviews,
        detected_event_ids=detected_ids,
    )
    selected_count = len(selected_ids)
    if target_keyframes is None:
        overrun_count: int | None = None
        overrun_ratio: float | None = None
    else:
        overrun_count = max(0, selected_count - target_keyframes)
        overrun_ratio = (
            round(overrun_count / target_keyframes, 6)
            if target_keyframes > 0
            else None
        )
    effective_shot_recall = ratio(
        len(candidate_shots) - len(missing_shots),
        len(candidate_shots),
    )
    detected_recall = ratio(satisfied_event_count, len(detected_events))
    hard_passed = (
        temporal.coverage_violation_count == 0
        and not missing_shots
        and detected_recall in {None, 1.0}
    )
    return {
        "video_id": canonical_video_id,
        "status": "passed" if hard_passed else "failed",
        "degraded": bool(degraded),
        "duration_seconds": round(duration, 6),
        "candidate_count": len(candidates_by_id),
        "selected_count": selected_count,
        "protected_selected_count": protected_selected_count,
        **temporal.to_dict(),
        "effective_shot_count": len(candidate_shots),
        "covered_effective_shot_count": len(candidate_shots) - len(missing_shots),
        "missing_shot_indices": missing_shots,
        "effective_shot_recall": effective_shot_recall,
        "detected_protected_event_count": len(detected_events),
        "satisfied_detected_protected_event_count": satisfied_event_count,
        "detected_protected_event_recall": detected_recall,
        "detected_event_type_counts": dict(sorted(event_type_counts.items())),
        **manual,
        **protection,
        "keyframes_per_minute": round(
            selected_count * 60.0 / duration,
            6,
        )
        if duration > 0
        else 0.0,
        "soft_budget_configured": target_keyframes is not None,
        "target_keyframes": target_keyframes,
        "soft_budget_overrun_count": overrun_count,
        "soft_budget_overrun_ratio": overrun_ratio,
        "runtime_sec": _optional_measurement(resource_usage, "runtime_sec"),
        "peak_ram_mb": _optional_measurement(resource_usage, "peak_ram_mb"),
        "disk_bytes": int(disk_bytes),
        "hard_targets": {
            "coverage_violation_count": 0,
            "effective_shot_recall": 1.0,
            "detected_protected_event_recall": 1.0,
        },
    }


def evaluate_retrieval_evidence(
    records: Iterable[Mapping[str, object]],
    *,
    split_video_ids: set[str],
    k_values: Sequence[int] = (1, 5, 10, 100),
) -> dict[str, object]:
    """Evaluate supplied ranked retrieval evidence against frame intervals."""

    ranked_hits: list[list[bool]] = []
    query_ids: set[str] = set()
    for position, record in enumerate(records):
        query_id = _required_text(
            record.get("query_id"),
            f"retrieval_evidence[{position}].query_id",
        )
        if query_id in query_ids:
            raise ValueError(f"duplicate retrieval query_id: {query_id}")
        query_ids.add(query_id)
        raw_relevant = record.get("relevant")
        if not isinstance(raw_relevant, Sequence) or isinstance(
            raw_relevant,
            (str, bytes),
        ):
            raise TypeError(f"retrieval evidence {query_id} relevant must be a list")
        relevant: list[tuple[str, int, int]] = []
        for item_position, item in enumerate(raw_relevant):
            if not isinstance(item, Mapping):
                raise TypeError(f"retrieval evidence {query_id} relevant item must map")
            relevant_video = _required_text(
                item.get("video_id"),
                f"retrieval evidence {query_id} relevant video_id",
            )
            if relevant_video not in split_video_ids:
                continue
            start = _non_negative_int(
                item.get("start_frame"),
                f"retrieval evidence {query_id} relevant[{item_position}].start_frame",
            )
            end = _non_negative_int(
                item.get("end_frame"),
                f"retrieval evidence {query_id} relevant[{item_position}].end_frame",
            )
            if start > end:
                raise ValueError(f"retrieval evidence {query_id} interval is reversed")
            relevant.append((relevant_video, start, end))
        if not relevant:
            continue
        raw_ranked = record.get("ranked_results")
        if not isinstance(raw_ranked, Sequence) or isinstance(raw_ranked, (str, bytes)):
            raise TypeError(
                f"retrieval evidence {query_id} ranked_results must be a list"
            )
        hits: list[bool] = []
        for item_position, item in enumerate(raw_ranked):
            if not isinstance(item, Mapping):
                raise TypeError(f"retrieval ranked result {query_id} must map")
            result_video = _required_text(
                item.get("video_id"),
                f"retrieval evidence {query_id} result video_id",
            )
            frame_index = _non_negative_int(
                item.get("frame_index"),
                f"retrieval evidence {query_id} result[{item_position}].frame_index",
            )
            hits.append(
                any(
                    result_video == relevant_video and start <= frame_index <= end
                    for relevant_video, start, end in relevant
                )
            )
        ranked_hits.append(hits)
    return {
        "retrieval_evidence_query_count": len(ranked_hits),
        **retrieval_hit_at_k(ranked_hits, k_values=k_values),
    }


def aggregate_keyframe_reports(
    reports: Sequence[Mapping[str, object]],
    *,
    retrieval_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not reports:
        raise ValueError("reports must not be empty")
    video_ids = [str(report.get("video_id") or "") for report in reports]
    if any(not value for value in video_ids) or len(video_ids) != len(set(video_ids)):
        raise ValueError("video reports require unique non-empty video_id values")

    total_candidates = sum(int(report["candidate_count"]) for report in reports)
    total_selected = sum(int(report["selected_count"]) for report in reports)
    shot_total = sum(int(report["effective_shot_count"]) for report in reports)
    shot_covered = sum(
        int(report["covered_effective_shot_count"]) for report in reports
    )
    event_total = sum(
        int(report["detected_protected_event_count"]) for report in reports
    )
    event_satisfied = sum(
        int(report["satisfied_detected_protected_event_count"])
        for report in reports
    )
    manual_total = sum(int(report["manual_event_count"]) for report in reports)
    manual_selected = sum(
        int(report["manual_selected_event_count"]) for report in reports
    )
    manual_detected = sum(
        int(report["manual_detected_event_count"]) for report in reports
    )
    reviewed_total = sum(
        int(report["reviewed_protected_event_count"]) for report in reports
    )
    false_total = sum(
        int(report["false_protected_event_count"]) for report in reports
    )
    all_gaps = [
        float(gap)
        for report in reports
        for gap in report.get("gaps_seconds", [])
    ]
    runtimes = [
        float(report["runtime_sec"])
        for report in reports
        if report.get("runtime_sec") is not None
    ]
    peak_ram = [
        float(report["peak_ram_mb"])
        for report in reports
        if report.get("peak_ram_mb") is not None
    ]
    duration = sum(float(report["duration_seconds"]) for report in reports)
    budget_reports = [
        report for report in reports if report.get("target_keyframes") is not None
    ]
    budget_target_total = sum(
        int(report["target_keyframes"]) for report in budget_reports
    )
    budget_overrun_total = sum(
        int(report.get("soft_budget_overrun_count") or 0) for report in reports
    )
    aggregate: dict[str, object] = {
        "status": (
            "passed"
            if all(report.get("status") == "passed" for report in reports)
            else "failed"
        ),
        "video_count": len(reports),
        "failed_video_ids": [
            str(report["video_id"])
            for report in reports
            if report.get("status") != "passed"
        ],
        "degraded_video_count": sum(bool(report.get("degraded")) for report in reports),
        "candidate_count": total_candidates,
        "selected_count": total_selected,
        "coverage_violation_count": sum(
            int(report["coverage_violation_count"]) for report in reports
        ),
        "max_gap_seconds": round(max(all_gaps, default=0.0), 6),
        "p95_gap_seconds": percentile(all_gaps, 95.0),
        "effective_shot_recall": ratio(shot_covered, shot_total),
        "detected_protected_event_recall": ratio(event_satisfied, event_total),
        "manual_end_to_end_event_recall": ratio(manual_selected, manual_total),
        "manual_detector_event_recall": ratio(manual_detected, manual_total),
        "manual_event_count": manual_total,
        "protection_review_coverage": ratio(reviewed_total, event_total),
        "false_protection_rate": ratio(false_total, reviewed_total),
        "reviewed_protected_event_count": reviewed_total,
        "keyframes_per_minute": round(total_selected * 60.0 / duration, 6)
        if duration > 0
        else 0.0,
        "soft_budget_target_total": budget_target_total,
        "soft_budget_overrun_count": budget_overrun_total,
        "soft_budget_overrun_ratio": round(
            budget_overrun_total / budget_target_total,
            6,
        )
        if budget_target_total > 0
        else None,
        "soft_budget_measurement_coverage": ratio(
            len(budget_reports),
            len(reports),
        ),
        "runtime_sec_sum": round(sum(runtimes), 6) if runtimes else None,
        "runtime_measurement_coverage": ratio(len(runtimes), len(reports)),
        "peak_ram_mb_max": round(max(peak_ram), 6) if peak_ram else None,
        "peak_ram_measurement_coverage": ratio(len(peak_ram), len(reports)),
        "disk_bytes": sum(int(report["disk_bytes"]) for report in reports),
    }
    aggregate["retrieval"] = dict(retrieval_metrics or {
        "retrieval_evidence_query_count": 0,
        "hit_at_1": None,
        "hit_at_5": None,
        "hit_at_10": None,
        "hit_at_100": None,
    })
    return aggregate


__all__ = [
    "aggregate_keyframe_reports",
    "evaluate_keyframe_video",
    "evaluate_retrieval_evidence",
]
