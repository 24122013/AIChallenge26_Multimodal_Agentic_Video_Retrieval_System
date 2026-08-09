"""Ground-truth schema and leaderboard-oriented retrieval metrics."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_retrieval_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            query_id = str(value.get("query_id") or "")
            task = str(value.get("task") or "").upper()
            relevant = value.get("relevant")
            if not query_id or query_id in labels:
                raise ValueError(f"Missing/duplicate query_id at {path}:{line_number}")
            if task not in {"TKIS", "VKIS"}:
                raise ValueError(f"Invalid task at {path}:{line_number}")
            if not isinstance(relevant, list) or not relevant:
                raise ValueError(f"Labels require non-empty relevant list at {path}:{line_number}")
            for target in relevant:
                if not isinstance(target, Mapping) or not target.get("video"):
                    raise ValueError(f"Invalid relevance target at {path}:{line_number}")
                if "frame" not in target:
                    raise ValueError(f"Relevance target needs frame at {path}:{line_number}")
            labels[query_id] = value
    if not labels:
        raise ValueError(f"No retrieval labels found: {path}")
    return labels


def load_submission_answers(path: Path) -> dict[str, list[tuple[str, int]]]:
    answers: dict[str, list[tuple[str, int]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or reader.fieldnames[0] != "query_id":
            raise ValueError("Submission must start with query_id")
        columns = reader.fieldnames[1:]
        for row in reader:
            query_id = str(row.get("query_id") or "")
            values: list[tuple[str, int]] = []
            for column in columns:
                raw = str(row.get(column) or "")
                video, frame = raw.rsplit(",", 1)
                values.append((video, int(frame)))
            answers[query_id] = values
    return answers


def evaluate_submission(
    *,
    submission_path: Path,
    labels_path: Path,
    allowed_video_ids: set[str] | None = None,
) -> dict[str, Any]:
    answers = load_submission_answers(submission_path)
    labels = load_retrieval_labels(labels_path)
    if allowed_video_ids is not None:
        normalized_allowed = {Path(value).stem for value in allowed_video_ids}
        selected: dict[str, dict[str, Any]] = {}
        for query_id, label in labels.items():
            label_videos = {
                Path(str(target["video"])).stem for target in label["relevant"]
            }
            if label_videos <= normalized_allowed:
                selected[query_id] = label
            elif label_videos & normalized_allowed:
                raise ValueError(
                    f"Label {query_id} crosses evaluation splits: {sorted(label_videos)}"
                )
        labels = selected
        if not labels:
            raise ValueError(
                "No retrieval labels belong to the requested evaluation split"
            )
    missing = sorted(set(labels) - set(answers))
    if missing:
        raise ValueError(f"Submission is missing labelled queries: {missing}")
    rows = [
        _query_metrics(answers[query_id], labels[query_id])
        for query_id in sorted(labels)
    ]
    return {
        "status": "passed",
        "query_count": len(rows),
        "macro": _macro(rows),
        "by_task": {
            task: _macro([row for row in rows if row["task"] == task])
            for task in ("TKIS", "VKIS")
            if any(row["task"] == task for row in rows)
        },
        "queries": rows,
    }


def _query_metrics(
    ranked: Sequence[tuple[str, int]],
    label: Mapping[str, Any],
) -> dict[str, Any]:
    relevant = list(label["relevant"])
    matches: list[int | None] = []
    used_targets: set[int] = set()
    for video, frame in ranked:
        matched: int | None = None
        for index, target in enumerate(relevant):
            if index in used_targets:
                continue
            tolerance = int(target.get("tolerance", 12 if label["task"] == "VKIS" else 0))
            if str(target["video"]) == video and abs(int(target["frame"]) - frame) <= tolerance:
                matched = index
                used_targets.add(index)
                break
        matches.append(matched)
    ranks = [index + 1 for index, value in enumerate(matches) if value is not None]
    first_rank = min(ranks, default=None)
    ideal = min(20, len(relevant))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, value in enumerate(matches[:20], start=1)
        if value is not None
    )
    metrics: dict[str, Any] = {
        "query_id": str(label["query_id"]),
        "task": str(label["task"]),
        "nDCG@20": dcg / ideal_dcg if ideal_dcg else 0.0,
        "MRR": 1.0 / first_rank if first_rank else 0.0,
    }
    for k in (1, 5, 10, 20, 100):
        matched = {value for value in matches[:k] if value is not None}
        metrics[f"Hit@{k}"] = float(bool(matched))
        metrics[f"Recall@{k}"] = len(matched) / len(relevant)
    if label["task"] == "TKIS" and label.get("temporal_chain"):
        metrics["temporal_Hit@20"] = _temporal_hit(
            ranked[:20],
            label["temporal_chain"],
        )
    if label["task"] == "VKIS":
        metrics["VKIS_Hit@100"] = metrics["Hit@100"]
    return metrics


def _temporal_hit(
    ranked: Sequence[tuple[str, int]],
    chain: Sequence[Mapping[str, Any]],
) -> float:
    for video in sorted({item[0] for item in ranked}):
        last_frame = -1
        satisfied = True
        for event in chain:
            tolerance = int(event.get("tolerance", 0))
            candidates = [
                frame
                for candidate_video, frame in ranked
                if candidate_video == video
                and frame >= last_frame
                and abs(frame - int(event["frame"])) <= tolerance
            ]
            if not candidates:
                satisfied = False
                break
            last_frame = min(candidates)
        if satisfied:
            return 1.0
    return 0.0


def _macro(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"query_id", "task"} and isinstance(value, (int, float))
        }
    )
    return {
        name: sum(float(row[name]) for row in rows if name in row)
        / sum(name in row for row in rows)
        for name in metric_names
    }
