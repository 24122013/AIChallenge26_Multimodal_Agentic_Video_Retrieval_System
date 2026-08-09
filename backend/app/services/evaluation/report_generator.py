"""Deterministic report assembly and atomic publication for Phase 5."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .evaluator import aggregate_keyframe_reports


PHASE5_REPORT_VERSION = 1


def build_keyframe_evaluation_report(
    *,
    split: str,
    split_manifest_sha256: str,
    config_sha256: str | None,
    canonical_sources: Sequence[Mapping[str, object]],
    video_reports: Sequence[Mapping[str, object]],
    retrieval_metrics: Mapping[str, object] | None,
    evidence: Mapping[str, object],
) -> dict[str, Any]:
    if split not in {"dev", "validation", "test"}:
        raise ValueError("split must be dev, validation, or test")
    aggregate = aggregate_keyframe_reports(
        video_reports,
        retrieval_metrics=retrieval_metrics,
    )
    return {
        "version": PHASE5_REPORT_VERSION,
        "status": aggregate["status"],
        "split": split,
        "split_manifest_sha256": split_manifest_sha256,
        "config_sha256": config_sha256,
        "canonical_sources": [dict(source) for source in canonical_sources],
        "evidence": dict(evidence),
        "aggregate": aggregate,
        "videos": [dict(report) for report in video_reports],
        "claims": {
            "detected_protected_event_recall": (
                "selector recall over feature-adapter events only"
            ),
            "manual_end_to_end_event_recall": (
                "human interval annotations; null when not provided"
            ),
            "retrieval_hit_at_k": (
                "computed only from supplied ranked retrieval evidence"
            ),
        },
    }


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "PHASE5_REPORT_VERSION",
    "build_keyframe_evaluation_report",
    "write_json_atomic",
]
