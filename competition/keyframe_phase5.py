"""Phase 5 evaluation split, lock, and artifact-backed report helpers."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.app.services.evaluation.evaluator import (
    evaluate_keyframe_video,
    evaluate_retrieval_evidence,
)
from backend.app.services.evaluation.report_generator import (
    build_keyframe_evaluation_report,
    write_json_atomic,
)
from competition.keyframe_phase3 import sha256_file, sha256_json


PHASE5_SPLIT_VERSION = 1
PHASE5_CONFIG_LOCK_VERSION = 1
PHASE5_REQUIRED_SPLIT_COUNTS = {"dev": 4, "validation": 4, "test": 8}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _required_video_ids(video_ids: Sequence[str]) -> list[str]:
    normalized = [str(value).strip() for value in video_ids]
    if len(normalized) != 16:
        raise ValueError("Phase 5 split requires exactly 16 video IDs")
    if any(not value for value in normalized):
        raise ValueError("video IDs must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("video IDs must be unique")
    return sorted(normalized)


def _seeded_order(video_ids: Sequence[str], seed: int) -> list[str]:
    return sorted(
        video_ids,
        key=lambda video_id: (
            hashlib.sha256(f"{seed}\0{video_id}".encode("utf-8")).hexdigest(),
            video_id,
        ),
    )


def create_split_manifest(
    video_ids: Sequence[str],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    source_ids = _required_video_ids(video_ids)
    ordered = _seeded_order(source_ids, seed)
    splits = {
        "dev": ordered[:4],
        "validation": ordered[4:8],
        "test": ordered[8:],
    }
    manifest = {
        "version": PHASE5_SPLIT_VERSION,
        "status": "locked",
        "algorithm": "sha256_seeded_order_v1",
        "seed": seed,
        "source_video_ids": source_ids,
        "splits": splits,
        "counts": dict(PHASE5_REQUIRED_SPLIT_COUNTS),
        "test_locked": True,
    }
    manifest["assignment_sha256"] = sha256_json(splits)
    return manifest


def validate_split_manifest(manifest: Mapping[str, object]) -> dict[str, Any]:
    if (
        manifest.get("version") != PHASE5_SPLIT_VERSION
        or manifest.get("status") != "locked"
        or manifest.get("algorithm") != "sha256_seeded_order_v1"
        or manifest.get("test_locked") is not True
    ):
        raise ValueError("Phase 5 split manifest is missing or outdated")
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Phase 5 split seed is invalid")
    raw_source = manifest.get("source_video_ids")
    if not isinstance(raw_source, list):
        raise ValueError("Phase 5 source_video_ids must be a list")
    expected = create_split_manifest([str(value) for value in raw_source], seed=seed)
    if dict(manifest) != expected:
        raise ValueError("Phase 5 split manifest was modified or is inconsistent")
    return expected


def write_split_manifest(
    path: Path,
    video_ids: Sequence[str],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(
            f"split manifest already exists and is immutable: {path}"
        )
    manifest = create_split_manifest(video_ids, seed=seed)
    write_json_atomic(path, manifest)
    return manifest


def load_split_manifest(path: Path) -> dict[str, Any]:
    return validate_split_manifest(_read_json(path))


def _validated_artifact_entry(raw: object, name: str) -> Path:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Phase 3 canonical artifact {name} is missing")
    raw_path = raw.get("path")
    expected_hash = raw.get("sha256")
    expected_size = raw.get("size")
    if (
        not isinstance(raw_path, str)
        or not isinstance(expected_hash, str)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
    ):
        raise ValueError(f"Phase 3 canonical artifact {name} is invalid")
    path = Path(raw_path)
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path) != expected_hash
    ):
        raise ValueError(f"Phase 3 canonical artifact {name} is stale: {path}")
    return path


def load_video_evaluation_artifacts(
    output_root: Path,
    video_id: str,
) -> dict[str, Any]:
    metadata_dir = output_root / "metadata"
    extract_report_path = metadata_dir / f"keyframes_{video_id}_extract_report.json"
    extract_report = _read_json(extract_report_path)
    if (
        extract_report.get("video_id") != video_id
        or extract_report.get("keyframe_strategy") != "multimodal_coverage"
        or extract_report.get("status") != "satisfied"
    ):
        raise ValueError(f"Phase 5 requires a passed Phase 3 publish: {video_id}")
    raw_manifest_path = extract_report.get("phase3_manifest_path")
    expected_manifest_hash = extract_report.get("phase3_manifest_sha256")
    if not isinstance(raw_manifest_path, str) or not isinstance(
        expected_manifest_hash,
        str,
    ):
        raise ValueError(f"Phase 3 commit marker is incomplete: {video_id}")
    phase3_manifest_path = Path(raw_manifest_path)
    if (
        not phase3_manifest_path.is_file()
        or sha256_file(phase3_manifest_path) != expected_manifest_hash
    ):
        raise ValueError(f"Phase 3 manifest is stale: {video_id}")
    manifest = _read_json(phase3_manifest_path)
    if (
        manifest.get("status") != "passed"
        or manifest.get("video_id") != video_id
        or manifest.get("selection_run_id")
        != extract_report.get("phase3_selection_run_id")
    ):
        raise ValueError(f"Phase 3 selection contract failed: {video_id}")
    raw_artifacts = manifest.get("canonical_artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise ValueError(f"Phase 3 artifact ledger is missing: {video_id}")
    metadata_path = _validated_artifact_entry(
        raw_artifacts.get("keyframe_metadata"),
        "keyframe_metadata",
    )
    candidate_path = _validated_artifact_entry(
        raw_artifacts.get("candidate_scores"),
        "candidate_scores",
    )
    event_path = _validated_artifact_entry(
        raw_artifacts.get("protected_events"),
        "protected_events",
    )
    final_records = _read_jsonl(metadata_path)
    candidate_records = _read_jsonl(candidate_path)
    event_records = _read_jsonl(event_path)
    selection_config = extract_report.get("selection_config")
    extract_config = extract_report.get("competition_extract_config")
    manifest_feature_config = manifest.get("feature_config")
    manifest_adapter_config = manifest.get("adapter_config")
    manifest_selection_config = manifest.get("selection_config")
    guarantees = extract_report.get("guarantees")
    if not isinstance(selection_config, Mapping) or not isinstance(
        extract_config,
        Mapping,
    ) or not isinstance(manifest_feature_config, Mapping) or not isinstance(
        manifest_adapter_config,
        Mapping,
    ) or not isinstance(manifest_selection_config, Mapping) or not isinstance(
        guarantees,
        Mapping,
    ):
        raise ValueError(f"Phase 3 evaluation config/report is incomplete: {video_id}")
    if dict(selection_config) != dict(manifest_selection_config):
        raise ValueError(f"Phase 3 selection config lineage is inconsistent: {video_id}")
    duration = extract_report.get("duration")
    if duration is None:
        raise ValueError(f"Phase 3 duration is missing: {video_id}")

    disk_bytes = sum(
        int(entry.get("size", 0))
        for name, entry in raw_artifacts.items()
        if name != "keyframe_images_sha256" and isinstance(entry, Mapping)
    )
    image_paths = {
        Path(str(record["keyframe_path"]))
        for record in final_records
        if record.get("keyframe_path")
    }
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(f"selected keyframe image is missing: {path}")
        disk_bytes += path.stat().st_size
    phase3_config = {
        "competition_extract_config": dict(extract_config),
        "feature_config": dict(manifest_feature_config),
        "adapter_config": dict(manifest_adapter_config),
        "selection_config": dict(manifest_selection_config),
        "allow_partial_features": manifest.get("allow_partial_features"),
    }
    return {
        "video_id": video_id,
        "extract_report_path": extract_report_path,
        "phase3_manifest_path": phase3_manifest_path,
        "extract_report": extract_report,
        "phase3_manifest": manifest,
        "final_records": final_records,
        "candidate_records": candidate_records,
        "event_records": event_records,
        "video_duration": duration,
        "max_gap_seconds": selection_config.get("max_gap_seconds"),
        "gap_tolerance_seconds": selection_config.get("gap_tolerance_seconds", 0.0),
        "target_keyframes": selection_config.get("target_keyframes"),
        "phase3_config": phase3_config,
        "config_sha256": sha256_json(phase3_config),
        "degraded": bool(manifest.get("degraded")),
        "disk_bytes": disk_bytes,
        "artifact_lineage": {
            "video_id": video_id,
            "selection_run_id": manifest.get("selection_run_id"),
            "extract_report_sha256": sha256_file(extract_report_path),
            "phase3_manifest_sha256": sha256_file(phase3_manifest_path),
            "candidate_scores_sha256": sha256_file(candidate_path),
            "protected_events_sha256": sha256_file(event_path),
        },
    }


def create_config_lock(
    *,
    output_root: Path,
    split_manifest_path: Path,
) -> dict[str, Any]:
    split_manifest = load_split_manifest(split_manifest_path)
    dev_ids = split_manifest["splits"]["dev"]
    artifacts = [
        load_video_evaluation_artifacts(output_root, video_id)
        for video_id in dev_ids
    ]
    config_hashes = {str(artifact["config_sha256"]) for artifact in artifacts}
    if len(config_hashes) != 1:
        raise ValueError("dev videos were produced with different extraction configs")
    config = artifacts[0]["phase3_config"]
    lock = {
        "version": PHASE5_CONFIG_LOCK_VERSION,
        "status": "locked",
        "source_split": "dev",
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "config_sha256": next(iter(config_hashes)),
        "phase3_config": config,
        "source_selection_runs": {
            artifact["video_id"]: artifact["phase3_manifest"].get("selection_run_id")
            for artifact in artifacts
        },
    }
    lock["payload_sha256"] = sha256_json(lock)
    return lock


def write_config_lock(
    path: Path,
    *,
    output_root: Path,
    split_manifest_path: Path,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"config lock already exists and is immutable: {path}")
    lock = create_config_lock(
        output_root=output_root,
        split_manifest_path=split_manifest_path,
    )
    write_json_atomic(path, lock)
    return lock


def validate_config_lock(
    path: Path,
    *,
    split_manifest_path: Path,
    expected_config_sha256: str,
) -> dict[str, Any]:
    lock = _read_json(path)
    expected_payload_hash = lock.get("payload_sha256")
    unhashed_payload = {
        key: value for key, value in lock.items() if key != "payload_sha256"
    }
    if (
        lock.get("version") != PHASE5_CONFIG_LOCK_VERSION
        or lock.get("status") != "locked"
        or lock.get("source_split") != "dev"
        or lock.get("split_manifest_sha256") != sha256_file(split_manifest_path)
        or lock.get("config_sha256") != expected_config_sha256
        or not isinstance(expected_payload_hash, str)
        or sha256_json(unhashed_payload) != expected_payload_hash
    ):
        raise ValueError(
            "Phase 5 config lock is missing, stale, or does not match artifacts"
        )
    raw_config = lock.get("phase3_config")
    if not isinstance(raw_config, Mapping) or sha256_json(dict(raw_config)) != expected_config_sha256:
        raise ValueError("Phase 5 locked config payload was modified")
    return lock


def _optional_jsonl(path: Path | None) -> list[dict[str, Any]]:
    return _read_jsonl(path) if path is not None else []


def _group_by_video(
    records: Sequence[Mapping[str, object]],
    *,
    name: str,
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for position, record in enumerate(records):
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{name}[{position}] requires video_id")
        grouped[video_id].append(record)
    return grouped


def evaluate_split_artifacts(
    *,
    output_root: Path,
    split_manifest_path: Path,
    split: str,
    canonical_sources: Sequence[Mapping[str, object]],
    config_lock_path: Path | None = None,
    manual_events_path: Path | None = None,
    protection_reviews_path: Path | None = None,
    retrieval_evidence_path: Path | None = None,
    resource_usage_path: Path | None = None,
    manual_tolerance_seconds: float = 0.0,
) -> dict[str, Any]:
    manifest = load_split_manifest(split_manifest_path)
    if split not in PHASE5_REQUIRED_SPLIT_COUNTS:
        raise ValueError("split must be dev, validation, or test")
    video_ids = list(manifest["splits"][split])
    if [source.get("video_id") for source in canonical_sources] != video_ids:
        raise ValueError("canonical source order does not match the evaluation split")
    artifacts = [
        load_video_evaluation_artifacts(output_root, video_id)
        for video_id in video_ids
    ]
    config_hashes = {str(artifact["config_sha256"]) for artifact in artifacts}
    if len(config_hashes) != 1:
        raise ValueError("split videos were produced with different extraction configs")
    config_hash = next(iter(config_hashes))
    if split in {"validation", "test"}:
        if config_lock_path is None:
            raise ValueError(f"{split} evaluation requires a dev-derived config lock")
        validate_config_lock(
            config_lock_path,
            split_manifest_path=split_manifest_path,
            expected_config_sha256=config_hash,
        )
    elif config_lock_path is not None:
        validate_config_lock(
            config_lock_path,
            split_manifest_path=split_manifest_path,
            expected_config_sha256=config_hash,
        )

    manual_records = _optional_jsonl(manual_events_path)
    review_records = _optional_jsonl(protection_reviews_path)
    resource_records = _optional_jsonl(resource_usage_path)
    manual_by_video = _group_by_video(manual_records, name="manual_events")
    reviews_by_video = _group_by_video(review_records, name="protection_reviews")
    resources_by_video = _group_by_video(resource_records, name="resource_usage")
    if any(len(values) != 1 for values in resources_by_video.values()):
        raise ValueError("resource_usage requires at most one record per video")
    unknown_evidence_videos = (
        set(manual_by_video) | set(reviews_by_video) | set(resources_by_video)
    ) - set(video_ids)
    if unknown_evidence_videos:
        raise ValueError(
            f"evidence contains videos outside split {split}: "
            f"{sorted(unknown_evidence_videos)}"
        )

    reports: list[dict[str, Any]] = []
    for artifact in artifacts:
        video_id = str(artifact["video_id"])
        resource = (
            resources_by_video[video_id][0]
            if resources_by_video.get(video_id)
            else None
        )
        reports.append(
            evaluate_keyframe_video(
                video_id=video_id,
                final_records=artifact["final_records"],
                candidate_records=artifact["candidate_records"],
                event_records=artifact["event_records"],
                video_duration=artifact["video_duration"],
                max_gap_seconds=artifact["max_gap_seconds"],
                gap_tolerance_seconds=artifact["gap_tolerance_seconds"],
                target_keyframes=artifact["target_keyframes"],
                degraded=bool(artifact["degraded"]),
                manual_events=manual_by_video.get(video_id, []),
                protection_reviews=reviews_by_video.get(video_id, []),
                manual_tolerance_seconds=manual_tolerance_seconds,
                resource_usage=resource,
                disk_bytes=int(artifact["disk_bytes"]),
            )
        )

    retrieval_records = _optional_jsonl(retrieval_evidence_path)
    retrieval_metrics = evaluate_retrieval_evidence(
        retrieval_records,
        split_video_ids=set(video_ids),
    )
    evidence_paths = {
        "manual_events": manual_events_path,
        "protection_reviews": protection_reviews_path,
        "retrieval_evidence": retrieval_evidence_path,
        "resource_usage": resource_usage_path,
    }
    evidence = {
        name: (
            {
                "path": path.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path is not None
            else None
        )
        for name, path in evidence_paths.items()
    }
    evidence["manual_tolerance_seconds"] = manual_tolerance_seconds
    report = build_keyframe_evaluation_report(
        split=split,
        split_manifest_sha256=sha256_file(split_manifest_path),
        config_sha256=config_hash,
        canonical_sources=canonical_sources,
        video_reports=reports,
        retrieval_metrics=retrieval_metrics,
        evidence=evidence,
    )
    report["artifact_lineage"] = [
        artifact["artifact_lineage"] for artifact in artifacts
    ]
    return report


__all__ = [
    "PHASE5_CONFIG_LOCK_VERSION",
    "PHASE5_REQUIRED_SPLIT_COUNTS",
    "PHASE5_SPLIT_VERSION",
    "create_config_lock",
    "create_split_manifest",
    "evaluate_split_artifacts",
    "load_split_manifest",
    "load_video_evaluation_artifacts",
    "validate_config_lock",
    "validate_split_manifest",
    "write_config_lock",
    "write_split_manifest",
]
