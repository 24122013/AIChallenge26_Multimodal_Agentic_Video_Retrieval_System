"""Artifact workspace and dense-pool materialization for keyframe Phase 3.

The public competition adapter historically selected temporal/shot keyframes
before running OCR, object detection, and SigLIP2.  Phase 3 needs a full dense
pool first, so this module materializes that pool in an isolated workspace and
binds it to the exact source/config with content hashes.  Canonical keyframe
artifacts are intentionally not written here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from backend.app.services.indexing.materialize_keyframe_candidates import (
    materialize_keyframe_candidates_for_video,
)


PHASE3_CANDIDATE_CONTRACT_VERSION = 2
PHASE3_FEATURE_CONTRACT_VERSION = 2
PHASE3_SELECTION_CONTRACT_VERSION = 2


@dataclass(frozen=True)
class CandidatePoolConfig:
    """Exact model-free configuration used to build one dense candidate pool."""

    phash_threshold: int = 6
    phash_window_sec: float = 12.0
    jpeg_quality: int = 95
    shot_threshold: float = 0.5
    shot_device: str = "auto"
    candidate_interval_sec: float = 0.5
    boundary_guard_sec: float = 0.2
    tiny_shot_max_sec: float = 0.5
    include_video_endpoints: bool = False

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        # Preserve the v2 cache key for the baseline candidate pool.  The key
        # changes only for the endpoint-on ablation that truly adds frames.
        if not self.include_video_endpoints:
            value.pop("include_video_endpoints")
        return value


@dataclass(frozen=True)
class Phase3WorkspacePaths:
    """All non-canonical artifacts for one source/config run."""

    run_id: str
    root: Path
    candidate_images_root: Path
    candidate_images_dir: Path
    candidate_metadata: Path
    candidate_report: Path
    candidate_validation: Path
    embeddings: Path
    embedding_metadata: Path
    embedding_benchmark: Path
    embedding_skipped: Path
    captions: Path
    caption_report: Path
    ocr: Path
    ocr_report: Path
    objects: Path
    object_report: Path
    asr: Path
    asr_segments: Path
    asr_report: Path
    feature_manifest: Path
    candidate_scores: Path
    protected_events: Path
    selection_report: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frame_ids_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    seen: set[str] = set()
    for offset, record in enumerate(records):
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError(f"candidate row {offset} has no frame_id")
        if frame_id in seen:
            raise ValueError(f"duplicate candidate frame_id: {frame_id}")
        seen.add(frame_id)
        digest.update(frame_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def images_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for offset, record in enumerate(records):
        frame_id = record.get("frame_id")
        raw_path = record.get("keyframe_path")
        if not isinstance(frame_id, str) or not isinstance(raw_path, str):
            raise ValueError(f"candidate row {offset} has invalid image identity")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"candidate image missing: {path}")
        digest.update(frame_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object in {path} at line {line_number}")
            records.append(value)
    return records


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(raw_path)


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npy(path: Path, embeddings: np.ndarray) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy one file and expose it only after the copy is complete."""

    temporary = _temporary_path(destination)
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                target_handle.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def candidate_run_contract(
    *,
    video_path: Path,
    video_id: str,
    frame_count: int,
    config: CandidatePoolConfig,
) -> dict[str, object]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return {
        "version": PHASE3_CANDIDATE_CONTRACT_VERSION,
        "video_id": video_id,
        "source_video_sha256": sha256_file(video_path),
        "source_video_size": video_path.stat().st_size,
        "frame_count": frame_count,
        "candidate_config": config.to_dict(),
    }


def workspace_paths(
    output_root: Path,
    video_id: str,
    contract: Mapping[str, object],
) -> Phase3WorkspacePaths:
    run_id = sha256_json(contract)[:20]
    root = output_root / "work" / "keyframe_v3" / video_id / run_id
    candidate_images_root = root / "candidate_keyframes"
    return Phase3WorkspacePaths(
        run_id=run_id,
        root=root,
        candidate_images_root=candidate_images_root,
        candidate_images_dir=candidate_images_root / video_id,
        candidate_metadata=root / "candidates.jsonl",
        candidate_report=root / "candidate_report.json",
        candidate_validation=root / "candidate_validation.json",
        embeddings=root / "siglip2.npy",
        embedding_metadata=root / "siglip2_metadata.jsonl",
        embedding_benchmark=root / "siglip2_benchmark.json",
        embedding_skipped=root / "siglip2_skipped.jsonl",
        captions=root / "captions.jsonl",
        caption_report=root / "captions_report.json",
        ocr=root / "ocr.jsonl",
        ocr_report=root / "ocr_report.json",
        objects=root / "objects.jsonl",
        object_report=root / "objects_report.json",
        asr=root / "asr.jsonl",
        asr_segments=root / "asr_segments.jsonl",
        asr_report=root / "asr_report.json",
        feature_manifest=root / "feature_manifest.json",
        candidate_scores=root / "candidate_scores.jsonl",
        protected_events=root / "protected_events.jsonl",
        selection_report=root / "selection_report.json",
    )


def validate_candidate_pool(
    *,
    paths: Phase3WorkspacePaths,
    expected_contract: Mapping[str, object],
) -> tuple[dict, list[dict]]:
    if not paths.candidate_metadata.is_file() or not paths.candidate_report.is_file():
        raise FileNotFoundError("candidate metadata/report is incomplete")
    report = read_json(paths.candidate_report)
    if report.get("phase3_candidate_contract") != dict(expected_contract):
        raise ValueError("candidate source/config contract changed")
    if report.get("phase3_candidate_contract_version") != PHASE3_CANDIDATE_CONTRACT_VERSION:
        raise ValueError("candidate contract version changed")
    if report.get("phase3_status") != "passed":
        raise ValueError("candidate pool did not pass materialization")
    records = read_jsonl(paths.candidate_metadata)
    expected_count = report.get("materialized_candidate_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("candidate report count is invalid")
    if not records or len(records) != expected_count:
        raise ValueError("candidate metadata count does not match report")
    video_id = expected_contract.get("video_id")
    candidate_ids: set[str] = set()
    for offset, record in enumerate(records):
        candidate_id = record.get("candidate_id")
        if record.get("video_id") != video_id:
            raise ValueError(f"candidate row {offset} has the wrong video_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"candidate row {offset} has no candidate_id")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        if record.get("artifact_role") != "dense_candidate":
            raise ValueError(f"candidate row {offset} has the wrong artifact_role")
    if report.get("candidate_metadata_sha256") != sha256_file(paths.candidate_metadata):
        raise ValueError("candidate metadata checksum changed")
    if report.get("candidate_frame_ids_sha256") != frame_ids_sha256(records):
        raise ValueError("candidate identity checksum changed")
    if report.get("candidate_images_sha256") != images_sha256(records):
        raise ValueError("candidate image checksum changed")
    return report, records


def candidate_pool_is_current(
    *,
    paths: Phase3WorkspacePaths,
    expected_contract: Mapping[str, object],
) -> bool:
    try:
        validate_candidate_pool(paths=paths, expected_contract=expected_contract)
    except (OSError, UnicodeError, ValueError):
        return False
    return True


def materialize_candidate_pool(
    *,
    video_path: Path,
    video_id: str,
    frame_count: int,
    output_root: Path,
    config: CandidatePoolConfig,
    resume: bool = False,
    extractor: Callable[..., dict] = materialize_keyframe_candidates_for_video,
) -> tuple[Phase3WorkspacePaths, dict, list[dict]]:
    """Decode every deterministic dense candidate into a lineage-bound workspace."""

    contract = candidate_run_contract(
        video_path=video_path,
        video_id=video_id,
        frame_count=frame_count,
        config=config,
    )
    paths = workspace_paths(output_root, video_id, contract)
    if resume and candidate_pool_is_current(paths=paths, expected_contract=contract):
        report, records = validate_candidate_pool(
            paths=paths,
            expected_contract=contract,
        )
        return paths, report, records

    paths.root.mkdir(parents=True, exist_ok=True)
    report = extractor(
        video_path=video_path,
        output_dir=paths.candidate_images_root,
        metadata_path=paths.candidate_metadata,
        report_path=paths.candidate_report,
        phash_threshold=config.phash_threshold,
        phash_window_sec=config.phash_window_sec,
        jpeg_quality=config.jpeg_quality,
        shot_threshold=config.shot_threshold,
        shot_device=config.shot_device,
        candidate_interval_sec=config.candidate_interval_sec,
        boundary_guard_sec=config.boundary_guard_sec,
        tiny_shot_max_sec=config.tiny_shot_max_sec,
        include_video_endpoints=config.include_video_endpoints,
    )
    records = read_jsonl(paths.candidate_metadata)
    records = [
        {
            **record,
            "artifact_role": "dense_candidate",
            "candidate_pool_run_id": paths.run_id,
        }
        for record in records
    ]
    atomic_write_jsonl(paths.candidate_metadata, records)

    planned_count = report.get("candidate_count")
    skipped_count = report.get("skipped_count")
    complete = (
        isinstance(planned_count, int)
        and not isinstance(planned_count, bool)
        and planned_count > 0
        and len(records) == planned_count
        and skipped_count == 0
        and report.get("constraints_satisfied") is True
        and report.get("coverage_satisfied") is True
    )
    report.update(
        {
            "phase3_candidate_contract_version": PHASE3_CANDIDATE_CONTRACT_VERSION,
            "phase3_candidate_contract": contract,
            "phase3_status": "passed" if complete else "partial",
            "candidate_pool_run_id": paths.run_id,
            "planned_candidate_count": planned_count,
            "materialized_candidate_count": len(records),
            "candidate_metadata_sha256": sha256_file(paths.candidate_metadata),
            "candidate_frame_ids_sha256": frame_ids_sha256(records),
            "candidate_images_sha256": images_sha256(records),
        }
    )
    atomic_write_json(paths.candidate_report, report)
    if not complete:
        raise RuntimeError(
            "dense candidate materialization was incomplete: "
            f"planned={planned_count}, materialized={len(records)}, "
            f"skipped={skipped_count}; see {paths.candidate_report}"
        )
    validated_report, validated_records = validate_candidate_pool(
        paths=paths,
        expected_contract=contract,
    )
    return paths, validated_report, validated_records


__all__ = [
    "CandidatePoolConfig",
    "PHASE3_CANDIDATE_CONTRACT_VERSION",
    "PHASE3_FEATURE_CONTRACT_VERSION",
    "PHASE3_SELECTION_CONTRACT_VERSION",
    "Phase3WorkspacePaths",
    "atomic_copy",
    "atomic_save_npy",
    "atomic_write_json",
    "atomic_write_jsonl",
    "candidate_pool_is_current",
    "candidate_run_contract",
    "frame_ids_sha256",
    "images_sha256",
    "materialize_candidate_pool",
    "read_json",
    "read_jsonl",
    "sha256_file",
    "sha256_json",
    "validate_candidate_pool",
    "workspace_paths",
]
