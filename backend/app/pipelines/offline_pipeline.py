"""Canonical sequential offline orchestration for multimodal video retrieval.

The expensive unit of work is one video.  A video completes dense candidate
generation, full-pool materialization, every configured feature extractor,
multimodal selection, canonical publication, and validation before the next
video starts.  Corpus indexes, selected-keyframe neighbor mappings, and
segment/event metadata are built only from the explicit set of videos that
completed that contract.

This module intentionally orchestrates existing algorithms.  Shot detection,
dense sampling, image decoding, feature inference, selection, and index
construction remain in their existing service modules.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.environment import load_project_env
from backend.app.services.indexing.build_bge_m3_index import (
    load_canonical_keyframe_records,
)
from backend.app.services.indexing.build_faiss_index import (
    ARTIFACT_TAG,
    build_faiss_artifacts,
    frame_map_record,
    require_faiss,
    validate_embedding_source,
)
from backend.app.services.indexing.build_siglip2_index import (
    DEFAULT_MODEL_CACHE_DIR as DEFAULT_SIGLIP_CACHE_DIR,
    DEFAULT_MODEL_NAME as DEFAULT_SIGLIP_MODEL,
    encode_keyframes,
    validate_embedding_artifacts,
)
from backend.app.services.indexing.build_text_index import write_text_index
from backend.app.services.indexing.extract_keyframes import (
    Shot,
    VideoInfo,
    detect_shots_transnetv2,
    read_video_info,
)
from backend.app.services.indexing.keyframe_candidates import (
    DEFAULT_BOUNDARY_GUARD_SEC,
    DEFAULT_INTERVAL_SEC,
    DEFAULT_TINY_SHOT_MAX_SEC,
    KeyframeCandidate,
    generate_keyframe_candidates,
)
from backend.app.services.indexing.keyframe_multimodal_pipeline import (
    MultimodalKeyframePipelineResult,
    run_multimodal_keyframe_pipeline,
)
from backend.app.services.indexing.keyframe_selection import (
    DEFAULT_MAX_GAP_SECONDS,
    SelectionConfig,
)
from backend.app.services.indexing.materialize_keyframe_candidates import (
    MATERIALIZATION_MODE,
    materialize_generated_keyframe_candidates_for_video,
)
from backend.app.services.indexing.neighbor_index import build_neighbor_index
from backend.app.services.indexing.extract_segments import (
    build_segment_records,
    build_segments,
)
from backend.app.services.indexing.validate_frame_map import validate_frame_map
from backend.app.services.indexing.validate_keyframes import validate_records
from backend.app.services.ingestion.caption_pipeline import (
    DEFAULT_MODEL_CACHE_DIR as DEFAULT_CAPTION_CACHE_DIR,
    DEFAULT_MODEL_NAME as DEFAULT_CAPTION_MODEL,
    DEFAULT_MODEL_REVISION as DEFAULT_CAPTION_REVISION,
    DEFAULT_TASK_PROMPT as DEFAULT_CAPTION_TASK_PROMPT,
    run_caption_file,
)
from backend.app.services.ingestion.object_pipeline import (
    DEFAULT_MODEL_NAME as DEFAULT_OBJECT_MODEL,
    DEFAULT_MODEL_REVISION as DEFAULT_OBJECT_REVISION,
    run_object_file,
)
from backend.app.services.ingestion.ocr_pipeline import (
    DEFAULT_DETECTION_MODEL,
    DEFAULT_MODEL_REVISION as DEFAULT_OCR_REVISION,
    DEFAULT_RECOGNITION_MODEL,
)
from backend.app.services.retrieval.bge_dense import (
    DEFAULT_BGE_M3_MODEL,
    DEFAULT_BGE_M3_REVISION,
    BgeM3ArtifactPaths,
    build_bge_m3_index,
    validate_bge_m3_artifacts,
)
from backend.app.services.retrieval.dense_candidate_index import (
    DENSE_ARTIFACT_ROLE,
    DENSE_FRAME_MAP_NAME,
    DENSE_INDEX_NAME,
    DENSE_MANIFEST_NAME,
    DENSE_MANIFEST_SCHEMA_VERSION,
    DENSE_METADATA_NAME,
    DENSE_REPORT_NAME,
)
from backend.app.services.retrieval.text_index import (
    MODALITIES as TEXT_MODALITIES,
    build_text_index as build_text_index_payload,
)


LOGGER = logging.getLogger("offline_pipeline")
PIPELINE_SCHEMA_VERSION = "1.0"
PIPELINE_NAME = "canonical_sequential_offline"
CORPUS_BUNDLE_SCHEMA_VERSION = "1.0"
SUPPORTED_VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".mov", ".webm")
DEFAULT_TARGET_DENSITY_PER_SECOND = 1.0 / DEFAULT_MAX_GAP_SECONDS
CHECKPOINT_INVALID_ERRORS = (
    OSError,
    EOFError,
    KeyError,
    TypeError,
    ValueError,
    RuntimeError,
)

STAGE_SHOTS = "shot_detection"
STAGE_CANDIDATES = "dense_candidate_generation"
STAGE_MATERIALIZATION = "dense_candidate_materialization"
STAGE_SIGLIP = "siglip2_features"
STAGE_OCR = "ocr_features"
STAGE_OBJECTS = "object_features"
STAGE_CAPTIONS = "caption_features"
STAGE_SELECTION = "multimodal_selection"
STAGE_PERSISTENCE = "canonical_persistence"
STAGE_VALIDATION = "per_video_validation"


@dataclass(frozen=True)
class OfflinePipelineConfig:
    """Runtime and artifact policy for the canonical offline pipeline."""

    output_dir: Path = Path("data")
    device: str = "auto"
    resume: bool = True
    force: bool = False
    allow_partial_corpus: bool = False
    build_corpus: bool = True

    shot_threshold: float = 0.5
    shot_device: str = "auto"
    dense_interval_sec: float = DEFAULT_INTERVAL_SEC
    boundary_guard_sec: float = DEFAULT_BOUNDARY_GUARD_SEC
    tiny_shot_max_sec: float = DEFAULT_TINY_SHOT_MAX_SEC
    include_video_endpoints: bool = False
    jpeg_quality: int = 95
    phash_threshold: int = 6
    phash_window_sec: float = 12.0

    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS
    gap_tolerance_seconds: float = 0.0
    target_keyframes: int | None = None
    target_density_per_second: float | None = DEFAULT_TARGET_DENSITY_PER_SECOND
    hard_max_keyframes: int | None = None
    enable_event_aware_dedup: bool = True
    dedup_similarity_threshold: float = 0.92
    dedup_temporal_window_seconds: float = 12.0

    siglip_model_name: str = DEFAULT_SIGLIP_MODEL
    siglip_model_revision: str | None = None
    siglip_model_cache_dir: Path = DEFAULT_SIGLIP_CACHE_DIR
    siglip_batch_size: str | int = "auto"
    siglip_num_workers: int = 0
    siglip_prefetch_factor: int = 2
    siglip_use_autocast: bool = True

    caption_model_name: str = DEFAULT_CAPTION_MODEL
    caption_model_revision: str | None = DEFAULT_CAPTION_REVISION
    caption_model_cache_dir: Path = DEFAULT_CAPTION_CACHE_DIR
    caption_batch_size: int = 2
    caption_max_new_tokens: int = 256
    caption_dtype: str = "auto"
    caption_quantization: str = "none"
    caption_task_prompt: str = DEFAULT_CAPTION_TASK_PROMPT

    ocr_detection_model: str = DEFAULT_DETECTION_MODEL
    ocr_recognition_model: str = DEFAULT_RECOGNITION_MODEL
    ocr_model_revision: str = DEFAULT_OCR_REVISION
    ocr_model_cache_dir: Path = Path("data/model_cache/ocr")
    ocr_batch_size: int = 4
    ocr_conf_threshold: float = 0.3

    object_model_name: str = DEFAULT_OBJECT_MODEL
    object_model_revision: str = DEFAULT_OBJECT_REVISION
    object_model_cache_dir: Path = Path("data/model_cache/objects")
    object_batch_size: int = 8
    object_conf_threshold: float = 0.25
    object_iou_threshold: float = 0.7

    bge_enabled: bool = True
    bge_model_name: str = DEFAULT_BGE_M3_MODEL
    bge_model_revision: str = DEFAULT_BGE_M3_REVISION
    bge_batch_size: int = 16
    bge_local_files_only: bool = False
    bge_model_cache_dir: Path = Path("data/model_cache/bge_m3")

    neighbor_window_seconds: float = 5.0
    segment_strategy: str = "auto"
    segment_fixed_duration_seconds: float = 10.0
    segment_caption_similarity_threshold: float = 0.92

    min_image_width: int = 16
    min_image_height: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for field_name in (
            "siglip_model_cache_dir",
            "caption_model_cache_dir",
            "ocr_model_cache_dir",
            "object_model_cache_dir",
            "bge_model_cache_dir",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.shot_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("shot_device must be one of: auto, cpu, cuda")
        for name in (
            "shot_threshold",
            "dense_interval_sec",
            "max_gap_seconds",
            "phash_window_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in (
            "boundary_guard_sec",
            "tiny_shot_max_sec",
            "gap_tolerance_seconds",
            "dedup_temporal_window_seconds",
            "neighbor_window_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if self.phash_threshold < 0:
            raise ValueError("phash_threshold must be >= 0")
        for name in (
            "siglip_num_workers",
            "siglip_prefetch_factor",
            "caption_batch_size",
            "caption_max_new_tokens",
            "ocr_batch_size",
            "object_batch_size",
            "bge_batch_size",
            "min_image_width",
            "min_image_height",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < (0 if name == "siglip_num_workers" else 1):
                raise ValueError(f"{name} has an invalid value: {value!r}")
        if self.caption_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(
                "caption_dtype must be auto, bfloat16, float16, or float32"
            )
        if self.caption_quantization != "none":
            raise ValueError(
                "Florence-2 caption quantization is not supported or tested; "
                "caption_quantization must be 'none'"
            )
        if not str(self.caption_task_prompt).strip():
            raise ValueError("caption_task_prompt must not be empty")
        if self.target_keyframes is not None and self.target_density_per_second is not None:
            raise ValueError(
                "target_keyframes and target_density_per_second are mutually exclusive"
            )
        if self.target_density_per_second is not None:
            density = float(self.target_density_per_second)
            if not math.isfinite(density) or density <= 0:
                raise ValueError("target_density_per_second must be positive")
        if self.segment_strategy not in {"auto", "boundary", "fixed"}:
            raise ValueError("segment_strategy must be one of: auto, boundary, fixed")
        if (
            not math.isfinite(float(self.segment_fixed_duration_seconds))
            or self.segment_fixed_duration_seconds <= 0
        ):
            raise ValueError("segment_fixed_duration_seconds must be positive")
        if not 0 <= float(self.segment_caption_similarity_threshold) <= 1:
            raise ValueError(
                "segment_caption_similarity_threshold must be between 0 and 1"
            )

    def selection_config(self) -> SelectionConfig:
        return SelectionConfig(
            max_gap_seconds=self.max_gap_seconds,
            target_keyframes=self.target_keyframes,
            target_density_per_second=self.target_density_per_second,
            hard_max_keyframes=self.hard_max_keyframes,
            gap_tolerance_seconds=self.gap_tolerance_seconds,
            protect_each_shot=True,
            enable_event_aware_dedup=self.enable_event_aware_dedup,
            dedup_similarity_threshold=self.dedup_similarity_threshold,
            dedup_temporal_window_seconds=self.dedup_temporal_window_seconds,
        )


@dataclass(frozen=True)
class PerVideoPaths:
    video_id: str
    output_dir: Path
    dense_metadata: Path
    dense_images_root: Path
    dense_images_dir: Path
    dense_features_dir: Path
    shot_report: Path
    candidate_plan: Path
    candidate_report: Path
    materialization_report: Path
    dense_embeddings: Path
    dense_embedding_metadata: Path
    dense_embedding_skipped: Path
    dense_embedding_report: Path
    dense_captions: Path
    dense_caption_report: Path
    dense_ocr: Path
    dense_ocr_report: Path
    dense_objects: Path
    dense_object_report: Path
    selected_images_dir: Path
    selected_metadata: Path
    selected_embeddings: Path
    selected_embedding_metadata: Path
    selected_captions: Path
    selected_ocr: Path
    selected_objects: Path
    candidate_ledger: Path
    event_ledger: Path
    selection_report: Path
    validation_report: Path
    completion_report: Path
    state_manifest: Path

    @classmethod
    def from_config(
        cls,
        video_id: str,
        config: OfflinePipelineConfig,
    ) -> "PerVideoPaths":
        root = config.output_dir
        report_dir = root / "reports" / "offline" / video_id
        dense_features = root / "candidate_features" / video_id
        metadata = root / "metadata"
        return cls(
            video_id=video_id,
            output_dir=root,
            dense_metadata=root / "candidates" / f"{video_id}.jsonl",
            dense_images_root=root / "dense_keyframes",
            dense_images_dir=root / "dense_keyframes" / video_id,
            dense_features_dir=dense_features,
            shot_report=report_dir / "shots.json",
            candidate_plan=report_dir / "candidate_plan.jsonl",
            candidate_report=report_dir / "candidate_report.json",
            materialization_report=report_dir / "materialization_report.json",
            dense_embeddings=dense_features / "siglip2.npy",
            dense_embedding_metadata=dense_features / "siglip2_metadata.jsonl",
            dense_embedding_skipped=dense_features / "siglip2_skipped.jsonl",
            dense_embedding_report=dense_features / "siglip2_benchmark.json",
            dense_captions=dense_features / "captions.jsonl",
            dense_caption_report=dense_features / "captions_report.json",
            dense_ocr=dense_features / "ocr.jsonl",
            dense_ocr_report=dense_features / "ocr_report.json",
            dense_objects=dense_features / "objects.jsonl",
            dense_object_report=dense_features / "objects_report.json",
            selected_images_dir=root / "keyframes" / video_id,
            selected_metadata=metadata / f"keyframes_{video_id}.jsonl",
            selected_embeddings=root / "embeddings" / f"{ARTIFACT_TAG}_{video_id}.npy",
            selected_embedding_metadata=(
                metadata / f"{ARTIFACT_TAG}_embeddings_{video_id}.jsonl"
            ),
            selected_captions=metadata / f"captions_{video_id}.jsonl",
            selected_ocr=metadata / f"ocr_{video_id}.jsonl",
            selected_objects=metadata / f"objects_{video_id}.jsonl",
            candidate_ledger=report_dir / "candidate_ledger.jsonl",
            event_ledger=report_dir / "protected_events.jsonl",
            selection_report=report_dir / "selection_report.json",
            validation_report=report_dir / "validation.json",
            completion_report=metadata / f"keyframes_{video_id}_extract_report.json",
            state_manifest=report_dir / "state.json",
        )


@dataclass(frozen=True)
class ShotStageResult:
    info: VideoInfo
    shots: tuple[Shot, ...]
    detector_name: str
    contract_sha256: str


@dataclass(frozen=True)
class CandidateStageResult:
    candidates: tuple[KeyframeCandidate, ...]
    contract_sha256: str


@dataclass(frozen=True)
class MaterializedStageResult:
    records: tuple[dict[str, Any], ...]
    report: Mapping[str, Any]
    contract_sha256: str


@dataclass(frozen=True)
class DenseFeatureArtifacts:
    embeddings: np.ndarray
    embedding_records: tuple[dict[str, Any], ...]
    caption_records: tuple[dict[str, Any], ...]
    ocr_records: tuple[dict[str, Any], ...]
    object_records: tuple[dict[str, Any], ...]
    contract_sha256: str


@dataclass(frozen=True)
class VideoArtifacts:
    video_id: str
    video_path: Path
    paths: PerVideoPaths
    selected_count: int
    dense_candidate_count: int
    skipped: bool = False
    validation: Mapping[str, Any] = field(default_factory=dict)

    @property
    def visual_source(self) -> tuple[Path, Path, str]:
        return (
            self.paths.selected_embeddings,
            self.paths.selected_embedding_metadata,
            self.video_id,
        )

    @property
    def dense_visual_source(self) -> tuple[Path, Path, str]:
        return (
            self.paths.dense_embeddings,
            self.paths.dense_embedding_metadata,
            self.video_id,
        )


@dataclass(frozen=True)
class VideoFailure:
    video_id: str
    video_path: Path | None
    stage: str
    error_type: str
    message: str
    traceback_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path.as_posix() if self.video_path else None,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback_text,
        }


@dataclass(frozen=True)
class DatasetProcessResult:
    requested_videos: tuple[Path, ...]
    successful_videos: tuple[VideoArtifacts, ...]
    failures: tuple[VideoFailure, ...]
    corpus_result: Mapping[str, Any] | None
    corpus_blocked: bool
    corpus_skipped: bool = False

    @property
    def complete(self) -> bool:
        return not self.failures and (
            self.corpus_result is not None or self.corpus_skipped
        )


class OfflineStageError(RuntimeError):
    """A required per-video stage failed and was not silently downgraded."""

    def __init__(self, video_id: str, stage: str, cause: BaseException) -> None:
        self.video_id = video_id
        self.stage = stage
        self.cause = cause
        super().__init__(f"{video_id} failed at {stage}: {type(cause).__name__}: {cause}")


class CorpusIndexError(RuntimeError):
    """Corpus index construction or validation failed."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_change_time(handle: Any) -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class FileBasicInfo(ctypes.Structure):
            _fields_ = (
                ("creation_time", ctypes.c_longlong),
                ("last_access_time", ctypes.c_longlong),
                ("last_write_time", ctypes.c_longlong),
                ("change_time", ctypes.c_longlong),
                ("file_attributes", wintypes.DWORD),
            )

        get_file_information = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        ).GetFileInformationByHandleEx
        get_file_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        get_file_information.restype = wintypes.BOOL
        info = FileBasicInfo()
        succeeded = get_file_information(
            wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno())),
            0,  # FILE_INFO_BY_HANDLE_CLASS.FileBasicInfo
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        return int(info.change_time) if succeeded else None
    except Exception:  # Optional OS optimization; fallback hashes without caching.
        return None


def _source_change_token(path: Path) -> int | None:
    if os.name != "nt":
        return int(path.stat().st_ctime_ns)
    with path.open("rb") as handle:
        return _windows_change_time(handle)


@lru_cache(maxsize=None)
def _cached_source_sha256(
    path: str,
    size_bytes: int,
    mtime_ns: int,
    change_token: int,
    device: int,
    inode: int,
) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        before = os.fstat(handle.fileno())
        before_token = (
            _windows_change_time(handle)
            if os.name == "nt"
            else int(before.st_ctime_ns)
        )
        before_identity = (
            int(before.st_size),
            int(before.st_mtime_ns),
            before_token,
            int(before.st_dev),
            int(before.st_ino),
        )
        expected_identity = (size_bytes, mtime_ns, change_token, device, inode)
        if before_identity != expected_identity:
            raise RuntimeError(f"Source video changed before hashing: {source}")
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
        after_token = (
            _windows_change_time(handle)
            if os.name == "nt"
            else int(after.st_ctime_ns)
        )
        after_identity = (
            int(after.st_size),
            int(after.st_mtime_ns),
            after_token,
            int(after.st_dev),
            int(after.st_ino),
        )
        if after_identity != expected_identity:
            raise RuntimeError(f"Source video changed while hashing: {source}")

    current = source.stat()
    current_token = _source_change_token(source)
    current_identity = (
        int(current.st_size),
        int(current.st_mtime_ns),
        current_token,
        int(current.st_dev),
        int(current.st_ino),
    )
    if current_identity != expected_identity:
        raise RuntimeError(f"Source video changed while it was being hashed: {source}")
    return digest.hexdigest()


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    resolved = path.resolve().as_posix()
    change_token = _source_change_token(path)
    if change_token is None:
        # Some non-NTFS Windows filesystems do not expose ChangeTime.  Correct
        # resume behavior is more important than caching in that uncommon case.
        source_sha256 = _sha256_file(path)
    else:
        source_sha256 = _cached_source_sha256(
            resolved,
            int(stat.st_size),
            int(stat.st_mtime_ns),
            change_token,
            int(stat.st_dev),
            int(stat.st_ino),
        )
    return {
        "path": resolved,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": source_sha256,
    }


def _source_content_identity(signature: Mapping[str, Any]) -> dict[str, object]:
    """Return the location-independent identity used by portable checkpoints."""

    size_bytes = int(signature["size_bytes"])
    sha256 = str(signature["sha256"])
    if size_bytes < 0 or not sha256:
        raise ValueError("source signature has an invalid size or SHA-256")
    return {"size_bytes": size_bytes, "sha256": sha256}


def _source_signatures_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    try:
        return _source_content_identity(left) == _source_content_identity(right)
    except (KeyError, TypeError, ValueError):
        return False


def _assert_source_unchanged(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    current = _file_signature(path)
    if not _source_signatures_match(current, expected):
        raise ValueError(
            "Source video changed during offline preprocessing: "
            f"expected_sha256={expected.get('sha256')} "
            f"actual_sha256={current.get('sha256')}"
        )


def _stage_contract(stage: str, **payload: object) -> dict[str, object]:
    contract = {
        "pipeline": PIPELINE_NAME,
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": stage,
        **payload,
    }
    return {**contract, "contract_sha256": _sha256_value(contract)}


def _contract_matches(report: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    stored = report.get("offline_contract")
    if not isinstance(stored, dict):
        return False
    expected = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if report.get("offline_contract_sha256") != _sha256_value(stored):
        return False
    if contract.get("contract_sha256") != _sha256_value(expected):
        return False

    # Schema 1.0 shot reports included an absolute path and mtime.  Normalize
    # both sides so those reports remain resumable after moving the same bytes
    # between Linux and Windows.  Returning the stored legacy hash from the
    # loader preserves all downstream contract-hash lineage.
    if stored.get("stage") == expected.get("stage") == STAGE_SHOTS:
        stored_source = stored.get("source_video")
        expected_source = expected.get("source_video")
        if not isinstance(stored_source, dict) or not isinstance(
            expected_source,
            dict,
        ):
            return False
        stored = {**stored, "source_video": _source_content_identity(stored_source)}
        expected = {
            **expected,
            "source_video": _source_content_identity(expected_source),
        }
    return stored == expected


def _atomic_write_json(path: Path, value: object) -> None:
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
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
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
            for record in records:
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_save_npy(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        handle.close()
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            records.append(value)
    return records


def _with_contract(
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(report),
        "offline_contract": {
            key: value for key, value in contract.items() if key != "contract_sha256"
        },
        "offline_contract_sha256": contract["contract_sha256"],
    }


def _stage_log(video_id: str, stage: str, status: str, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    LOGGER.info("[%s] %s :: %s%s", status, video_id, stage, suffix)


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_state(
    paths: PerVideoPaths,
    *,
    video_path: Path,
    stage: str,
    status: str,
    detail: Mapping[str, Any] | None = None,
) -> None:
    try:
        state = _read_json(paths.state_manifest) if paths.state_manifest.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        state = {}
    stages = state.get("stages")
    if not isinstance(stages, dict):
        stages = {}
    stages[stage] = {
        "status": status,
        "updated_at_epoch": time.time(),
        **dict(detail or {}),
    }
    _atomic_write_json(
        paths.state_manifest,
        {
            "pipeline": PIPELINE_NAME,
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "video_id": paths.video_id,
            "video_path": video_path.resolve().as_posix(),
            "stages": stages,
        },
    )


def _candidate_to_record(candidate: KeyframeCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "video_id": candidate.video_id,
        "shot_index": candidate.shot_index,
        "frame_index": candidate.frame_index,
        "timestamp": candidate.timestamp_sec,
        "shot_start": candidate.shot_start_sec,
        "shot_end": candidate.shot_end_sec,
        "candidate_reasons": list(candidate.reasons),
        "artifact_role": "dense_candidate_plan",
    }


def _candidate_from_record(record: Mapping[str, Any]) -> KeyframeCandidate:
    return KeyframeCandidate(
        candidate_id=str(record["candidate_id"]),
        video_id=str(record["video_id"]),
        shot_index=int(record["shot_index"]),
        frame_index=int(record["frame_index"]),
        timestamp_sec=float(record["timestamp"]),
        shot_start_sec=float(record["shot_start"]),
        shot_end_sec=float(record["shot_end"]),
        reasons=tuple(str(reason) for reason in record["candidate_reasons"]),
    )


def _candidate_ids(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    ids: list[str] = []
    for index, record in enumerate(records):
        value = record.get("candidate_id")
        if not isinstance(value, str) or not value:
            raise ValueError(f"record {index} has no non-empty candidate_id")
        ids.append(value)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique")
    return tuple(ids)


def _frame_ids(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    ids: list[str] = []
    for index, record in enumerate(records):
        value = record.get("frame_id")
        if not isinstance(value, str) or not value:
            raise ValueError(f"record {index} has no non-empty frame_id")
        ids.append(value)
    if len(set(ids)) != len(ids):
        raise ValueError("frame_id values must be unique")
    return tuple(ids)


def _images_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        identity = str(record.get("frame_id") or record.get("candidate_id") or "")
        path = _resolve_record_image(record)
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_modality_records(
    *,
    label: str,
    records: Sequence[Mapping[str, Any]],
    expected_candidates: Sequence[Mapping[str, Any]],
    video_id: str,
    require_success: bool = True,
    required_nonempty_field: str | None = None,
) -> None:
    expected_ids = set(_candidate_ids(expected_candidates))
    expected_by_id = {
        str(record["candidate_id"]): record for record in expected_candidates
    }
    actual_ids = _candidate_ids(records)
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        missing = sorted(expected_ids - set(actual_ids))
        extra = sorted(set(actual_ids) - expected_ids)
        raise ValueError(
            f"{label} candidate alignment mismatch: missing={missing}; extra={extra}"
        )
    failures: list[str] = []
    empty_values: list[str] = []
    for index, record in enumerate(records):
        if str(record.get("video_id") or "") != video_id:
            raise ValueError(f"{label} record {index} has the wrong video_id")
        candidate_id = str(record["candidate_id"])
        expected = expected_by_id[candidate_id]
        if str(record.get("frame_id") or "") != str(expected.get("frame_id") or ""):
            raise ValueError(
                f"{label} record {candidate_id} has the wrong frame_id"
            )
        if record.get("frame_index") is not None and expected.get("frame_index") is not None:
            if int(record["frame_index"]) != int(expected["frame_index"]):
                raise ValueError(
                    f"{label} record {candidate_id} has the wrong frame_index"
                )
        if require_success and record.get("status") != "success":
            failures.append(f"{record.get('candidate_id')}:{record.get('status')}")
        if required_nonempty_field is not None and not str(
            record.get(required_nonempty_field) or ""
        ).strip():
            empty_values.append(candidate_id)
    if failures:
        raise ValueError(f"{label} contains unsuccessful records: {failures[:20]}")
    if empty_values:
        raise ValueError(
            f"{label} contains empty {required_nonempty_field} values: "
            f"{empty_values[:20]}"
        )


def _safe_json_error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _shot_contract(
    video_path: Path,
    config: OfflinePipelineConfig,
    *,
    source_signature: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    source = dict(source_signature or _file_signature(video_path))
    return _stage_contract(
        STAGE_SHOTS,
        source_video=_source_content_identity(source),
        shot_threshold=config.shot_threshold,
        shot_device=config.shot_device,
    )


def _load_shot_stage(
    path: Path,
    *,
    contract: Mapping[str, Any],
    expected_video_id: str,
) -> ShotStageResult:
    report = _read_json(path)
    if not _contract_matches(report, contract):
        raise ValueError("shot detection contract changed")
    raw_info = report.get("video_info")
    raw_shots = report.get("shots")
    detector_name = report.get("detector_name")
    if not isinstance(raw_info, dict) or not isinstance(raw_shots, list):
        raise ValueError("shot report is missing video_info or shots")
    if not isinstance(detector_name, str) or not detector_name:
        raise ValueError("shot report has no detector_name")
    info = VideoInfo(
        video_id=str(raw_info["video_id"]),
        fps=float(raw_info["fps"]),
        frame_count=int(raw_info["frame_count"]),
    )
    if info.video_id != expected_video_id or info.fps <= 0 or info.frame_count <= 0:
        raise ValueError("shot report has invalid video identity or dimensions")
    shots = tuple(
        Shot(
            shot_index=int(raw["shot_index"]),
            start_frame=int(raw["start_frame"]),
            end_frame=int(raw["end_frame"]),
            fps=float(raw["fps"]),
        )
        for raw in raw_shots
        if isinstance(raw, dict)
    )
    if len(shots) != len(raw_shots) or not shots:
        raise ValueError("shot report must contain at least one valid shot")
    # The existing candidate generator is also the strongest shot-range validator.
    generate_keyframe_candidates(
        info.video_id,
        shots,
        info.fps,
        interval_sec=DEFAULT_INTERVAL_SEC,
        frame_count=info.frame_count,
    )
    return ShotStageResult(
        info=info,
        shots=shots,
        detector_name=detector_name,
        # Preserve a valid legacy hash so candidate/materialization contracts
        # created from it keep matching after a cross-platform move.
        contract_sha256=str(report["offline_contract_sha256"]),
    )


def _load_or_run_shot_detection(
    video_path: Path,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
    *,
    source_signature: Mapping[str, Any] | None = None,
) -> ShotStageResult:
    contract = _shot_contract(
        video_path,
        config,
        source_signature=source_signature,
    )
    if config.resume and not config.force:
        try:
            result = _load_shot_stage(
                paths.shot_report,
                contract=contract,
                expected_video_id=paths.video_id,
            )
        except CHECKPOINT_INVALID_ERRORS as exc:
            _stage_log(paths.video_id, STAGE_SHOTS, "RUN", f"checkpoint invalid: {exc}")
        else:
            _stage_log(paths.video_id, STAGE_SHOTS, "SKIP", f"shots={len(result.shots)}")
            return result
    else:
        _stage_log(paths.video_id, STAGE_SHOTS, "RUN")

    info = read_video_info(video_path)
    if info.video_id != paths.video_id:
        raise ValueError(f"video_id mismatch: {info.video_id} != {paths.video_id}")
    shots, detector_name = detect_shots_transnetv2(
        video_path,
        info,
        threshold=config.shot_threshold,
        device=config.shot_device,
    )
    if not shots:
        raise RuntimeError("shot detection produced no shots")
    report = _with_contract(
        {
            "status": "passed",
            "video_id": info.video_id,
            "video_path": video_path.resolve().as_posix(),
            "video_info": {
                "video_id": info.video_id,
                "fps": info.fps,
                "frame_count": info.frame_count,
                "duration": info.duration,
            },
            "detector_name": detector_name,
            "shot_count": len(shots),
            "shots": [
                {
                    **asdict(shot),
                    "start_sec": shot.start_sec,
                    "end_sec": shot.end_sec,
                    "duration": shot.duration,
                }
                for shot in shots
            ],
        },
        contract,
    )
    _atomic_write_json(paths.shot_report, report)
    _write_state(
        paths,
        video_path=video_path,
        stage=STAGE_SHOTS,
        status="passed",
        detail={"shot_count": len(shots), "contract_sha256": contract["contract_sha256"]},
    )
    return _load_shot_stage(
        paths.shot_report,
        contract=contract,
        expected_video_id=paths.video_id,
    )


def _candidate_contract(
    shot_stage: ShotStageResult,
    config: OfflinePipelineConfig,
) -> dict[str, object]:
    return _stage_contract(
        STAGE_CANDIDATES,
        shot_contract_sha256=shot_stage.contract_sha256,
        dense_interval_sec=config.dense_interval_sec,
        boundary_guard_sec=config.boundary_guard_sec,
        tiny_shot_max_sec=config.tiny_shot_max_sec,
        include_video_endpoints=config.include_video_endpoints,
    )


def _generate_candidates(
    shot_stage: ShotStageResult,
    config: OfflinePipelineConfig,
) -> tuple[KeyframeCandidate, ...]:
    return tuple(
        generate_keyframe_candidates(
            shot_stage.info.video_id,
            shot_stage.shots,
            shot_stage.info.fps,
            interval_sec=config.dense_interval_sec,
            boundary_guard_sec=config.boundary_guard_sec,
            tiny_shot_max_sec=config.tiny_shot_max_sec,
            frame_count=shot_stage.info.frame_count,
            include_video_endpoints=config.include_video_endpoints,
        )
    )


def _load_candidate_stage(
    paths: PerVideoPaths,
    *,
    shot_stage: ShotStageResult,
    config: OfflinePipelineConfig,
    contract: Mapping[str, Any],
) -> CandidateStageResult:
    report = _read_json(paths.candidate_report)
    if not _contract_matches(report, contract):
        raise ValueError("candidate-generation contract changed")
    if report.get("status") != "passed":
        raise ValueError("candidate report is not passed")
    if report.get("candidate_plan_sha256") != _sha256_file(paths.candidate_plan):
        raise ValueError("candidate plan checksum changed")
    records = _read_jsonl(paths.candidate_plan)
    candidates = tuple(_candidate_from_record(record) for record in records)
    expected = _generate_candidates(shot_stage, config)
    if candidates != expected:
        raise ValueError("candidate plan no longer matches deterministic generation")
    if report.get("candidate_count") != len(candidates) or not candidates:
        raise ValueError("candidate report count mismatch")
    return CandidateStageResult(
        candidates=candidates,
        contract_sha256=str(contract["contract_sha256"]),
    )


def _load_or_run_dense_candidate_generation(
    video_path: Path,
    shot_stage: ShotStageResult,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
) -> CandidateStageResult:
    contract = _candidate_contract(shot_stage, config)
    if config.resume and not config.force:
        try:
            result = _load_candidate_stage(
                paths,
                shot_stage=shot_stage,
                config=config,
                contract=contract,
            )
        except CHECKPOINT_INVALID_ERRORS as exc:
            _stage_log(paths.video_id, STAGE_CANDIDATES, "RUN", f"checkpoint invalid: {exc}")
        else:
            _stage_log(
                paths.video_id,
                STAGE_CANDIDATES,
                "SKIP",
                f"candidates={len(result.candidates)}",
            )
            return result
    else:
        _stage_log(paths.video_id, STAGE_CANDIDATES, "RUN")

    candidates = _generate_candidates(shot_stage, config)
    if not candidates:
        raise RuntimeError("dense candidate generation produced an empty pool")
    _atomic_write_jsonl(paths.candidate_plan, map(_candidate_to_record, candidates))
    report = _with_contract(
        {
            "status": "passed",
            "video_id": paths.video_id,
            "candidate_count": len(candidates),
            "candidate_plan_path": paths.candidate_plan.as_posix(),
            "candidate_plan_sha256": _sha256_file(paths.candidate_plan),
            "first_frame_index": candidates[0].frame_index,
            "last_frame_index": candidates[-1].frame_index,
        },
        contract,
    )
    _atomic_write_json(paths.candidate_report, report)
    _write_state(
        paths,
        video_path=video_path,
        stage=STAGE_CANDIDATES,
        status="passed",
        detail={
            "candidate_count": len(candidates),
            "contract_sha256": contract["contract_sha256"],
        },
    )
    return _load_candidate_stage(
        paths,
        shot_stage=shot_stage,
        config=config,
        contract=contract,
    )


def _materialization_contract(
    paths: PerVideoPaths,
    candidate_stage: CandidateStageResult,
    config: OfflinePipelineConfig,
) -> dict[str, object]:
    return _stage_contract(
        STAGE_MATERIALIZATION,
        candidate_contract_sha256=candidate_stage.contract_sha256,
        candidate_plan_sha256=_sha256_file(paths.candidate_plan),
        jpeg_quality=config.jpeg_quality,
        phash_threshold=config.phash_threshold,
        phash_window_sec=config.phash_window_sec,
    )


def _validate_materialized_stage(
    paths: PerVideoPaths,
    *,
    candidate_stage: CandidateStageResult,
    config: OfflinePipelineConfig,
    contract: Mapping[str, Any],
) -> MaterializedStageResult:
    report = _read_json(paths.materialization_report)
    if not _contract_matches(report, contract):
        raise ValueError("materialization contract changed")
    if (
        report.get("status") != "satisfied"
        or report.get("constraints_satisfied") is not True
        or report.get("selection_applied") is not False
        or report.get("materialization_mode") != MATERIALIZATION_MODE
    ):
        raise ValueError("dense candidate materialization is incomplete")
    records = _read_jsonl(paths.dense_metadata)
    expected_ids = tuple(candidate.candidate_id for candidate in candidate_stage.candidates)
    actual_ids = _candidate_ids(records)
    if actual_ids != expected_ids:
        raise ValueError("materialized candidate order/identity mismatch")
    if any(str(record.get("video_id") or "") != paths.video_id for record in records):
        raise ValueError("materialized candidate metadata has the wrong video_id")
    if report.get("candidate_count") != len(records) or report.get("keyframe_count") != len(records):
        raise ValueError("materialization count mismatch")
    if report.get("dense_metadata_sha256") != _sha256_file(paths.dense_metadata):
        raise ValueError("dense metadata checksum changed")
    if any(record.get("artifact_role") != "dense_candidate" for record in records):
        raise ValueError("materialized metadata must be artifact_role=dense_candidate")
    image_validation = validate_records(
        records,
        min_width=config.min_image_width,
        min_height=config.min_image_height,
    )
    if image_validation.get("valid") is not True:
        raise ValueError(
            f"materialized candidate image validation failed: {image_validation.get('errors')}"
        )
    if report.get("dense_images_sha256") != _images_sha256(records):
        raise ValueError("materialized candidate image content changed")
    return MaterializedStageResult(
        records=tuple(records),
        report=report,
        contract_sha256=str(contract["contract_sha256"]),
    )


def _load_or_run_dense_materialization(
    video_path: Path,
    shot_stage: ShotStageResult,
    candidate_stage: CandidateStageResult,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
) -> MaterializedStageResult:
    contract = _materialization_contract(paths, candidate_stage, config)
    if config.resume and not config.force:
        try:
            result = _validate_materialized_stage(
                paths,
                candidate_stage=candidate_stage,
                config=config,
                contract=contract,
            )
        except CHECKPOINT_INVALID_ERRORS as exc:
            _stage_log(paths.video_id, STAGE_MATERIALIZATION, "RUN", f"checkpoint invalid: {exc}")
        else:
            _stage_log(
                paths.video_id,
                STAGE_MATERIALIZATION,
                "SKIP",
                f"materialized={len(result.records)}",
            )
            return result
    else:
        _stage_log(paths.video_id, STAGE_MATERIALIZATION, "RUN")

    report = materialize_generated_keyframe_candidates_for_video(
        video_path=video_path,
        info=shot_stage.info,
        shots=shot_stage.shots,
        detector_name=shot_stage.detector_name,
        candidates=candidate_stage.candidates,
        output_dir=paths.dense_images_root,
        metadata_path=paths.dense_metadata,
        report_path=paths.materialization_report,
        phash_threshold=config.phash_threshold,
        phash_window_sec=config.phash_window_sec,
        jpeg_quality=config.jpeg_quality,
        shot_threshold=config.shot_threshold,
        shot_device=config.shot_device,
        candidate_interval_sec=config.dense_interval_sec,
        boundary_guard_sec=config.boundary_guard_sec,
        tiny_shot_max_sec=config.tiny_shot_max_sec,
        include_video_endpoints=config.include_video_endpoints,
    )
    records = [
        {
            **record,
            "artifact_role": "dense_candidate",
            "offline_candidate_contract_sha256": candidate_stage.contract_sha256,
        }
        for record in _read_jsonl(paths.dense_metadata)
    ]
    _atomic_write_jsonl(paths.dense_metadata, records)
    report = _with_contract(
        {
            **dict(report),
            "dense_metadata_sha256": _sha256_file(paths.dense_metadata),
            "dense_candidate_ids_sha256": _sha256_value(_candidate_ids(records)),
            "dense_images_sha256": _images_sha256(records),
        },
        contract,
    )
    _atomic_write_json(paths.materialization_report, report)
    result = _validate_materialized_stage(
        paths,
        candidate_stage=candidate_stage,
        config=config,
        contract=contract,
    )
    _write_state(
        paths,
        video_path=video_path,
        stage=STAGE_MATERIALIZATION,
        status="passed",
        detail={
            "candidate_count": len(result.records),
            "contract_sha256": contract["contract_sha256"],
        },
    )
    return result


def _siglip_contract(
    paths: PerVideoPaths,
    materialized: MaterializedStageResult,
    config: OfflinePipelineConfig,
) -> dict[str, object]:
    return _stage_contract(
        STAGE_SIGLIP,
        materialization_contract_sha256=materialized.contract_sha256,
        dense_metadata_sha256=_sha256_file(paths.dense_metadata),
        dense_images_sha256=str(materialized.report.get("dense_images_sha256")),
        model_name=config.siglip_model_name,
        requested_model_revision=config.siglip_model_revision,
        use_autocast=config.siglip_use_autocast,
    )


def _validate_siglip_stage(
    paths: PerVideoPaths,
    *,
    materialized: MaterializedStageResult,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    report = _read_json(paths.dense_embedding_report)
    if not _contract_matches(report, contract) or report.get("status") != "passed":
        raise ValueError("SigLIP2 feature contract is not current")
    embeddings = np.load(paths.dense_embeddings, allow_pickle=False)
    records = _read_jsonl(paths.dense_embedding_metadata)
    skipped = _read_jsonl(paths.dense_embedding_skipped)
    if skipped:
        raise ValueError("SigLIP2 skipped records are not allowed in the canonical path")
    validate_embedding_artifacts(embeddings, records)
    expected_ids = _candidate_ids(materialized.records)
    actual_ids = _candidate_ids(records)
    if actual_ids != expected_ids or embeddings.shape[0] != len(expected_ids):
        raise ValueError("SigLIP2 must contain exactly one aligned row per dense candidate")
    if _frame_ids(records) != _frame_ids(materialized.records):
        raise ValueError("SigLIP2 frame IDs do not align with dense candidates")
    if any(str(record.get("video_id") or "") != paths.video_id for record in records):
        raise ValueError("SigLIP2 metadata has the wrong video_id")
    if report.get("embedding_sha256") != _sha256_file(paths.dense_embeddings):
        raise ValueError("SigLIP2 embedding checksum changed")
    if report.get("embedding_metadata_sha256") != _sha256_file(
        paths.dense_embedding_metadata
    ):
        raise ValueError("SigLIP2 metadata checksum changed")
    return embeddings, tuple(records)


def _load_or_run_siglip_features(
    video_path: Path,
    materialized: MaterializedStageResult,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...], str]:
    contract = _siglip_contract(paths, materialized, config)
    if config.resume and not config.force:
        try:
            embeddings, records = _validate_siglip_stage(
                paths,
                materialized=materialized,
                contract=contract,
            )
        except CHECKPOINT_INVALID_ERRORS as exc:
            _stage_log(paths.video_id, STAGE_SIGLIP, "RUN", f"checkpoint invalid: {exc}")
        else:
            _stage_log(paths.video_id, STAGE_SIGLIP, "SKIP", f"rows={len(records)}")
            return embeddings, records, str(contract["contract_sha256"])
    else:
        _stage_log(paths.video_id, STAGE_SIGLIP, "RUN")

    try:
        embeddings, records, skipped, benchmark = encode_keyframes(
            records=[dict(record) for record in materialized.records],
            model_name=config.siglip_model_name,
            model_revision=config.siglip_model_revision,
            batch_size=config.siglip_batch_size,
            num_workers=config.siglip_num_workers,
            device=config.device,
            use_autocast=config.siglip_use_autocast,
            model_cache_dir=config.siglip_model_cache_dir,
            prefetch_factor=config.siglip_prefetch_factor,
        )
        _atomic_save_npy(paths.dense_embeddings, embeddings)
        _atomic_write_jsonl(paths.dense_embedding_metadata, records)
        _atomic_write_jsonl(paths.dense_embedding_skipped, skipped)
        status = "passed" if not skipped and len(records) == len(materialized.records) else "failed"
        benchmark = _with_contract(
            {
                **dict(benchmark),
                "status": status,
                "embedding_sha256": _sha256_file(paths.dense_embeddings),
                "embedding_metadata_sha256": _sha256_file(
                    paths.dense_embedding_metadata
                ),
                "skipped_sha256": _sha256_file(paths.dense_embedding_skipped),
            },
            contract,
        )
        _atomic_write_json(paths.dense_embedding_report, benchmark)
        if status != "passed":
            raise RuntimeError(
                "SigLIP2 did not encode the complete dense pool: "
                f"encoded={len(records)} expected={len(materialized.records)} "
                f"skipped={len(skipped)}"
            )
        embeddings, loaded_records = _validate_siglip_stage(
            paths,
            materialized=materialized,
            contract=contract,
        )
    finally:
        _release_accelerator_memory()
    _write_state(
        paths,
        video_path=video_path,
        stage=STAGE_SIGLIP,
        status="passed",
        detail={"row_count": len(loaded_records), "contract_sha256": contract["contract_sha256"]},
    )
    return embeddings, loaded_records, str(contract["contract_sha256"])


def _modality_contract(
    stage: str,
    paths: PerVideoPaths,
    materialized: MaterializedStageResult,
    **model_contract: object,
) -> dict[str, object]:
    return _stage_contract(
        stage,
        materialization_contract_sha256=materialized.contract_sha256,
        dense_metadata_sha256=_sha256_file(paths.dense_metadata),
        dense_images_sha256=str(materialized.report.get("dense_images_sha256")),
        **model_contract,
    )


def _validate_modality_stage(
    *,
    label: str,
    output_path: Path,
    report_path: Path,
    materialized: MaterializedStageResult,
    video_id: str,
    contract: Mapping[str, Any],
    required_nonempty_field: str | None = None,
) -> tuple[dict[str, Any], ...]:
    report = _read_json(report_path)
    if not _contract_matches(report, contract) or report.get("offline_status") != "passed":
        raise ValueError(f"{label} feature contract is not current")
    records = _read_jsonl(output_path)
    _validate_modality_records(
        label=label,
        records=records,
        expected_candidates=materialized.records,
        video_id=video_id,
        required_nonempty_field=required_nonempty_field,
    )
    if report.get("output_sha256") != _sha256_file(output_path):
        raise ValueError(f"{label} output checksum changed")
    return tuple(records)


def _can_append_modality_resume(
    output_path: Path,
    *,
    expected_candidates: Sequence[Mapping[str, Any]],
    video_id: str,
) -> bool:
    if not output_path.is_file():
        return True
    try:
        records = _read_jsonl(output_path)
        expected_ids = set(_candidate_ids(expected_candidates))
        expected_by_id = {
            str(record["candidate_id"]): record for record in expected_candidates
        }
        actual_ids = _candidate_ids(records)
    except (OSError, KeyError, TypeError, ValueError):
        return False
    if not set(actual_ids).issubset(expected_ids):
        return False
    return all(
        str(record.get("video_id") or "") == video_id
        and str(record.get("frame_id") or "")
        == str(expected_by_id[str(record["candidate_id"])].get("frame_id") or "")
        for record in records
    )


def _can_resume_modality_stage(
    output_path: Path,
    report_path: Path,
    *,
    contract: Mapping[str, Any],
    expected_candidates: Sequence[Mapping[str, Any]],
    video_id: str,
) -> bool:
    if not output_path.is_file():
        return True
    try:
        report = _read_json(report_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not _contract_matches(report, contract):
        return False
    # A matching ``running`` sidecar proves this is an interrupted appendable
    # run.  If a previously passed artifact reached this branch, its validator
    # already found corruption; recompute instead of blessing modified rows.
    if report.get("offline_status") != "running":
        return False
    if report.get("append_allowed") is not True:
        return False
    return _can_append_modality_resume(
        output_path,
        expected_candidates=expected_candidates,
        video_id=video_id,
    )


def _mark_modality_running(
    report_path: Path,
    contract: Mapping[str, Any],
    *,
    append_allowed: bool,
) -> None:
    _atomic_write_json(
        report_path,
        _with_contract(
            {
                "offline_status": "running",
                "append_allowed": append_allowed,
            },
            contract,
        ),
    )


def _mark_modality_failed(
    report_path: Path,
    contract: Mapping[str, Any],
    exc: Exception,
) -> None:
    _atomic_write_json(
        report_path,
        _with_contract(
            {
                "offline_status": "failed",
                "error": _safe_json_error(exc),
            },
            contract,
        ),
    )


def _finalize_modality_report(
    *,
    output_path: Path,
    report_path: Path,
    generated_report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    report = _with_contract(
        {
            **dict(generated_report),
            "offline_status": "passed",
            "total_validated_count": len(records),
            "output_sha256": _sha256_file(output_path),
        },
        contract,
    )
    _atomic_write_json(report_path, report)


def _run_ocr_file_isolated(
    *,
    metadata_path: Path,
    output_path: Path,
    report_path: Path,
    config: OfflinePipelineConfig,
    overwrite: bool,
) -> dict[str, Any]:
    """Run PaddleOCR outside the Torch process to avoid Windows cuDNN clashes."""

    command = [
        sys.executable,
        "-m",
        "backend.app.services.ingestion.run_ocr",
        "--metadata-path",
        str(metadata_path.resolve()),
        "--output-path",
        str(output_path.resolve()),
        "--report-path",
        str(report_path.resolve()),
        "--device",
        config.device,
        "--batch-size",
        str(config.ocr_batch_size),
        "--conf-threshold",
        str(config.ocr_conf_threshold),
        "--detection-model",
        config.ocr_detection_model,
        "--recognition-model",
        config.ocr_recognition_model,
        "--model-revision",
        config.ocr_model_revision,
        "--model-cache-dir",
        str(config.ocr_model_cache_dir.resolve()),
        "--isolate-paddle-runtime",
    ]
    if overwrite:
        command.append("--overwrite")

    project_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(command, cwd=project_root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Isolated PaddleOCR worker failed with exit code "
            f"{completed.returncode}."
        )
    return _read_json(report_path)


def _load_or_run_ocr_features(
    video_path: Path,
    materialized: MaterializedStageResult,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
) -> tuple[tuple[dict[str, Any], ...], str]:
    contract = _modality_contract(
        STAGE_OCR,
        paths,
        materialized,
        detection_model=config.ocr_detection_model,
        recognition_model=config.ocr_recognition_model,
        model_revision=config.ocr_model_revision,
        confidence_threshold=config.ocr_conf_threshold,
    )
    if config.resume and not config.force:
        try:
            records = _validate_modality_stage(
                label="OCR",
                output_path=paths.dense_ocr,
                report_path=paths.dense_ocr_report,
                materialized=materialized,
                video_id=paths.video_id,
                contract=contract,
            )
        except CHECKPOINT_INVALID_ERRORS as exc:
            _stage_log(paths.video_id, STAGE_OCR, "RUN", f"checkpoint invalid: {exc}")
        else:
            _stage_log(paths.video_id, STAGE_OCR, "SKIP", f"rows={len(records)}")
            return records, str(contract["contract_sha256"])
    else:
        _stage_log(paths.video_id, STAGE_OCR, "RUN")

    overwrite = config.force or not config.resume or not _can_resume_modality_stage(
        paths.dense_ocr,
        paths.dense_ocr_report,
        contract=contract,
        expected_candidates=materialized.records,
        video_id=paths.video_id,
    )
    _mark_modality_running(
        paths.dense_ocr_report,
        contract,
        append_allowed=not overwrite,
    )
    try:
        generated_report = _run_ocr_file_isolated(
            metadata_path=paths.dense_metadata,
            output_path=paths.dense_ocr,
            report_path=paths.dense_ocr_report,
            config=config,
            overwrite=overwrite,
        )
        records = _read_jsonl(paths.dense_ocr)
        _validate_modality_records(
            label="OCR",
            records=records,
            expected_candidates=materialized.records,
            video_id=paths.video_id,
        )
        _finalize_modality_report(
            output_path=paths.dense_ocr,
            report_path=paths.dense_ocr_report,
            generated_report=generated_report,
            records=records,
            contract=contract,
        )
    except Exception as exc:
        _mark_modality_failed(paths.dense_ocr_report, contract, exc)
        raise
    finally:
        _release_accelerator_memory()
    validated = _validate_modality_stage(
        label="OCR",
        output_path=paths.dense_ocr,
        report_path=paths.dense_ocr_report,
        materialized=materialized,
        video_id=paths.video_id,
        contract=contract,
    )
    _write_state(
        paths,
        video_path=video_path,
        stage=STAGE_OCR,
        status="passed",
        detail={"row_count": len(validated), "contract_sha256": contract["contract_sha256"]},
    )
    return validated, str(contract["contract_sha256"])


def _load_or_run_object_features(
    video_path: Path,
    materialized: MaterializedStageResult,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
) -> tuple[tuple[dict[str, Any], ...], str]:
    contract = _modality_contract(
        STAGE_OBJECTS,
        paths,
        materialized,
        model_name=config.object_model_name,
        model_revision=config.object_model_revision,
        confidence_threshold=config.object_conf_threshold,
        iou_threshold=config.object_iou_threshold,
    )
    if config.resume and not config.force:
        try:
            records = _validate_modality_stage(
                label="object",
                output_path=paths.dense_objects,
                report_path=paths.dense_object_report,
                materialized=materialized,
                video_id=paths.video_id,
                contract=contract,
            )
        except CHECKPOINT_INVALID_ERRORS as exc:
            _stage_log(paths.video_id, STAGE_OBJECTS, "RUN", f"checkpoint invalid: {exc}")
        else:
            _stage_log(paths.video_id, STAGE_OBJECTS, "SKIP", f"rows={len(records)}")
            return records, str(contract["contract_sha256"])
    else:
        _stage_log(paths.video_id, STAGE_OBJECTS, "RUN")

    overwrite = config.force or not config.resume or not _can_resume_modality_stage(
        paths.dense_objects,
        paths.dense_object_report,
        contract=contract,
        expected_candidates=materialized.records,
        video_id=paths.video_id,
    )
    _mark_modality_running(
        paths.dense_object_report,
        contract,
        append_allowed=not overwrite,
    )
    try:
        generated_report = run_object_file(
            paths.dense_metadata,
            output_path=paths.dense_objects,
            report_path=paths.dense_object_report,
            device=config.device,
            batch_size=config.object_batch_size,
            conf_threshold=config.object_conf_threshold,
            iou_threshold=config.object_iou_threshold,
            overwrite=overwrite,
            model_name=config.object_model_name,
            revision=config.object_model_revision,
            model_cache_dir=config.object_model_cache_dir,
        )
        records = _read_jsonl(paths.dense_objects)
        _validate_modality_records(
            label="object",
            records=records,
            expected_candidates=materialized.records,
            video_id=paths.video_id,
        )
        _finalize_modality_report(
            output_path=paths.dense_objects,
            report_path=paths.dense_object_report,
            generated_report=generated_report,
            records=records,
            contract=contract,
        )
    except Exception as exc:
        _mark_modality_failed(paths.dense_object_report, contract, exc)
        raise
    finally:
        _release_accelerator_memory()
    validated = _validate_modality_stage(
        label="object",
        output_path=paths.dense_objects,
        report_path=paths.dense_object_report,
        materialized=materialized,
        video_id=paths.video_id,
        contract=contract,
    )
    _write_state(
        paths,
        video_path=video_path,
        stage=STAGE_OBJECTS,
        status="passed",
        detail={"row_count": len(validated), "contract_sha256": contract["contract_sha256"]},
    )
    return validated, str(contract["contract_sha256"])


def _load_or_run_caption_features(
    video_path: Path,
    materialized: MaterializedStageResult,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
) -> tuple[tuple[dict[str, Any], ...], str]:
    contract = _modality_contract(
        STAGE_CAPTIONS,
        paths,
        materialized,
        model_name=config.caption_model_name,
        requested_model_revision=config.caption_model_revision,
        max_new_tokens=config.caption_max_new_tokens,
        dtype=config.caption_dtype,
        quantization=config.caption_quantization,
        task_prompt=config.caption_task_prompt,
    )
    if config.resume and not config.force:
        try:
            records = _validate_modality_stage(
                label="caption",
                output_path=paths.dense_captions,
                report_path=paths.dense_caption_report,
                materialized=materialized,
                video_id=paths.video_id,
                contract=contract,
                required_nonempty_field="caption",
            )
        except CHECKPOINT_INVALID_ERRORS as exc:
            _stage_log(paths.video_id, STAGE_CAPTIONS, "RUN", f"checkpoint invalid: {exc}")
        else:
            _stage_log(paths.video_id, STAGE_CAPTIONS, "SKIP", f"rows={len(records)}")
            return records, str(contract["contract_sha256"])
    else:
        _stage_log(paths.video_id, STAGE_CAPTIONS, "RUN")

    overwrite = config.force or not config.resume or not _can_resume_modality_stage(
        paths.dense_captions,
        paths.dense_caption_report,
        contract=contract,
        expected_candidates=materialized.records,
        video_id=paths.video_id,
    )
    _mark_modality_running(
        paths.dense_caption_report,
        contract,
        append_allowed=not overwrite,
    )
    try:
        generated_report = run_caption_file(
            paths.dense_metadata,
            output_path=paths.dense_captions,
            report_path=paths.dense_caption_report,
            device=config.device,
            batch_size=config.caption_batch_size,
            overwrite=overwrite,
            model_name=config.caption_model_name,
            revision=config.caption_model_revision,
            max_new_tokens=config.caption_max_new_tokens,
            dtype=config.caption_dtype,
            quantization=config.caption_quantization,
            task_prompt=config.caption_task_prompt,
            model_cache_dir=config.caption_model_cache_dir,
        )
        records = _read_jsonl(paths.dense_captions)
        # Caption is soft inside the selector, but canonical default requires it
        # for every dense candidate, so the orchestrator closes that gap here.
        _validate_modality_records(
            label="caption",
            records=records,
            expected_candidates=materialized.records,
            video_id=paths.video_id,
            required_nonempty_field="caption",
        )
        _finalize_modality_report(
            output_path=paths.dense_captions,
            report_path=paths.dense_caption_report,
            generated_report=generated_report,
            records=records,
            contract=contract,
        )
    except Exception as exc:
        _mark_modality_failed(paths.dense_caption_report, contract, exc)
        raise
    finally:
        _release_accelerator_memory()
    validated = _validate_modality_stage(
        label="caption",
        output_path=paths.dense_captions,
        report_path=paths.dense_caption_report,
        materialized=materialized,
        video_id=paths.video_id,
        contract=contract,
        required_nonempty_field="caption",
    )
    _write_state(
        paths,
        video_path=video_path,
        stage=STAGE_CAPTIONS,
        status="passed",
        detail={"row_count": len(validated), "contract_sha256": contract["contract_sha256"]},
    )
    return validated, str(contract["contract_sha256"])


def _extract_all_dense_features(
    video_path: Path,
    materialized: MaterializedStageResult,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
) -> DenseFeatureArtifacts:
    try:
        embeddings, embedding_records, siglip_contract = _load_or_run_siglip_features(
            video_path,
            materialized,
            config,
            paths,
        )
    except Exception as exc:
        raise OfflineStageError(paths.video_id, STAGE_SIGLIP, exc) from exc
    try:
        ocr_records, ocr_contract = _load_or_run_ocr_features(
            video_path,
            materialized,
            config,
            paths,
        )
    except Exception as exc:
        raise OfflineStageError(paths.video_id, STAGE_OCR, exc) from exc
    try:
        object_records, object_contract = _load_or_run_object_features(
            video_path,
            materialized,
            config,
            paths,
        )
    except Exception as exc:
        raise OfflineStageError(paths.video_id, STAGE_OBJECTS, exc) from exc
    try:
        caption_records, caption_contract = _load_or_run_caption_features(
            video_path,
            materialized,
            config,
            paths,
        )
    except Exception as exc:
        raise OfflineStageError(paths.video_id, STAGE_CAPTIONS, exc) from exc
    expected_count = len(materialized.records)
    if not (
        embeddings.shape[0]
        == len(embedding_records)
        == len(ocr_records)
        == len(object_records)
        == len(caption_records)
        == expected_count
    ):
        raise RuntimeError("dense multimodal feature bundle is not fully aligned")
    contract_sha256 = _sha256_value(
        {
            "siglip": siglip_contract,
            "ocr": ocr_contract,
            "objects": object_contract,
            "captions": caption_contract,
        }
    )
    return DenseFeatureArtifacts(
        embeddings=embeddings,
        embedding_records=embedding_records,
        caption_records=caption_records,
        ocr_records=ocr_records,
        object_records=object_records,
        contract_sha256=contract_sha256,
    )


def _selection_contract(
    paths: PerVideoPaths,
    materialized: MaterializedStageResult,
    features: DenseFeatureArtifacts,
    config: OfflinePipelineConfig,
) -> dict[str, object]:
    selection_config = asdict(config.selection_config())
    return _stage_contract(
        STAGE_SELECTION,
        materialization_contract_sha256=materialized.contract_sha256,
        feature_contract_sha256=features.contract_sha256,
        dense_embeddings_sha256=_sha256_file(paths.dense_embeddings),
        dense_embedding_metadata_sha256=_sha256_file(paths.dense_embedding_metadata),
        dense_captions_sha256=_sha256_file(paths.dense_captions),
        dense_ocr_sha256=_sha256_file(paths.dense_ocr),
        dense_objects_sha256=_sha256_file(paths.dense_objects),
        selection_config=selection_config,
    )


def _run_multimodal_selection(
    shot_stage: ShotStageResult,
    materialized: MaterializedStageResult,
    features: DenseFeatureArtifacts,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
) -> tuple[MultimodalKeyframePipelineResult, dict[str, object]]:
    contract = _selection_contract(paths, materialized, features, config)
    _stage_log(
        paths.video_id,
        STAGE_SELECTION,
        "RUN",
        f"full_dense_pool={len(materialized.records)}",
    )
    expected_ids = _candidate_ids(materialized.records)
    if _candidate_ids(features.embedding_records) != expected_ids:
        raise ValueError("selector input embedding order is not aligned with dense candidates")
    _validate_modality_records(
        label="OCR",
        records=features.ocr_records,
        expected_candidates=materialized.records,
        video_id=paths.video_id,
    )
    _validate_modality_records(
        label="object",
        records=features.object_records,
        expected_candidates=materialized.records,
        video_id=paths.video_id,
    )
    _validate_modality_records(
        label="caption",
        records=features.caption_records,
        expected_candidates=materialized.records,
        video_id=paths.video_id,
        required_nonempty_field="caption",
    )
    result = run_multimodal_keyframe_pipeline(
        materialized.records,
        embeddings=features.embeddings,
        embedding_records=features.embedding_records,
        ocr_records=features.ocr_records,
        object_records=features.object_records,
        caption_records=features.caption_records,
        video_duration=shot_stage.info.duration,
        selection_config=config.selection_config(),
        allow_partial_features=False,
    )
    if not result.final_records:
        raise RuntimeError("multimodal selection returned no final keyframes")
    if result.guarantee_report.constraints_satisfied is not True:
        raise RuntimeError("multimodal selection failed its independent guarantee audit")
    if len(result.candidate_ledger) != len(materialized.records):
        raise RuntimeError("selector candidate ledger does not cover the full dense pool")
    return result, contract


def _resolve_record_image(record: Mapping[str, Any]) -> Path:
    raw = record.get("keyframe_path") or record.get("frame_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"selected record has no image path: {record.get('candidate_id')}")
    path = Path(raw)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.is_file():
        return cwd_path
    return path


def _rebase_selected_record(
    record: Mapping[str, Any],
    *,
    destination: Path,
) -> dict[str, Any]:
    value = dict(record)
    source = str(value.get("keyframe_path") or value.get("frame_path") or "")
    canonical_path = destination.as_posix()
    value.update(
        {
            "artifact_role": "selected_keyframe",
            "source_dense_keyframe_path": source,
            "image_path": canonical_path,
            "keyframe_path": canonical_path,
            "frame_path": canonical_path,
            "thumbnail_path": canonical_path,
        }
    )
    return value


def _rebase_selected_feature_record(
    record: Mapping[str, Any],
    canonical_record: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(record)
    for key in (
        "candidate_id",
        "frame_id",
        "video_id",
        "shot_id",
        "segment_id",
        "timestamp",
        "frame_index",
        "keyframe_path",
        "frame_path",
        "thumbnail_path",
    ):
        if key in canonical_record:
            value[key] = canonical_record[key]
    value["source_artifact_role"] = "selected_keyframe"
    return value


def _persist_selected_artifacts(
    video_path: Path,
    result: MultimodalKeyframePipelineResult,
    selection_contract: Mapping[str, Any],
    paths: PerVideoPaths,
    *,
    source_signature: Mapping[str, Any] | None = None,
) -> None:
    _stage_log(paths.video_id, STAGE_PERSISTENCE, "RUN")
    canonical_records: list[dict[str, Any]] = []
    canonical_by_candidate: dict[str, dict[str, Any]] = {}
    for record in result.final_records:
        source = _resolve_record_image(record)
        if not source.is_file():
            raise FileNotFoundError(f"selected dense image does not exist: {source}")
        destination = paths.selected_images_dir / source.name
        _atomic_copy(source, destination)
        canonical = _rebase_selected_record(record, destination=destination)
        candidate_id = str(canonical["candidate_id"])
        canonical_records.append(canonical)
        canonical_by_candidate[candidate_id] = canonical

    selected_ids = _candidate_ids(canonical_records)
    if len(canonical_by_candidate) != len(canonical_records):
        raise ValueError("selected candidate IDs must be unique")

    embedding_records = [
        _rebase_selected_feature_record(
            {**record, "embedding_index": index},
            canonical_by_candidate[str(record["candidate_id"])],
        )
        for index, record in enumerate(result.final_embedding_records)
    ]
    caption_records = [
        _rebase_selected_feature_record(
            record,
            canonical_by_candidate[str(record["candidate_id"])],
        )
        for record in result.final_caption_records
    ]
    ocr_records = [
        _rebase_selected_feature_record(
            record,
            canonical_by_candidate[str(record["candidate_id"])],
        )
        for record in result.final_ocr_records
    ]
    object_records = [
        _rebase_selected_feature_record(
            record,
            canonical_by_candidate[str(record["candidate_id"])],
        )
        for record in result.final_object_records
    ]
    for label, records in (
        ("selected embeddings", embedding_records),
        ("selected captions", caption_records),
        ("selected OCR", ocr_records),
        ("selected objects", object_records),
    ):
        if set(_candidate_ids(records)) != set(selected_ids) or len(records) != len(selected_ids):
            raise ValueError(f"{label} do not align with the selected keyframes")

    validate_embedding_artifacts(result.final_embeddings, embedding_records)
    _atomic_write_jsonl(paths.selected_metadata, canonical_records)
    _atomic_save_npy(paths.selected_embeddings, result.final_embeddings)
    _atomic_write_jsonl(paths.selected_embedding_metadata, embedding_records)
    _atomic_write_jsonl(paths.selected_captions, caption_records)
    _atomic_write_jsonl(paths.selected_ocr, ocr_records)
    _atomic_write_jsonl(paths.selected_objects, object_records)
    _atomic_write_jsonl(paths.candidate_ledger, result.candidate_ledger)
    _atomic_write_jsonl(paths.event_ledger, result.event_ledger)
    selection_report = _with_contract(
        {
            **result.to_report(),
            "status": "passed",
            "source_video": _source_content_identity(
                dict(source_signature or _file_signature(video_path))
            ),
            "selected_metadata_sha256": _sha256_file(paths.selected_metadata),
            "selected_images_sha256": _images_sha256(canonical_records),
            "selected_embeddings_sha256": _sha256_file(paths.selected_embeddings),
            "selected_embedding_metadata_sha256": _sha256_file(
                paths.selected_embedding_metadata
            ),
            "selected_captions_sha256": _sha256_file(paths.selected_captions),
            "selected_ocr_sha256": _sha256_file(paths.selected_ocr),
            "selected_objects_sha256": _sha256_file(paths.selected_objects),
        },
        selection_contract,
    )
    _atomic_write_json(paths.selection_report, selection_report)
    _write_state(
        paths,
        video_path=video_path,
        stage=STAGE_PERSISTENCE,
        status="passed",
        detail={
            "selected_count": len(canonical_records),
            "selection_contract_sha256": selection_contract["contract_sha256"],
        },
    )


def _per_video_config_payload(config: OfflinePipelineConfig) -> dict[str, Any]:
    excluded = {
        "output_dir",
        "resume",
        "force",
        "allow_partial_corpus",
        "build_corpus",
        "bge_enabled",
        "bge_model_name",
        "bge_model_revision",
        "bge_batch_size",
        "bge_local_files_only",
        "bge_model_cache_dir",
    }
    return {
        key: (value.as_posix() if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
        if key not in excluded
    }


def _validate_selected_bundle(
    *,
    video_path: Path,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
    require_completion: bool,
    source_signature: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    selection_report = _read_json(paths.selection_report)
    if selection_report.get("status") != "passed":
        raise ValueError("selection report is not passed")
    guarantees = selection_report.get("guarantees")
    if not isinstance(guarantees, dict) or guarantees.get("constraints_satisfied") is not True:
        raise ValueError("selection hard guarantees are not satisfied")
    shot_report = _read_json(paths.shot_report)
    shot_contract = shot_report.get("offline_contract")
    shot_source = (
        shot_contract.get("source_video")
        if isinstance(shot_contract, dict)
        else None
    )
    if not isinstance(shot_source, dict):
        raise ValueError("shot report has no immutable source-video lineage")
    current_source = dict(source_signature or _file_signature(video_path))
    if not _source_signatures_match(shot_source, current_source):
        raise ValueError("Source video changed since shot detection")
    selected_source = selection_report.get("source_video")
    if not isinstance(selected_source, dict) or not _source_signatures_match(
        selected_source,
        shot_source,
    ):
        raise ValueError("selection artifacts came from a different source video")
    # Retain the checkpoint's representation in validation output. Legacy
    # reports therefore continue to match byte-for-byte, while new reports use
    # the portable size+SHA-256 representation emitted by _shot_contract.
    expected_source = dict(shot_source)
    _assert_source_unchanged(video_path, expected_source)

    records = _read_jsonl(paths.selected_metadata)
    if not records:
        raise ValueError("selected keyframe metadata is empty")
    candidate_ids = _candidate_ids(records)
    frame_ids = _frame_ids(records)
    if any(record.get("artifact_role") != "selected_keyframe" for record in records):
        raise ValueError("canonical metadata contains a non-selected artifact role")
    if any(str(record.get("video_id") or "") != paths.video_id for record in records):
        raise ValueError("canonical metadata contains the wrong video_id")
    image_validation = validate_records(
        records,
        min_width=config.min_image_width,
        min_height=config.min_image_height,
    )
    if image_validation.get("valid") is not True:
        raise ValueError(f"selected image validation failed: {image_validation.get('errors')}")

    embeddings = np.load(paths.selected_embeddings, allow_pickle=False)
    embedding_records = _read_jsonl(paths.selected_embedding_metadata)
    validate_embedding_artifacts(embeddings, embedding_records)
    if _candidate_ids(embedding_records) != candidate_ids:
        raise ValueError("selected embedding order does not match canonical metadata")
    if _frame_ids(embedding_records) != frame_ids:
        raise ValueError("selected embedding frame IDs do not match canonical metadata")
    if any(
        str(record.get("video_id") or "") != paths.video_id
        for record in embedding_records
    ):
        raise ValueError("selected embedding metadata has the wrong video_id")

    for label, path in (
        ("caption", paths.selected_captions),
        ("OCR", paths.selected_ocr),
        ("object", paths.selected_objects),
    ):
        modality_records = _read_jsonl(path)
        _validate_modality_records(
            label=f"selected {label}",
            records=modality_records,
            expected_candidates=records,
            video_id=paths.video_id,
            required_nonempty_field="caption" if label == "caption" else None,
        )
        if set(_frame_ids(modality_records)) != set(frame_ids):
            raise ValueError(f"selected {label} frame IDs do not align")

    # Exercise the exact downstream canonical loader contract without scanning
    # or accepting metadata from unrelated videos.
    load_canonical_keyframe_records(
        paths.selected_metadata.parent,
        video_ids=(paths.video_id,),
    )

    artifact_hashes = {
        "dense_embeddings": _sha256_file(paths.dense_embeddings),
        "dense_embedding_metadata": _sha256_file(paths.dense_embedding_metadata),
        "dense_captions": _sha256_file(paths.dense_captions),
        "dense_ocr": _sha256_file(paths.dense_ocr),
        "dense_objects": _sha256_file(paths.dense_objects),
        "candidate_ledger": _sha256_file(paths.candidate_ledger),
        "selected_metadata": _sha256_file(paths.selected_metadata),
        "selected_images": _images_sha256(records),
        "selected_embeddings": _sha256_file(paths.selected_embeddings),
        "selected_embedding_metadata": _sha256_file(paths.selected_embedding_metadata),
        "selected_captions": _sha256_file(paths.selected_captions),
        "selected_ocr": _sha256_file(paths.selected_ocr),
        "selected_objects": _sha256_file(paths.selected_objects),
        "selection_report": _sha256_file(paths.selection_report),
    }
    validation = {
        "status": "passed",
        "video_id": paths.video_id,
        "source_video": expected_source,
        "dense_candidate_count": int(selection_report.get("candidate_count", -1)),
        "selected_count": len(records),
        "keyframe_validation": image_validation,
        "embedding_validation": validate_embedding_artifacts(embeddings, embedding_records),
        "artifact_hashes": artifact_hashes,
        "per_video_config_sha256": _sha256_value(_per_video_config_payload(config)),
    }
    if require_completion:
        completion = _read_json(paths.completion_report)
        if completion.get("status") != "passed":
            raise ValueError("completion report is not passed")
        completion_source = completion.get("source_video")
        if not isinstance(completion_source, dict) or not _source_signatures_match(
            completion_source,
            _file_signature(video_path),
        ):
            raise ValueError("source video changed since completion")
        if completion.get("per_video_config_sha256") != validation["per_video_config_sha256"]:
            raise ValueError("per-video configuration changed since completion")
        if completion.get("artifact_hashes") != artifact_hashes:
            raise ValueError("canonical artifact hashes changed since completion")
        if completion.get("selected_count") != len(records):
            raise ValueError("completion selected_count mismatch")
        if not paths.validation_report.is_file() or completion.get(
            "validation_report_sha256"
        ) != _sha256_file(paths.validation_report):
            raise ValueError("per-video validation report is missing or changed")
        if _read_json(paths.validation_report) != validation:
            raise ValueError("per-video validation report no longer matches artifacts")
    return validation, tuple(records)


def _validate_and_commit_video(
    video_path: Path,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
    *,
    source_signature: Mapping[str, Any] | None = None,
) -> VideoArtifacts:
    _stage_log(paths.video_id, STAGE_VALIDATION, "RUN")
    validation, records = _validate_selected_bundle(
        video_path=video_path,
        config=config,
        paths=paths,
        require_completion=False,
        source_signature=source_signature,
    )
    _atomic_write_json(paths.validation_report, validation)
    completion = {
        "pipeline": PIPELINE_NAME,
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "status": "passed",
        "video_id": paths.video_id,
        "source_video": validation["source_video"],
        "dense_candidate_count": validation["dense_candidate_count"],
        "selected_count": validation["selected_count"],
        "artifact_hashes": validation["artifact_hashes"],
        "validation_report": paths.validation_report.as_posix(),
        "validation_report_sha256": _sha256_file(paths.validation_report),
        "per_video_config_sha256": validation["per_video_config_sha256"],
    }
    # This is the per-video commit marker and is deliberately written last.
    _atomic_write_json(paths.completion_report, completion)
    verified, verified_records = _validate_selected_bundle(
        video_path=video_path,
        config=config,
        paths=paths,
        require_completion=True,
        source_signature=source_signature,
    )
    _write_state(
        paths,
        video_path=video_path,
        stage=STAGE_VALIDATION,
        status="passed",
        detail={"selected_count": len(verified_records)},
    )
    return VideoArtifacts(
        video_id=paths.video_id,
        video_path=video_path,
        paths=paths,
        selected_count=len(verified_records),
        dense_candidate_count=int(verified["dense_candidate_count"]),
        skipped=False,
        validation=verified,
    )


def _try_load_complete_video(
    video_path: Path,
    config: OfflinePipelineConfig,
    paths: PerVideoPaths,
    *,
    source_signature: Mapping[str, Any] | None = None,
) -> VideoArtifacts | None:
    if not config.resume or config.force:
        return None
    try:
        validation, records = _validate_selected_bundle(
            video_path=video_path,
            config=config,
            paths=paths,
            require_completion=True,
            source_signature=source_signature,
        )
    except CHECKPOINT_INVALID_ERRORS:
        return None
    return VideoArtifacts(
        video_id=paths.video_id,
        video_path=video_path,
        paths=paths,
        selected_count=len(records),
        dense_candidate_count=int(validation["dense_candidate_count"]),
        skipped=True,
        validation=validation,
    )


def process_video(
    video_path: str | Path,
    config: OfflinePipelineConfig,
) -> VideoArtifacts:
    """Run the complete canonical preprocessing pipeline for exactly one video.

    raw video -> shots -> dense candidates -> all images -> all multimodal
    features -> multimodal selection -> canonical artifacts -> validation
    """

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if path.suffix.casefold() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video suffix for {path}")
    video_id = path.stem
    paths = PerVideoPaths.from_config(video_id, config)
    source_signature = _file_signature(path)
    existing = _try_load_complete_video(
        path,
        config,
        paths,
        source_signature=source_signature,
    )
    if existing is not None:
        _stage_log(
            video_id,
            "complete_per_video_pipeline",
            "SKIP",
            f"validated selected={existing.selected_count}",
        )
        return existing

    current_stage = STAGE_SHOTS
    try:
        shots = _load_or_run_shot_detection(
            path,
            config,
            paths,
            source_signature=source_signature,
        )
        current_stage = STAGE_CANDIDATES
        candidates = _load_or_run_dense_candidate_generation(
            path,
            shots,
            config,
            paths,
        )
        current_stage = STAGE_MATERIALIZATION
        _assert_source_unchanged(path, source_signature)
        materialized = _load_or_run_dense_materialization(
            path,
            shots,
            candidates,
            config,
            paths,
        )
        current_stage = STAGE_SIGLIP
        features = _extract_all_dense_features(path, materialized, config, paths)
        current_stage = STAGE_SELECTION
        result, selection_contract = _run_multimodal_selection(
            shots,
            materialized,
            features,
            config,
            paths,
        )
        current_stage = STAGE_PERSISTENCE
        _assert_source_unchanged(path, source_signature)
        _persist_selected_artifacts(
            path,
            result,
            selection_contract,
            paths,
            source_signature=source_signature,
        )
        current_stage = STAGE_VALIDATION
        artifacts = _validate_and_commit_video(
            path,
            config,
            paths,
            source_signature=source_signature,
        )
    except Exception as exc:
        failed_stage = exc.stage if isinstance(exc, OfflineStageError) else current_stage
        failed_cause = exc.cause if isinstance(exc, OfflineStageError) else exc
        _stage_log(video_id, failed_stage, "FAIL", str(failed_cause))
        try:
            _write_state(
                paths,
                video_path=path,
                stage=failed_stage,
                status="failed",
                detail={"error": _safe_json_error(failed_cause)},
            )
        except OSError:
            LOGGER.exception("Could not persist failed stage state for %s", video_id)
        if isinstance(exc, OfflineStageError):
            raise
        raise OfflineStageError(video_id, current_stage, exc) from exc
    finally:
        _release_accelerator_memory()

    _stage_log(
        video_id,
        "complete_per_video_pipeline",
        "DONE",
        f"dense={artifacts.dense_candidate_count} selected={artifacts.selected_count}",
    )
    return artifacts


def _failure_from_exception(
    video_path: Path | None,
    exc: BaseException,
) -> VideoFailure:
    if isinstance(exc, OfflineStageError):
        video_id = exc.video_id
        stage = exc.stage
        cause = exc.cause
    else:
        video_id = video_path.stem if video_path else "<corpus>"
        stage = "process_video" if video_path else "corpus_indexes"
        cause = exc
    return VideoFailure(
        video_id=video_id,
        video_path=video_path,
        stage=stage,
        error_type=type(cause).__name__,
        message=str(cause),
        traceback_text="".join(traceback.format_exception(exc)),
    )


def _ordered_video_paths(video_paths: Iterable[str | Path]) -> tuple[Path, ...]:
    paths = tuple(Path(path) for path in video_paths)
    if not paths:
        raise ValueError("At least one video path is required")
    ordered = tuple(
        sorted(
            paths,
            key=lambda path: (
                path.stem.casefold(),
                path.resolve().as_posix().casefold(),
            ),
        )
    )
    video_ids = [path.stem for path in ordered]
    ids_by_casefold: dict[str, list[str]] = {}
    for video_id in video_ids:
        ids_by_casefold.setdefault(video_id.casefold(), []).append(video_id)
    duplicate_groups = [
        values for values in ids_by_casefold.values() if len(values) > 1
    ]
    if duplicate_groups:
        raise ValueError(
            "Duplicate video_id values would overwrite per-video artifacts: "
            f"{duplicate_groups}"
        )
    return ordered


def process_dataset(
    video_paths: Iterable[str | Path],
    config: OfflinePipelineConfig,
) -> DatasetProcessResult:
    """Process sorted videos sequentially, then build indexes once at the end."""

    ordered = _ordered_video_paths(video_paths)
    successful: list[VideoArtifacts] = []
    failures: list[VideoFailure] = []
    for index, video_path in enumerate(ordered, start=1):
        LOGGER.info(
            "[VIDEO %d/%d] START %s",
            index,
            len(ordered),
            video_path.as_posix(),
        )
        try:
            artifacts = process_video(video_path, config)
        except Exception as exc:
            failure = _failure_from_exception(video_path, exc)
            failures.append(failure)
            LOGGER.error(
                "[VIDEO %d/%d] FAILED %s :: %s",
                index,
                len(ordered),
                failure.video_id,
                failure.message,
            )
            continue
        successful.append(artifacts)
        LOGGER.info(
            "[VIDEO %d/%d] DONE %s",
            index,
            len(ordered),
            artifacts.video_id,
        )

    corpus_result: Mapping[str, Any] | None = None
    corpus_skipped = not config.build_corpus
    corpus_blocked = bool(
        config.build_corpus and failures and not config.allow_partial_corpus
    )
    if corpus_skipped:
        LOGGER.info(
            "[OPTIONAL DISABLED] corpus indexing | per-video artifacts preserved"
        )
    elif corpus_blocked:
        LOGGER.error(
            "[BLOCK] Corpus indexing requires every requested video; failures=%d",
            len(failures),
        )
    elif not successful:
        corpus_blocked = True
        LOGGER.error("[BLOCK] Corpus indexing has no successfully processed videos")
    else:
        try:
            corpus_result = build_corpus_indexes(tuple(successful), config)
            corpus_result = {
                **dict(corpus_result),
                "partial_corpus": bool(failures),
                "excluded_failed_video_ids": [failure.video_id for failure in failures],
            }
        except Exception as exc:
            failure = _failure_from_exception(None, exc)
            failures.append(failure)
            LOGGER.error("[CORPUS] FAILED :: %s", failure.message)
            corpus_result = None

    return DatasetProcessResult(
        requested_videos=ordered,
        successful_videos=tuple(successful),
        failures=tuple(failures),
        corpus_result=corpus_result,
        corpus_blocked=corpus_blocked,
        corpus_skipped=corpus_skipped,
    )


@dataclass(frozen=True)
class CorpusPaths:
    visual_index: Path
    visual_metadata: Path
    visual_frame_map: Path
    visual_manifest: Path
    visual_report: Path
    dense_index: Path
    dense_metadata: Path
    dense_frame_map: Path
    dense_manifest: Path
    dense_report: Path
    text_index: Path
    neighbor_metadata: Path
    segment_metadata: Path
    bge_root: Path
    state_manifest: Path
    validation_report: Path
    corpus_manifest: Path

    @classmethod
    def from_config(cls, config: OfflinePipelineConfig) -> "CorpusPaths":
        root = config.output_dir
        metadata = root / "metadata"
        return cls(
            visual_index=root / "indexes" / f"{ARTIFACT_TAG}_flat_ip.faiss",
            visual_metadata=metadata / f"{ARTIFACT_TAG}_faiss_metadata.jsonl",
            visual_frame_map=metadata / f"{ARTIFACT_TAG}_frame_map.json",
            visual_manifest=metadata / f"{ARTIFACT_TAG}_faiss_manifest.json",
            visual_report=metadata / f"{ARTIFACT_TAG}_index_report.json",
            dense_index=root / "indexes" / DENSE_INDEX_NAME,
            dense_metadata=metadata / DENSE_METADATA_NAME,
            dense_frame_map=metadata / DENSE_FRAME_MAP_NAME,
            dense_manifest=metadata / DENSE_MANIFEST_NAME,
            dense_report=metadata / DENSE_REPORT_NAME,
            text_index=root / "indexes" / "retrieval_text_index.json",
            neighbor_metadata=metadata / "neighbors_all.jsonl",
            segment_metadata=metadata / "segments_all.jsonl",
            bge_root=root / "indexes" / "bge_m3",
            state_manifest=root / "reports" / "offline" / "corpus_state.json",
            validation_report=root / "reports" / "offline" / "corpus_report.json",
            corpus_manifest=metadata / "offline_corpus_manifest.json",
        )


def _corpus_source_contract(
    videos: Sequence[VideoArtifacts],
    config: OfflinePipelineConfig,
) -> dict[str, object]:
    sources: list[dict[str, object]] = []
    for artifact in sorted(videos, key=lambda item: item.video_id):
        completion = _read_json(artifact.paths.completion_report)
        dense_inputs = {
            "embeddings": artifact.paths.dense_embeddings,
            "embedding_metadata": artifact.paths.dense_embedding_metadata,
            "captions": artifact.paths.dense_captions,
            "ocr": artifact.paths.dense_ocr,
            "objects": artifact.paths.dense_objects,
            "candidate_ledger": artifact.paths.candidate_ledger,
        }
        sources.append(
            {
                "video_id": artifact.video_id,
                "completion_report_sha256": _sha256_file(artifact.paths.completion_report),
                "artifact_hashes": completion.get("artifact_hashes"),
                "selected_count": artifact.selected_count,
                "dense_candidate_count": artifact.dense_candidate_count,
                "dense_input_hashes": {
                    label: _sha256_file(path) for label, path in dense_inputs.items()
                },
            }
        )
    contract = {
        "pipeline": PIPELINE_NAME,
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "video_ids": [item.video_id for item in sorted(videos, key=lambda item: item.video_id)],
        "sources": sources,
        "bge_enabled": config.bge_enabled,
        "bge_model_name": config.bge_model_name,
        "bge_requested_revision": config.bge_model_revision,
        "neighbor_window_seconds": config.neighbor_window_seconds,
        "segment_strategy": config.segment_strategy,
        "segment_fixed_duration_seconds": config.segment_fixed_duration_seconds,
        "segment_caption_similarity_threshold": (
            config.segment_caption_similarity_threshold
        ),
    }
    return {**contract, "contract_sha256": _sha256_value(contract)}


def _load_corpus_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value


def _write_corpus_stage_state(
    paths: CorpusPaths,
    *,
    source_contract: Mapping[str, Any],
    stage: str,
    detail: Mapping[str, Any],
) -> None:
    state = _load_corpus_state(paths.state_manifest)
    if state.get("source_contract_sha256") != source_contract.get("contract_sha256"):
        state = {}
    stages = state.get("stages")
    if not isinstance(stages, dict):
        stages = {}
    stages[stage] = {"status": "passed", **dict(detail)}
    _atomic_write_json(
        paths.state_manifest,
        {
            "pipeline": PIPELINE_NAME,
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "source_contract": {
                key: value
                for key, value in source_contract.items()
                if key != "contract_sha256"
            },
            "source_contract_sha256": source_contract["contract_sha256"],
            "stages": stages,
        },
    )


def _corpus_stage_state(
    paths: CorpusPaths,
    source_contract: Mapping[str, Any],
    stage: str,
) -> Mapping[str, Any] | None:
    state = _load_corpus_state(paths.state_manifest)
    if state.get("source_contract_sha256") != source_contract.get("contract_sha256"):
        return None
    stages = state.get("stages")
    if not isinstance(stages, dict):
        return None
    value = stages.get(stage)
    return value if isinstance(value, dict) and value.get("status") == "passed" else None


def _artifact_hash_map(paths: Sequence[Path]) -> dict[str, str]:
    return {path.as_posix(): _sha256_file(path) for path in paths}


def _hash_map_matches(expected: object, paths: Sequence[Path]) -> bool:
    if not isinstance(expected, dict):
        return False
    try:
        return expected == _artifact_hash_map(paths)
    except OSError:
        return False


def _expected_visual_index_records(
    videos: Sequence[VideoArtifacts],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    records: list[dict[str, Any]] = []
    vector_batches: list[np.ndarray] = []
    for video in sorted(videos, key=lambda item: item.video_id):
        vectors, source_records, _, _ = validate_embedding_source(
            video.paths.selected_embeddings,
            video.paths.selected_embedding_metadata,
            video.video_id,
        )
        vector_batches.append(vectors)
        for record in source_records:
            indexed = dict(record)
            indexed["faiss_index"] = len(records)
            records.append(indexed)
    if not vector_batches:
        raise ValueError("visual corpus requires at least one embedding source")
    expected_vectors = np.ascontiguousarray(
        np.concatenate(vector_batches, axis=0),
        dtype=np.float32,
    )
    return records, expected_vectors


def _validate_visual_bundle_manifest(
    manifest: Mapping[str, Any],
    paths: CorpusPaths,
) -> str:
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("visual manifest is missing bundle artifact checksums")
    actual_hashes: dict[str, str] = {}
    for label, path in (
        ("index", paths.visual_index),
        ("metadata", paths.visual_metadata),
        ("frame_map", paths.visual_frame_map),
        ("report", paths.visual_report),
    ):
        item = declared.get(label)
        if not isinstance(item, dict) or item.get("filename") != path.name:
            raise ValueError(f"visual manifest has invalid {label} artifact lineage")
        actual_sha256 = _sha256_file(path)
        if item.get("sha256") != actual_sha256:
            raise ValueError(f"visual {label} checksum does not match manifest")
        actual_hashes[label] = actual_sha256
    generation = _sha256_value(actual_hashes)
    if manifest.get("bundle_generation") != generation:
        raise ValueError("visual bundle generation does not match artifact checksums")
    return generation


def _validate_visual_corpus_index(
    paths: CorpusPaths,
    *,
    videos: Sequence[VideoArtifacts],
) -> dict[str, Any]:
    for path in (
        paths.visual_index,
        paths.visual_metadata,
        paths.visual_frame_map,
        paths.visual_manifest,
        paths.visual_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing visual index artifact: {path}")
    faiss = require_faiss()
    index = faiss.read_index(paths.visual_index.as_posix())
    expected_records, expected_vectors = _expected_visual_index_records(videos)
    expected_count = len(expected_records)
    if int(index.ntotal) != expected_count:
        raise ValueError(
            f"visual FAISS count mismatch: {index.ntotal} != {expected_count}"
        )
    reconstructed = np.empty_like(expected_vectors)
    index.reconstruct_n(0, expected_count, reconstructed)
    if not np.allclose(reconstructed, expected_vectors, atol=1e-6, rtol=1e-6):
        raise ValueError("visual FAISS vectors do not match selected embeddings")
    raw_frame_map = _read_json(paths.visual_frame_map)
    frame_map = {int(key): value for key, value in raw_frame_map.items()}
    frame_report, _ = validate_frame_map(frame_map, int(index.ntotal), fix=False)
    if frame_report.get("status") != "passed":
        raise ValueError(f"visual frame-map validation failed: {frame_report.get('errors')}")
    report = _read_json(paths.visual_report)
    manifest = _read_json(paths.visual_manifest)
    if report.get("status") != "passed":
        raise ValueError("visual index build report is not passed")
    if manifest.get("vector_count") != expected_count:
        raise ValueError("visual index manifest count mismatch")
    generation = _validate_visual_bundle_manifest(manifest, paths)
    indexed_records = _read_jsonl(paths.visual_metadata)
    if [_sha256_value(record) for record in indexed_records] != [
        _sha256_value(record) for record in expected_records
    ]:
        raise ValueError("visual index metadata does not match selected embeddings")
    expected_frame_map = {
        str(index): frame_map_record(record)
        for index, record in enumerate(expected_records)
    }
    if raw_frame_map != expected_frame_map:
        raise ValueError("visual frame map does not match selected embedding order")
    allowed_ids = {video.video_id for video in videos}
    actual_ids = {str(record.get("video_id") or "") for record in indexed_records}
    if actual_ids != allowed_ids:
        raise ValueError(
            f"visual index video set mismatch: actual={sorted(actual_ids)} "
            f"expected={sorted(allowed_ids)}"
        )
    return {
        "status": "passed",
        "vector_count": expected_count,
        "video_ids": sorted(actual_ids),
        "frame_map": frame_report,
        "bundle_generation": generation,
    }


def _build_visual_corpus_index(
    paths: CorpusPaths,
    videos: Sequence[VideoArtifacts],
    source_contract: Mapping[str, Any],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    stage = "visual_faiss"
    artifact_paths = (
        paths.visual_index,
        paths.visual_metadata,
        paths.visual_frame_map,
        paths.visual_manifest,
        paths.visual_report,
    )
    state = _corpus_stage_state(paths, source_contract, stage)
    if config.resume and not config.force and state is not None:
        try:
            result = _validate_visual_corpus_index(paths, videos=videos)
            if not _hash_map_matches(state.get("artifact_hashes"), artifact_paths):
                raise ValueError("visual index artifact checksum changed")
        except CHECKPOINT_INVALID_ERRORS as exc:
            LOGGER.info("[RUN] corpus :: visual FAISS | checkpoint invalid: %s", exc)
        else:
            LOGGER.info("[SKIP] corpus :: visual FAISS | vectors=%d", result["vector_count"])
            return result
    else:
        LOGGER.info("[RUN] corpus :: visual FAISS")

    sources = [video.visual_source for video in sorted(videos, key=lambda item: item.video_id)]
    for embeddings_path, metadata_path, video_id in sources:
        validate_embedding_source(embeddings_path, metadata_path, video_id)

    staging_parent = config.output_dir / "reports" / "offline" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_parent) as temporary_dir:
        root = Path(temporary_dir)
        staged = {
            "index": root / paths.visual_index.name,
            "metadata": root / paths.visual_metadata.name,
            "frame_map": root / paths.visual_frame_map.name,
            "manifest": root / paths.visual_manifest.name,
            "report": root / paths.visual_report.name,
        }
        built = build_faiss_artifacts(
            sources=sources,
            index_path=staged["index"],
            index_metadata_path=staged["metadata"],
            frame_map_path=staged["frame_map"],
            manifest_path=staged["manifest"],
            report_path=staged["report"],
            metric="ip",
            normalize_for_index=True,
        )
        frame_report, _ = validate_frame_map(
            {int(key): value for key, value in built["frame_map"].items()},
            int(built["index"].ntotal),
            fix=False,
        )
        if frame_report.get("status") != "passed":
            raise CorpusIndexError("staged visual frame map failed validation")
        report = _read_json(staged["report"])
        report.update(
            {
                "manifest_path": paths.visual_manifest.as_posix(),
                "index_path": paths.visual_index.as_posix(),
                "frame_map_path": paths.visual_frame_map.as_posix(),
            }
        )
        _atomic_write_json(staged["report"], report)
        bundle_hashes = {
            label: _sha256_file(staged[label])
            for label in ("index", "metadata", "frame_map", "report")
        }
        manifest = _read_json(staged["manifest"])
        manifest.update(
            {
                "index_path": paths.visual_index.as_posix(),
                "index_metadata_path": paths.visual_metadata.as_posix(),
                "frame_map_path": paths.visual_frame_map.as_posix(),
                "report_path": paths.visual_report.as_posix(),
                "bundle_generation": _sha256_value(bundle_hashes),
                "artifacts": {
                    label: {
                        "filename": final_path.name,
                        "sha256": bundle_hashes[label],
                    }
                    for label, final_path in (
                        ("index", paths.visual_index),
                        ("metadata", paths.visual_metadata),
                        ("frame_map", paths.visual_frame_map),
                        ("report", paths.visual_report),
                    )
                },
            }
        )
        _atomic_write_json(staged["manifest"], manifest)
        # Publish the commit marker last.  First install an invalid/publishing
        # sentinel so a crash during file replacement makes readers fail closed
        # instead of pairing a new vector row with an old frame map.  This also
        # protects the first migration from a legacy unhashed manifest.
        paths.visual_manifest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            paths.visual_manifest,
            {
                "schema_version": "1.2",
                "publication_status": "publishing",
                "bundle_generation": "publishing",
                "artifacts": {},
            },
        )
        for final, staged_path in (
            (paths.visual_index, staged["index"]),
            (paths.visual_metadata, staged["metadata"]),
            (paths.visual_frame_map, staged["frame_map"]),
            (paths.visual_report, staged["report"]),
            (paths.visual_manifest, staged["manifest"]),
        ):
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, final)

    result = _validate_visual_corpus_index(paths, videos=videos)
    _write_corpus_stage_state(
        paths,
        source_contract=source_contract,
        stage=stage,
        detail={"artifact_hashes": _artifact_hash_map(artifact_paths), **result},
    )
    return result


def _dense_object_labels(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    labels: list[str] = []
    for item in value:
        label = ""
        if isinstance(item, str):
            label = item.strip()
        elif isinstance(item, Mapping):
            for key in ("class_name", "label", "name", "class"):
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip():
                    label = raw.strip()
                    break
        if label and label not in labels:
            labels.append(label)
    return labels


def _dense_enriched_source_records(
    video: VideoArtifacts,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Join already-computed dense modalities without running inference."""

    vectors, embedding_records, _, _ = validate_embedding_source(
        video.paths.dense_embeddings,
        video.paths.dense_embedding_metadata,
        video.video_id,
    )
    if len(embedding_records) != video.dense_candidate_count:
        raise ValueError(
            f"dense candidate count mismatch for {video.video_id}: "
            f"{len(embedding_records)} != {video.dense_candidate_count}"
        )
    expected_ids = _candidate_ids(embedding_records)
    expected_id_set = set(expected_ids)

    captions = _read_jsonl(video.paths.dense_captions)
    ocr = _read_jsonl(video.paths.dense_ocr)
    objects = _read_jsonl(video.paths.dense_objects)
    for label, records, required in (
        ("dense caption", captions, "caption"),
        ("dense OCR", ocr, None),
        ("dense object", objects, None),
    ):
        _validate_modality_records(
            label=label,
            records=records,
            expected_candidates=embedding_records,
            video_id=video.video_id,
            required_nonempty_field=required,
        )
    ledger = _read_jsonl(video.paths.candidate_ledger)
    if set(_candidate_ids(ledger)) != expected_id_set or len(ledger) != len(expected_ids):
        raise ValueError(f"candidate ledger does not align with dense pool: {video.video_id}")
    if any(str(record.get("video_id") or "") != video.video_id for record in ledger):
        raise ValueError(f"candidate ledger has wrong video_id: {video.video_id}")

    caption_by_id = {str(record["candidate_id"]): record for record in captions}
    ocr_by_id = {str(record["candidate_id"]): record for record in ocr}
    objects_by_id = {str(record["candidate_id"]): record for record in objects}
    ledger_by_id = {str(record["candidate_id"]): record for record in ledger}
    enriched: list[dict[str, Any]] = []
    for record in embedding_records:
        candidate_id = str(record["candidate_id"])
        candidate_ledger = ledger_by_id[candidate_id]
        protected_event_ids = candidate_ledger.get("feature_protected_event_ids", [])
        if not isinstance(protected_event_ids, list) or any(
            not isinstance(value, str) for value in protected_event_ids
        ):
            raise ValueError(
                f"candidate ledger has invalid protected events: {candidate_id}"
            )
        caption = str(caption_by_id[candidate_id].get("caption") or "").strip()
        if not caption:
            raise ValueError(f"dense caption evidence is empty: {candidate_id}")
        value = dict(record)
        value.update(
            {
                "artifact_role": "dense_candidate",
                "caption": caption,
                "ocr_text": str(ocr_by_id[candidate_id].get("ocr_text") or ""),
                "objects": _dense_object_labels(
                    objects_by_id[candidate_id].get("objects")
                ),
                "protected_event_ids": list(dict.fromkeys(protected_event_ids)),
                "protected": bool(protected_event_ids),
                "importance_score": candidate_ledger.get("importance_score"),
                "semantic_novelty": candidate_ledger.get("semantic_novelty"),
                "component_scores": dict(
                    candidate_ledger.get("component_scores")
                    if isinstance(candidate_ledger.get("component_scores"), Mapping)
                    else {}
                ),
                "available_modalities": list(
                    candidate_ledger.get("available_modalities")
                    if isinstance(candidate_ledger.get("available_modalities"), list)
                    else []
                ),
                "offline_selected": bool(candidate_ledger.get("selected")),
                "selection_rank": candidate_ledger.get("selection_rank"),
                "selection_phase": candidate_ledger.get("selection_phase"),
                "selection_reasons": list(
                    candidate_ledger.get("selection_reasons")
                    if isinstance(candidate_ledger.get("selection_reasons"), list)
                    else []
                ),
            }
        )
        enriched.append(value)
    return vectors, enriched


def _expected_dense_index_records(
    videos: Sequence[VideoArtifacts],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    records: list[dict[str, Any]] = []
    vector_batches: list[np.ndarray] = []
    for video in sorted(videos, key=lambda item: item.video_id):
        vectors, source_records = _dense_enriched_source_records(video)
        vector_batches.append(vectors)
        for record in source_records:
            indexed = dict(record)
            indexed["faiss_index"] = len(records)
            records.append(indexed)
    if not vector_batches:
        raise ValueError("dense corpus requires at least one embedding source")
    expected_vectors = np.ascontiguousarray(
        np.concatenate(vector_batches, axis=0),
        dtype=np.float32,
    )
    norms = np.linalg.norm(expected_vectors, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("dense corpus contains an invalid vector")
    expected_vectors = np.ascontiguousarray(expected_vectors / norms, dtype=np.float32)
    return records, expected_vectors


def _validate_dense_bundle_manifest(
    manifest: Mapping[str, Any],
    paths: CorpusPaths,
) -> str:
    if manifest.get("schema_version") != DENSE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("dense manifest has an unsupported schema version")
    if manifest.get("artifact_role") != DENSE_ARTIFACT_ROLE:
        raise ValueError("dense manifest has the wrong artifact role")
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("dense manifest is missing bundle artifact checksums")
    actual_hashes: dict[str, str] = {}
    for label, path in (
        ("index", paths.dense_index),
        ("metadata", paths.dense_metadata),
        ("frame_map", paths.dense_frame_map),
        ("report", paths.dense_report),
    ):
        item = declared.get(label)
        if not isinstance(item, dict) or item.get("filename") != path.name:
            raise ValueError(f"dense manifest has invalid {label} artifact lineage")
        actual_sha256 = _sha256_file(path)
        if item.get("sha256") != actual_sha256:
            raise ValueError(f"dense {label} checksum does not match manifest")
        actual_hashes[label] = actual_sha256
    generation = _sha256_value(actual_hashes)
    if manifest.get("bundle_generation") != generation:
        raise ValueError("dense bundle generation does not match artifact checksums")
    return generation


def _validate_dense_corpus_index(
    paths: CorpusPaths,
    *,
    videos: Sequence[VideoArtifacts],
) -> dict[str, Any]:
    for path in (
        paths.dense_index,
        paths.dense_metadata,
        paths.dense_frame_map,
        paths.dense_manifest,
        paths.dense_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dense index artifact: {path}")
    faiss = require_faiss()
    index = faiss.read_index(paths.dense_index.as_posix())
    if type(index).__name__ != "IndexFlatIP":
        raise ValueError("dense FAISS index must be IndexFlatIP")
    expected_records, expected_vectors = _expected_dense_index_records(videos)
    expected_count = len(expected_records)
    if int(index.ntotal) != expected_count or int(index.d) != expected_vectors.shape[1]:
        raise ValueError("dense FAISS count or dimension mismatch")
    reconstructed = np.empty_like(expected_vectors)
    index.reconstruct_n(0, expected_count, reconstructed)
    if not np.allclose(reconstructed, expected_vectors, atol=1e-6, rtol=1e-6):
        raise ValueError("dense FAISS vectors do not match full candidate embeddings")

    indexed_records = _read_jsonl(paths.dense_metadata)
    if [_sha256_value(record) for record in indexed_records] != [
        _sha256_value(record) for record in expected_records
    ]:
        raise ValueError("dense index metadata does not match enriched dense candidates")
    raw_frame_map = _read_json(paths.dense_frame_map)
    expected_frame_map = {
        str(row): frame_map_record(record)
        for row, record in enumerate(expected_records)
    }
    if raw_frame_map != expected_frame_map:
        raise ValueError("dense frame map does not match dense metadata order")
    frame_report, _ = validate_frame_map(
        {int(key): value for key, value in raw_frame_map.items()},
        int(index.ntotal),
        fix=False,
    )
    if frame_report.get("status") != "passed":
        raise ValueError(f"dense frame-map validation failed: {frame_report.get('errors')}")

    report = _read_json(paths.dense_report)
    manifest = _read_json(paths.dense_manifest)
    if report.get("status") != "passed":
        raise ValueError("dense index build report is not passed")
    if (
        manifest.get("vector_count") != expected_count
        or manifest.get("metadata_record_count") != expected_count
        or report.get("vector_count") != expected_count
        or report.get("metadata_record_count") != expected_count
    ):
        raise ValueError("dense index manifest/report count mismatch")
    clip_keys = {
        (
            str(record.get("video_id") or ""),
            str(record.get("segment_id") or record.get("shot_id") or ""),
        )
        for record in expected_records
    }
    if any(not video_id or not clip_id for video_id, clip_id in clip_keys):
        raise ValueError("dense candidate metadata has an empty clip identity")
    if manifest.get("clip_count") != len(clip_keys):
        raise ValueError("dense index manifest clip count mismatch")
    generation = _validate_dense_bundle_manifest(manifest, paths)
    return {
        "status": "passed",
        "vector_count": expected_count,
        "clip_count": len(clip_keys),
        "video_ids": sorted({key[0] for key in clip_keys}),
        "frame_map": frame_report,
        "bundle_generation": generation,
        "enriched_fields": [
            "caption",
            "ocr_text",
            "objects",
            "protected_event_ids",
        ],
    }


def _build_dense_corpus_index(
    paths: CorpusPaths,
    videos: Sequence[VideoArtifacts],
    source_contract: Mapping[str, Any],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    stage = "dense_candidate_faiss"
    artifact_paths = (
        paths.dense_index,
        paths.dense_metadata,
        paths.dense_frame_map,
        paths.dense_manifest,
        paths.dense_report,
    )
    state = _corpus_stage_state(paths, source_contract, stage)
    if config.resume and not config.force and state is not None:
        try:
            result = _validate_dense_corpus_index(paths, videos=videos)
            if not _hash_map_matches(state.get("artifact_hashes"), artifact_paths):
                raise ValueError("dense index artifact checksum changed")
        except CHECKPOINT_INVALID_ERRORS as exc:
            LOGGER.info("[RUN] corpus :: dense candidate FAISS | checkpoint invalid: %s", exc)
        else:
            LOGGER.info(
                "[SKIP] corpus :: dense candidate FAISS | vectors=%d",
                result["vector_count"],
            )
            return result
    else:
        LOGGER.info("[RUN] corpus :: dense candidate FAISS")

    sources = [
        video.dense_visual_source for video in sorted(videos, key=lambda item: item.video_id)
    ]
    for embeddings_path, metadata_path, video_id in sources:
        validate_embedding_source(embeddings_path, metadata_path, video_id)
    expected_records, _ = _expected_dense_index_records(videos)

    staging_parent = config.output_dir / "reports" / "offline" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_parent) as temporary_dir:
        root = Path(temporary_dir)
        staged = {
            "index": root / paths.dense_index.name,
            "metadata": root / paths.dense_metadata.name,
            "frame_map": root / paths.dense_frame_map.name,
            "manifest": root / paths.dense_manifest.name,
            "report": root / paths.dense_report.name,
        }
        built = build_faiss_artifacts(
            sources=sources,
            index_path=staged["index"],
            index_metadata_path=staged["metadata"],
            frame_map_path=staged["frame_map"],
            manifest_path=staged["manifest"],
            report_path=staged["report"],
            metric="ip",
            normalize_for_index=True,
        )
        built_ids = [str(record.get("candidate_id") or "") for record in built["index_records"]]
        expected_ids = [str(record["candidate_id"]) for record in expected_records]
        if built_ids != expected_ids:
            raise CorpusIndexError("dense FAISS row order changed during enrichment")
        _atomic_write_jsonl(staged["metadata"], expected_records)
        dense_frame_map = {
            str(row): frame_map_record(record)
            for row, record in enumerate(expected_records)
        }
        _atomic_write_json(staged["frame_map"], dense_frame_map)
        frame_report, _ = validate_frame_map(
            {int(key): value for key, value in dense_frame_map.items()},
            int(built["index"].ntotal),
            fix=False,
        )
        if frame_report.get("status") != "passed":
            raise CorpusIndexError("staged dense frame map failed validation")

        clip_keys = {
            (
                str(record.get("video_id") or ""),
                str(record.get("segment_id") or record.get("shot_id") or ""),
            )
            for record in expected_records
        }
        report = _read_json(staged["report"])
        report.update(
            {
                "artifact_role": DENSE_ARTIFACT_ROLE,
                "metadata_record_count": len(expected_records),
                "clip_count": len(clip_keys),
                "enriched_fields": [
                    "caption",
                    "ocr_text",
                    "objects",
                    "protected_event_ids",
                ],
                "manifest_path": paths.dense_manifest.as_posix(),
                "index_path": paths.dense_index.as_posix(),
                "metadata_path": paths.dense_metadata.as_posix(),
                "frame_map_path": paths.dense_frame_map.as_posix(),
            }
        )
        _atomic_write_json(staged["report"], report)
        bundle_hashes = {
            label: _sha256_file(staged[label])
            for label in ("index", "metadata", "frame_map", "report")
        }
        manifest = _read_json(staged["manifest"])
        manifest.update(
            {
                "artifact_role": DENSE_ARTIFACT_ROLE,
                "source_kind": "full_dense_candidates",
                "clip_count": len(clip_keys),
                "index_path": paths.dense_index.as_posix(),
                "index_metadata_path": paths.dense_metadata.as_posix(),
                "frame_map_path": paths.dense_frame_map.as_posix(),
                "report_path": paths.dense_report.as_posix(),
                "bundle_generation": _sha256_value(bundle_hashes),
                "enrichment": {
                    "inference_performed": False,
                    "join_key": "candidate_id",
                    "fields": [
                        "caption",
                        "ocr_text",
                        "objects",
                        "protected_event_ids",
                    ],
                },
                "artifacts": {
                    label: {
                        "filename": final_path.name,
                        "sha256": bundle_hashes[label],
                    }
                    for label, final_path in (
                        ("index", paths.dense_index),
                        ("metadata", paths.dense_metadata),
                        ("frame_map", paths.dense_frame_map),
                        ("report", paths.dense_report),
                    )
                },
            }
        )
        _atomic_write_json(staged["manifest"], manifest)

        paths.dense_manifest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            paths.dense_manifest,
            {
                "schema_version": DENSE_MANIFEST_SCHEMA_VERSION,
                "artifact_role": DENSE_ARTIFACT_ROLE,
                "publication_status": "publishing",
                "bundle_generation": "publishing",
                "artifacts": {},
            },
        )
        for final, staged_path in (
            (paths.dense_index, staged["index"]),
            (paths.dense_metadata, staged["metadata"]),
            (paths.dense_frame_map, staged["frame_map"]),
            (paths.dense_report, staged["report"]),
            (paths.dense_manifest, staged["manifest"]),
        ):
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, final)

    result = _validate_dense_corpus_index(paths, videos=videos)
    _write_corpus_stage_state(
        paths,
        source_contract=source_contract,
        stage=stage,
        detail={"artifact_hashes": _artifact_hash_map(artifact_paths), **result},
    )
    return result


def _validate_unique_canonical_records(
    records: Sequence[Mapping[str, Any]],
    *,
    allowed_video_ids: set[str],
) -> None:
    if not records:
        raise ValueError("No selected keyframe has retrievable caption/OCR/object text")
    seen: set[tuple[str, str]] = set()
    covered_video_ids: set[str] = set()
    for index, record in enumerate(records):
        video_id = str(record.get("video_id") or "")
        frame_id = str(record.get("frame_id") or "")
        if video_id not in allowed_video_ids or not frame_id:
            raise ValueError(f"canonical text record {index} has invalid lineage")
        identity = (video_id, frame_id)
        if identity in seen:
            raise ValueError(f"duplicate canonical text identity: {identity}")
        seen.add(identity)
        if record.get("artifact_role") != "selected_keyframe":
            raise ValueError("text index input contains a non-selected artifact")
        covered_video_ids.add(video_id)
    if covered_video_ids != allowed_video_ids:
        raise ValueError(
            "canonical text input does not cover every successful video: "
            f"actual={sorted(covered_video_ids)} "
            f"expected={sorted(allowed_video_ids)}"
        )


def _validate_text_corpus_index(
    paths: CorpusPaths,
    canonical_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    payload = _read_json(paths.text_index)
    expected = build_text_index_payload(canonical_records)
    if payload != expected:
        raise ValueError("text index payload does not match canonical selected records")
    modalities = payload.get("modalities")
    if not isinstance(modalities, dict) or set(modalities) != set(TEXT_MODALITIES):
        raise ValueError("text index does not contain caption/OCR/object modalities")
    return {
        "status": "passed",
        "input_record_count": len(canonical_records),
        "modalities": {
            modality: modalities[modality].get("stats", {})
            for modality in TEXT_MODALITIES
        },
    }


def _build_text_corpus_index(
    paths: CorpusPaths,
    canonical_records: list[dict[str, Any]],
    source_contract: Mapping[str, Any],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    stage = "text_bm25"
    state = _corpus_stage_state(paths, source_contract, stage)
    if config.resume and not config.force and state is not None:
        try:
            result = _validate_text_corpus_index(paths, canonical_records)
            if not _hash_map_matches(state.get("artifact_hashes"), (paths.text_index,)):
                raise ValueError("text index checksum changed")
        except CHECKPOINT_INVALID_ERRORS as exc:
            LOGGER.info("[RUN] corpus :: caption/OCR/object BM25 | checkpoint invalid: %s", exc)
        else:
            LOGGER.info("[SKIP] corpus :: caption/OCR/object BM25")
            return result
    else:
        LOGGER.info("[RUN] corpus :: caption/OCR/object BM25")
    write_text_index(canonical_records, paths.text_index)
    result = _validate_text_corpus_index(paths, canonical_records)
    _write_corpus_stage_state(
        paths,
        source_contract=source_contract,
        stage=stage,
        detail={"artifact_hashes": _artifact_hash_map((paths.text_index,)), **result},
    )
    return result


def _canonical_identity(
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, str]:
    video_id = str(record.get("video_id") or "").strip()
    frame_id = str(record.get("frame_id") or record.get("keyframe_id") or "").strip()
    if not video_id or not frame_id:
        raise ValueError(f"{label} record is missing video_id/frame_id")
    return video_id, frame_id


def _load_explicit_selected_modalities(
    videos: Sequence[VideoArtifacts],
    canonical_records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Load only modality files belonging to the explicit successful videos."""

    expected = {
        _canonical_identity(record, label="canonical selected")
        for record in canonical_records
    }
    path_fields = {
        "captions": "selected_captions",
        "ocr": "selected_ocr",
        "objects": "selected_objects",
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for label, path_field in path_fields.items():
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for video in sorted(videos, key=lambda item: item.video_id):
            path = getattr(video.paths, path_field)
            for record in _read_jsonl(path):
                identity = _canonical_identity(record, label=label)
                if identity[0] != video.video_id:
                    raise ValueError(
                        f"{label} file for {video.video_id} contains {identity[0]}"
                    )
                if identity in seen:
                    raise ValueError(f"duplicate selected {label} identity: {identity}")
                seen.add(identity)
                records.append(record)
        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise ValueError(
                f"selected {label} identities do not match canonical keyframes: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        output[label] = records
    return output


def _neighbor_sort_key(record: Mapping[str, Any]) -> tuple[str, float, int, str]:
    identity = _canonical_identity(record, label="neighbor source")
    frame_index = record.get("frame_index")
    normalized_index = -1 if frame_index in {None, ""} else int(frame_index)
    return identity[0], float(record.get("timestamp")), normalized_index, identity[1]


def _expected_neighbor_records(
    canonical_records: Sequence[Mapping[str, Any]],
    *,
    window_seconds: float,
) -> list[dict[str, Any]]:
    ordered = sorted(canonical_records, key=_neighbor_sort_key)
    by_video: dict[str, list[Mapping[str, Any]]] = {}
    for record in ordered:
        video_id, _ = _canonical_identity(record, label="neighbor source")
        by_video.setdefault(video_id, []).append(record)

    output: list[dict[str, Any]] = []
    for center in ordered:
        video_id, frame_id = _canonical_identity(center, label="neighbor source")
        timestamp = round(float(center.get("timestamp")), 6)
        lower = max(0.0, timestamp - window_seconds)
        upper = timestamp + window_seconds
        before = [
            record
            for record in by_video[video_id]
            if lower <= float(record.get("timestamp")) < timestamp
        ]
        after = [
            record
            for record in by_video[video_id]
            if timestamp < float(record.get("timestamp")) <= upper
        ]

        def compact(record: Mapping[str, Any]) -> dict[str, Any]:
            _, neighbor_id = _canonical_identity(record, label="neighbor source")
            return {
                "frame_id": neighbor_id,
                "delta_seconds": round(float(record.get("timestamp")) - timestamp, 6),
            }

        value: dict[str, Any] = {
            "schema_version": "1.0",
            "video_id": video_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "timestamp_source": str(center.get("timestamp_source") or "metadata"),
            "neighbors_before": [compact(record) for record in before],
            "neighbors_after": [compact(record) for record in after],
        }
        if center.get("frame_index") not in {None, ""}:
            value["frame_index"] = int(center["frame_index"])
        output.append(value)
    return output


def _validate_neighbor_corpus_metadata(
    paths: CorpusPaths,
    *,
    canonical_records: Sequence[Mapping[str, Any]],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    actual = _read_jsonl(paths.neighbor_metadata)
    expected = _expected_neighbor_records(
        canonical_records,
        window_seconds=config.neighbor_window_seconds,
    )
    if actual != expected:
        raise ValueError(
            "neighbor mapping does not exactly match selected keyframe timestamps"
        )
    return {
        "status": "passed",
        "record_count": len(actual),
        "neighbor_reference_count": sum(
            len(record["neighbors_before"]) + len(record["neighbors_after"])
            for record in actual
        ),
        "window_seconds": config.neighbor_window_seconds,
    }


def _build_neighbor_corpus_metadata(
    paths: CorpusPaths,
    canonical_records: Sequence[Mapping[str, Any]],
    source_contract: Mapping[str, Any],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    stage = "neighbor_mapping"
    artifact_paths = (paths.neighbor_metadata,)
    state = _corpus_stage_state(paths, source_contract, stage)
    if config.resume and not config.force and state is not None:
        try:
            result = _validate_neighbor_corpus_metadata(
                paths,
                canonical_records=canonical_records,
                config=config,
            )
            if not _hash_map_matches(state.get("artifact_hashes"), artifact_paths):
                raise ValueError("neighbor mapping checksum changed")
        except CHECKPOINT_INVALID_ERRORS as exc:
            LOGGER.info("[RUN] corpus :: neighbor mapping | checkpoint invalid: %s", exc)
        else:
            LOGGER.info("[SKIP] corpus :: neighbor mapping")
            return result
    else:
        LOGGER.info("[RUN] corpus :: neighbor mapping")

    staging_parent = config.output_dir / "reports" / "offline" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    paths.neighbor_metadata.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_parent) as temporary_dir:
        selected_input = Path(temporary_dir) / "selected_keyframes.jsonl"
        _atomic_write_jsonl(selected_input, canonical_records)
        build_neighbor_index(
            selected_input,
            paths.neighbor_metadata,
            window_seconds=config.neighbor_window_seconds,
        )
    result = _validate_neighbor_corpus_metadata(
        paths,
        canonical_records=canonical_records,
        config=config,
    )
    _write_corpus_stage_state(
        paths,
        source_contract=source_contract,
        stage=stage,
        detail={"artifact_hashes": _artifact_hash_map(artifact_paths), **result},
    )
    return result


def _expected_segment_records(
    canonical_records: Sequence[Mapping[str, Any]],
    modalities: Mapping[str, Sequence[dict[str, Any]]],
    config: OfflinePipelineConfig,
) -> list[dict[str, Any]]:
    segments = build_segments(
        [dict(record) for record in canonical_records],
        strategy=config.segment_strategy,
        fixed_duration_seconds=config.segment_fixed_duration_seconds,
    )
    return build_segment_records(
        segments,
        captions=modalities["captions"],
        ocr=modalities["ocr"],
        objects=modalities["objects"],
        caption_similarity_threshold=config.segment_caption_similarity_threshold,
    )


def _validate_segment_corpus_metadata(
    paths: CorpusPaths,
    *,
    canonical_records: Sequence[Mapping[str, Any]],
    modalities: Mapping[str, Sequence[dict[str, Any]]],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    actual = _read_jsonl(paths.segment_metadata)
    expected = _expected_segment_records(canonical_records, modalities, config)
    if actual != expected:
        raise ValueError(
            "segment metadata does not match selected keyframes and modalities"
        )
    identities = [
        (str(record.get("video_id") or ""), str(record.get("segment_id") or ""))
        for record in actual
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("segment metadata contains duplicate video/segment identities")
    covered_keyframes = [
        (str(record.get("video_id") or ""), str(frame_id))
        for record in actual
        for frame_id in record.get("keyframe_ids", [])
    ]
    expected_keyframes = [
        _canonical_identity(record, label="segment source")
        for record in canonical_records
    ]
    if len(set(covered_keyframes)) != len(covered_keyframes) or set(
        covered_keyframes
    ) != set(expected_keyframes):
        raise ValueError("segments do not cover every selected keyframe exactly once")
    return {
        "status": "passed",
        "record_count": len(actual),
        "selected_keyframe_count": len(covered_keyframes),
        "video_ids": sorted({identity[0] for identity in identities}),
        "strategy": config.segment_strategy,
    }


def _build_segment_corpus_metadata(
    paths: CorpusPaths,
    canonical_records: Sequence[Mapping[str, Any]],
    modalities: Mapping[str, Sequence[dict[str, Any]]],
    source_contract: Mapping[str, Any],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    stage = "segments_events"
    artifact_paths = (paths.segment_metadata,)
    state = _corpus_stage_state(paths, source_contract, stage)
    if config.resume and not config.force and state is not None:
        try:
            result = _validate_segment_corpus_metadata(
                paths,
                canonical_records=canonical_records,
                modalities=modalities,
                config=config,
            )
            if not _hash_map_matches(state.get("artifact_hashes"), artifact_paths):
                raise ValueError("segment metadata checksum changed")
        except CHECKPOINT_INVALID_ERRORS as exc:
            LOGGER.info("[RUN] corpus :: segments/events | checkpoint invalid: %s", exc)
        else:
            LOGGER.info("[SKIP] corpus :: segments/events")
            return result
    else:
        LOGGER.info("[RUN] corpus :: segments/events")

    records = _expected_segment_records(canonical_records, modalities, config)
    _atomic_write_jsonl(paths.segment_metadata, records)
    result = _validate_segment_corpus_metadata(
        paths,
        canonical_records=canonical_records,
        modalities=modalities,
        config=config,
    )
    _write_corpus_stage_state(
        paths,
        source_contract=source_contract,
        stage=stage,
        detail={"artifact_hashes": _artifact_hash_map(artifact_paths), **result},
    )
    return result


def _validate_bge_corpus_index(
    paths: CorpusPaths,
    *,
    canonical_records: Sequence[dict[str, Any]],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    artifacts = validate_bge_m3_artifacts(
        paths.bge_root,
        expected_model_name=config.bge_model_name,
        expected_model_revision=None,
    )
    if len(artifacts.frame_records) != len(canonical_records):
        raise ValueError("BGE-M3 vector count does not match canonical text records")
    source_contract = artifacts.manifest.get("source_contract")
    if not isinstance(source_contract, dict) or source_contract.get(
        "canonical_only"
    ) is not True:
        raise ValueError("BGE-M3 index is not canonical-only")
    if source_contract.get("source_kind") != "selected_keyframes":
        raise ValueError("BGE-M3 index does not come from selected keyframes")

    expected_by_identity = {
        (str(record.get("video_id") or ""), str(record.get("frame_id") or "")): record
        for record in canonical_records
    }
    actual_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in artifacts.frame_records:
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("BGE-M3 frame map contains invalid metadata")
        identity = (
            str(metadata.get("video_id") or ""),
            str(metadata.get("frame_id") or ""),
        )
        if identity in actual_by_identity:
            raise ValueError(f"BGE-M3 contains duplicate source lineage: {identity}")
        actual_by_identity[identity] = metadata
    if set(actual_by_identity) != set(expected_by_identity):
        raise ValueError("BGE-M3 source identities do not match canonical records")
    for identity, expected in expected_by_identity.items():
        if _sha256_value(actual_by_identity[identity]) != _sha256_value(expected):
            raise ValueError(
                f"BGE-M3 source metadata does not match canonical record {identity}"
            )
    return {
        "status": "passed",
        "vector_count": len(artifacts.frame_records),
        "model": dict(artifacts.manifest.get("model", {})),
    }


def _build_bge_corpus_index(
    paths: CorpusPaths,
    canonical_records: list[dict[str, Any]],
    source_contract: Mapping[str, Any],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    stage = "bge_m3"
    bge_paths = BgeM3ArtifactPaths.from_root(paths.bge_root)
    artifact_paths = (bge_paths.index, bge_paths.frame_map, bge_paths.manifest)
    state = _corpus_stage_state(paths, source_contract, stage)
    if config.resume and not config.force and state is not None:
        try:
            result = _validate_bge_corpus_index(
                paths,
                canonical_records=canonical_records,
                config=config,
            )
            if not _hash_map_matches(state.get("artifact_hashes"), artifact_paths):
                raise ValueError("BGE-M3 artifact checksum changed")
        except CHECKPOINT_INVALID_ERRORS as exc:
            LOGGER.info("[RUN] corpus :: BGE-M3 | checkpoint invalid: %s", exc)
        else:
            LOGGER.info("[SKIP] corpus :: BGE-M3 | vectors=%d", result["vector_count"])
            return result
    else:
        LOGGER.info("[RUN] corpus :: BGE-M3")

    staging_parent = config.output_dir / "reports" / "offline" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_parent) as temporary_dir:
        staged_root = Path(temporary_dir) / "bge_m3"
        build_bge_m3_index(
            canonical_records,
            staged_root,
            model_name=config.bge_model_name,
            model_revision=config.bge_model_revision,
            batch_size=config.bge_batch_size,
            device=config.device,
            cache_dir=config.bge_model_cache_dir,
            local_files_only=config.bge_local_files_only,
            canonical_only=True,
        )
        validate_bge_m3_artifacts(
            staged_root,
            expected_model_name=config.bge_model_name,
            expected_model_revision=None,
        )
        staged_paths = BgeM3ArtifactPaths.from_root(staged_root)
        paths.bge_root.mkdir(parents=True, exist_ok=True)
        for source, destination in (
            (staged_paths.index, bge_paths.index),
            (staged_paths.frame_map, bge_paths.frame_map),
            (staged_paths.manifest, bge_paths.manifest),
        ):
            os.replace(source, destination)
    _release_accelerator_memory()
    result = _validate_bge_corpus_index(
        paths,
        canonical_records=canonical_records,
        config=config,
    )
    _write_corpus_stage_state(
        paths,
        source_contract=source_contract,
        stage=stage,
        detail={"artifact_hashes": _artifact_hash_map(artifact_paths), **result},
    )
    return result


def _corpus_bundle_artifacts(
    paths: CorpusPaths,
    *,
    bge_enabled: bool,
) -> dict[str, Path]:
    artifacts = {
        "visual_index": paths.visual_index,
        "visual_metadata": paths.visual_metadata,
        "visual_frame_map": paths.visual_frame_map,
        "visual_manifest": paths.visual_manifest,
        "visual_report": paths.visual_report,
        "dense_index": paths.dense_index,
        "dense_metadata": paths.dense_metadata,
        "dense_frame_map": paths.dense_frame_map,
        "dense_manifest": paths.dense_manifest,
        "dense_report": paths.dense_report,
        "text_index": paths.text_index,
        "neighbor_metadata": paths.neighbor_metadata,
        "segment_metadata": paths.segment_metadata,
        "corpus_state": paths.state_manifest,
        "corpus_report": paths.validation_report,
    }
    if bge_enabled:
        bge_paths = BgeM3ArtifactPaths.from_root(paths.bge_root)
        artifacts.update(
            {
                "bge_index": bge_paths.index,
                "bge_frame_map": bge_paths.frame_map,
                "bge_manifest": bge_paths.manifest,
            }
        )
    return artifacts


def _rebase_staged_visual_bundle(
    staged: CorpusPaths,
    final: CorpusPaths,
) -> None:
    report = _read_json(staged.visual_report)
    report.update(
        {
            "manifest_path": final.visual_manifest.as_posix(),
            "index_path": final.visual_index.as_posix(),
            "frame_map_path": final.visual_frame_map.as_posix(),
        }
    )
    _atomic_write_json(staged.visual_report, report)
    hashes = {
        label: _sha256_file(path)
        for label, path in (
            ("index", staged.visual_index),
            ("metadata", staged.visual_metadata),
            ("frame_map", staged.visual_frame_map),
            ("report", staged.visual_report),
        )
    }
    manifest = _read_json(staged.visual_manifest)
    manifest.update(
        {
            "index_path": final.visual_index.as_posix(),
            "index_metadata_path": final.visual_metadata.as_posix(),
            "frame_map_path": final.visual_frame_map.as_posix(),
            "report_path": final.visual_report.as_posix(),
            "bundle_generation": _sha256_value(hashes),
            "artifacts": {
                label: {
                    "filename": final_path.name,
                    "sha256": hashes[label],
                }
                for label, final_path in (
                    ("index", final.visual_index),
                    ("metadata", final.visual_metadata),
                    ("frame_map", final.visual_frame_map),
                    ("report", final.visual_report),
                )
            },
        }
    )
    _atomic_write_json(staged.visual_manifest, manifest)


def _rebase_staged_dense_bundle(
    staged: CorpusPaths,
    final: CorpusPaths,
) -> None:
    report = _read_json(staged.dense_report)
    report.update(
        {
            "manifest_path": final.dense_manifest.as_posix(),
            "index_path": final.dense_index.as_posix(),
            "metadata_path": final.dense_metadata.as_posix(),
            "frame_map_path": final.dense_frame_map.as_posix(),
        }
    )
    _atomic_write_json(staged.dense_report, report)
    hashes = {
        label: _sha256_file(path)
        for label, path in (
            ("index", staged.dense_index),
            ("metadata", staged.dense_metadata),
            ("frame_map", staged.dense_frame_map),
            ("report", staged.dense_report),
        )
    }
    manifest = _read_json(staged.dense_manifest)
    manifest.update(
        {
            "index_path": final.dense_index.as_posix(),
            "index_metadata_path": final.dense_metadata.as_posix(),
            "frame_map_path": final.dense_frame_map.as_posix(),
            "report_path": final.dense_report.as_posix(),
            "bundle_generation": _sha256_value(hashes),
            "artifacts": {
                label: {
                    "filename": final_path.name,
                    "sha256": hashes[label],
                }
                for label, final_path in (
                    ("index", final.dense_index),
                    ("metadata", final.dense_metadata),
                    ("frame_map", final.dense_frame_map),
                    ("report", final.dense_report),
                )
            },
        }
    )
    _atomic_write_json(staged.dense_manifest, manifest)


def _corpus_bundle_manifest_payload(
    staged: CorpusPaths,
    final: CorpusPaths,
    *,
    source_contract: Mapping[str, Any],
    video_ids: Sequence[str],
    bge_enabled: bool,
) -> dict[str, Any]:
    staged_artifacts = _corpus_bundle_artifacts(staged, bge_enabled=bge_enabled)
    final_artifacts = _corpus_bundle_artifacts(final, bge_enabled=bge_enabled)
    hashes = {
        role: _sha256_file(staged_artifacts[role])
        for role in staged_artifacts
    }
    return {
        "schema_version": CORPUS_BUNDLE_SCHEMA_VERSION,
        "status": "passed",
        "pipeline": PIPELINE_NAME,
        "source_contract_sha256": source_contract["contract_sha256"],
        "video_ids": list(video_ids),
        "bge_enabled": bge_enabled,
        "bundle_generation": _sha256_value(hashes),
        "artifacts": {
            role: {
                "path": path.relative_to(final.visual_index.parents[1]).as_posix(),
                "sha256": hashes[role],
            }
            for role, path in final_artifacts.items()
        },
    }


def _validate_corpus_bundle_commit(
    paths: CorpusPaths,
    *,
    source_contract: Mapping[str, Any],
    video_ids: Sequence[str],
    bge_enabled: bool,
) -> dict[str, Any]:
    manifest = _read_json(paths.corpus_manifest)
    if (
        manifest.get("schema_version") != CORPUS_BUNDLE_SCHEMA_VERSION
        or manifest.get("status") != "passed"
        or manifest.get("source_contract_sha256")
        != source_contract.get("contract_sha256")
        or manifest.get("video_ids") != list(video_ids)
        or manifest.get("bge_enabled") is not bge_enabled
    ):
        raise ValueError("corpus bundle commit does not match the requested sources")
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("corpus bundle manifest has no artifact declarations")
    artifacts = _corpus_bundle_artifacts(paths, bge_enabled=bge_enabled)
    if set(declared) != set(artifacts):
        raise ValueError("corpus bundle artifact set is incomplete")
    hashes: dict[str, str] = {}
    root = paths.visual_index.parents[1]
    for role, path in artifacts.items():
        item = declared.get(role)
        if not isinstance(item, dict):
            raise ValueError(f"corpus bundle declaration is invalid: {role}")
        if item.get("path") != path.relative_to(root).as_posix():
            raise ValueError(f"corpus bundle path lineage changed: {role}")
        digest = _sha256_file(path)
        if item.get("sha256") != digest:
            raise ValueError(f"corpus bundle checksum changed: {role}")
        hashes[role] = digest
    if manifest.get("bundle_generation") != _sha256_value(hashes):
        raise ValueError("corpus bundle generation does not match its artifacts")
    return manifest


def _publish_staged_corpus_bundle(
    staged: CorpusPaths,
    final: CorpusPaths,
    *,
    manifest: Mapping[str, Any],
    bge_enabled: bool,
) -> None:
    staged_artifacts = _corpus_bundle_artifacts(staged, bge_enabled=bge_enabled)
    final_artifacts = _corpus_bundle_artifacts(final, bge_enabled=bge_enabled)
    for path in staged_artifacts.values():
        if not path.is_file():
            raise FileNotFoundError(f"staged corpus artifact is missing: {path}")

    sentinel = {
        "schema_version": CORPUS_BUNDLE_SCHEMA_VERSION,
        "status": "publishing",
    }
    _atomic_write_json(final.corpus_manifest, sentinel)
    _atomic_write_json(
        final.visual_manifest,
        {
            "schema_version": "1.2",
            "publication_status": "publishing",
            "bundle_generation": "publishing",
            "artifacts": {},
        },
    )
    _atomic_write_json(
        final.dense_manifest,
        {
            "schema_version": DENSE_MANIFEST_SCHEMA_VERSION,
            "artifact_role": DENSE_ARTIFACT_ROLE,
            "publication_status": "publishing",
            "bundle_generation": "publishing",
            "artifacts": {},
        },
    )
    if bge_enabled:
        _atomic_write_json(
            BgeM3ArtifactPaths.from_root(final.bge_root).manifest,
            {"status": "publishing"},
        )

    commit_roles = {"visual_manifest", "dense_manifest", "bge_manifest"}
    for role, source in staged_artifacts.items():
        if role in commit_roles:
            continue
        destination = final_artifacts[role]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    for role in ("visual_manifest", "dense_manifest", "bge_manifest"):
        if role not in staged_artifacts:
            continue
        destination = final_artifacts[role]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_artifacts[role], destination)
    _atomic_write_json(final.corpus_manifest, manifest)


def build_corpus_indexes(
    videos: Sequence[VideoArtifacts],
    config: OfflinePipelineConfig,
) -> dict[str, Any]:
    """Build and atomically publish indexes plus neighbor/segment metadata."""

    if not videos:
        raise ValueError("Corpus indexing requires at least one successful video")
    ordered = tuple(sorted(videos, key=lambda item: item.video_id))
    if len({video.video_id.casefold() for video in ordered}) != len(ordered):
        raise ValueError("Corpus indexing received duplicate video IDs")

    # Revalidate the published bundles at the handoff boundary.  Existing files
    # alone are never sufficient evidence that a video is indexable.
    validated: list[VideoArtifacts] = []
    for video in ordered:
        validation, records = _validate_selected_bundle(
            video_path=video.video_path,
            config=config,
            paths=video.paths,
            require_completion=True,
        )
        validated.append(
            VideoArtifacts(
                video_id=video.video_id,
                video_path=video.video_path,
                paths=video.paths,
                selected_count=len(records),
                dense_candidate_count=int(validation["dense_candidate_count"]),
                skipped=video.skipped,
                validation=validation,
            )
        )
    ordered = tuple(validated)
    paths = CorpusPaths.from_config(config)
    source_contract = _corpus_source_contract(ordered, config)
    video_ids = tuple(video.video_id for video in ordered)
    canonical_records = load_canonical_keyframe_records(
        config.output_dir / "metadata",
        video_ids=video_ids,
    )
    _validate_unique_canonical_records(
        canonical_records,
        allowed_video_ids=set(video_ids),
    )
    selected_modalities = _load_explicit_selected_modalities(
        ordered,
        canonical_records,
    )

    if config.resume and not config.force:
        try:
            _validate_corpus_bundle_commit(
                paths,
                source_contract=source_contract,
                video_ids=video_ids,
                bge_enabled=config.bge_enabled,
            )
            _validate_visual_corpus_index(paths, videos=ordered)
            _validate_dense_corpus_index(paths, videos=ordered)
            _validate_text_corpus_index(paths, canonical_records)
            _validate_neighbor_corpus_metadata(
                paths,
                canonical_records=canonical_records,
                config=config,
            )
            _validate_segment_corpus_metadata(
                paths,
                canonical_records=canonical_records,
                modalities=selected_modalities,
                config=config,
            )
            if config.bge_enabled:
                _validate_bge_corpus_index(
                    paths,
                    canonical_records=canonical_records,
                    config=config,
                )
            report = _read_json(paths.validation_report)
            if report.get("status") != "passed":
                raise ValueError("corpus report is not passed")
        except CHECKPOINT_INVALID_ERRORS as exc:
            LOGGER.info("[RUN] corpus bundle | checkpoint invalid: %s", exc)
        else:
            LOGGER.info("[SKIP] corpus bundle | generation is fully validated")
            return report

    staging_parent = config.output_dir / "reports" / "offline" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_parent) as temporary_dir:
        staged_root = Path(temporary_dir) / "corpus"
        staged_config = replace(
            config,
            output_dir=staged_root,
            resume=False,
            force=True,
            build_corpus=True,
        )
        staged_paths = CorpusPaths.from_config(staged_config)
        visual = _build_visual_corpus_index(
            staged_paths,
            ordered,
            source_contract,
            staged_config,
        )
        dense = _build_dense_corpus_index(
            staged_paths,
            ordered,
            source_contract,
            staged_config,
        )
        text = _build_text_corpus_index(
            staged_paths,
            canonical_records,
            source_contract,
            staged_config,
        )
        neighbors = _build_neighbor_corpus_metadata(
            staged_paths,
            canonical_records,
            source_contract,
            staged_config,
        )
        segments = _build_segment_corpus_metadata(
            staged_paths,
            canonical_records,
            selected_modalities,
            source_contract,
            staged_config,
        )
        bge: Mapping[str, Any] | None = None
        if config.bge_enabled:
            bge = _build_bge_corpus_index(
                staged_paths,
                canonical_records,
                source_contract,
                staged_config,
            )
        else:
            LOGGER.info("[SKIP] corpus :: BGE-M3 | disabled by configuration")

        _rebase_staged_visual_bundle(staged_paths, paths)
        _rebase_staged_dense_bundle(staged_paths, paths)
        visual = _validate_visual_corpus_index(staged_paths, videos=ordered)
        dense = _validate_dense_corpus_index(staged_paths, videos=ordered)
        report = {
            "pipeline": PIPELINE_NAME,
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": "passed",
            "partial_corpus_allowed": config.allow_partial_corpus,
            "video_ids": list(video_ids),
            "video_count": len(video_ids),
            "dense_candidate_count": sum(
                video.dense_candidate_count for video in ordered
            ),
            "selected_keyframe_count": sum(video.selected_count for video in ordered),
            "source_contract_sha256": source_contract["contract_sha256"],
            "visual_index": visual,
            "dense_candidate_index": dense,
            "text_index": text,
            "neighbor_mapping": neighbors,
            "segments_events": segments,
            "bge_m3": bge,
            "artifacts": {
                "visual_index": paths.visual_index.as_posix(),
                "visual_frame_map": paths.visual_frame_map.as_posix(),
                "visual_manifest": paths.visual_manifest.as_posix(),
                "dense_index": paths.dense_index.as_posix(),
                "dense_metadata": paths.dense_metadata.as_posix(),
                "dense_frame_map": paths.dense_frame_map.as_posix(),
                "dense_manifest": paths.dense_manifest.as_posix(),
                "text_index": paths.text_index.as_posix(),
                "neighbor_metadata": paths.neighbor_metadata.as_posix(),
                "segment_metadata": paths.segment_metadata.as_posix(),
                "bge_root": paths.bge_root.as_posix() if config.bge_enabled else None,
                "corpus_manifest": paths.corpus_manifest.as_posix(),
            },
        }
        _atomic_write_json(staged_paths.validation_report, report)
        _atomic_write_json(
            staged_paths.state_manifest,
            {
                "pipeline": PIPELINE_NAME,
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "status": "passed",
                "source_contract_sha256": source_contract["contract_sha256"],
                "video_ids": list(video_ids),
                "visual_index": visual,
                "dense_candidate_index": dense,
                "text_index": text,
                "neighbor_mapping": neighbors,
                "segments_events": segments,
                "bge_m3": bge,
            },
        )
        bundle_manifest = _corpus_bundle_manifest_payload(
            staged_paths,
            paths,
            source_contract=source_contract,
            video_ids=video_ids,
            bge_enabled=config.bge_enabled,
        )
        _publish_staged_corpus_bundle(
            staged_paths,
            paths,
            manifest=bundle_manifest,
            bge_enabled=config.bge_enabled,
        )

    _validate_corpus_bundle_commit(
        paths,
        source_contract=source_contract,
        video_ids=video_ids,
        bge_enabled=config.bge_enabled,
    )
    _validate_visual_corpus_index(paths, videos=ordered)
    _validate_dense_corpus_index(paths, videos=ordered)
    _validate_text_corpus_index(paths, canonical_records)
    _validate_neighbor_corpus_metadata(
        paths,
        canonical_records=canonical_records,
        config=config,
    )
    _validate_segment_corpus_metadata(
        paths,
        canonical_records=canonical_records,
        modalities=selected_modalities,
        config=config,
    )
    if config.bge_enabled:
        _validate_bge_corpus_index(
            paths,
            canonical_records=canonical_records,
            config=config,
        )
    report = _read_json(paths.validation_report)
    LOGGER.info(
        "[DONE] OFFLINE PIPELINE COMPLETE | videos=%d selected=%d",
        len(video_ids),
        report["selected_keyframe_count"],
    )
    return report


def _parse_siglip_batch_size(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "batch size must be a positive integer or 'auto'"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical sequential offline pipeline: finish every stage "
            "for video A before video B, then build corpus indexes once."
        )
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/raw/video"),
        help="Directory used for full-dataset discovery or --video-id lookup.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video-path", type=Path, help="Quick mode: process one video path.")
    source.add_argument("--video-id", help="Quick mode: process one video ID from --video-dir.")
    parser.add_argument("--video-glob", default="*.mp4", help="Full-dataset direct-file glob.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--dense-interval", type=float, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--boundary-guard-sec", type=float, default=DEFAULT_BOUNDARY_GUARD_SEC)
    parser.add_argument("--tiny-shot-max-sec", type=float, default=DEFAULT_TINY_SHOT_MAX_SEC)
    parser.add_argument("--include-video-endpoints", action="store_true")
    parser.add_argument("--shot-threshold", type=float, default=0.5)
    parser.add_argument("--shot-device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)

    parser.add_argument("--max-gap-seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS)
    parser.add_argument("--gap-tolerance-seconds", type=float, default=0.0)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target-keyframes", type=int)
    target.add_argument("--target-density-per-second", type=float)
    parser.add_argument("--hard-max-keyframes", type=int)
    parser.add_argument(
        "--no-event-aware-dedup",
        dest="enable_event_aware_dedup",
        action="store_false",
        default=True,
        help="Disable the canonical event-aware dedup pass.",
    )

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--batch-size",
        "--siglip-batch-size",
        dest="siglip_batch_size",
        type=_parse_siglip_batch_size,
        default="auto",
        help="SigLIP2 batch size or auto.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--caption-batch-size", type=int, default=2)
    parser.add_argument("--ocr-batch-size", type=int, default=4)
    parser.add_argument("--object-batch-size", type=int, default=8)
    parser.add_argument("--bge-batch-size", type=int, default=16)
    parser.add_argument("--siglip-model-revision", default=None)
    parser.add_argument(
        "--caption-model-name",
        default=os.getenv("CAPTION_MODEL", DEFAULT_CAPTION_MODEL),
    )
    parser.add_argument(
        "--caption-model-revision",
        default=os.getenv("CAPTION_MODEL_REVISION", DEFAULT_CAPTION_REVISION),
    )
    parser.add_argument(
        "--caption-model-cache-dir",
        type=Path,
        default=Path(
            os.getenv("CAPTION_MODEL_CACHE_DIR", str(DEFAULT_CAPTION_CACHE_DIR))
        ),
    )
    parser.add_argument(
        "--caption-task-prompt",
        default=os.getenv("CAPTION_TASK_PROMPT", DEFAULT_CAPTION_TASK_PROMPT),
        help="Florence-2 task token used for keyframe captioning.",
    )
    parser.add_argument("--caption-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--caption-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--caption-quantization",
        choices=("none", "8bit", "4bit"),
        default="none",
        help="Only 'none' is supported for the Florence-2 caption backend.",
    )
    parser.add_argument("--bge-model-revision", default=DEFAULT_BGE_M3_REVISION)
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--skip-bge", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--neighbor-window-seconds",
        type=float,
        default=5.0,
        help="Map selected keyframes to selected neighbors within this time window.",
    )
    parser.add_argument(
        "--segment-strategy",
        choices=("auto", "boundary", "fixed"),
        default="auto",
        help="Use shot/segment lineage when available, or fixed temporal windows.",
    )
    parser.add_argument(
        "--segment-fixed-duration-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--segment-caption-similarity-threshold",
        type=float,
        default=0.92,
    )

    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute per-video and corpus stages even when checkpoints validate.",
    )
    parser.add_argument(
        "--allow-partial-corpus",
        action="store_true",
        help="Build indexes from successful requested videos when another video failed.",
    )
    corpus_mode = parser.add_mutually_exclusive_group()
    corpus_mode.add_argument(
        "--build-corpus",
        dest="build_corpus",
        action="store_true",
        help=(
            "Build/replace global indexes from exactly the requested videos. "
            "This is the full-dataset default and must be explicit in quick mode."
        ),
    )
    corpus_mode.add_argument(
        "--skip-corpus",
        dest="build_corpus",
        action="store_false",
        help="Publish only per-video artifacts and leave existing global indexes untouched.",
    )
    parser.set_defaults(build_corpus=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _discover_videos(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.video_path is not None:
        path = Path(args.video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video does not exist: {path}")
        return (path,)
    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    if args.video_id is not None:
        video_id = str(args.video_id).strip()
        if not video_id:
            raise ValueError("--video-id must not be empty")
        if Path(video_id).name != video_id or any(char in video_id for char in "*?[]"):
            raise ValueError("--video-id must be a plain filename stem")
        matches = sorted(
            path
            for path in video_dir.glob(f"{video_id}.*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_VIDEO_SUFFIXES
        )
        if len(matches) != 1:
            raise ValueError(
                f"--video-id {video_id!r} resolved to {len(matches)} supported files "
                f"under {video_dir}: {[path.name for path in matches]}"
            )
        return tuple(matches)
    matches = tuple(
        path
        for path in sorted(video_dir.glob(args.video_glob))
        if path.is_file() and path.suffix.casefold() in SUPPORTED_VIDEO_SUFFIXES
    )
    if not matches:
        raise FileNotFoundError(
            f"No supported videos matched {args.video_glob!r} under {video_dir}"
        )
    return matches


def _config_from_args(args: argparse.Namespace) -> OfflinePipelineConfig:
    output_dir = Path(args.output_dir)
    target_density = args.target_density_per_second
    if args.target_keyframes is None and target_density is None:
        target_density = DEFAULT_TARGET_DENSITY_PER_SECOND
    quick_mode = args.video_path is not None or args.video_id is not None
    build_corpus = args.build_corpus
    if build_corpus is None:
        build_corpus = not quick_mode
    return OfflinePipelineConfig(
        output_dir=output_dir,
        device=args.device,
        resume=args.resume,
        force=args.force,
        allow_partial_corpus=args.allow_partial_corpus,
        build_corpus=build_corpus,
        shot_threshold=args.shot_threshold,
        shot_device=args.shot_device or args.device,
        dense_interval_sec=args.dense_interval,
        boundary_guard_sec=args.boundary_guard_sec,
        tiny_shot_max_sec=args.tiny_shot_max_sec,
        include_video_endpoints=args.include_video_endpoints,
        jpeg_quality=args.jpeg_quality,
        max_gap_seconds=args.max_gap_seconds,
        gap_tolerance_seconds=args.gap_tolerance_seconds,
        target_keyframes=args.target_keyframes,
        target_density_per_second=target_density,
        hard_max_keyframes=args.hard_max_keyframes,
        enable_event_aware_dedup=args.enable_event_aware_dedup,
        siglip_model_revision=args.siglip_model_revision,
        siglip_model_cache_dir=output_dir / "model_cache" / "siglip2",
        siglip_batch_size=args.siglip_batch_size,
        siglip_num_workers=args.num_workers,
        siglip_use_autocast=not args.no_autocast,
        caption_model_name=args.caption_model_name,
        caption_model_revision=args.caption_model_revision,
        caption_model_cache_dir=args.caption_model_cache_dir,
        caption_batch_size=args.caption_batch_size,
        caption_max_new_tokens=args.caption_max_new_tokens,
        caption_dtype=args.caption_dtype,
        caption_quantization=args.caption_quantization,
        caption_task_prompt=args.caption_task_prompt,
        ocr_model_cache_dir=output_dir / "model_cache" / "ocr",
        ocr_batch_size=args.ocr_batch_size,
        object_model_cache_dir=output_dir / "model_cache" / "objects",
        object_batch_size=args.object_batch_size,
        bge_enabled=not args.skip_bge,
        bge_model_revision=args.bge_model_revision,
        bge_batch_size=args.bge_batch_size,
        bge_local_files_only=args.local_files_only,
        bge_model_cache_dir=output_dir / "model_cache" / "bge_m3",
        neighbor_window_seconds=args.neighbor_window_seconds,
        segment_strategy=args.segment_strategy,
        segment_fixed_duration_seconds=args.segment_fixed_duration_seconds,
        segment_caption_similarity_threshold=(
            args.segment_caption_similarity_threshold
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    try:
        videos = _discover_videos(args)
        config = _config_from_args(args)
        result = process_dataset(videos, config)
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.error("[FATAL] %s: %s", type(exc).__name__, exc)
        return 2

    summary = {
        "status": "passed" if result.complete else "failed",
        "requested_video_count": len(result.requested_videos),
        "successful_video_count": len(result.successful_videos),
        "failed_video_count": len(result.failures),
        "corpus_blocked": result.corpus_blocked,
        "corpus_skipped": result.corpus_skipped,
        "corpus_indexed": result.corpus_result is not None,
        "failures": [failure.to_dict() for failure in result.failures],
    }
    LOGGER.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.complete else 1


__all__ = [
    "CorpusIndexError",
    "DatasetProcessResult",
    "DenseFeatureArtifacts",
    "MaterializedStageResult",
    "OfflinePipelineConfig",
    "OfflineStageError",
    "PerVideoPaths",
    "VideoArtifacts",
    "VideoFailure",
    "build_corpus_indexes",
    "build_parser",
    "main",
    "process_dataset",
    "process_video",
]


if __name__ == "__main__":
    raise SystemExit(main())
