"""Run the public TKIS/VKIS competition with the repository's existing services.

This module deliberately contains only competition-specific orchestration:
CSV contracts, artifact paths, multimodal stage wiring, VKIS frame refinement,
and submission writing. Extraction, SigLIP2, FAISS, caption/OCR/object,
segment/text indexing, hybrid reranking, and image MSE are delegated to the
implementations already present under ``backend.app`` and ``src.indexing``.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.indexing.build_text_index import (
    load_records as load_text_records,
    write_text_index,
)
from backend.app.services.indexing.build_faiss_index import build_faiss_artifacts
from backend.app.services.indexing.build_siglip2_index import (
    ARTIFACT_TAG,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    choose_device,
    compute_dtype_for,
    dtype_name,
    encode_keyframes,
    is_cuda_oom,
    load_jsonl,
    load_siglip2_model_processor,
    parse_batch_size,
    resolve_model_revision,
    validate_embedding_artifacts,
    write_json,
    write_jsonl,
)
from backend.app.services.indexing.extract_keyframes import (
    KEYFRAME_STRATEGIES,
    KEYFRAME_STRATEGY_DENSE_COVERAGE,
    extract_keyframes_for_video,
)
from backend.app.services.indexing.keyframe_candidates import (
    DEFAULT_BOUNDARY_GUARD_SEC,
    DEFAULT_INTERVAL_SEC,
    DEFAULT_TINY_SHOT_MAX_SEC,
)
from backend.app.services.indexing.keyframe_feature_adapter import (
    FeatureAdapterConfig,
)
from backend.app.services.indexing.keyframe_multimodal_pipeline import (
    run_multimodal_keyframe_pipeline,
)
from backend.app.services.indexing.keyframe_selection import (
    DEFAULT_MAX_GAP_SECONDS,
    SelectionConfig,
)
from backend.app.services.indexing.normalize_keyframe_metadata import (
    image_to_small_array,
    mse,
)
from backend.app.services.indexing.validate_keyframes import validate_records
from backend.app.services.ingestion.caption_pipeline import (
    DEFAULT_MODEL_NAME as DEFAULT_CAPTION_MODEL_NAME,
    DEFAULT_MODEL_REVISION as DEFAULT_CAPTION_MODEL_REVISION,
    DEFAULT_PROMPT as DEFAULT_CAPTION_PROMPT,
    QwenCaptionBackend,
    run_caption_file,
)
from backend.app.services.ingestion.object_pipeline import (
    DEFAULT_MODEL_NAME as DEFAULT_OBJECT_MODEL_NAME,
    DEFAULT_MODEL_REVISION as DEFAULT_OBJECT_MODEL_REVISION,
    DEFAULT_VOCABULARY as DEFAULT_OBJECT_VOCABULARY,
    YoloEBackend,
    run_object_file,
)
from backend.app.services.ingestion.ocr_pipeline import (
    DEFAULT_DETECTION_MODEL as DEFAULT_OCR_DETECTION_MODEL,
    DEFAULT_MODEL_REVISION as DEFAULT_OCR_MODEL_REVISION,
    DEFAULT_RECOGNITION_MODEL as DEFAULT_OCR_RECOGNITION_MODEL,
    PaddleOcrBackend,
    run_ocr_file,
)
from backend.app.services.metadata.metadata_store import FrameRecord, MetadataStore
from backend.app.services.agent.query_expansion import (
    build_production_query_expansion_provider,
)
from backend.app.services.retrieval.hybrid_search import (
    HybridSearchConfig,
    HybridSearchEngine,
)
from backend.app.services.retrieval.rerank import HybridReranker
from backend.app.services.retrieval.retrieval_config import (
    DEFAULT_RETRIEVAL_CONFIG_PATH,
    load_retrieval_runtime_config,
)
from backend.app.services.retrieval.search_caption import CaptionSearchEngine
from backend.app.services.retrieval.search_object import ObjectSearchEngine
from backend.app.services.retrieval.search_ocr import OcrSearchEngine
from backend.app.services.retrieval.text_index import (
    INDEX_VERSION as TEXT_INDEX_VERSION,
    TextIndexSearcher,
)
from backend.app.services.retrieval.advanced_search import (
    AdvancedSearchConfig,
    advanced_text_search,
    advanced_vector_search,
)
from backend.app.services.retrieval.advanced_rerank import AdvancedRankedFrame
from backend.app.services.retrieval.query_plan import QueryPlan, build_query_plan
from backend.app.services.retrieval.vlm_reranker import (
    DEFAULT_VLM_MODEL,
    VLM_MODES,
    build_local_vlm_runner,
    rerank_with_vlm,
)
from backend.app.services.retrieval.temporal_search import decompose_temporal_query
from backend.app.services.retrieval.search_visual import (
    FaissVectorSearcher,
    Siglip2TextEncoder,
    VisualSearchConfig,
    VisualSearchEngine,
    load_encoder_contract,
    normalize_query_vector,
)
from src.indexing.build_neighbor_index import build_neighbor_index
from src.indexing.build_segment_metadata import build_segment_metadata

from competition.downstream_lineage import (
    validate_stage_manifest,
    write_stage_manifest,
)
from competition.dense_index import (
    DenseCandidateIndex,
    build_dense_index,
    resolve_run_reference,
    validate_dense_index,
)
from competition.keyframe_phase3 import (
    PHASE3_FEATURE_CONTRACT_VERSION,
    PHASE3_SELECTION_CONTRACT_VERSION,
    CandidatePoolConfig,
    Phase3WorkspacePaths,
    atomic_copy,
    atomic_save_npy,
    atomic_write_json,
    atomic_write_jsonl,
    materialize_candidate_pool,
    read_json as read_phase3_json,
    read_jsonl as read_phase3_jsonl,
    sha256_file as phase3_sha256_file,
    sha256_json,
    validate_candidate_pool,
    workspace_paths,
)
from competition.keyframe_phase5 import (
    evaluate_split_artifacts,
    load_split_manifest,
    write_config_lock,
    write_split_manifest,
)
from competition.run_manifest import (
    initialize_run_manifest,
    git_fingerprint,
    promote_active_run,
    record_leaderboard_score,
    record_submission,
    set_active_baseline,
    update_run_manifest,
)


DEFAULT_PUBLIC_ROOT = Path("data/public")
DEFAULT_OUTPUT_ROOT = Path("competition")
ANSWER_COUNT = 100
SUPPORTED_TASKS = {"TKIS", "VKIS"}
EXTRACTOR_CONTRACT_VERSION = 1
EMBEDDING_LINEAGE_VERSION = 1
INDEX_LINEAGE_VERSION = 1


@dataclass(frozen=True)
class CorpusVideo:
    filename: str
    relative_path: Path
    fps: float
    frame_count: int

    @property
    def video_id(self) -> str:
        return Path(self.filename).stem


@dataclass(frozen=True)
class Question:
    query_id: str
    task: str
    text: str
    query_image: str


@dataclass(frozen=True)
class RankedFrame:
    record: FrameRecord
    score: float


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def load_corpus(public_root: Path) -> list[CorpusVideo]:
    expected = [
        "video",
        "path",
        "duration_seconds",
        "fps",
        "frame_count",
        "width",
        "height",
    ]
    header, rows = _read_csv(public_root / "corpus.csv")
    if header != expected:
        raise ValueError(f"Unexpected corpus.csv header: {header}; expected {expected}")

    corpus: list[CorpusVideo] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        filename = (row.get("video") or "").strip()
        relative_path = (row.get("path") or "").strip()
        if not filename or not relative_path:
            raise ValueError(f"corpus.csv line {line_number}: video/path must not be empty")
        if filename in seen:
            raise ValueError(f"corpus.csv line {line_number}: duplicate video {filename!r}")
        try:
            fps = float(row["fps"])
            frame_count = int(row["frame_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"corpus.csv line {line_number}: invalid fps/frame_count"
            ) from exc
        if fps <= 0 or frame_count <= 0:
            raise ValueError(
                f"corpus.csv line {line_number}: fps/frame_count must be positive"
            )
        seen.add(filename)
        corpus.append(
            CorpusVideo(
                filename=filename,
                relative_path=Path(relative_path),
                fps=fps,
                frame_count=frame_count,
            )
        )
    if not corpus:
        raise ValueError("corpus.csv contains no videos")
    if len(corpus) != 250:
        raise ValueError(f"corpus.csv must contain exactly 250 videos, got {len(corpus)}")
    return corpus


def load_questions(public_root: Path) -> list[Question]:
    expected = ["query_id", "task", "text", "query_image"]
    header, rows = _read_csv(public_root / "questions.csv")
    if header != expected:
        raise ValueError(f"Unexpected questions.csv header: {header}; expected {expected}")

    questions: list[Question] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        query_id = (row.get("query_id") or "").strip()
        task = (row.get("task") or "").strip().upper()
        text = (row.get("text") or "").strip()
        query_image = (row.get("query_image") or "").strip()
        if not query_id or query_id in seen:
            raise ValueError(
                f"questions.csv line {line_number}: query_id is empty or duplicated"
            )
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"questions.csv line {line_number}: unsupported task {task!r}"
            )
        if task == "TKIS" and not text:
            raise ValueError(f"questions.csv line {line_number}: TKIS text is empty")
        if task == "VKIS" and not query_image:
            raise ValueError(f"questions.csv line {line_number}: VKIS image is empty")
        seen.add(query_id)
        questions.append(Question(query_id, task, text, query_image))
    if not questions:
        raise ValueError("questions.csv contains no questions")
    if len(questions) != 100:
        raise ValueError(
            f"questions.csv must contain exactly 100 questions, got {len(questions)}"
        )
    task_counts = {
        task: sum(question.task == task for question in questions)
        for task in SUPPORTED_TASKS
    }
    if task_counts != {"TKIS": 50, "VKIS": 50}:
        raise ValueError(
            "questions.csv must contain exactly 50 TKIS and 50 VKIS questions; "
            f"got {task_counts}"
        )
    return questions


def submission_columns(public_root: Path) -> list[str]:
    header, _ = _read_csv(public_root / "sample_submission.csv")
    expected = ["query_id", *[f"answer_{index:03d}" for index in range(1, 101)]]
    if header != expected:
        raise ValueError("sample_submission.csv does not have the required 100 answer columns")
    return expected


def validate_input(public_root: Path, *, require_files: bool = True) -> dict:
    corpus = load_corpus(public_root)
    questions = load_questions(public_root)
    submission_columns(public_root)

    missing: list[str] = []
    if require_files:
        for video in corpus:
            path = public_root / video.relative_path
            if not path.is_file():
                missing.append(path.as_posix())
        for question in questions:
            if question.task == "VKIS":
                path = public_root / question.query_image
                if not path.is_file():
                    missing.append(path.as_posix())
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} public files; first entries: {preview}")

    task_counts = {
        task: sum(question.task == task for question in questions)
        for task in sorted(SUPPORTED_TASKS)
    }
    return {
        "public_root": public_root.as_posix(),
        "video_count": len(corpus),
        "question_count": len(questions),
        "task_counts": task_counts,
        "answer_columns": ANSWER_COUNT,
        "status": "passed",
    }


def extract_command(args: argparse.Namespace) -> None:
    public_root = args.public_root
    output_root = args.output_root
    corpus = load_corpus(public_root)
    for number, video in enumerate(corpus, start=1):
        video_path = public_root / video.relative_path
        metadata_path = output_root / "metadata" / f"keyframes_{video.video_id}.jsonl"
        report_path = output_root / "metadata" / f"keyframes_{video.video_id}_extract_report.json"
        if args.resume and _can_resume_extraction(
            video=video,
            video_path=video_path,
            metadata_path=metadata_path,
            report_path=report_path,
            requested_config=_competition_extract_config(args),
        ):
            print(
                f"[{number}/{len(corpus)}] skip {video.filename}: "
                "matching artifacts passed integrity checks"
            )
            continue
        print(f"[{number}/{len(corpus)}] extracting {video.filename}")
        report = extract_keyframes_for_video(
            video_path=video_path,
            output_dir=output_root / "keyframes",
            metadata_path=metadata_path,
            report_path=report_path,
            phash_threshold=args.phash_threshold,
            phash_window_sec=args.phash_window_sec,
            jpeg_quality=args.jpeg_quality,
            shot_threshold=args.shot_threshold,
            shot_device=args.device,
            strategy=args.keyframe_strategy,
            candidate_interval_sec=args.candidate_interval_sec,
            boundary_guard_sec=args.boundary_guard_sec,
            tiny_shot_max_sec=args.tiny_shot_max_sec,
            max_gap_seconds=args.max_gap_seconds,
            gap_tolerance_seconds=args.gap_tolerance_seconds,
            target_keyframes=args.target_keyframes,
            hard_max_keyframes=args.hard_max_keyframes,
        )
        requested_config = _competition_extract_config(args)
        _require_extractor_report_contract(
            report,
            requested_config,
            report_path,
            require_satisfied=False,
        )
        records = _require_keyframe_metadata_integrity(
            video=video,
            metadata_path=metadata_path,
            report=report,
        )
        report["extractor_contract_version"] = EXTRACTOR_CONTRACT_VERSION
        report["source_video_fingerprint"] = _file_stat_fingerprint(video_path)
        report["keyframe_metadata_sha256"] = _sha256_file(metadata_path)
        report["keyframe_frame_ids_sha256"] = _frame_ids_sha256(records)
        report["keyframe_images_sha256"] = _keyframe_images_sha256(records)
        report["competition_extract_config"] = requested_config
        write_json(report, report_path)
        _require_extractor_report_contract(report, requested_config, report_path)


def _competition_extract_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the exact competition extraction contract used for resume checks."""

    return {
        "keyframe_strategy": args.keyframe_strategy,
        "phash_threshold": args.phash_threshold,
        "phash_window_sec": args.phash_window_sec,
        "jpeg_quality": args.jpeg_quality,
        "shot_threshold": args.shot_threshold,
        "shot_device": args.device,
        "candidate_interval_sec": args.candidate_interval_sec,
        "boundary_guard_sec": args.boundary_guard_sec,
        "tiny_shot_max_sec": args.tiny_shot_max_sec,
        "max_gap_seconds": args.max_gap_seconds,
        "gap_tolerance_seconds": args.gap_tolerance_seconds,
        "target_keyframes": args.target_keyframes,
        "hard_max_keyframes": args.hard_max_keyframes,
    }


def _phase3_candidate_config(args: argparse.Namespace) -> CandidatePoolConfig:
    return CandidatePoolConfig(
        phash_threshold=args.phash_threshold,
        phash_window_sec=args.phash_window_sec,
        jpeg_quality=args.jpeg_quality,
        shot_threshold=args.shot_threshold,
        shot_device=args.device,
        candidate_interval_sec=args.candidate_interval_sec,
        boundary_guard_sec=args.boundary_guard_sec,
        tiny_shot_max_sec=args.tiny_shot_max_sec,
        include_video_endpoints=args.endpoint_protection == "on",
    )


def _phase3_adapter_config(args: argparse.Namespace) -> FeatureAdapterConfig:
    return FeatureAdapterConfig(
        ocr_min_confidence=args.ocr_event_min_confidence,
        object_min_confidence=args.object_event_min_confidence,
        transition_absolute_floor=args.transition_absolute_floor,
        transition_mad_multiplier=args.transition_mad_multiplier,
    )


def _phase3_selection_config(args: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        max_gap_seconds=args.max_gap_seconds,
        target_keyframes=args.target_keyframes,
        hard_max_keyframes=args.hard_max_keyframes,
        gap_tolerance_seconds=args.gap_tolerance_seconds,
        importance_weight=args.importance_weight,
        novelty_weight=args.novelty_weight,
        protect_each_shot=True,
        protect_video_endpoints=args.endpoint_protection == "on",
        enable_event_aware_dedup=True,
        target_density_per_second=args.target_density_per_second,
        dedup_similarity_threshold=args.dedup_similarity_threshold,
        dedup_temporal_window_seconds=args.dedup_temporal_window_seconds,
    )


def _phase3_feature_config(
    args: argparse.Namespace,
    *,
    resolved_device: str,
    resolved_model_revision: str,
) -> dict[str, object]:
    return {
        "siglip2": {
            "model_name": args.model_name,
            "requested_model_revision": args.model_revision,
            "resolved_model_revision": resolved_model_revision,
            "device": resolved_device,
            "use_autocast": not args.no_autocast,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
        },
        "caption": {
            "model_name": args.caption_model_name,
            "model_revision": args.caption_model_revision,
            "batch_size": args.caption_batch_size,
            "max_new_tokens": args.caption_max_new_tokens,
            "dtype": args.caption_dtype,
            "quantization": args.caption_quantization,
            "prompt": DEFAULT_CAPTION_PROMPT,
            "segment_caption": not args.no_segment_caption,
        },
        "ocr": {
            "detection_model": args.ocr_detection_model,
            "recognition_model": args.ocr_recognition_model,
            "model_revision": args.ocr_model_revision,
            "languages": ["vi", "en"],
            "batch_size": args.ocr_batch_size,
            "confidence_threshold": args.ocr_conf_threshold,
        },
        "objects": {
            "model_name": args.object_model_name,
            "model_revision": args.object_model_revision,
            "prompt_mode": args.object_prompt_mode,
            "vocabulary": list(args.object_vocabulary),
            "batch_size": args.object_batch_size,
            "confidence_threshold": args.object_conf_threshold,
            "iou_threshold": args.object_iou_threshold,
        },
        "device": resolved_device,
    }


def _phase3_record_candidate_id(record: Mapping[str, object]) -> str:
    value = record.get("candidate_id") or record.get("frame_id")
    if not isinstance(value, str) or not value:
        raise RuntimeError("Phase 3 feature record has no candidate/frame identity")
    return value


def _phase3_exact_frame_artifact(
    records: Sequence[Mapping[str, object]],
    candidate_ids: Sequence[str],
    *,
    require_success: bool,
    name: str,
) -> bool:
    try:
        identities = [_phase3_record_candidate_id(record) for record in records]
    except RuntimeError:
        return False
    if len(identities) != len(set(identities)) or set(identities) != set(candidate_ids):
        return False
    if require_success and any(record.get("status", "success") != "success" for record in records):
        return False
    if any(
        record.get("video_id") != records[0].get("video_id")
        for record in records[1:]
    ):
        return False
    return True


def _phase3_artifact_entry(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": phase3_sha256_file(path),
        "size": path.stat().st_size,
    }


def _phase3_load_and_validate_features(
    *,
    video: CorpusVideo,
    paths: Phase3WorkspacePaths,
    candidate_report: Mapping[str, object],
    candidate_records: Sequence[Mapping[str, object]],
    feature_config: Mapping[str, object],
    allow_partial_features: bool,
    require_manifest: bool,
) -> dict[str, object]:
    required_paths = (
        paths.embeddings,
        paths.embedding_metadata,
        paths.captions,
        paths.ocr,
        paths.objects,
    )
    if any(not path.is_file() for path in required_paths):
        raise FileNotFoundError("one or more Phase 3 feature artifacts are missing")

    embeddings = np.load(paths.embeddings, allow_pickle=False)
    embedding_records = read_phase3_jsonl(paths.embedding_metadata)
    validation = validate_embedding_artifacts(embeddings, embedding_records)
    candidate_ids = [
        _phase3_record_candidate_id(record) for record in candidate_records
    ]
    embedded_ids = [
        _phase3_record_candidate_id(record) for record in embedding_records
    ]
    if embedded_ids != candidate_ids:
        raise RuntimeError("dense SigLIP2 identities/order do not match candidate pool")

    captions = read_phase3_jsonl(paths.captions)
    ocr = read_phase3_jsonl(paths.ocr)
    objects = read_phase3_jsonl(paths.objects)
    caption_complete = _phase3_exact_frame_artifact(
        captions,
        candidate_ids,
        require_success=False,
        name="caption",
    )
    ocr_complete = _phase3_exact_frame_artifact(
        ocr,
        candidate_ids,
        require_success=True,
        name="ocr",
    )
    object_complete = _phase3_exact_frame_artifact(
        objects,
        candidate_ids,
        require_success=True,
        name="objects",
    )
    if not caption_complete:
        raise RuntimeError("caption artifact identities do not match candidate pool")
    if (not ocr_complete or not object_complete) and not allow_partial_features:
        raise RuntimeError(
            "OCR/object hard-feature extraction is incomplete; rerun Phase 3 or "
            "use --allow-partial-features for an explicit degraded run"
        )

    hard_feature_complete = ocr_complete and object_complete
    artifact_paths = {
        "embeddings": paths.embeddings,
        "embedding_metadata": paths.embedding_metadata,
        "captions": paths.captions,
        "ocr": paths.ocr,
        "objects": paths.objects,
    }
    artifacts = {
        name: _phase3_artifact_entry(path)
        for name, path in artifact_paths.items()
    }
    expected_manifest = {
        "version": PHASE3_FEATURE_CONTRACT_VERSION,
        "video_id": video.video_id,
        "candidate_pool_run_id": paths.run_id,
        "candidate_metadata_sha256": candidate_report["candidate_metadata_sha256"],
        "candidate_images_sha256": candidate_report["candidate_images_sha256"],
        "candidate_frame_ids_sha256": candidate_report[
            "candidate_frame_ids_sha256"
        ],
        "candidate_count": len(candidate_records),
        "feature_config": dict(feature_config),
        "allow_partial_features": allow_partial_features,
        "hard_feature_complete": hard_feature_complete,
        "status": "passed" if hard_feature_complete else "degraded",
        "embedding_validation": validation,
        "artifacts": artifacts,
    }
    if require_manifest:
        stored = read_phase3_json(paths.feature_manifest)
        if stored != expected_manifest:
            raise RuntimeError("Phase 3 feature manifest is stale or was tampered with")
    return {
        "embeddings": embeddings,
        "embedding_records": embedding_records,
        "caption_records": captions,
        "ocr_records": ocr,
        "object_records": objects,
        "manifest": expected_manifest,
    }


def _phase3_feature_artifacts_current(
    **kwargs: object,
) -> bool:
    paths = kwargs.get("paths")
    if not isinstance(paths, Phase3WorkspacePaths) or not paths.feature_manifest.is_file():
        return False
    try:
        _phase3_load_and_validate_features(require_manifest=True, **kwargs)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        return False
    return True


def _phase3_siglip_artifacts_current(
    *,
    paths: Phase3WorkspacePaths,
    candidate_records: Sequence[Mapping[str, object]],
    feature_config: Mapping[str, object],
) -> bool:
    """Validate the durable SigLIP2 sub-stage before reusing it.

    The complete feature manifest is written after all visual modalities.  A
    process-level CUDA/native crash during the earlier SigLIP2 sweep therefore
    needs this narrower checkpoint or every already-encoded video would be run
    again on resume.
    """

    required = (
        paths.embeddings,
        paths.embedding_metadata,
        paths.embedding_skipped,
        paths.embedding_benchmark,
    )
    if any(not path.is_file() for path in required):
        return False
    siglip_config = feature_config.get("siglip2")
    if not isinstance(siglip_config, Mapping):
        return False
    try:
        embeddings = np.load(paths.embeddings, allow_pickle=False)
        embedding_records = read_phase3_jsonl(paths.embedding_metadata)
        skipped = read_phase3_jsonl(paths.embedding_skipped)
        benchmark = read_phase3_json(paths.embedding_benchmark)
        validate_embedding_artifacts(embeddings, embedding_records)
        candidate_ids = [
            _phase3_record_candidate_id(record) for record in candidate_records
        ]
        embedded_ids = [
            _phase3_record_candidate_id(record) for record in embedding_records
        ]
        device = str(siglip_config["device"])
        expected = {
            "model_family": "siglip2",
            "model_name": siglip_config["model_name"],
            "model_revision": siglip_config["resolved_model_revision"],
            "device": device,
            "compute_dtype": dtype_name(
                compute_dtype_for(device, bool(siglip_config["use_autocast"]))
            ),
            "output_dtype": "float32",
            "normalized": True,
            "requested_batch_size": siglip_config["batch_size"],
            "num_workers": siglip_config["num_workers"],
            "prefetch_factor": siglip_config["prefetch_factor"],
            "input_record_count": len(candidate_records),
            "encoded_count": len(candidate_records),
            "skipped_count": 0,
            "embedding_shape": list(embeddings.shape),
        }
    except (KeyError, OSError, UnicodeError, TypeError, ValueError, RuntimeError):
        return False
    return (
        not skipped
        and embedded_ids == candidate_ids
        and all(benchmark.get(key) == value for key, value in expected.items())
    )


def _phase3_path_matches(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return Path(value).resolve() == expected.resolve()
    except OSError:
        return False


def _phase3_frame_modality_artifacts_current(
    *,
    paths: Phase3WorkspacePaths,
    candidate_records: Sequence[Mapping[str, object]],
    output_path: Path,
    report_path: Path,
    pipeline: str,
    expected_report: Mapping[str, object],
    require_success: bool,
) -> bool:
    """Validate a complete frame-level modality without a final manifest."""

    if not output_path.is_file() or not report_path.is_file():
        return False
    try:
        records = read_phase3_jsonl(output_path)
        report = read_phase3_json(report_path)
        candidate_ids = [
            _phase3_record_candidate_id(record) for record in candidate_records
        ]
    except (OSError, UnicodeError, ValueError, RuntimeError):
        return False
    if not _phase3_exact_frame_artifact(
        records,
        candidate_ids,
        require_success=require_success,
        name=pipeline,
    ):
        return False
    required_report = {
        "pipeline": pipeline,
        "input_record_count": len(candidate_records),
        "input_path": paths.candidate_metadata,
        "output_path": output_path,
        **dict(expected_report),
    }
    for key, expected in required_report.items():
        if key in {"input_path", "output_path"}:
            if not _phase3_path_matches(report.get(key), expected):
                return False
        elif report.get(key) != expected:
            return False
    if require_success and report.get("error_count") != 0:
        return False
    return True


def _phase3_release_model_memory(*owners: object) -> None:
    """Best-effort release between heavyweight Phase 3 modalities."""

    for owner in owners:
        if owner is None:
            continue
        if hasattr(owner, "to"):
            try:
                owner.to("cpu")
            except Exception:  # noqa: BLE001 - cleanup must not hide stage errors.
                pass
        for attribute in ("_model", "_reader", "_processor"):
            if not hasattr(owner, attribute):
                continue
            value = getattr(owner, attribute, None)
            if value is not None and hasattr(value, "to"):
                try:
                    value.to("cpu")
                except Exception:  # noqa: BLE001 - backend-specific cleanup.
                    pass
            try:
                setattr(owner, attribute, None)
            except Exception:  # noqa: BLE001 - some third-party objects are immutable.
                pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - SigLIP already requires torch in production.
        pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_ids_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    seen: set[str] = set()
    for offset, record in enumerate(records):
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise RuntimeError(f"Missing frame_id at metadata row {offset}")
        if frame_id in seen:
            raise RuntimeError(f"Duplicate frame_id in metadata: {frame_id}")
        seen.add(frame_id)
        digest.update(frame_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _keyframe_images_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for offset, record in enumerate(records):
        frame_id = record.get("frame_id")
        keyframe_path = record.get("keyframe_path")
        if not isinstance(frame_id, str) or not isinstance(keyframe_path, str):
            raise RuntimeError(f"Invalid keyframe record at metadata row {offset}")
        digest.update(frame_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(Path(keyframe_path)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_stat_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_extractor_report_contract(
    report: Mapping[str, object],
    requested_config: Mapping[str, object],
    report_path: Path,
    *,
    require_satisfied: bool = True,
) -> None:
    strategy = requested_config["keyframe_strategy"]
    if report.get("keyframe_strategy") != strategy:
        raise RuntimeError(
            f"Extractor strategy mismatch for {report_path}: "
            f"{report.get('keyframe_strategy')!r} != {strategy!r}"
        )
    for field in (
        "phash_threshold",
        "phash_window_sec",
        "jpeg_quality",
        "shot_threshold",
        "shot_device",
    ):
        if report.get(field) != requested_config[field]:
            raise RuntimeError(
                f"Extractor report mismatch for {field} in {report_path}: "
                f"{report.get(field)!r} != {requested_config[field]!r}"
            )
    if strategy != KEYFRAME_STRATEGY_DENSE_COVERAGE:
        return

    direct_fields = (
        "candidate_interval_sec",
        "boundary_guard_sec",
        "tiny_shot_max_sec",
    )
    for field in direct_fields:
        if report.get(field) != requested_config[field]:
            raise RuntimeError(
                f"Extractor report mismatch for {field} in {report_path}: "
                f"{report.get(field)!r} != {requested_config[field]!r}"
            )
    selection_config = report.get("selection_config")
    if not isinstance(selection_config, Mapping):
        raise RuntimeError(f"Dense extraction report has no selection_config: {report_path}")
    for field in (
        "max_gap_seconds",
        "gap_tolerance_seconds",
        "target_keyframes",
        "hard_max_keyframes",
    ):
        if selection_config.get(field) != requested_config[field]:
            raise RuntimeError(
                f"Extractor selection mismatch for {field} in {report_path}: "
                f"{selection_config.get(field)!r} != {requested_config[field]!r}"
            )
    if require_satisfied and (
        report.get("status") != "satisfied"
        or report.get("constraints_satisfied") is not True
        or report.get("coverage_satisfied") is not True
    ):
        selection = report.get("selection")
        reason = (
            selection.get("stop_reason")
            if isinstance(selection, Mapping)
            else report.get("status", "unknown")
        )
        raise RuntimeError(
            f"Dense keyframe constraints were not satisfied: {reason}; "
            f"see {report_path}"
        )


def _require_keyframe_metadata_integrity(
    *,
    video: CorpusVideo,
    metadata_path: Path,
    report: Mapping[str, object],
) -> list[dict]:
    records = load_jsonl(metadata_path)
    expected_count = report.get("keyframe_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise RuntimeError(f"Invalid keyframe_count in extraction report: {expected_count!r}")
    if len(records) != expected_count:
        raise RuntimeError(
            f"Keyframe metadata count mismatch for {video.video_id}: "
            f"{len(records)} != {expected_count}"
        )
    frame_ids: set[str] = set()
    for offset, record in enumerate(records):
        frame_id = record.get("frame_id")
        if record.get("video_id") != video.video_id or not isinstance(frame_id, str):
            raise RuntimeError(
                f"Invalid keyframe identity for {video.video_id} at row {offset}"
            )
        if frame_id in frame_ids:
            raise RuntimeError(f"Duplicate keyframe frame_id for {video.video_id}: {frame_id}")
        frame_ids.add(frame_id)
        keyframe_path = record.get("keyframe_path")
        if not isinstance(keyframe_path, str) or not Path(keyframe_path).is_file():
            raise RuntimeError(
                f"Missing keyframe image for {video.video_id} at row {offset}: "
                f"{keyframe_path!r}"
            )
    return records


def _load_valid_extraction_report(
    *,
    video: CorpusVideo,
    video_path: Path,
    metadata_path: Path,
    report_path: Path,
    requested_config: Mapping[str, object] | None = None,
) -> dict:
    if not metadata_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            f"Extraction artifacts missing for {video.video_id}: "
            f"{metadata_path}, {report_path}"
        )
    report = _read_json_object(report_path)
    stored_config = report.get("competition_extract_config")
    if not isinstance(stored_config, Mapping):
        raise RuntimeError(f"Extraction lineage missing from {report_path}")
    if requested_config is not None and dict(stored_config) != dict(requested_config):
        raise RuntimeError(f"Extraction config changed for {video.video_id}")
    if report.get("extractor_contract_version") != EXTRACTOR_CONTRACT_VERSION:
        raise RuntimeError(f"Extractor contract version mismatch for {video.video_id}")
    if report.get("source_video_fingerprint") != _file_stat_fingerprint(video_path):
        raise RuntimeError(f"Source video changed for {video.video_id}")
    _require_extractor_report_contract(report, stored_config, report_path)
    records = _require_keyframe_metadata_integrity(
        video=video,
        metadata_path=metadata_path,
        report=report,
    )
    if report.get("keyframe_metadata_sha256") != _sha256_file(metadata_path):
        raise RuntimeError(f"Keyframe metadata changed for {video.video_id}")
    if report.get("keyframe_frame_ids_sha256") != _frame_ids_sha256(records):
        raise RuntimeError(f"Keyframe identities changed for {video.video_id}")
    if report.get("keyframe_images_sha256") != _keyframe_images_sha256(records):
        raise RuntimeError(f"Keyframe image content changed for {video.video_id}")
    return report


def _can_resume_extraction(
    *,
    video: CorpusVideo,
    video_path: Path,
    metadata_path: Path,
    report_path: Path,
    requested_config: Mapping[str, object],
) -> bool:
    try:
        _load_valid_extraction_report(
            video=video,
            video_path=video_path,
            metadata_path=metadata_path,
            report_path=report_path,
            requested_config=requested_config,
        )
    except (OSError, UnicodeError, ValueError, RuntimeError):
        return False
    return True


def _embedding_lineage(
    args: argparse.Namespace,
    extract_report: Mapping[str, object],
    *,
    resolved_device: str,
    resolved_model_revision: str,
) -> dict[str, object]:
    return {
        "version": EMBEDDING_LINEAGE_VERSION,
        "source_keyframe_metadata_sha256": extract_report["keyframe_metadata_sha256"],
        "source_keyframe_images_sha256": extract_report["keyframe_images_sha256"],
        "source_frame_ids_sha256": extract_report["keyframe_frame_ids_sha256"],
        "source_keyframe_count": extract_report["keyframe_count"],
        "source_extractor_contract_version": extract_report[
            "extractor_contract_version"
        ],
        "source_extract_config": dict(extract_report["competition_extract_config"]),
        "encoder_config": {
            "model_name": args.model_name,
            "requested_model_revision": args.model_revision,
            "resolved_model_revision": resolved_model_revision,
            "device": resolved_device,
            "use_autocast": not args.no_autocast,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
        },
    }


def _embedding_artifacts_match(
    *,
    embeddings_path: Path,
    embedding_metadata_path: Path,
    artifact_report_path: Path,
    expected_lineage: Mapping[str, object],
) -> bool:
    if not (
        embeddings_path.is_file()
        and embedding_metadata_path.is_file()
        and artifact_report_path.is_file()
    ):
        return False
    try:
        artifact_report = _read_json_object(artifact_report_path)
        if artifact_report.get("status") != "passed":
            return False
        if artifact_report.get("embedding_lineage") != dict(expected_lineage):
            return False
        embeddings = np.load(embeddings_path, allow_pickle=False)
        records = load_jsonl(embedding_metadata_path)
        validation = validate_embedding_artifacts(embeddings, records)
        source_count = expected_lineage.get("source_keyframe_count")
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or len(records) != source_count
            or validation["vector_count"] != source_count
        ):
            return False
        frame_ids_sha256 = _frame_ids_sha256(records)
        if frame_ids_sha256 != expected_lineage.get("source_frame_ids_sha256"):
            return False
        if (
            artifact_report.get("source_keyframe_count") != source_count
            or artifact_report.get("embedding_file_sha256")
            != _sha256_file(embeddings_path)
            or artifact_report.get("embedding_metadata_sha256")
            != _sha256_file(embedding_metadata_path)
            or artifact_report.get("embedded_frame_ids_sha256")
            != frame_ids_sha256
        ):
            return False
        encoder_config = expected_lineage.get("encoder_config")
        if not isinstance(encoder_config, Mapping):
            return False
        if any(
            record.get("model_name") != encoder_config.get("model_name")
            or record.get("model_revision")
            != encoder_config.get("resolved_model_revision")
            for record in records
        ):
            return False
    except (OSError, UnicodeError, ValueError, RuntimeError):
        return False
    return True


def _require_embedding_matches_extraction(
    *,
    video: CorpusVideo,
    extract_report: Mapping[str, object],
    embeddings_path: Path,
    embedding_metadata_path: Path,
    artifact_report_path: Path,
) -> None:
    if not artifact_report_path.is_file():
        raise FileNotFoundError(
            f"Embedding validation report missing for {video.video_id}: "
            f"{artifact_report_path}"
        )
    artifact_report = _read_json_object(artifact_report_path)
    if artifact_report.get("status") != "passed":
        raise RuntimeError(f"Embedding validation did not pass for {video.video_id}")
    lineage = artifact_report.get("embedding_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError(f"Embedding lineage missing for {video.video_id}")
    if (
        lineage.get("version") != EMBEDDING_LINEAGE_VERSION
        or lineage.get("source_keyframe_metadata_sha256")
        != extract_report.get("keyframe_metadata_sha256")
        or lineage.get("source_keyframe_images_sha256")
        != extract_report.get("keyframe_images_sha256")
        or lineage.get("source_frame_ids_sha256")
        != extract_report.get("keyframe_frame_ids_sha256")
        or lineage.get("source_keyframe_count")
        != extract_report.get("keyframe_count")
        or lineage.get("source_extractor_contract_version")
        != extract_report.get("extractor_contract_version")
        or lineage.get("source_extract_config")
        != extract_report.get("competition_extract_config")
    ):
        raise RuntimeError(
            f"Embedding artifacts are stale for {video.video_id}; rerun the embed step"
        )
    if not _embedding_artifacts_match(
        embeddings_path=embeddings_path,
        embedding_metadata_path=embedding_metadata_path,
        artifact_report_path=artifact_report_path,
        expected_lineage=lineage,
    ):
        raise RuntimeError(
            f"Embedding artifact integrity failed for {video.video_id}; "
            "rerun the embed step"
        )


def _embedding_source_lineage(
    video: CorpusVideo,
    artifact_report_path: Path,
) -> dict[str, object]:
    report = _read_json_object(artifact_report_path)
    lineage = report.get("embedding_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError(f"Embedding lineage missing for {video.video_id}")
    return {
        "video_id": video.video_id,
        "embedding_file_sha256": report.get("embedding_file_sha256"),
        "embedding_metadata_sha256": report.get("embedding_metadata_sha256"),
        "embedding_lineage": dict(lineage),
    }


def embed_command(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.public_root)
    output_root = args.output_root
    resolved_device = choose_device(args.device)
    prepared: list[
        tuple[int, CorpusVideo, Path, Path, Path, Path, dict[str, object]]
    ] = []
    for number, video in enumerate(corpus, start=1):
        video_path = args.public_root / video.relative_path
        metadata_path = output_root / "metadata" / f"keyframes_{video.video_id}.jsonl"
        extract_report_path = (
            output_root / "metadata" / f"keyframes_{video.video_id}_extract_report.json"
        )
        embeddings_path = output_root / "embeddings" / f"{ARTIFACT_TAG}_{video.video_id}.npy"
        embedding_metadata_path = (
            output_root / "metadata" / f"{ARTIFACT_TAG}_embeddings_{video.video_id}.jsonl"
        )
        artifact_report_path = (
            output_root
            / "metadata"
            / f"{ARTIFACT_TAG}_artifacts_{video.video_id}_validation.json"
        )
        extract_report = _load_valid_extraction_report(
            video=video,
            video_path=video_path,
            metadata_path=metadata_path,
            report_path=extract_report_path,
        )
        prepared.append(
            (
                number,
                video,
                metadata_path,
                embeddings_path,
                embedding_metadata_path,
                artifact_report_path,
                extract_report,
            )
        )

    if not prepared:
        return
    model, processor = load_siglip2_model_processor(
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=resolved_device,
        model_cache_dir=args.model_cache_dir,
        use_autocast=not args.no_autocast,
    )
    resolved_revision = resolve_model_revision(model, args.model_revision)
    jobs: list[tuple[int, CorpusVideo, Path, Path, Path, Path, dict[str, object]]] = []
    for (
        number,
        video,
        metadata_path,
        embeddings_path,
        embedding_metadata_path,
        artifact_report_path,
        extract_report,
    ) in prepared:
        lineage = _embedding_lineage(
            args,
            extract_report,
            resolved_device=resolved_device,
            resolved_model_revision=resolved_revision,
        )
        if args.resume and _embedding_artifacts_match(
            embeddings_path=embeddings_path,
            embedding_metadata_path=embedding_metadata_path,
            artifact_report_path=artifact_report_path,
            expected_lineage=lineage,
        ):
            print(
                f"[{number}/{len(corpus)}] skip {video.filename}: "
                "matching embeddings passed lineage checks"
            )
            continue
        jobs.append(
            (
                number,
                video,
                metadata_path,
                embeddings_path,
                embedding_metadata_path,
                artifact_report_path,
                lineage,
            )
        )

    if not jobs:
        return

    for (
        number,
        video,
        metadata_path,
        embeddings_path,
        embedding_metadata_path,
        artifact_report_path,
        lineage,
    ) in jobs:
        print(f"[{number}/{len(corpus)}] encoding {video.filename}")
        records = load_jsonl(metadata_path)
        validation = validate_records(records, min_width=16, min_height=16)
        validation_path = (
            output_root / "metadata" / f"keyframes_{video.video_id}_validation.json"
        )
        write_json(validation, validation_path)
        if not validation["valid"]:
            raise ValueError(f"Invalid keyframes; see {validation_path}")

        embeddings, embedding_records, skipped, benchmark = encode_keyframes(
            records=records,
            model_name=args.model_name,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=resolved_device,
            use_autocast=not args.no_autocast,
            model_cache_dir=args.model_cache_dir,
            prefetch_factor=args.prefetch_factor,
            model=model,
            processor=processor,
        )
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings)
        write_jsonl(embedding_records, embedding_metadata_path)
        write_jsonl(
            skipped,
            output_root / "metadata" / f"{ARTIFACT_TAG}_skipped_{video.video_id}.jsonl",
        )
        write_json(
            benchmark,
            output_root / "metadata" / f"{ARTIFACT_TAG}_benchmark_{video.video_id}.json",
        )
        artifact_report = validate_embedding_artifacts(embeddings, embedding_records)
        embedded_frame_ids_sha256 = _frame_ids_sha256(embedding_records)
        encoder_config = lineage["encoder_config"]
        assert isinstance(encoder_config, Mapping)
        contract_matches = all(
            record.get("model_name") == encoder_config["model_name"]
            and record.get("model_revision")
            == encoder_config["resolved_model_revision"]
            for record in embedding_records
        )
        artifact_report["source_keyframe_count"] = len(records)
        artifact_report["skipped_count"] = len(skipped)
        artifact_report["embedding_file_sha256"] = _sha256_file(embeddings_path)
        artifact_report["embedding_metadata_sha256"] = _sha256_file(
            embedding_metadata_path
        )
        artifact_report["embedded_frame_ids_sha256"] = embedded_frame_ids_sha256
        complete = (
            not skipped
            and len(embedding_records) == len(records)
            and embedded_frame_ids_sha256 == lineage["source_frame_ids_sha256"]
            and contract_matches
        )
        artifact_report["status"] = "passed" if complete else "partial"
        artifact_report["embedding_lineage"] = lineage
        write_json(artifact_report, artifact_report_path)
        if not complete:
            raise RuntimeError(
                f"Embedding completeness/contract failed for {video.video_id}: "
                f"encoded={len(embedding_records)}, source={len(records)}, "
                f"skipped={len(skipped)}, "
                f"identities_match="
                f"{embedded_frame_ids_sha256 == lineage['source_frame_ids_sha256']}, "
                f"encoder_contract_matches={contract_matches}; "
                f"see {artifact_report_path}"
            )


def index_command(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.public_root)
    _require_current_canonical_publish(
        corpus,
        public_root=args.public_root,
        output_root=args.output_root,
    )
    sources: list[tuple[Path, Path, str]] = []
    source_lineage: list[dict[str, object]] = []
    for video in corpus:
        keyframe_metadata_path = (
            args.output_root / "metadata" / f"keyframes_{video.video_id}.jsonl"
        )
        extract_report_path = (
            args.output_root
            / "metadata"
            / f"keyframes_{video.video_id}_extract_report.json"
        )
        embeddings_path = (
            args.output_root / "embeddings" / f"{ARTIFACT_TAG}_{video.video_id}.npy"
        )
        metadata_path = (
            args.output_root
            / "metadata"
            / f"{ARTIFACT_TAG}_embeddings_{video.video_id}.jsonl"
        )
        artifact_report_path = (
            args.output_root
            / "metadata"
            / f"{ARTIFACT_TAG}_artifacts_{video.video_id}_validation.json"
        )
        if not embeddings_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Embedding artifacts missing for {video.filename}: "
                f"{embeddings_path}, {metadata_path}"
            )
        extract_report = _load_valid_extraction_report(
            video=video,
            video_path=args.public_root / video.relative_path,
            metadata_path=keyframe_metadata_path,
            report_path=extract_report_path,
        )
        _require_embedding_matches_extraction(
            video=video,
            extract_report=extract_report,
            embeddings_path=embeddings_path,
            embedding_metadata_path=metadata_path,
            artifact_report_path=artifact_report_path,
        )
        source_lineage.append(
            _embedding_source_lineage(video, artifact_report_path)
        )
        sources.append((embeddings_path, metadata_path, video.video_id))

    paths = competition_index_paths(args.output_root)
    result = build_faiss_artifacts(
        sources=sources,
        index_path=paths["index"],
        index_metadata_path=paths["index_metadata"],
        frame_map_path=paths["frame_map"],
        manifest_path=paths["manifest"],
        report_path=paths["report"],
        metric="ip",
        normalize_for_index=True,
    )
    index_lineage = {
        "version": INDEX_LINEAGE_VERSION,
        "sources": source_lineage,
        "artifacts": {
            "index_sha256": _sha256_file(paths["index"]),
            "index_metadata_sha256": _sha256_file(paths["index_metadata"]),
            "frame_map_sha256": _sha256_file(paths["frame_map"]),
        },
    }
    result["manifest"]["competition_index_lineage"] = index_lineage
    result["report"]["competition_index_lineage"] = index_lineage
    write_json(result["manifest"], paths["manifest"])
    write_json(result["report"], paths["report"])
    print(f"FAISS index: {paths['index']} vectors={result['index'].ntotal}")


def competition_index_paths(output_root: Path) -> dict[str, Path]:
    return {
        "index": output_root / "indexes" / f"{ARTIFACT_TAG}_flat_ip.faiss",
        "index_metadata": output_root / "metadata" / f"{ARTIFACT_TAG}_faiss_metadata.jsonl",
        "frame_map": output_root / "metadata" / f"{ARTIFACT_TAG}_frame_map.json",
        "manifest": output_root / "metadata" / f"{ARTIFACT_TAG}_faiss_manifest.json",
        "report": output_root / "metadata" / f"{ARTIFACT_TAG}_index_report.json",
        "neighbors": output_root / "metadata" / "neighbors_all.jsonl",
        "neighbors_manifest": output_root
        / "metadata"
        / "neighbors_all_phase4_manifest.json",
        "segments": output_root / "metadata" / "segments_all.jsonl",
        "segments_manifest": output_root
        / "metadata"
        / "segments_all_phase4_manifest.json",
        "text_index": output_root / "indexes" / "retrieval_text_index.json",
        "text_index_manifest": output_root
        / "indexes"
        / "retrieval_text_index_phase4_manifest.json",
    }


def _require_current_index_lineage(
    *,
    corpus: Sequence[CorpusVideo],
    public_root: Path,
    output_root: Path,
    manifest_path: Path,
) -> None:
    manifest = _read_json_object(manifest_path)
    stored = manifest.get("competition_index_lineage")
    if not isinstance(stored, Mapping) or stored.get("version") != INDEX_LINEAGE_VERSION:
        raise RuntimeError("FAISS index lineage is missing or outdated; rerun index")

    paths = competition_index_paths(output_root)
    try:
        current_artifacts = {
            "index_sha256": _sha256_file(paths["index"]),
            "index_metadata_sha256": _sha256_file(paths["index_metadata"]),
            "frame_map_sha256": _sha256_file(paths["frame_map"]),
        }
    except OSError as exc:
        raise RuntimeError("FAISS index artifacts are missing; rerun index") from exc
    if stored.get("artifacts") != current_artifacts:
        raise RuntimeError("FAISS index artifacts changed; rerun index")

    expected_sources: list[dict[str, object]] = []
    for video in corpus:
        keyframe_metadata_path = (
            output_root / "metadata" / f"keyframes_{video.video_id}.jsonl"
        )
        extract_report_path = (
            output_root
            / "metadata"
            / f"keyframes_{video.video_id}_extract_report.json"
        )
        embeddings_path = (
            output_root / "embeddings" / f"{ARTIFACT_TAG}_{video.video_id}.npy"
        )
        embedding_metadata_path = (
            output_root
            / "metadata"
            / f"{ARTIFACT_TAG}_embeddings_{video.video_id}.jsonl"
        )
        artifact_report_path = (
            output_root
            / "metadata"
            / f"{ARTIFACT_TAG}_artifacts_{video.video_id}_validation.json"
        )
        extract_report = _load_valid_extraction_report(
            video=video,
            video_path=public_root / video.relative_path,
            metadata_path=keyframe_metadata_path,
            report_path=extract_report_path,
        )
        _require_embedding_matches_extraction(
            video=video,
            extract_report=extract_report,
            embeddings_path=embeddings_path,
            embedding_metadata_path=embedding_metadata_path,
            artifact_report_path=artifact_report_path,
        )
        expected_sources.append(
            _embedding_source_lineage(video, artifact_report_path)
        )

    if stored.get("sources") != expected_sources:
        raise RuntimeError("FAISS index is stale for current embeddings; rerun index")


def _keyframe_metadata_path(output_root: Path, video: CorpusVideo) -> Path:
    return output_root / "metadata" / f"keyframes_{video.video_id}.jsonl"


def enrich_command(args: argparse.Namespace) -> None:
    """Run caption, OCR, and object ingestion over the corpus."""
    corpus = load_corpus(args.public_root)
    metadata_dir = args.output_root / "metadata"
    resolved_device = choose_device(args.device)

    for video in corpus:
        metadata_path = _keyframe_metadata_path(args.output_root, video)
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Keyframe metadata missing for {video.filename}: {metadata_path}"
            )

    if "caption" in args.modalities:
        print("[caption] loading one shared Qwen multimodal backend for the corpus")
        backend = QwenCaptionBackend(
            model_name=args.caption_model_name,
            revision=args.caption_model_revision,
            device=resolved_device,
            cache_dir=args.model_cache_root / "caption",
            max_new_tokens=args.caption_max_new_tokens,
            dtype=args.caption_dtype,
            quantization=args.caption_quantization,
        )
        for number, video in enumerate(corpus, start=1):
            print(f"[caption {number}/{len(corpus)}] {video.filename}")
            run_caption_file(
                _keyframe_metadata_path(args.output_root, video),
                output_dir=metadata_dir,
                device=resolved_device,
                batch_size=args.caption_batch_size,
                overwrite=args.overwrite,
                include_segment_caption=not args.no_segment_caption,
                backend=backend,
                model_name=args.caption_model_name,
                revision=args.caption_model_revision,
                max_new_tokens=args.caption_max_new_tokens,
                dtype=args.caption_dtype,
                quantization=args.caption_quantization,
                model_cache_dir=args.model_cache_root / "caption",
            )

    if "ocr" in args.modalities:
        print("[ocr] loading one shared PP-OCRv5 backend for the corpus")
        backend = PaddleOcrBackend(
            device=resolved_device,
            detection_model=args.ocr_detection_model,
            recognition_model=args.ocr_recognition_model,
            revision=args.ocr_model_revision,
            cache_dir=args.model_cache_root / "ocr",
        )
        for number, video in enumerate(corpus, start=1):
            print(f"[ocr {number}/{len(corpus)}] {video.filename}")
            run_ocr_file(
                _keyframe_metadata_path(args.output_root, video),
                output_dir=metadata_dir,
                device=resolved_device,
                batch_size=args.ocr_batch_size,
                conf_threshold=args.ocr_conf_threshold,
                overwrite=args.overwrite,
                backend=backend,
                detection_model=args.ocr_detection_model,
                recognition_model=args.ocr_recognition_model,
                revision=args.ocr_model_revision,
                model_cache_dir=args.model_cache_root / "ocr",
            )

    if "objects" in args.modalities:
        print("[objects] loading one shared YOLOE backend for the corpus")
        object_cache = args.model_cache_root / "objects"
        backend = YoloEBackend(
            model_name=args.object_model_name,
            revision=args.object_model_revision,
            device=resolved_device,
            conf_threshold=args.object_conf_threshold,
            iou_threshold=args.object_iou_threshold,
            cache_dir=object_cache,
            vocabulary=args.object_vocabulary,
            prompt_mode=args.object_prompt_mode,
        )
        for number, video in enumerate(corpus, start=1):
            print(f"[objects {number}/{len(corpus)}] {video.filename}")
            run_object_file(
                _keyframe_metadata_path(args.output_root, video),
                output_dir=metadata_dir,
                device=resolved_device,
                batch_size=args.object_batch_size,
                conf_threshold=args.object_conf_threshold,
                iou_threshold=args.object_iou_threshold,
                overwrite=args.overwrite,
                backend=backend,
                model_name=args.object_model_name,
                revision=args.object_model_revision,
                model_cache_dir=object_cache,
                vocabulary=args.object_vocabulary,
                prompt_mode=args.object_prompt_mode,
            )

def _phase3_generate_features(
    args: argparse.Namespace,
    *,
    corpus: Sequence[CorpusVideo],
    work_items: Sequence[
        tuple[CorpusVideo, Phase3WorkspacePaths, Mapping[str, object]]
    ],
    resolved_device: str,
) -> tuple[dict[str, object], str]:
    """Generate complete multimodal artifacts for stale candidate workspaces."""

    print("[SigLIP2] loading one shared model for Phase 3")
    model, processor = load_siglip2_model_processor(
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=resolved_device,
        model_cache_dir=args.model_cache_dir,
        use_autocast=not args.no_autocast,
    )
    resolved_model_revision = resolve_model_revision(model, args.model_revision)
    feature_config = _phase3_feature_config(
        args,
        resolved_device=resolved_device,
        resolved_model_revision=resolved_model_revision,
    )
    pending: list[
        tuple[CorpusVideo, Phase3WorkspacePaths, Mapping[str, object]]
    ] = []
    for video, paths, candidate_contract in work_items:
        candidate_report, candidate_records = validate_candidate_pool(
            paths=paths,
            expected_contract=candidate_contract,
        )
        current = args.resume and _phase3_feature_artifacts_current(
            video=video,
            paths=paths,
            candidate_report=candidate_report,
            candidate_records=candidate_records,
            feature_config=feature_config,
            allow_partial_features=args.allow_partial_features,
        )
        if current:
            print(f"[features] skip {video.filename}: exact manifest passed")
        else:
            pending.append((video, paths, candidate_contract))

    if not pending:
        _phase3_release_model_memory(model, processor)
        del model, processor
        return feature_config, resolved_model_revision

    try:
        for number, (video, paths, candidate_contract) in enumerate(
            pending,
            start=1,
        ):
            print(f"[SigLIP2 {number}/{len(pending)}] {video.filename}")
            _candidate_report, records = validate_candidate_pool(
                paths=paths,
                expected_contract=candidate_contract,
            )
            if args.resume and _phase3_siglip_artifacts_current(
                paths=paths,
                candidate_records=records,
                feature_config=feature_config,
            ):
                print(
                    f"[SigLIP2 {number}/{len(pending)}] skip {video.filename}: "
                    "exact embedding checkpoint passed"
                )
                continue
            validation = validate_records(records, min_width=16, min_height=16)
            atomic_write_json(paths.candidate_validation, validation)
            if not validation["valid"]:
                raise RuntimeError(
                    f"Invalid dense candidate images; see {paths.candidate_validation}"
                )
            embeddings, embedding_records, skipped, benchmark = encode_keyframes(
                records=records,
                model_name=args.model_name,
                model_revision=args.model_revision,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=resolved_device,
                use_autocast=not args.no_autocast,
                model_cache_dir=args.model_cache_dir,
                prefetch_factor=args.prefetch_factor,
                model=model,
                processor=processor,
            )
            if skipped or len(embedding_records) != len(records):
                atomic_write_jsonl(paths.embedding_skipped, skipped)
                raise RuntimeError(
                    f"Dense SigLIP2 encoding was incomplete for {video.video_id}: "
                    f"encoded={len(embedding_records)}, candidates={len(records)}, "
                    f"skipped={len(skipped)}"
                )
            atomic_save_npy(paths.embeddings, embeddings)
            atomic_write_jsonl(paths.embedding_metadata, embedding_records)
            atomic_write_jsonl(paths.embedding_skipped, skipped)
            atomic_write_json(paths.embedding_benchmark, benchmark)
    finally:
        _phase3_release_model_memory(model, processor)
        del model, processor

    caption_pending = []
    for number, (video, paths, contract) in enumerate(pending, start=1):
        _report, records = validate_candidate_pool(
            paths=paths,
            expected_contract=contract,
        )
        current = args.resume and _phase3_frame_modality_artifacts_current(
            paths=paths,
            candidate_records=records,
            output_path=paths.captions,
            report_path=paths.caption_report,
            pipeline="caption",
            expected_report={
                "model_name": args.caption_model_name,
                "requested_model_revision": args.caption_model_revision,
                "device": resolved_device,
                "batch_size": args.caption_batch_size,
                "max_new_tokens": args.caption_max_new_tokens,
                "dtype": args.caption_dtype,
                "quantization": args.caption_quantization,
                "segment_caption_enabled": not args.no_segment_caption,
            },
            require_success=False,
        )
        if current:
            print(
                f"[caption {number}/{len(pending)}] skip {video.filename}: "
                "exact checkpoint passed"
            )
        else:
            caption_pending.append((video, paths, contract))
    if caption_pending:
        print("[caption] loading one shared Qwen multimodal backend")
        caption_backend = QwenCaptionBackend(
            model_name=args.caption_model_name,
            revision=args.caption_model_revision,
            device=resolved_device,
            cache_dir=args.model_cache_root / "caption",
            max_new_tokens=args.caption_max_new_tokens,
            dtype=args.caption_dtype,
            quantization=args.caption_quantization,
        )
        try:
            for number, (video, paths, _contract) in enumerate(
                caption_pending,
                start=1,
            ):
                print(f"[caption {number}/{len(caption_pending)}] {video.filename}")
                run_caption_file(
                    paths.candidate_metadata,
                    output_path=paths.captions,
                    report_path=paths.caption_report,
                    device=resolved_device,
                    batch_size=args.caption_batch_size,
                    overwrite=True,
                    include_segment_caption=not args.no_segment_caption,
                    backend=caption_backend,
                    model_name=args.caption_model_name,
                    revision=args.caption_model_revision,
                    max_new_tokens=args.caption_max_new_tokens,
                    dtype=args.caption_dtype,
                    quantization=args.caption_quantization,
                    model_cache_dir=args.model_cache_root / "caption",
                )
        finally:
            _phase3_release_model_memory(caption_backend)
            del caption_backend

    ocr_pending = []
    for number, (video, paths, contract) in enumerate(pending, start=1):
        _report, records = validate_candidate_pool(
            paths=paths,
            expected_contract=contract,
        )
        current = args.resume and _phase3_frame_modality_artifacts_current(
            paths=paths,
            candidate_records=records,
            output_path=paths.ocr,
            report_path=paths.ocr_report,
            pipeline="ocr",
            expected_report={
                "model_name": (
                    f"{args.ocr_detection_model}+{args.ocr_recognition_model}"
                ),
                "model_revision": args.ocr_model_revision,
                "device": resolved_device,
                "batch_size": args.ocr_batch_size,
                "confidence_threshold": args.ocr_conf_threshold,
                "detection_model": args.ocr_detection_model,
                "recognition_model": args.ocr_recognition_model,
            },
            require_success=not args.allow_partial_features,
        )
        if current:
            print(
                f"[ocr {number}/{len(pending)}] skip {video.filename}: "
                "exact checkpoint passed"
            )
        else:
            ocr_pending.append((video, paths, contract))
    if ocr_pending:
        print("[ocr] loading one shared PP-OCRv5 backend")
        ocr_backend = PaddleOcrBackend(
            device=resolved_device,
            detection_model=args.ocr_detection_model,
            recognition_model=args.ocr_recognition_model,
            revision=args.ocr_model_revision,
            cache_dir=args.model_cache_root / "ocr",
        )
        try:
            for number, (video, paths, _contract) in enumerate(ocr_pending, start=1):
                print(f"[ocr {number}/{len(ocr_pending)}] {video.filename}")
                run_ocr_file(
                    paths.candidate_metadata,
                    output_path=paths.ocr,
                    report_path=paths.ocr_report,
                    device=resolved_device,
                    batch_size=args.ocr_batch_size,
                    conf_threshold=args.ocr_conf_threshold,
                    overwrite=True,
                    backend=ocr_backend,
                    detection_model=args.ocr_detection_model,
                    recognition_model=args.ocr_recognition_model,
                    revision=args.ocr_model_revision,
                    model_cache_dir=args.model_cache_root / "ocr",
                )
        finally:
            _phase3_release_model_memory(ocr_backend)
            del ocr_backend

    object_pending = []
    for number, (video, paths, contract) in enumerate(pending, start=1):
        _report, records = validate_candidate_pool(
            paths=paths,
            expected_contract=contract,
        )
        current = args.resume and _phase3_frame_modality_artifacts_current(
            paths=paths,
            candidate_records=records,
            output_path=paths.objects,
            report_path=paths.object_report,
            pipeline="objects",
            expected_report={
                "model_name": args.object_model_name,
                "model_revision": args.object_model_revision,
                "device": resolved_device,
                "batch_size": args.object_batch_size,
                "confidence_threshold": args.object_conf_threshold,
                "iou_threshold": args.object_iou_threshold,
                "open_vocabulary_mode": args.object_prompt_mode,
                "vocabulary": list(args.object_vocabulary),
                "evidence_only": True,
            },
            require_success=not args.allow_partial_features,
        )
        if current:
            print(
                f"[objects {number}/{len(pending)}] skip {video.filename}: "
                "exact checkpoint passed"
            )
        else:
            object_pending.append((video, paths, contract))
    if object_pending:
        print("[objects] loading one shared YOLOE open-vocabulary backend")
        object_cache = args.model_cache_root / "objects"
        object_backend = YoloEBackend(
            model_name=args.object_model_name,
            revision=args.object_model_revision,
            device=resolved_device,
            conf_threshold=args.object_conf_threshold,
            iou_threshold=args.object_iou_threshold,
            cache_dir=object_cache,
            vocabulary=args.object_vocabulary,
            prompt_mode=args.object_prompt_mode,
        )
        try:
            for number, (video, paths, _contract) in enumerate(object_pending, start=1):
                print(f"[objects {number}/{len(object_pending)}] {video.filename}")
                run_object_file(
                    paths.candidate_metadata,
                    output_path=paths.objects,
                    report_path=paths.object_report,
                    device=resolved_device,
                    batch_size=args.object_batch_size,
                    conf_threshold=args.object_conf_threshold,
                    iou_threshold=args.object_iou_threshold,
                    overwrite=True,
                    backend=object_backend,
                    model_name=args.object_model_name,
                    revision=args.object_model_revision,
                    model_cache_dir=object_cache,
                    vocabulary=args.object_vocabulary,
                    prompt_mode=args.object_prompt_mode,
                )
        finally:
            _phase3_release_model_memory(object_backend)
            del object_backend

    for video, paths, candidate_contract in pending:
        # Commit a full manifest after every configured visual modality passes.
        candidate_report, candidate_records = validate_candidate_pool(
            paths=paths,
            expected_contract=candidate_contract,
        )
        loaded = _phase3_load_and_validate_features(
            video=video,
            paths=paths,
            candidate_report=candidate_report,
            candidate_records=candidate_records,
            feature_config=feature_config,
            allow_partial_features=args.allow_partial_features,
            require_manifest=False,
        )
        manifest = loaded["manifest"]
        assert isinstance(manifest, Mapping)
        atomic_write_json(paths.feature_manifest, manifest)
        _phase3_load_and_validate_features(
            video=video,
            paths=paths,
            candidate_report=candidate_report,
            candidate_records=candidate_records,
            feature_config=feature_config,
            allow_partial_features=args.allow_partial_features,
            require_manifest=True,
        )
    return feature_config, resolved_model_revision


def _phase3_final_record_paths(
    records: Sequence[Mapping[str, object]],
    *,
    output_root: Path,
    video_id: str,
    selection_run_id: str,
) -> tuple[list[dict], dict[str, Path]]:
    final_dir = (
        output_root
        / "keyframes"
        / video_id
        / f"phase3_{selection_run_id}"
    )
    rewritten: list[dict] = []
    paths_by_candidate: dict[str, Path] = {}
    for offset, record in enumerate(records):
        candidate_id = _phase3_record_candidate_id(record)
        frame_id = record.get("frame_id")
        source = record.get("keyframe_path")
        if not isinstance(frame_id, str) or not isinstance(source, str):
            raise RuntimeError(f"Invalid selected record at offset {offset}")
        destination = final_dir / f"{frame_id}.jpg"
        atomic_copy(Path(source), destination)
        normalized = destination.as_posix()
        paths_by_candidate[candidate_id] = destination
        rewritten.append(
            {
                **dict(record),
                "keyframe_path": normalized,
                "frame_path": normalized,
                "thumbnail_path": normalized,
                "artifact_role": "selected_keyframe",
                "phase3_selection_run_id": selection_run_id,
            }
        )
    return rewritten, paths_by_candidate


def _phase3_rewrite_embedding_paths(
    records: Sequence[Mapping[str, object]],
    final_records: Sequence[Mapping[str, object]],
) -> list[dict]:
    final_by_id = {
        _phase3_record_candidate_id(record): record for record in final_records
    }
    rewritten: list[dict] = []
    for offset, record in enumerate(records):
        candidate_id = _phase3_record_candidate_id(record)
        final = final_by_id.get(candidate_id)
        if final is None:
            raise RuntimeError(f"Embedding references unselected candidate: {candidate_id}")
        value = dict(record)
        value["embedding_index"] = offset
        for field in (
            "frame_id",
            "video_id",
            "shot_id",
            "segment_id",
            "shot_index",
            "shot_start",
            "shot_end",
            "timestamp",
            "timestamp_source",
            "timestamp_confidence",
            "frame_index",
            "keyframe_path",
            "thumbnail_path",
            "source_video_path",
            "video_path",
            "selection_reason",
            "candidate_index",
            "candidate_id",
            "candidate_reasons",
            "keyframe_strategy",
            "selection_phase",
            "selection_rank",
            "selection_reasons",
            "covered_event_ids",
            "selection_score",
            "protected",
            "coverage_added",
            "importance_score",
            "semantic_novelty",
            "component_scores",
            "available_modalities",
            "protected_event_ids",
            "selection_provenance",
        ):
            if field in final:
                value[field] = final[field]
        rewritten.append(value)
    return rewritten


def _phase3_rewrite_frame_artifact_paths(
    records: Sequence[Mapping[str, object]],
    final_records: Sequence[Mapping[str, object]],
) -> list[dict]:
    final_by_id = {
        _phase3_record_candidate_id(record): record for record in final_records
    }
    rewritten: list[dict] = []
    for record in records:
        candidate_id = _phase3_record_candidate_id(record)
        final = final_by_id.get(candidate_id)
        if final is None:
            raise RuntimeError(f"Feature references unselected candidate: {candidate_id}")
        value = dict(record)
        for field in ("keyframe_path", "frame_path", "thumbnail_path"):
            if field in final:
                value[field] = final[field]
        rewritten.append(value)
    return rewritten


def _phase3_publish_video(
    args: argparse.Namespace,
    *,
    video: CorpusVideo,
    paths: Phase3WorkspacePaths,
    candidate_report: Mapping[str, object],
    candidate_records: Sequence[Mapping[str, object]],
    features: Mapping[str, object],
    feature_config: Mapping[str, object],
    resolved_device: str,
    resolved_model_revision: str,
) -> dict[str, object]:
    adapter_config = _phase3_adapter_config(args)
    selection_config = _phase3_selection_config(args)
    feature_manifest = features.get("manifest")
    if not isinstance(feature_manifest, Mapping):
        raise RuntimeError("Phase 3 feature manifest is unavailable")
    feature_manifest_sha256 = phase3_sha256_file(paths.feature_manifest)
    selection_lineage = {
        "version": PHASE3_SELECTION_CONTRACT_VERSION,
        "video_id": video.video_id,
        "candidate_pool_run_id": paths.run_id,
        "feature_manifest_sha256": feature_manifest_sha256,
        "adapter_config": asdict(adapter_config),
        "selection_config": asdict(selection_config),
        "allow_partial_features": args.allow_partial_features,
    }
    selection_run_id = sha256_json(selection_lineage)[:20]
    result = run_multimodal_keyframe_pipeline(
        candidate_records,
        embeddings=features["embeddings"],
        embedding_records=features["embedding_records"],
        ocr_records=features["ocr_records"],
        object_records=features["object_records"],
        caption_records=features["caption_records"],
        video_duration=video.frame_count / video.fps,
        selection_config=selection_config,
        adapter_config=adapter_config,
        allow_partial_features=args.allow_partial_features,
    )
    if result.video_id != video.video_id:
        raise RuntimeError(
            f"Phase 3 core returned wrong video_id: {result.video_id!r}"
        )

    final_records, _paths_by_candidate = _phase3_final_record_paths(
        result.final_records,
        output_root=args.output_root,
        video_id=video.video_id,
        selection_run_id=selection_run_id,
    )
    final_embedding_records = _phase3_rewrite_embedding_paths(
        result.final_embedding_records,
        final_records,
    )
    final_ocr = _phase3_rewrite_frame_artifact_paths(
        result.final_ocr_records,
        final_records,
    )
    final_objects = _phase3_rewrite_frame_artifact_paths(
        result.final_object_records,
        final_records,
    )
    final_captions = _phase3_rewrite_frame_artifact_paths(
        result.final_caption_records,
        final_records,
    )
    atomic_write_jsonl(paths.candidate_scores, result.candidate_ledger)
    atomic_write_jsonl(paths.protected_events, result.event_ledger)

    metadata_dir = args.output_root / "metadata"
    metadata_path = metadata_dir / f"keyframes_{video.video_id}.jsonl"
    extract_report_path = (
        metadata_dir / f"keyframes_{video.video_id}_extract_report.json"
    )
    embeddings_path = (
        args.output_root / "embeddings" / f"{ARTIFACT_TAG}_{video.video_id}.npy"
    )
    embedding_metadata_path = (
        metadata_dir / f"{ARTIFACT_TAG}_embeddings_{video.video_id}.jsonl"
    )
    embedding_report_path = (
        metadata_dir
        / f"{ARTIFACT_TAG}_artifacts_{video.video_id}_validation.json"
    )
    caption_path = metadata_dir / f"captions_{video.video_id}.jsonl"
    ocr_path = metadata_dir / f"ocr_{video.video_id}.jsonl"
    object_path = metadata_dir / f"objects_{video.video_id}.jsonl"
    phase3_manifest_path = (
        metadata_dir / f"keyframes_{video.video_id}_phase3_manifest.json"
    )
    candidate_scores_path = (
        metadata_dir / f"keyframe_candidate_scores_{video.video_id}.jsonl"
    )
    protected_events_path = (
        metadata_dir / f"keyframe_protected_events_{video.video_id}.jsonl"
    )

    # Publish data artifacts first.  The extraction report is the commit marker
    # and is atomically replaced only after every checksum below is available.
    atomic_write_jsonl(metadata_path, final_records)
    atomic_save_npy(embeddings_path, result.final_embeddings)
    atomic_write_jsonl(embedding_metadata_path, final_embedding_records)
    atomic_write_jsonl(caption_path, final_captions)
    atomic_write_jsonl(ocr_path, final_ocr)
    atomic_write_jsonl(object_path, final_objects)
    atomic_write_jsonl(candidate_scores_path, result.candidate_ledger)
    atomic_write_jsonl(protected_events_path, result.event_ledger)

    extract_config = {
        "keyframe_strategy": "multimodal_coverage",
        "phash_threshold": args.phash_threshold,
        "phash_window_sec": args.phash_window_sec,
        "jpeg_quality": args.jpeg_quality,
        "shot_threshold": args.shot_threshold,
        "shot_device": args.device,
        "candidate_interval_sec": args.candidate_interval_sec,
        "boundary_guard_sec": args.boundary_guard_sec,
        "tiny_shot_max_sec": args.tiny_shot_max_sec,
        "max_gap_seconds": args.max_gap_seconds,
        "gap_tolerance_seconds": args.gap_tolerance_seconds,
        "target_keyframes": args.target_keyframes,
        "hard_max_keyframes": args.hard_max_keyframes,
        "adapter_config": asdict(adapter_config),
        "allow_partial_features": args.allow_partial_features,
    }
    selection_report = result.selection_result.to_report()
    extract_report: dict[str, object] = {
        "video_id": video.video_id,
        "video_path": (args.public_root / video.relative_path).as_posix(),
        "fps": video.fps,
        "frame_count": video.frame_count,
        "duration": round(video.frame_count / video.fps, 3),
        "shot_detector": candidate_report.get("shot_detector"),
        "shot_count": candidate_report.get("shot_count"),
        "keyframe_count": len(final_records),
        "skipped_count": 0,
        "phash_threshold": args.phash_threshold,
        "phash_window_sec": args.phash_window_sec,
        "jpeg_quality": args.jpeg_quality,
        "shot_threshold": args.shot_threshold,
        "shot_device": args.device,
        "frame_extractor": candidate_report.get("frame_extractor", "ffmpeg"),
        "keyframe_strategy": "multimodal_coverage",
        "status": selection_report["status"],
        "constraints_satisfied": selection_report["constraints_satisfied"],
        "coverage_satisfied": selection_report["coverage_satisfied"],
        "candidate_count": len(candidate_records),
        "candidate_pool_run_id": paths.run_id,
        "phase3_selection_run_id": selection_run_id,
        "feature_manifest_sha256": feature_manifest_sha256,
        "allow_partial_features": args.allow_partial_features,
        "selection_config": asdict(selection_config),
        "feature_adapter_config": asdict(adapter_config),
        "feature_adapter": result.feature_adapter_result.report.to_dict(),
        "selection": selection_report,
        "guarantees": result.guarantee_report.to_dict(),
        "candidate_scores_path": candidate_scores_path.as_posix(),
        "protected_events_path": protected_events_path.as_posix(),
        "extractor_contract_version": EXTRACTOR_CONTRACT_VERSION,
        "source_video_fingerprint": _file_stat_fingerprint(
            args.public_root / video.relative_path
        ),
        "keyframe_metadata_sha256": _sha256_file(metadata_path),
        "keyframe_frame_ids_sha256": _frame_ids_sha256(final_records),
        "keyframe_images_sha256": _keyframe_images_sha256(final_records),
        "competition_extract_config": extract_config,
    }
    if (
        extract_report["status"] != "satisfied"
        or extract_report["constraints_satisfied"] is not True
        or extract_report["coverage_satisfied"] is not True
    ):
        raise RuntimeError("Phase 3 selection failed hard guarantees; refusing publish")

    embedding_validation = validate_embedding_artifacts(
        result.final_embeddings,
        final_embedding_records,
    )
    embedding_lineage = _embedding_lineage(
        args,
        extract_report,
        resolved_device=resolved_device,
        resolved_model_revision=resolved_model_revision,
    )
    embedding_report = {
        **embedding_validation,
        "source_keyframe_count": len(final_records),
        "skipped_count": 0,
        "embedding_file_sha256": _sha256_file(embeddings_path),
        "embedding_metadata_sha256": _sha256_file(embedding_metadata_path),
        "embedded_frame_ids_sha256": _frame_ids_sha256(final_embedding_records),
        "status": "passed",
        "embedding_lineage": embedding_lineage,
        "phase3_source_dense_embeddings_sha256": phase3_sha256_file(
            paths.embeddings
        ),
    }
    atomic_write_json(embedding_report_path, embedding_report)

    canonical_artifacts = {
        "keyframe_metadata": _phase3_artifact_entry(metadata_path),
        "keyframe_images_sha256": extract_report["keyframe_images_sha256"],
        "embeddings": _phase3_artifact_entry(embeddings_path),
        "embedding_metadata": _phase3_artifact_entry(embedding_metadata_path),
        "embedding_report": _phase3_artifact_entry(embedding_report_path),
        "captions": _phase3_artifact_entry(caption_path),
        "ocr": _phase3_artifact_entry(ocr_path),
        "objects": _phase3_artifact_entry(object_path),
        "candidate_scores": _phase3_artifact_entry(candidate_scores_path),
        "protected_events": _phase3_artifact_entry(protected_events_path),
    }
    phase3_manifest = {
        **selection_lineage,
        "selection_run_id": selection_run_id,
        "status": "passed",
        "degraded": args.allow_partial_features
        and feature_manifest.get("hard_feature_complete") is not True,
        "candidate_count": len(candidate_records),
        "selected_count": len(final_records),
        "selected_candidate_ids": [
            _phase3_record_candidate_id(record) for record in final_records
        ],
        "feature_config": dict(feature_config),
        "feature_adapter_report": result.feature_adapter_result.report.to_dict(),
        "selection_report": selection_report,
        "guarantees": result.guarantee_report.to_dict(),
        "candidate_scores_sha256": phase3_sha256_file(candidate_scores_path),
        "protected_events_sha256": phase3_sha256_file(protected_events_path),
        "canonical_artifacts": canonical_artifacts,
    }
    atomic_write_json(phase3_manifest_path, phase3_manifest)
    extract_report["phase3_manifest_path"] = phase3_manifest_path.as_posix()
    extract_report["phase3_manifest_sha256"] = _sha256_file(phase3_manifest_path)
    atomic_write_json(paths.selection_report, phase3_manifest)
    staged_extract_report_path = paths.root / "canonical_extract_report.json"
    atomic_write_json(staged_extract_report_path, extract_report)

    # Reuse the existing downstream contracts before exposing the extraction
    # report commit marker.  A crash or contract failure leaves the previous
    # canonical report in place, so downstream commands fail closed on hashes.
    _load_valid_extraction_report(
        video=video,
        video_path=args.public_root / video.relative_path,
        metadata_path=metadata_path,
        report_path=staged_extract_report_path,
    )
    _require_embedding_matches_extraction(
        video=video,
        extract_report=extract_report,
        embeddings_path=embeddings_path,
        embedding_metadata_path=embedding_metadata_path,
        artifact_report_path=embedding_report_path,
    )
    atomic_write_json(extract_report_path, extract_report)
    _load_valid_extraction_report(
        video=video,
        video_path=args.public_root / video.relative_path,
        metadata_path=metadata_path,
        report_path=extract_report_path,
    )
    return phase3_manifest


def keyframes_command(args: argparse.Namespace) -> None:
    """Run candidate materialization -> multimodal selection -> canonical publish."""

    corpus = load_corpus(args.public_root)
    candidate_config = _phase3_candidate_config(args)
    work_items: list[
        tuple[CorpusVideo, Phase3WorkspacePaths, Mapping[str, object]]
    ] = []
    for number, video in enumerate(corpus, start=1):
        print(f"[candidates {number}/{len(corpus)}] {video.filename}")
        paths, report, records = materialize_candidate_pool(
            video_path=args.public_root / video.relative_path,
            video_id=video.video_id,
            frame_count=video.frame_count,
            output_root=args.output_root,
            config=candidate_config,
            resume=args.resume,
        )
        candidate_contract = report.get("phase3_candidate_contract")
        if not isinstance(candidate_contract, Mapping):
            raise RuntimeError(
                f"Candidate contract missing for {video.video_id}: "
                f"{paths.candidate_report}"
            )
        work_items.append((video, paths, dict(candidate_contract)))
        del report, records

    resolved_device = choose_device(args.device)
    feature_config, resolved_revision = _phase3_generate_features(
        args,
        corpus=corpus,
        work_items=work_items,
        resolved_device=resolved_device,
    )
    for number, (video, paths, candidate_contract) in enumerate(
        work_items,
        start=1,
    ):
        print(f"[select {number}/{len(corpus)}] {video.filename}")
        candidate_report, candidate_records = validate_candidate_pool(
            paths=paths,
            expected_contract=candidate_contract,
        )
        features = _phase3_load_and_validate_features(
            video=video,
            paths=paths,
            candidate_report=candidate_report,
            candidate_records=candidate_records,
            feature_config=feature_config,
            allow_partial_features=args.allow_partial_features,
            require_manifest=True,
        )
        manifest = _phase3_publish_video(
            args,
            video=video,
            paths=paths,
            candidate_report=candidate_report,
            candidate_records=candidate_records,
            features=features,
            feature_config=feature_config,
            resolved_device=resolved_device,
            resolved_model_revision=resolved_revision,
        )
        print(
            f"[selected {number}/{len(corpus)}] {video.filename}: "
            f"{manifest['selected_count']}/{manifest['candidate_count']} keyframes"
        )


def reselect_keyframes_command(args: argparse.Namespace) -> None:
    """Reselect cached Phase-3 features into an isolated experiment run.

    The source workspace remains read-only.  Only selection ledgers and
    canonical outputs are written below ``run_root``, so offline ablations can
    reuse the expensive five-modality artifacts without mutating the 5s
    baseline.
    """

    public_root = args.public_root.resolve()
    source_output_root = args.source_output_root.resolve()
    run_root = args.run_root.resolve()
    if run_root == source_output_root:
        raise ValueError("--run-root must differ from --source-output-root")
    args.public_root = public_root
    args.output_root = run_root
    corpus = load_corpus(public_root)

    prepared: list[
        tuple[
            CorpusVideo,
            Phase3WorkspacePaths,
            Phase3WorkspacePaths,
            dict[str, object],
            list[dict],
            dict[str, object],
        ]
    ] = []
    canonical_feature_config: dict[str, object] | None = None
    missing_endpoints: list[str] = []
    for video in corpus:
        phase3_manifest_path = (
            source_output_root
            / "metadata"
            / f"keyframes_{video.video_id}_phase3_manifest.json"
        )
        phase3_manifest = read_phase3_json(phase3_manifest_path)
        run_id = str(phase3_manifest.get("candidate_pool_run_id") or "")
        if not run_id:
            raise RuntimeError(
                f"Missing candidate_pool_run_id in {phase3_manifest_path}"
            )
        workspace_root = (
            source_output_root / "work" / "keyframe_v3" / video.video_id / run_id
        )
        candidate_report = read_phase3_json(workspace_root / "candidate_report.json")
        candidate_contract = candidate_report.get("phase3_candidate_contract")
        if not isinstance(candidate_contract, Mapping):
            raise RuntimeError(
                f"Missing Phase-3 candidate contract for {video.video_id}"
            )
        source_paths = workspace_paths(
            source_output_root,
            video.video_id,
            candidate_contract,
        )
        if source_paths.run_id != run_id:
            raise RuntimeError(
                f"Candidate run lineage mismatch for {video.video_id}: "
                f"canonical={run_id}, contract={source_paths.run_id}"
            )
        validated_report, candidate_records = validate_candidate_pool(
            paths=source_paths,
            expected_contract=candidate_contract,
        )
        feature_manifest = read_phase3_json(source_paths.feature_manifest)
        feature_config = feature_manifest.get("feature_config")
        if not isinstance(feature_config, dict):
            raise RuntimeError(
                f"Missing feature_config in {source_paths.feature_manifest}"
            )
        if canonical_feature_config is None:
            canonical_feature_config = dict(feature_config)
        elif feature_config != canonical_feature_config:
            raise RuntimeError(
                f"Feature config drift detected at {video.video_id}; "
                "refusing a mixed offline run"
            )

        if args.endpoint_protection == "on":
            reasons = {
                str(reason)
                for record in candidate_records
                for reason in (record.get("candidate_reasons") or ())
            }
            absent = {"video_start", "video_end"} - reasons
            if absent:
                missing_endpoints.append(
                    f"{video.video_id}({','.join(sorted(absent))})"
                )

        target_workspace = run_root / "work" / "keyframe_v3" / video.video_id / run_id
        publish_paths = replace(
            source_paths,
            root=target_workspace,
            candidate_scores=target_workspace / "candidate_scores.jsonl",
            protected_events=target_workspace / "protected_events.jsonl",
            selection_report=target_workspace / "selection_report.json",
        )
        prepared.append(
            (
                video,
                source_paths,
                publish_paths,
                validated_report,
                candidate_records,
                dict(feature_config),
            )
        )

    if missing_endpoints:
        preview = ", ".join(missing_endpoints[:10])
        raise RuntimeError(
            "Endpoint-on reselect requires endpoint candidates and features. "
            "The cached pool is missing them for "
            f"{len(missing_endpoints)} videos ({preview}). Build an endpoint-enabled "
            "candidate/feature cache first; no canonical files were written."
        )
    if canonical_feature_config is None:
        raise RuntimeError("No Phase-3 feature configuration found")

    siglip_config = canonical_feature_config.get("siglip2")
    if not isinstance(siglip_config, Mapping):
        raise RuntimeError("Cached feature config has no SigLIP2 contract")
    resolved_device = str(siglip_config.get("device") or "cpu")
    resolved_revision = str(siglip_config.get("resolved_model_revision") or "")
    if not resolved_revision:
        raise RuntimeError("Cached feature config has no resolved SigLIP2 revision")
    args.model_name = str(siglip_config.get("model_name") or args.model_name)
    args.model_revision = siglip_config.get("requested_model_revision")
    args.no_autocast = not bool(siglip_config.get("use_autocast", True))
    args.batch_size = siglip_config.get("batch_size", args.batch_size)
    args.num_workers = int(siglip_config.get("num_workers", args.num_workers))
    args.prefetch_factor = int(
        siglip_config.get("prefetch_factor", args.prefetch_factor)
    )

    started = time.perf_counter()
    initialize_run_manifest(
        run_root=run_root,
        repo_root=Path(__file__).resolve().parents[1],
        public_root=public_root,
        offline_config={
            "source_output_root": Path(
                os.path.relpath(source_output_root, run_root)
            ).as_posix(),
            "selection_config": asdict(_phase3_selection_config(args)),
            "feature_adapter_config": asdict(_phase3_adapter_config(args)),
            "feature_config": canonical_feature_config,
        },
    )
    selected_total = 0
    candidate_total = 0
    for number, (
        video,
        source_paths,
        publish_paths,
        candidate_report,
        candidate_records,
        feature_config,
    ) in enumerate(prepared, start=1):
        print(f"[reselect {number}/{len(prepared)}] {video.filename}")
        features = _phase3_load_and_validate_features(
            video=video,
            paths=source_paths,
            candidate_report=candidate_report,
            candidate_records=candidate_records,
            feature_config=feature_config,
            allow_partial_features=args.allow_partial_features,
            require_manifest=True,
        )
        manifest = _phase3_publish_video(
            args,
            video=video,
            paths=publish_paths,
            candidate_report=candidate_report,
            candidate_records=candidate_records,
            features=features,
            feature_config=feature_config,
            resolved_device=resolved_device,
            resolved_model_revision=resolved_revision,
        )
        selected_total += int(manifest["selected_count"])
        candidate_total += int(manifest["candidate_count"])

    elapsed = round(time.perf_counter() - started, 3)
    manifest = update_run_manifest(
        run_root,
        git=git_fingerprint(Path(__file__).resolve().parents[1]),
        offline={
            "candidate_count": candidate_total,
            "selected_count": selected_total,
            "source_feature_cache_reused": True,
        },
        stages={
            "offline_reselect": {
                "status": "passed",
                "elapsed_seconds": elapsed,
                "videos": len(prepared),
            }
        },
        status="offline_reselect_passed",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def neighbors_command(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.public_root)
    canonical_sources = _require_current_canonical_publish(
        corpus,
        public_root=args.public_root,
        output_root=args.output_root,
    )
    paths = competition_index_paths(args.output_root)
    result = build_neighbor_index(
        args.output_root / "metadata",
        paths["neighbors"],
        window_seconds=args.window_seconds,
        fps=25.0,
    )
    write_stage_manifest(
        paths["neighbors_manifest"],
        stage="neighbors",
        canonical_sources=canonical_sources,
        input_paths={},
        output_paths={"neighbors": paths["neighbors"]},
        config={"window_seconds": args.window_seconds, "fallback_fps": 25.0},
    )
    result["phase4_manifest_path"] = paths["neighbors_manifest"].as_posix()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _require_multimodal_artifacts(
    corpus: Sequence[CorpusVideo],
    metadata_dir: Path,
) -> dict[str, Path]:
    missing: list[Path] = []
    artifacts: dict[str, Path] = {}
    prefixes = ("captions", "ocr", "objects")
    for video in corpus:
        for prefix in prefixes:
            path = metadata_dir / f"{prefix}_{video.video_id}.jsonl"
            if not path.exists():
                missing.append(path)
            else:
                artifacts[f"{prefix}:{video.video_id}"] = path
    if missing:
        preview = ", ".join(path.as_posix() for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} multimodal artifacts; run enrich first. "
            f"First entries: {preview}"
        )
    return artifacts


def _require_current_canonical_publish(
    corpus: Sequence[CorpusVideo],
    *,
    public_root: Path,
    output_root: Path,
) -> list[dict[str, object]]:
    """Reject mixed/stale canonical files before downstream aggregation.

    Phase 3 publishes files with per-file atomic replacement and exposes the
    extraction report last as its commit marker.  Downstream commands that do
    not otherwise consume embedding lineage must validate that marker and the
    manifest-bound canonical checksums before reading the metadata directory.
    """

    metadata_dir = output_root / "metadata"
    canonical_sources: list[dict[str, object]] = []
    for video in corpus:
        report_path = metadata_dir / f"keyframes_{video.video_id}_extract_report.json"
        metadata_path = _keyframe_metadata_path(output_root, video)
        report = _load_valid_extraction_report(
            video=video,
            video_path=public_root / video.relative_path,
            metadata_path=metadata_path,
            report_path=report_path,
        )
        source: dict[str, object] = {
            "video_id": video.video_id,
            "keyframe_strategy": str(report.get("keyframe_strategy") or "legacy"),
            "extraction_report": _phase3_artifact_entry(report_path),
            "keyframe_metadata": _phase3_artifact_entry(metadata_path),
            "keyframe_images_sha256": report.get("keyframe_images_sha256"),
        }
        if report.get("keyframe_strategy") != "multimodal_coverage":
            canonical_sources.append(source)
            continue
        raw_manifest_path = report.get("phase3_manifest_path")
        expected_manifest_hash = report.get("phase3_manifest_sha256")
        if not isinstance(raw_manifest_path, str) or not isinstance(
            expected_manifest_hash,
            str,
        ):
            raise RuntimeError(f"Phase 3 commit marker is incomplete: {report_path}")
        manifest_path = Path(raw_manifest_path)
        if not manifest_path.is_file() or _sha256_file(manifest_path) != expected_manifest_hash:
            raise RuntimeError(f"Phase 3 manifest is missing or stale: {manifest_path}")
        manifest = _read_json_object(manifest_path)
        if (
            manifest.get("status") != "passed"
            or manifest.get("video_id") != video.video_id
            or manifest.get("selection_run_id")
            != report.get("phase3_selection_run_id")
        ):
            raise RuntimeError(f"Phase 3 manifest contract failed: {manifest_path}")
        artifacts = manifest.get("canonical_artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(f"Canonical artifact ledger missing: {manifest_path}")
        if artifacts.get("keyframe_images_sha256") != report.get(
            "keyframe_images_sha256"
        ):
            raise RuntimeError(f"Canonical image ledger is stale: {manifest_path}")
        for name, raw_entry in artifacts.items():
            if name == "keyframe_images_sha256":
                continue
            if not isinstance(raw_entry, Mapping):
                raise RuntimeError(
                    f"Invalid canonical artifact entry {name!r}: {manifest_path}"
                )
            raw_path = raw_entry.get("path")
            expected_hash = raw_entry.get("sha256")
            expected_size = raw_entry.get("size")
            if (
                not isinstance(raw_path, str)
                or not isinstance(expected_hash, str)
                or isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
            ):
                raise RuntimeError(
                    f"Incomplete canonical artifact entry {name!r}: {manifest_path}"
                )
            artifact_path = Path(raw_path)
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != expected_size
                or _sha256_file(artifact_path) != expected_hash
            ):
                raise RuntimeError(
                    f"Canonical Phase 3 artifact is stale ({name}): {artifact_path}"
                )
        source["phase3_selection_run_id"] = report.get("phase3_selection_run_id")
        source["phase3_manifest"] = _phase3_artifact_entry(manifest_path)
        canonical_sources.append(source)
    return canonical_sources


def segments_command(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.public_root)
    metadata_dir = args.output_root / "metadata"
    multimodal_paths = _require_multimodal_artifacts(corpus, metadata_dir)
    canonical_sources = _require_current_canonical_publish(
        corpus,
        public_root=args.public_root,
        output_root=args.output_root,
    )
    paths = competition_index_paths(args.output_root)
    result = build_segment_metadata(
        metadata_dir,
        paths["segments"],
        captions_path=metadata_dir,
        ocr_path=metadata_dir,
        objects_path=metadata_dir,
        strategy=args.strategy,
        fixed_duration_seconds=args.fixed_duration_seconds,
        fps=25.0,
        caption_similarity_threshold=args.caption_similarity_threshold,
    )
    write_stage_manifest(
        paths["segments_manifest"],
        stage="segments",
        canonical_sources=canonical_sources,
        input_paths=multimodal_paths,
        output_paths={"segments": paths["segments"]},
        config={
            "strategy": args.strategy,
            "fixed_duration_seconds": args.fixed_duration_seconds,
            "fallback_fps": 25.0,
            "caption_similarity_threshold": args.caption_similarity_threshold,
        },
    )
    result["phase4_manifest_path"] = paths["segments_manifest"].as_posix()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _require_current_segments_lineage(
    *,
    corpus: Sequence[CorpusVideo],
    public_root: Path,
    output_root: Path,
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    metadata_dir = output_root / "metadata"
    multimodal_paths = _require_multimodal_artifacts(corpus, metadata_dir)
    canonical_sources = _require_current_canonical_publish(
        corpus,
        public_root=public_root,
        output_root=output_root,
    )
    paths = competition_index_paths(output_root)
    validate_stage_manifest(
        paths["segments_manifest"],
        stage="segments",
        canonical_sources=canonical_sources,
        input_paths=multimodal_paths,
        output_paths={"segments": paths["segments"]},
    )
    return canonical_sources, multimodal_paths


def _require_current_text_index_lineage(
    *,
    corpus: Sequence[CorpusVideo],
    public_root: Path,
    output_root: Path,
) -> None:
    canonical_sources, _ = _require_current_segments_lineage(
        corpus=corpus,
        public_root=public_root,
        output_root=output_root,
    )
    paths = competition_index_paths(output_root)
    validate_stage_manifest(
        paths["text_index_manifest"],
        stage="text-index",
        canonical_sources=canonical_sources,
        input_paths={"segments": paths["segments"]},
        output_paths={"text_index": paths["text_index"]},
    )


def text_index_command(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.public_root)
    paths = competition_index_paths(args.output_root)
    canonical_sources, _ = _require_current_segments_lineage(
        corpus=corpus,
        public_root=args.public_root,
        output_root=args.output_root,
    )
    summary = write_text_index(
        load_text_records(paths["segments"]),
        paths["text_index"],
    )
    write_stage_manifest(
        paths["text_index_manifest"],
        stage="text-index",
        canonical_sources=canonical_sources,
        input_paths={"segments": paths["segments"]},
        output_paths={"text_index": paths["text_index"]},
        config={"text_index_version": TEXT_INDEX_VERSION, "modalities": [
            "caption",
            "ocr",
            "objects",
        ]},
    )
    summary["phase4_manifest_path"] = paths["text_index_manifest"].as_posix()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _phase5_split_path(args: argparse.Namespace) -> Path:
    return (
        args.split_manifest
        if args.split_manifest is not None
        else args.output_root / "evaluation" / "keyframe_phase5_split.json"
    )


def _phase5_config_lock_path(args: argparse.Namespace) -> Path:
    return (
        args.config_lock
        if args.config_lock is not None
        else args.output_root / "evaluation" / "keyframe_phase5_config_lock.json"
    )


def _phase5_corpus_subset(
    corpus: Sequence[CorpusVideo],
    video_ids: Sequence[str],
) -> list[CorpusVideo]:
    by_id = {video.video_id: video for video in corpus}
    missing = [video_id for video_id in video_ids if video_id not in by_id]
    if missing:
        raise ValueError(f"Phase 5 split references videos outside corpus: {missing}")
    return [by_id[video_id] for video_id in video_ids]


def phase5_init_command(args: argparse.Namespace) -> None:
    if args.video_ids is not None:
        video_ids = list(args.video_ids)
    else:
        assert args.video_ids_file is not None
        video_ids = [
            line.strip()
            for line in args.video_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    corpus = load_corpus(args.public_root)
    _phase5_corpus_subset(corpus, video_ids)
    split_path = _phase5_split_path(args)
    manifest = write_split_manifest(split_path, video_ids, seed=args.seed)
    print(
        json.dumps(
            {"split_manifest_path": split_path.as_posix(), **manifest},
            ensure_ascii=False,
            indent=2,
        )
    )


def phase5_lock_command(args: argparse.Namespace) -> None:
    split_path = _phase5_split_path(args)
    manifest = load_split_manifest(split_path)
    corpus = load_corpus(args.public_root)
    dev_videos = _phase5_corpus_subset(corpus, manifest["splits"]["dev"])
    _require_current_canonical_publish(
        dev_videos,
        public_root=args.public_root,
        output_root=args.output_root,
    )
    lock_path = _phase5_config_lock_path(args)
    lock = write_config_lock(
        lock_path,
        output_root=args.output_root,
        split_manifest_path=split_path,
    )
    print(
        json.dumps(
            {"config_lock_path": lock_path.as_posix(), **lock},
            ensure_ascii=False,
            indent=2,
        )
    )


def phase5_evaluate_command(args: argparse.Namespace) -> None:
    if args.split == "test" and not args.confirm_locked_test:
        raise ValueError(
            "locked test evaluation requires --confirm-locked-test; inspect dev/"
            "validation reports and freeze config first"
        )
    split_path = _phase5_split_path(args)
    manifest = load_split_manifest(split_path)
    video_ids = manifest["splits"][args.split]
    corpus = load_corpus(args.public_root)
    videos = _phase5_corpus_subset(corpus, video_ids)
    canonical_sources = _require_current_canonical_publish(
        videos,
        public_root=args.public_root,
        output_root=args.output_root,
    )
    report_path = (
        args.report_path
        if args.report_path is not None
        else args.output_root
        / "evaluation"
        / f"keyframe_phase5_{args.split}_report.json"
    )
    if report_path.exists():
        if args.split == "test":
            raise FileExistsError(
                f"locked test report already exists and cannot be overwritten: {report_path}"
            )
        if not args.overwrite:
            raise FileExistsError(
                f"evaluation report already exists; pass --overwrite: {report_path}"
            )
    config_lock_path: Path | None = None
    candidate_lock = _phase5_config_lock_path(args)
    if args.split in {"validation", "test"} or candidate_lock.exists():
        config_lock_path = candidate_lock
    report = evaluate_split_artifacts(
        output_root=args.output_root,
        split_manifest_path=split_path,
        split=args.split,
        canonical_sources=canonical_sources,
        config_lock_path=config_lock_path,
        manual_events_path=args.manual_events,
        protection_reviews_path=args.protection_reviews,
        retrieval_evidence_path=args.retrieval_evidence,
        resource_usage_path=args.resource_usage,
        manual_tolerance_seconds=args.manual_tolerance_seconds,
    )
    atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "report_path": report_path.as_posix(),
                "split": args.split,
                "status": report["status"],
                "aggregate": report["aggregate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _ranked_records(
    vector: np.ndarray,
    *,
    searcher: FaissVectorSearcher,
    metadata_store: MetadataStore,
    top_k: int,
) -> list[RankedFrame]:
    scores, indices = searcher.search(normalize_query_vector(vector), top_k)
    ranked: list[RankedFrame] = []
    for raw_score, raw_index in zip(scores[0], indices[0]):
        faiss_index = int(raw_index)
        if faiss_index < 0:
            continue
        record = metadata_store.get_by_faiss_index(faiss_index)
        if record is not None:
            ranked.append(RankedFrame(record=record, score=float(raw_score)))
    return ranked


def encode_vkis_questions(
    questions: Sequence[Question],
    *,
    public_root: Path,
    model_name: str,
    model_revision: str,
    batch_size: str | int,
    num_workers: int,
    device: str,
    use_autocast: bool,
    model_cache_dir: Path,
    model: object,
    processor: object,
) -> dict[str, np.ndarray]:
    vkis = [question for question in questions if question.task == "VKIS"]
    records = [
        {
            "frame_id": question.query_id,
            "video_id": "VKIS_QUERY",
            "shot_id": question.query_id,
            "segment_id": question.query_id,
            "timestamp": 0.0,
            "timestamp_source": "query_image",
            "timestamp_confidence": 1.0,
            "frame_index": 0,
            "keyframe_path": (public_root / question.query_image).as_posix(),
        }
        for question in vkis
    ]
    if not records:
        return {}
    embeddings, embedding_records, skipped, _ = encode_keyframes(
        records=records,
        model_name=model_name,
        model_revision=model_revision,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        use_autocast=use_autocast,
        model_cache_dir=model_cache_dir,
        model=model,
        processor=processor,
    )
    if skipped:
        details = ", ".join(record.get("frame_id", "") for record in skipped)
        raise ValueError(f"Failed to encode VKIS query images: {details}")
    return {
        record["frame_id"]: embeddings[int(record["embedding_index"])]
        for record in embedding_records
    }


def localize_vkis_frame(
    query_image: Path,
    video_path: Path,
    candidate: FrameRecord,
    *,
    fps: float,
    frame_count: int,
    radius_frames: int,
    resize_size: tuple[int, int] = (96, 96),
) -> int:
    """Find the closest decoded frame around one FAISS keyframe candidate."""
    if candidate.frame_index is None:
        center = int(round(candidate.timestamp * fps))
    else:
        center = int(candidate.frame_index)
    start = max(0, center - radius_frames)
    end = min(frame_count - 1, center + radius_frames)
    if candidate.shot_start is not None:
        start = max(start, int(math.floor(candidate.shot_start * fps)))
    if candidate.shot_end is not None:
        end = min(end, int(math.ceil(candidate.shot_end * fps)))
    if end < start:
        return max(0, min(center, frame_count - 1))

    query = image_to_small_array(query_image, resize_size)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video for VKIS refinement: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    best_index = start
    best_mse = float("inf")
    try:
        for frame_index in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, resize_size, interpolation=cv2.INTER_AREA)
            score = mse(query, resized.astype(np.float32))
            if score < best_mse:
                best_mse = score
                best_index = frame_index
    finally:
        capture.release()
    return best_index


def _append_answer(
    answers: list[str],
    seen: set[tuple[str, int]],
    *,
    filename: str,
    frame_index: int,
    frame_count: int,
) -> None:
    bounded_index = max(0, min(int(frame_index), frame_count - 1))
    key = (filename, bounded_index)
    if key not in seen:
        seen.add(key)
        answers.append(f"{filename},{bounded_index}")


def answers_from_ranked(
    ranked: Sequence[RankedFrame],
    *,
    corpus: Sequence[CorpusVideo],
    public_root: Path,
    query_image: Path | None = None,
    refine_top_k: int = 0,
    refine_radius_frames: int = 75,
) -> list[str]:
    by_video_id = {video.video_id: video for video in corpus}
    answers: list[str] = []
    seen: set[tuple[str, int]] = set()
    refinement_cache: dict[tuple[str, str, int], int] = {}

    for rank, item in enumerate(ranked):
        video = by_video_id.get(item.record.video_id)
        if video is None:
            continue
        frame_index = item.record.frame_index
        if frame_index is None:
            frame_index = int(round(item.record.timestamp * video.fps))
        if query_image is not None and rank < refine_top_k:
            cache_key = (video.video_id, item.record.shot_id, int(frame_index))
            if cache_key not in refinement_cache:
                refinement_cache[cache_key] = localize_vkis_frame(
                    query_image=query_image,
                    video_path=public_root / video.relative_path,
                    candidate=item.record,
                    fps=video.fps,
                    frame_count=video.frame_count,
                    radius_frames=refine_radius_frames,
                )
            frame_index = refinement_cache[cache_key]
        _append_answer(
            answers,
            seen,
            filename=video.filename,
            frame_index=frame_index,
            frame_count=video.frame_count,
        )
        if len(answers) == ANSWER_COUNT:
            return answers

    # A valid deterministic tail is preferable to an invalid short row. These
    # entries are only reached if retrieval returned fewer than 100 unique pairs.
    for video in corpus:
        _append_answer(
            answers,
            seen,
            filename=video.filename,
            frame_index=0,
            frame_count=video.frame_count,
        )
        if len(answers) == ANSWER_COUNT:
            return answers
    raise ValueError("Could not produce 100 unique valid answers")


def answers_from_results(
    results: Sequence[RetrievalResult],
    *,
    corpus: Sequence[CorpusVideo],
) -> list[str]:
    """Convert hybrid frame/segment results to the competition answer contract."""
    by_video_id = {video.video_id: video for video in corpus}
    answers: list[str] = []
    seen: set[tuple[str, int]] = set()
    for result in results:
        video = by_video_id.get(result.video_id)
        if video is None:
            continue
        frame_index = result.frame_index
        if frame_index is None:
            frame_index = int(round(result.timestamp * video.fps))
        _append_answer(
            answers,
            seen,
            filename=video.filename,
            frame_index=frame_index,
            frame_count=video.frame_count,
        )
        if len(answers) == ANSWER_COUNT:
            return answers

    for video in corpus:
        _append_answer(
            answers,
            seen,
            filename=video.filename,
            frame_index=0,
            frame_count=video.frame_count,
        )
        if len(answers) == ANSWER_COUNT:
            return answers
    raise ValueError("Could not produce 100 unique valid hybrid answers")


def answers_from_advanced(
    ranked: Sequence[AdvancedRankedFrame],
    *,
    corpus: Sequence[CorpusVideo],
    public_root: Path,
    run_root: Path,
    query_image: Path | None = None,
    refine_top_k: int = 0,
    refine_radius_frames: int = 75,
) -> list[str]:
    """Convert dense results without fabricating a frame-zero padding tail."""
    by_video_id = {video.video_id: video for video in corpus}
    answers: list[str] = []
    seen: set[tuple[str, int]] = set()
    for rank, item in enumerate(ranked):
        record = item.record
        video = by_video_id.get(str(record.get("video_id") or ""))
        if video is None:
            continue
        frame_index = int(record.get("frame_index") or 0)
        if query_image is not None and rank < refine_top_k:
            frame_record = FrameRecord.from_dict(
                item.dense_row,
                {
                    **dict(record),
                    "keyframe_path": resolve_run_reference(
                        run_root,
                        str(record.get("candidate_image") or ""),
                    ).as_posix(),
                },
            )
            frame_index = localize_vkis_frame(
                query_image=query_image,
                video_path=public_root / video.relative_path,
                candidate=frame_record,
                fps=video.fps,
                frame_count=video.frame_count,
                radius_frames=refine_radius_frames,
            )
        _append_answer(
            answers,
            seen,
            filename=video.filename,
            frame_index=frame_index,
            frame_count=video.frame_count,
        )
        if len(answers) == ANSWER_COUNT:
            return answers
    raise ValueError(
        "Advanced retrieval produced fewer than 100 unique answers; "
        f"got {len(answers)}. Increase the coarse/dense reserve instead of padding."
    )


def build_competition_hybrid_engine(
    visual_engine: VisualSearchEngine,
    *,
    text_index_path: Path,
    retrieval_config_path: Path,
    search_depth: int,
) -> HybridSearchEngine:
    """Build the visual+caption+OCR+object retrieval stack."""
    if not text_index_path.exists():
        raise FileNotFoundError(
            f"Competition text index not found: {text_index_path}; "
            "run enrich, segments, and text-index before predict"
        )
    runtime = load_retrieval_runtime_config(retrieval_config_path)
    shared_searcher = TextIndexSearcher(text_index_path)
    text_top_k = max(
        runtime.text_index.max_top_k,
        runtime.hybrid.text_stage1_top_k,
    )
    text_engines = {
        "caption": CaptionSearchEngine(
            text_index_path,
            default_top_k=text_top_k,
            max_top_k=text_top_k,
            searcher=shared_searcher,
        ),
        "ocr": OcrSearchEngine(
            text_index_path,
            default_top_k=text_top_k,
            max_top_k=text_top_k,
            searcher=shared_searcher,
        ),
        "objects": ObjectSearchEngine(
            text_index_path,
            default_top_k=text_top_k,
            max_top_k=text_top_k,
            searcher=shared_searcher,
        ),
    }
    config = HybridSearchConfig(
        stage1_top_k=runtime.hybrid.stage1_top_k,
        text_stage1_top_k=runtime.hybrid.text_stage1_top_k,
        rerank_pool_size=max(runtime.hybrid.rerank_pool_size, search_depth),
        default_top_k=runtime.hybrid.default_top_k,
        max_top_k=max(runtime.hybrid.max_top_k, search_depth),
        max_gap_seconds=runtime.hybrid.max_gap_seconds,
    )
    return HybridSearchEngine(
        visual_engine=visual_engine,
        text_engines=text_engines,
        reranker=HybridReranker(runtime.rerank),
        config=config,
    )


def write_submission(
    output_path: Path,
    *,
    columns: Sequence[str],
    questions: Sequence[Question],
    predictions: dict[str, list[str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for question in questions:
        answers = predictions.get(question.query_id)
        if answers is None or len(answers) != ANSWER_COUNT:
            raise ValueError(f"Missing 100 answers for {question.query_id}")
    descriptor, raw_temp = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(columns)
            for question in questions:
                writer.writerow([question.query_id, *predictions[question.query_id]])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def init_run_command(args: argparse.Namespace) -> None:
    baseline: dict[str, object] = {}
    if args.baseline_source_root is not None:
        source_root = args.baseline_source_root.resolve()
        baseline["source_output_root"] = Path(
            os.path.relpath(source_root, args.run_root.resolve())
        ).as_posix()
        baseline["expected_max_gap_seconds"] = args.baseline_max_gap_seconds
    manifest = initialize_run_manifest(
        run_root=args.run_root,
        repo_root=Path(__file__).resolve().parents[1],
        public_root=args.public_root,
        baseline=baseline,
    )
    if args.baseline_submission is not None:
        report = validate_submission(args.baseline_submission, args.public_root)
        destination = args.run_root.resolve() / "results" / "submission.csv"
        atomic_copy(args.baseline_submission.resolve(), destination)
        record_submission(
            args.run_root,
            destination,
            query_count=int(report["query_count"]),
            answers_per_query=int(report["answers_per_query"]),
        )
        if args.baseline_score is not None:
            record_leaderboard_score(
                args.run_root,
                score=args.baseline_score,
                split="public_private",
                source="user_reported",
            )
            set_active_baseline(
                run_root=args.run_root,
                runs_root=args.run_root.resolve().parent,
            )
        manifest = update_run_manifest(args.run_root, status="baseline_registered")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def dense_index_command(args: argparse.Namespace) -> None:
    initialize_run_manifest(
        run_root=args.run_root,
        repo_root=Path(__file__).resolve().parents[1],
        public_root=args.public_root,
    )
    started = time.perf_counter()
    manifest = build_dense_index(
        run_root=args.run_root,
        source_workspace=args.source_workspace,
        source_output_root=args.source_output_root,
    )
    report = validate_dense_index(args.run_root, verify_sources=True)
    elapsed = round(time.perf_counter() - started, 3)
    update_run_manifest(
        args.run_root,
        offline={
            "config_sha256": manifest["offline_config_sha256"],
            "encoder": manifest["encoder"],
            "candidate_count": manifest["candidate_count"],
        },
        artifacts={
            "dense_manifest": {
                "path": "dense/dense_manifest.json",
                "sha256": _sha256_file(
                    args.run_root.resolve() / "dense" / "dense_manifest.json"
                ),
            }
        },
        stages={
            "dense_index": {
                "status": "passed",
                "elapsed_seconds": elapsed,
                "candidate_count": report["candidate_count"],
            }
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def validate_dense_index_command(args: argparse.Namespace) -> None:
    report = validate_dense_index(
        args.run_root,
        verify_sources=not args.skip_source_validation,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def record_score_command(args: argparse.Namespace) -> None:
    manifest = record_leaderboard_score(
        args.run_root,
        score=args.score,
        split=args.split,
        source=args.source,
    )
    print(json.dumps(manifest["leaderboard"][-1], ensure_ascii=False, indent=2))


def promote_run_command(args: argparse.Namespace) -> None:
    active = promote_active_run(
        run_root=args.run_root,
        runs_root=args.runs_root,
        minimum_score=args.minimum_score,
    )
    print(json.dumps(active, ensure_ascii=False, indent=2))


def precompute_tkis_query_plans(
    questions: Sequence[Question],
    *,
    profile: str,
    expansion_config,
    device: str,
    cache_dir: Path,
    model_cache_dir: Path,
    local_files_only: bool,
    provider_factory: Callable[..., object] | None = None,
) -> dict[str, QueryPlan]:
    """Expand TKIS queries once before SigLIP allocation; VKIS never calls it."""
    tkis_questions = [question for question in questions if question.task == "TKIS"]
    if not tkis_questions:
        return {}
    factory = provider_factory or build_production_query_expansion_provider
    provider = None
    if expansion_config.enabled:
        provider = factory(
            config=expansion_config,
            device=device,
            cache_dir=cache_dir,
            model_cache_dir=model_cache_dir,
            local_files_only=local_files_only,
        )
    try:
        return {
            question.query_id: build_query_plan(
                question.text,
                profile=profile,
                expansion_provider=provider,
                expansion_config=expansion_config,
            )
            for question in tkis_questions
        }
    finally:
        if provider is not None:
            provider.close()


def predict_command(args: argparse.Namespace) -> None:
    public_root = args.public_root.resolve()
    output_root = args.output_root.resolve()
    advanced_mode = args.retrieval_mode == "advanced"
    if advanced_mode and args.run_root is None:
        raise ValueError("--run-root is required for --retrieval-mode advanced")
    run_root = args.run_root.resolve() if args.run_root is not None else None
    dense_run_root = (
        args.dense_run_root.resolve()
        if args.dense_run_root is not None
        else run_root
    )
    submission_path = (
        args.submission_path
        if args.submission_path is not None
        else (
            run_root / "results" / "submission.csv"
            if run_root is not None
            else output_root / "results" / "submission.csv"
        )
    )
    corpus = load_corpus(public_root)
    questions = load_questions(public_root)
    columns = submission_columns(public_root)
    paths = competition_index_paths(output_root)
    runtime = load_retrieval_runtime_config(args.retrieval_config)
    resolved_device = choose_device(args.device)

    _require_current_index_lineage(
        corpus=corpus,
        public_root=public_root,
        output_root=output_root,
        manifest_path=paths["manifest"],
    )
    _require_current_text_index_lineage(
        corpus=corpus,
        public_root=public_root,
        output_root=output_root,
    )
    query_plans: dict[str, QueryPlan] = {}
    if advanced_mode:
        expansion_config = replace(
            runtime.query_expansion,
            enabled=(
                runtime.query_expansion.enabled
                and not args.no_query_expansion
                and not args.no_query_plan
            ),
        )
        query_plans = precompute_tkis_query_plans(
            questions,
            profile=args.retrieval_profile,
            expansion_config=expansion_config,
            device=resolved_device,
            cache_dir=args.query_expansion_cache_dir,
            model_cache_dir=args.query_expansion_model_cache_dir,
            local_files_only=args.offline_model_cache,
        )
        gc.collect()
    contract = load_encoder_contract(paths["manifest"])
    model, processor = load_siglip2_model_processor(
        model_name=contract.model_name,
        model_revision=contract.model_revision,
        device=resolved_device,
        model_cache_dir=args.model_cache_dir,
        use_autocast=not args.no_autocast,
        local_files_only=args.offline_model_cache,
    )
    text_encoder = Siglip2TextEncoder(
        contract=contract,
        device=resolved_device,
        model_cache_dir=args.model_cache_dir,
        no_autocast=args.no_autocast,
        model=model,
        processor=processor,
    )
    searcher = FaissVectorSearcher(paths["index"], expected_dim=contract.vector_dim)
    metadata_store = MetadataStore.from_frame_map(paths["frame_map"])
    text_engine = VisualSearchEngine(
        config=VisualSearchConfig(
            index_path=paths["index"],
            frame_map_path=paths["frame_map"],
            manifest_path=paths["manifest"],
            device=resolved_device,
            model_cache_dir=args.model_cache_dir,
            no_autocast=args.no_autocast,
            default_top_k=args.search_depth,
            max_top_k=max(args.search_depth, runtime.hybrid.stage1_top_k),
        ),
        encoder=text_encoder,
        searcher=searcher,
        metadata_store=metadata_store,
    )
    hybrid_engine = build_competition_hybrid_engine(
        text_engine,
        text_index_path=paths["text_index"],
        retrieval_config_path=args.retrieval_config,
        search_depth=args.search_depth,
    )
    print(
        "TKIS retrieval modalities: "
        + ", ".join(hybrid_engine.available_modalities)
    )
    vkis_vectors = encode_vkis_questions(
        questions,
        public_root=public_root,
        model_name=contract.model_name,
        model_revision=contract.model_revision,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=resolved_device,
        use_autocast=not args.no_autocast,
        model_cache_dir=args.model_cache_dir,
        model=model,
        processor=processor,
    )

    dense_index: DenseCandidateIndex | None = None
    advanced_config: AdvancedSearchConfig | None = None
    if advanced_mode:
        assert run_root is not None
        initialize_run_manifest(
            run_root=run_root,
            repo_root=Path(__file__).resolve().parents[1],
            public_root=public_root,
        )
        assert dense_run_root is not None
        dense_index = DenseCandidateIndex(dense_run_root, verify_sources=True)
        dense_encoder = dense_index.manifest["encoder"]
        if (
            str(dense_encoder.get("model_name")) != contract.model_name
            or str(dense_encoder.get("resolved_model_revision")) != contract.model_revision
            or int(dense_index.manifest["vector_dim"]) != contract.vector_dim
        ):
            raise RuntimeError("Dense and coarse encoder lineage do not match")
        advanced_config = AdvancedSearchConfig(
            coarse_top_n=args.coarse_top_n,
            dense_global_top_k=args.dense_global_top_k,
            dense_rescue_clips=args.dense_rescue_clips,
            max_total_clips=args.max_candidate_clips,
            dense_frames_per_clip=args.dense_frames_per_clip,
            rrf_k=args.rrf_k,
            modality_hint_boost=args.modality_hint_boost,
            similarity_threshold=args.cses_similarity_threshold,
            temporal_window_seconds=args.cses_temporal_window_seconds,
            max_event_gap_seconds=runtime.hybrid.max_gap_seconds,
            query_plan_enabled=not args.no_query_plan,
            rrf_enabled=not args.no_rrf,
            dense_rescue_enabled=not args.no_dense_rescue,
            cses_enabled=not args.no_cses,
            deterministic_rerank_enabled=not args.no_deterministic_rerank,
            query_expansion=expansion_config,
        )

    predictions: dict[str, list[str]] = {}
    advanced_results: dict[str, list[AdvancedRankedFrame]] = {}
    advanced_traces: dict[str, dict[str, object]] = {}
    question_lookup = {question.query_id: question for question in questions}
    predict_started = time.perf_counter()
    if advanced_mode and resolved_device.startswith("cuda"):
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    for number, question in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] {question.query_id} {question.task}")
        if advanced_mode:
            assert dense_index is not None and advanced_config is not None
            if question.task == "TKIS":
                response = advanced_text_search(
                    question.text,
                    hybrid_engine=hybrid_engine,
                    text_encoder=text_encoder,
                    dense_index=dense_index,
                    profile=args.retrieval_profile,
                    config=advanced_config,
                    plan=query_plans[question.query_id],
                )
            else:
                vector = vkis_vectors.get(question.query_id)
                if vector is None:
                    raise ValueError(f"Missing VKIS embedding for {question.query_id}")
                coarse = _ranked_records(
                    vector,
                    searcher=searcher,
                    metadata_store=metadata_store,
                    top_k=args.search_depth,
                )
                coarse_results = [
                    RetrievalResult(
                        video_id=item.record.video_id,
                        frame_id=item.record.frame_id,
                        timestamp=item.record.timestamp,
                        score=item.score,
                        segment_id=item.record.segment_id,
                        shot_id=item.record.shot_id,
                        faiss_index=item.record.faiss_index,
                        frame_index=item.record.frame_index,
                        keyframe_path=item.record.keyframe_path,
                    )
                    for item in coarse
                ]
                response = advanced_vector_search(
                    vector,
                    coarse_results=coarse_results,
                    dense_index=dense_index,
                    config=advanced_config,
                )
            advanced_results[question.query_id] = list(response.results)
            advanced_traces[question.query_id] = response.trace()
            continue
        if question.task == "TKIS":
            events = decompose_temporal_query(question.text)
            if args.tkis_routing == "auto-temporal" and len(events) > 1:
                matches = hybrid_engine.temporal_search(
                    question.text,
                    top_k=args.search_depth,
                )
                tkis_results = [
                    event
                    for match in matches
                    for event in match.events
                ]
            else:
                response = hybrid_engine.search(
                    question.text,
                    top_k=args.search_depth,
                )
                tkis_results = response.results
            predictions[question.query_id] = answers_from_results(
                tkis_results,
                corpus=corpus,
            )
        else:
            vector = vkis_vectors.get(question.query_id)
            if vector is None:
                raise ValueError(f"Missing VKIS embedding for {question.query_id}")
            ranked = _ranked_records(
                vector,
                searcher=searcher,
                metadata_store=metadata_store,
                top_k=args.search_depth,
            )
            predictions[question.query_id] = answers_from_ranked(
                ranked,
                corpus=corpus,
                public_root=public_root,
                query_image=public_root / question.query_image,
                refine_top_k=args.vkis_refine_top_k,
                refine_radius_frames=args.vkis_refine_radius_frames,
            )

    vlm_reports: dict[str, dict[str, object]] = {}
    if advanced_mode:
        assert run_root is not None and dense_index is not None
        vlm_runner = None
        vlm_initialization_error = ""
        if args.vlm_mode != "off":
            # All SigLIP queries have been encoded.  Release the shared model
            # before allocating a local VLM on the 6 GB target GPU.
            del hybrid_engine, text_engine, text_encoder, model, processor
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                vlm_runner = build_local_vlm_runner(
                    args.vlm_model_name,
                    args.vlm_model_revision,
                )
            except Exception as exc:
                vlm_initialization_error = f"{type(exc).__name__}: {exc}"
                if args.vlm_mode == "required":
                    raise RuntimeError(
                        f"Required VLM initialization failed: {vlm_initialization_error}"
                    ) from exc

        trace_rows: list[dict[str, object]] = []
        for question in questions:
            ranked = advanced_results[question.query_id]
            if args.vlm_mode != "off" and vlm_runner is not None:
                ranked, vlm_report = rerank_with_vlm(
                    ranked,
                    query=question.text or "visual query image",
                    mode=args.vlm_mode,
                    cache_root=run_root / "cache" / "vlm",
                    image_resolver=lambda item: resolve_run_reference(
                        dense_run_root,
                        str(item.record.get("candidate_image") or ""),
                    ),
                    runner=vlm_runner,
                    model_name=args.vlm_model_name,
                    model_revision=args.vlm_model_revision,
                    top_m=args.vlm_top_m,
                    timeout_seconds=args.vlm_timeout_seconds,
                )
                vlm_reports[question.query_id] = asdict(vlm_report)
            elif args.vlm_mode != "off":
                vlm_reports[question.query_id] = {
                    "mode": args.vlm_mode,
                    "status": "fallback",
                    "fallback_reason": vlm_initialization_error,
                }
            else:
                vlm_reports[question.query_id] = {
                    "mode": "off",
                    "status": "disabled",
                }
            advanced_results[question.query_id] = ranked
            query_image = (
                public_root / question.query_image
                if question.task == "VKIS"
                else None
            )
            predictions[question.query_id] = answers_from_advanced(
                ranked,
                corpus=corpus,
                public_root=public_root,
                run_root=dense_run_root,
                query_image=query_image,
                refine_top_k=(args.vkis_refine_top_k if query_image is not None else 0),
                refine_radius_frames=args.vkis_refine_radius_frames,
            )
            trace = dict(advanced_traces[question.query_id])
            trace["query_id"] = question.query_id
            trace["task"] = question.task
            trace["vlm"] = vlm_reports[question.query_id]
            trace["final_results"] = [item.to_dict() for item in ranked]
            trace_rows.append(trace)
        atomic_write_jsonl(run_root / "results" / "query_traces.jsonl", trace_rows)

    write_submission(
        submission_path,
        columns=columns,
        questions=questions,
        predictions=predictions,
    )
    validation = validate_submission(submission_path, public_root)
    if advanced_mode:
        assert run_root is not None and advanced_config is not None
        elapsed = round(time.perf_counter() - predict_started, 3)
        peak_vram_mib: float | None = None
        if resolved_device.startswith("cuda"):
            try:
                import torch

                peak_vram_mib = round(torch.cuda.max_memory_allocated() / (1024**2), 3)
            except Exception:
                peak_vram_mib = None
        dense_manifest_path = dense_run_root / "dense" / "dense_manifest.json"
        artifact_ledger = {
            "coarse_index": {
                "path": Path(os.path.relpath(paths["index"], run_root)).as_posix(),
                "sha256": _sha256_file(paths["index"]),
            },
            "coarse_manifest": {
                "path": Path(os.path.relpath(paths["manifest"], run_root)).as_posix(),
                "sha256": _sha256_file(paths["manifest"]),
            },
            "text_index": {
                "path": Path(os.path.relpath(paths["text_index"], run_root)).as_posix(),
                "sha256": _sha256_file(paths["text_index"]),
            },
            "dense_manifest": {
                "path": Path(os.path.relpath(dense_manifest_path, run_root)).as_posix(),
                "sha256": _sha256_file(dense_manifest_path),
            },
        }
        record_submission(
            run_root,
            submission_path,
            query_count=int(validation["query_count"]),
            answers_per_query=int(validation["answers_per_query"]),
        )
        update_run_manifest(
            run_root,
            git=git_fingerprint(Path(__file__).resolve().parents[1]),
            artifacts=artifact_ledger,
            retrieval={
                "mode": "advanced",
                "config": asdict(advanced_config),
                "profile": args.retrieval_profile,
                "vlm_mode": args.vlm_mode,
                "vlm_model": args.vlm_model_name,
                "vlm_model_revision": args.vlm_model_revision,
            },
            stages={
                "predict": {
                    "status": "passed",
                    "elapsed_seconds": elapsed,
                    "vlm_fallback_queries": sum(
                        report.get("status") == "fallback"
                        for report in vlm_reports.values()
                    ),
                    "device": resolved_device,
                    "peak_vram_mib": peak_vram_mib,
                    "oom": False,
                }
            },
            status="submission_validated",
        )
    print(f"Submission written: {submission_path}")


def validate_submission(path: Path, public_root: Path) -> dict:
    expected_columns = submission_columns(public_root)
    questions = load_questions(public_root)
    corpus = load_corpus(public_root)
    corpus_by_name = {video.filename: video for video in corpus}
    header, rows = _read_csv(path)
    if header != expected_columns:
        raise ValueError("Submission header does not exactly match sample_submission.csv")
    expected_ids = [question.query_id for question in questions]
    actual_ids = [(row.get("query_id") or "").strip() for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("Submission query_id rows/order do not match questions.csv")

    duplicate_count = 0
    for row_number, row in enumerate(rows, start=2):
        seen: set[tuple[str, int]] = set()
        for column in expected_columns[1:]:
            value = row.get(column)
            if value is None or "," not in value:
                raise ValueError(f"Submission row {row_number} {column}: invalid answer")
            filename, raw_frame_index = value.rsplit(",", 1)
            video = corpus_by_name.get(filename)
            if video is None:
                raise ValueError(
                    f"Submission row {row_number} {column}: unknown video {filename!r}"
                )
            try:
                frame_index = int(raw_frame_index)
            except ValueError as exc:
                raise ValueError(
                    f"Submission row {row_number} {column}: frame index is not an integer"
                ) from exc
            if frame_index < 0 or frame_index >= video.frame_count:
                raise ValueError(
                    f"Submission row {row_number} {column}: frame {frame_index} is outside "
                    f"[0, {video.frame_count - 1}]"
                )
            key = (filename, frame_index)
            duplicate_count += int(key in seen)
            seen.add(key)
    return {
        "status": "passed",
        "submission": path.as_posix(),
        "query_count": len(rows),
        "answers_per_query": ANSWER_COUNT,
        "exact_duplicate_answers": duplicate_count,
    }


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def _add_phase3_keyframe_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_paths(parser)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--phash-window-sec", type=float, default=12.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--shot-threshold", type=float, default=0.5)
    parser.add_argument(
        "--candidate-interval-sec",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
    )
    parser.add_argument(
        "--boundary-guard-sec",
        type=float,
        default=DEFAULT_BOUNDARY_GUARD_SEC,
    )
    parser.add_argument(
        "--tiny-shot-max-sec",
        type=float,
        default=DEFAULT_TINY_SHOT_MAX_SEC,
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=DEFAULT_MAX_GAP_SECONDS,
    )
    parser.add_argument("--gap-tolerance-seconds", type=float, default=0.0)
    parser.add_argument("--target-keyframes", type=int, default=None)
    parser.add_argument(
        "--target-density-per-second",
        type=float,
        default=0.5,
        help="Soft MMR target used when --target-keyframes is omitted.",
    )
    parser.add_argument("--hard-max-keyframes", type=int, default=None)
    parser.add_argument("--importance-weight", type=float, default=0.65)
    parser.add_argument("--novelty-weight", type=float, default=0.35)
    parser.add_argument(
        "--endpoint-protection",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument("--dedup-similarity-threshold", type=float, default=0.92)
    parser.add_argument("--dedup-temporal-window-seconds", type=float, default=12.0)

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--batch-size", type=parse_batch_size, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument(
        "--model-cache-root",
        type=Path,
        default=Path("data/model_cache"),
    )
    parser.add_argument("--caption-model-name", default=DEFAULT_CAPTION_MODEL_NAME)
    parser.add_argument(
        "--caption-model-revision",
        default=DEFAULT_CAPTION_MODEL_REVISION,
    )
    parser.add_argument("--caption-batch-size", type=int, default=2)
    parser.add_argument("--caption-max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--caption-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--caption-quantization",
        choices=("none", "8bit", "4bit"),
        default="none",
    )
    parser.add_argument("--no-segment-caption", action="store_true")
    parser.add_argument("--ocr-detection-model", default=DEFAULT_OCR_DETECTION_MODEL)
    parser.add_argument("--ocr-recognition-model", default=DEFAULT_OCR_RECOGNITION_MODEL)
    parser.add_argument("--ocr-model-revision", default=DEFAULT_OCR_MODEL_REVISION)
    parser.add_argument("--ocr-batch-size", type=int, default=4)
    parser.add_argument("--ocr-conf-threshold", type=float, default=0.3)
    parser.add_argument("--object-model-name", default=DEFAULT_OBJECT_MODEL_NAME)
    parser.add_argument("--object-model-revision", default=DEFAULT_OBJECT_MODEL_REVISION)
    parser.add_argument("--object-batch-size", type=int, default=8)
    parser.add_argument("--object-conf-threshold", type=float, default=0.25)
    parser.add_argument("--object-iou-threshold", type=float, default=0.7)
    parser.add_argument(
        "--object-prompt-mode",
        choices=("text", "internal"),
        default="text",
    )
    parser.add_argument(
        "--object-vocabulary",
        nargs="+",
        default=list(DEFAULT_OBJECT_VOCABULARY),
    )
    parser.add_argument("--ocr-event-min-confidence", type=float, default=0.75)
    parser.add_argument("--object-event-min-confidence", type=float, default=0.65)
    parser.add_argument("--transition-absolute-floor", type=float, default=0.18)
    parser.add_argument("--transition-mad-multiplier", type=float, default=2.5)
    parser.add_argument(
        "--allow-partial-features",
        action="store_true",
        help=(
            "Explicit degraded mode: selection may proceed after OCR/object errors. "
            "Detected-event guarantees then apply only to successful artifacts."
        ),
    )
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_run_parser = subparsers.add_parser(
        "init-run",
        description="Create an immutable experiment manifest or register the 5s baseline.",
    )
    init_run_parser.add_argument("--run-root", type=Path, required=True)
    init_run_parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    init_run_parser.add_argument("--baseline-source-root", type=Path, default=None)
    init_run_parser.add_argument("--baseline-submission", type=Path, default=None)
    init_run_parser.add_argument("--baseline-score", type=float, default=None)
    init_run_parser.add_argument("--baseline-max-gap-seconds", type=float, default=5.0)

    dense_index_parser = subparsers.add_parser(
        "dense-index",
        description="Publish the cached Phase-3 dense candidates as a global safety index.",
    )
    dense_index_parser.add_argument("--run-root", type=Path, required=True)
    dense_index_parser.add_argument(
        "--source-workspace",
        type=Path,
        default=Path("competition/work/keyframe_v3"),
    )
    dense_index_parser.add_argument(
        "--source-output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    dense_index_parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)

    validate_dense_parser = subparsers.add_parser(
        "validate-dense-index",
        description="Fail closed on dense row, checksum, encoder or offline lineage drift.",
    )
    validate_dense_parser.add_argument("--run-root", type=Path, required=True)
    validate_dense_parser.add_argument("--skip-source-validation", action="store_true")

    record_score_parser = subparsers.add_parser(
        "record-score",
        description="Bind one leaderboard score to the current submission checksum.",
    )
    record_score_parser.add_argument("--run-root", type=Path, required=True)
    record_score_parser.add_argument("--score", type=float, required=True)
    record_score_parser.add_argument("--split", default="public")
    record_score_parser.add_argument("--source", default="user_reported")

    promote_run_parser = subparsers.add_parser(
        "promote-run",
        description="Atomically update active_run.json after the score gate passes.",
    )
    promote_run_parser.add_argument("--run-root", type=Path, required=True)
    promote_run_parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("competition/runs"),
    )
    promote_run_parser.add_argument("--minimum-score", type=float, default=0.818)

    validate_input_parser = subparsers.add_parser(
        "validate-input",
        description="Validate the public CSV/file contract without processing videos.",
    )
    validate_input_parser.add_argument(
        "--public-root",
        type=Path,
        default=DEFAULT_PUBLIC_ROOT,
    )

    keyframes_parser = subparsers.add_parser(
        "keyframes",
        description=(
            "Phase 3: materialize the dense pool, run multimodal features, enforce "
            "event/shot/temporal guarantees, and publish canonical keyframes."
        ),
    )
    _add_phase3_keyframe_arguments(keyframes_parser)

    reselect_parser = subparsers.add_parser(
        "reselect-keyframes",
        description=(
            "Run offline selector ablations from the validated Phase-3 feature "
            "cache and publish into an isolated run root without model inference."
        ),
    )
    _add_phase3_keyframe_arguments(reselect_parser)
    reselect_parser.add_argument("--run-root", type=Path, required=True)
    reselect_parser.add_argument(
        "--source-output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    extract_parser = subparsers.add_parser(
        "extract",
        description=(
            "Extract competition keyframes with dense temporal/shot coverage by default."
        ),
    )
    _add_common_paths(extract_parser)
    extract_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    extract_parser.add_argument("--phash-threshold", type=int, default=6)
    extract_parser.add_argument("--phash-window-sec", type=float, default=12.0)
    extract_parser.add_argument("--jpeg-quality", type=int, default=95)
    extract_parser.add_argument("--shot-threshold", type=float, default=0.5)
    extract_parser.add_argument(
        "--keyframe-strategy",
        choices=KEYFRAME_STRATEGIES,
        default=KEYFRAME_STRATEGY_DENSE_COVERAGE,
        help="Use 'legacy' as an explicit rollback path.",
    )
    extract_parser.add_argument(
        "--candidate-interval-sec",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
    )
    extract_parser.add_argument(
        "--boundary-guard-sec",
        type=float,
        default=DEFAULT_BOUNDARY_GUARD_SEC,
    )
    extract_parser.add_argument(
        "--tiny-shot-max-sec",
        type=float,
        default=DEFAULT_TINY_SHOT_MAX_SEC,
    )
    extract_parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=DEFAULT_MAX_GAP_SECONDS,
    )
    extract_parser.add_argument("--gap-tolerance-seconds", type=float, default=0.0)
    extract_parser.add_argument("--target-keyframes", type=int, default=None)
    extract_parser.add_argument("--hard-max-keyframes", type=int, default=None)
    extract_parser.add_argument("--resume", action="store_true")

    embed_parser = subparsers.add_parser(
        "embed", description="Encode extracted keyframes with the existing SigLIP2 encoder."
    )
    _add_common_paths(embed_parser)
    embed_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    embed_parser.add_argument("--model-revision", default=None)
    embed_parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    embed_parser.add_argument("--batch-size", type=parse_batch_size, default="auto")
    embed_parser.add_argument("--num-workers", type=int, default=0)
    embed_parser.add_argument("--prefetch-factor", type=int, default=2)
    embed_parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    embed_parser.add_argument("--no-autocast", action="store_true")
    embed_parser.add_argument("--resume", action="store_true")

    index_parser = subparsers.add_parser(
        "index", description="Build the competition FAISS index from encoded keyframes."
    )
    _add_common_paths(index_parser)

    enrich_parser = subparsers.add_parser(
        "enrich",
        description="Run Qwen caption, PP-OCRv5, and YOLOE object ingestion.",
    )
    _add_common_paths(enrich_parser)
    enrich_parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("caption", "ocr", "objects"),
        default=["caption", "ocr", "objects"],
    )
    enrich_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    enrich_parser.add_argument(
        "--model-cache-root",
        type=Path,
        default=Path("data/model_cache"),
    )
    enrich_parser.add_argument(
        "--caption-model-name",
        default=DEFAULT_CAPTION_MODEL_NAME,
    )
    enrich_parser.add_argument(
        "--caption-model-revision",
        default=DEFAULT_CAPTION_MODEL_REVISION,
    )
    enrich_parser.add_argument("--caption-batch-size", type=int, default=2)
    enrich_parser.add_argument("--caption-max-new-tokens", type=int, default=384)
    enrich_parser.add_argument(
        "--caption-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    enrich_parser.add_argument(
        "--caption-quantization",
        choices=("none", "8bit", "4bit"),
        default="none",
    )
    enrich_parser.add_argument("--no-segment-caption", action="store_true")
    enrich_parser.add_argument(
        "--ocr-detection-model",
        default=DEFAULT_OCR_DETECTION_MODEL,
    )
    enrich_parser.add_argument(
        "--ocr-recognition-model",
        default=DEFAULT_OCR_RECOGNITION_MODEL,
    )
    enrich_parser.add_argument(
        "--ocr-model-revision",
        default=DEFAULT_OCR_MODEL_REVISION,
    )
    enrich_parser.add_argument("--ocr-batch-size", type=int, default=4)
    enrich_parser.add_argument("--ocr-conf-threshold", type=float, default=0.3)
    enrich_parser.add_argument(
        "--object-model-name",
        default=DEFAULT_OBJECT_MODEL_NAME,
    )
    enrich_parser.add_argument(
        "--object-model-revision",
        default=DEFAULT_OBJECT_MODEL_REVISION,
    )
    enrich_parser.add_argument("--object-batch-size", type=int, default=8)
    enrich_parser.add_argument("--object-conf-threshold", type=float, default=0.25)
    enrich_parser.add_argument("--object-iou-threshold", type=float, default=0.7)
    enrich_parser.add_argument(
        "--object-prompt-mode",
        choices=("text", "internal"),
        default="text",
    )
    enrich_parser.add_argument(
        "--object-vocabulary",
        nargs="+",
        default=list(DEFAULT_OBJECT_VOCABULARY),
    )
    enrich_parser.add_argument("--overwrite", action="store_true")

    neighbors_parser = subparsers.add_parser(
        "neighbors",
        description="Build the original temporal neighbor metadata index.",
    )
    _add_common_paths(neighbors_parser)
    neighbors_parser.add_argument("--window-seconds", type=float, default=5.0)

    segments_parser = subparsers.add_parser(
        "segments",
        description="Aggregate keyframes and all multimodal artifacts into segments.",
    )
    _add_common_paths(segments_parser)
    segments_parser.add_argument(
        "--strategy",
        choices=("auto", "boundary", "fixed"),
        default="auto",
    )
    segments_parser.add_argument("--fixed-duration-seconds", type=float, default=10.0)
    segments_parser.add_argument(
        "--caption-similarity-threshold",
        type=float,
        default=0.92,
    )

    text_index_parser = subparsers.add_parser(
        "text-index",
        description="Build the caption/OCR/object lexical index.",
    )
    _add_common_paths(text_index_parser)

    predict_parser = subparsers.add_parser(
        "predict", description="Run all TKIS/VKIS queries and write the final submission."
    )
    _add_common_paths(predict_parser)
    predict_parser.add_argument(
        "--submission-path",
        type=Path,
        default=None,
    )
    predict_parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE_DIR,
    )
    predict_parser.add_argument("--batch-size", type=parse_batch_size, default="auto")
    predict_parser.add_argument("--num-workers", type=int, default=0)
    predict_parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    predict_parser.add_argument("--no-autocast", action="store_true")
    predict_parser.add_argument(
        "--offline-model-cache",
        action="store_true",
        help="Refuse network access and load the resolved SigLIP2 revision from cache.",
    )
    predict_parser.add_argument(
        "--query-expansion-cache-dir",
        type=Path,
        default=Path("data/model_cache/query_expansion"),
        help="Cache for the production TKIS paraphrase provider.",
    )
    predict_parser.add_argument(
        "--query-expansion-model-cache-dir",
        type=Path,
        default=Path("data/model_cache/caption"),
        help="Hugging Face cache shared with the Qwen caption stage.",
    )
    predict_parser.add_argument("--search-depth", type=int, default=200)
    predict_parser.add_argument(
        "--tkis-routing",
        choices=("hybrid", "auto-temporal"),
        default="auto-temporal",
    )
    predict_parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
    )
    predict_parser.add_argument("--vkis-refine-top-k", type=int, default=20)
    predict_parser.add_argument("--vkis-refine-radius-frames", type=int, default=75)

    phase5_init_parser = subparsers.add_parser(
        "phase5-init",
        description="Create the immutable deterministic 4/4/8 evaluation split.",
    )
    _add_common_paths(phase5_init_parser)
    phase5_init_parser.add_argument("--split-manifest", type=Path, default=None)
    video_source = phase5_init_parser.add_mutually_exclusive_group(required=True)
    video_source.add_argument("--video-ids", nargs=16)
    video_source.add_argument("--video-ids-file", type=Path)
    phase5_init_parser.add_argument("--seed", type=int, default=42)

    phase5_lock_parser = subparsers.add_parser(
        "phase5-lock",
        description="Freeze the common dev extraction config before validation/test.",
    )
    _add_common_paths(phase5_lock_parser)
    phase5_lock_parser.add_argument("--split-manifest", type=Path, default=None)
    phase5_lock_parser.add_argument("--config-lock", type=Path, default=None)

    phase5_evaluate_parser = subparsers.add_parser(
        "phase5-evaluate",
        description="Evaluate Phase 3 keyframe artifacts with Phase 5 safeguards.",
    )
    _add_common_paths(phase5_evaluate_parser)
    phase5_evaluate_parser.add_argument(
        "--split",
        choices=("dev", "validation", "test"),
        required=True,
    )
    phase5_evaluate_parser.add_argument("--split-manifest", type=Path, default=None)
    phase5_evaluate_parser.add_argument("--config-lock", type=Path, default=None)
    phase5_evaluate_parser.add_argument("--manual-events", type=Path, default=None)
    phase5_evaluate_parser.add_argument(
        "--protection-reviews",
        type=Path,
        default=None,
    )
    predict_parser.add_argument(
        "--retrieval-mode",
        choices=("legacy", "advanced"),
        default="legacy",
    )
    predict_parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Required in advanced mode; holds dense artifacts, traces and submission.",
    )
    predict_parser.add_argument(
        "--dense-run-root",
        type=Path,
        default=None,
        help="Reuse a validated dense index from another immutable run.",
    )
    predict_parser.add_argument(
        "--retrieval-profile",
        choices=("auto", "kis", "avs", "qa", "temporal"),
        default="auto",
    )
    predict_parser.add_argument("--coarse-top-n", type=int, default=50)
    predict_parser.add_argument("--dense-global-top-k", type=int, default=300)
    predict_parser.add_argument("--dense-rescue-clips", type=int, default=10)
    predict_parser.add_argument("--max-candidate-clips", type=int, default=60)
    predict_parser.add_argument("--dense-frames-per-clip", type=int, default=12)
    predict_parser.add_argument("--rrf-k", type=int, default=60)
    predict_parser.add_argument("--modality-hint-boost", type=float, default=1.5)
    predict_parser.add_argument("--cses-similarity-threshold", type=float, default=0.92)
    predict_parser.add_argument(
        "--cses-temporal-window-seconds",
        type=float,
        default=2.0,
    )
    predict_parser.add_argument("--vlm-mode", choices=VLM_MODES, default="off")
    predict_parser.add_argument("--vlm-model-name", default=DEFAULT_VLM_MODEL)
    predict_parser.add_argument("--vlm-model-revision", default="main")
    predict_parser.add_argument("--vlm-top-m", type=int, default=20)
    predict_parser.add_argument("--vlm-timeout-seconds", type=float, default=120.0)
    predict_parser.add_argument("--no-query-plan", action="store_true")
    predict_parser.add_argument(
        "--no-query-expansion",
        action="store_true",
        help="Explicit TKIS ablation: use the original query only.",
    )
    predict_parser.add_argument("--no-rrf", action="store_true")
    predict_parser.add_argument("--no-dense-rescue", action="store_true")
    predict_parser.add_argument("--no-cses", action="store_true")
    predict_parser.add_argument("--no-deterministic-rerank", action="store_true")
    phase5_evaluate_parser.add_argument(
        "--retrieval-evidence",
        type=Path,
        default=None,
    )
    phase5_evaluate_parser.add_argument(
        "--resource-usage",
        type=Path,
        default=None,
    )
    phase5_evaluate_parser.add_argument(
        "--manual-tolerance-seconds",
        type=float,
        default=0.0,
    )
    phase5_evaluate_parser.add_argument("--report-path", type=Path, default=None)
    phase5_evaluate_parser.add_argument("--confirm-locked-test", action="store_true")
    phase5_evaluate_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate-submission", description="Strictly validate a generated submission."
    )
    validate_parser.add_argument(
        "--submission-path",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "results" / "submission.csv",
    )
    validate_parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "init-run":
        if args.baseline_score is not None and args.baseline_submission is None:
            parser.error("--baseline-score requires --baseline-submission")
        init_run_command(args)
    elif args.command == "dense-index":
        dense_index_command(args)
    elif args.command == "validate-dense-index":
        validate_dense_index_command(args)
    elif args.command == "record-score":
        record_score_command(args)
    elif args.command == "promote-run":
        promote_run_command(args)
    elif args.command == "validate-input":
        print(json.dumps(validate_input(args.public_root), ensure_ascii=False, indent=2))
    elif args.command == "keyframes":
        adapter_config = _phase3_adapter_config(args)
        if args.candidate_interval_sec > adapter_config.transition_max_pair_gap_seconds:
            parser.error(
                "--candidate-interval-sec must not exceed the semantic transition "
                f"pair gap ({adapter_config.transition_max_pair_gap_seconds}s)"
            )
        _phase3_selection_config(args)
        keyframes_command(args)
    elif args.command == "reselect-keyframes":
        adapter_config = _phase3_adapter_config(args)
        if args.candidate_interval_sec > adapter_config.transition_max_pair_gap_seconds:
            parser.error(
                "--candidate-interval-sec must not exceed the semantic transition "
                f"pair gap ({adapter_config.transition_max_pair_gap_seconds}s)"
            )
        _phase3_selection_config(args)
        reselect_keyframes_command(args)
    elif args.command == "extract":
        extract_command(args)
    elif args.command == "embed":
        embed_command(args)
    elif args.command == "index":
        index_command(args)
    elif args.command == "enrich":
        enrich_command(args)
    elif args.command == "neighbors":
        neighbors_command(args)
    elif args.command == "segments":
        segments_command(args)
    elif args.command == "text-index":
        text_index_command(args)
    elif args.command == "predict":
        if args.search_depth < ANSWER_COUNT:
            parser.error("--search-depth must be at least 100")
        if args.vkis_refine_top_k < 0 or args.vkis_refine_radius_frames < 0:
            parser.error("VKIS refinement parameters must be non-negative")
        if args.retrieval_mode == "advanced":
            positive = {
                "--coarse-top-n": args.coarse_top_n,
                "--dense-global-top-k": args.dense_global_top_k,
                "--dense-rescue-clips": args.dense_rescue_clips,
                "--max-candidate-clips": args.max_candidate_clips,
                "--dense-frames-per-clip": args.dense_frames_per_clip,
                "--rrf-k": args.rrf_k,
                "--modality-hint-boost": args.modality_hint_boost,
                "--cses-temporal-window-seconds": args.cses_temporal_window_seconds,
                "--vlm-top-m": args.vlm_top_m,
                "--vlm-timeout-seconds": args.vlm_timeout_seconds,
            }
            invalid = [name for name, value in positive.items() if value <= 0]
            if invalid:
                parser.error(f"Advanced retrieval values must be positive: {invalid}")
            if not 0.0 <= args.cses_similarity_threshold <= 1.0:
                parser.error("--cses-similarity-threshold must be within [0, 1]")
            if args.max_candidate_clips < args.coarse_top_n:
                parser.error("--max-candidate-clips must be >= --coarse-top-n")
        predict_command(args)
    elif args.command == "phase5-init":
        phase5_init_command(args)
    elif args.command == "phase5-lock":
        phase5_lock_command(args)
    elif args.command == "phase5-evaluate":
        if args.split == "test" and args.overwrite:
            parser.error("locked test reports cannot be overwritten")
        phase5_evaluate_command(args)
    elif args.command == "validate-submission":
        report = validate_submission(args.submission_path, args.public_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if is_cuda_oom(exc):
            print(
                "CUDA out of memory; the resumable runner may retry after GPU "
                "memory is released.",
                file=sys.stderr,
            )
            raise SystemExit(75) from exc
        raise
