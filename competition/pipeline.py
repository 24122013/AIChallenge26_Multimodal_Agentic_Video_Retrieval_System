"""Run the public TKIS/VKIS competition with the repository's existing services.

This module deliberately contains only competition-specific orchestration:
CSV contracts, artifact paths, multimodal stage wiring, VKIS frame refinement,
and submission writing. Extraction, SigLIP2, FAISS, caption/OCR/object/ASR,
segment/text indexing, hybrid reranking, and image MSE are delegated to the
implementations already present under ``backend.app`` and ``src.indexing``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    encode_keyframes,
    load_jsonl,
    load_siglip2_model_processor,
    parse_batch_size,
    validate_embedding_artifacts,
    write_json,
    write_jsonl,
)
from backend.app.services.indexing.extract_keyframes import (
    DEFAULT_DEDUP_TEMPORAL_WINDOW_SEC,
    DEFAULT_LONG_SHOT_INTERVAL_SEC,
    DEFAULT_PHASH_THRESHOLD,
    DEFAULT_REGULAR_SHOT_MAX_SEC,
    DEFAULT_SHORT_SHOT_MAX_SEC,
    extract_keyframes_for_video,
)
from backend.app.services.indexing.normalize_keyframe_metadata import (
    image_to_small_array,
    mse,
)
from backend.app.services.indexing.validate_keyframes import validate_records
from backend.app.services.ingestion.asr_pipeline import (
    DEFAULT_MODEL_SIZE as DEFAULT_ASR_MODEL_SIZE,
    WhisperBackend,
    run_asr_file,
)
from backend.app.services.ingestion.caption_pipeline import (
    DEFAULT_MODEL_NAME as DEFAULT_CAPTION_MODEL_NAME,
    BlipCaptionBackend,
    run_caption_file,
)
from backend.app.services.ingestion.object_pipeline import (
    DEFAULT_MODEL_NAME as DEFAULT_OBJECT_MODEL_NAME,
    YoloBackend,
    run_object_file,
)
from backend.app.services.ingestion.ocr_pipeline import EasyOcrBackend, run_ocr_file
from backend.app.services.metadata.metadata_store import FrameRecord, MetadataStore
from backend.app.services.retrieval.hybrid_search import (
    HybridSearchConfig,
    HybridSearchEngine,
)
from backend.app.services.retrieval.rerank import HybridReranker
from backend.app.services.retrieval.retrieval_config import (
    DEFAULT_RETRIEVAL_CONFIG_PATH,
    load_retrieval_runtime_config,
)
from backend.app.services.retrieval.search_asr import AsrSearchEngine
from backend.app.services.retrieval.search_caption import CaptionSearchEngine
from backend.app.services.retrieval.search_object import ObjectSearchEngine
from backend.app.services.retrieval.search_ocr import OcrSearchEngine
from backend.app.services.retrieval.text_index import TextIndexSearcher
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


DEFAULT_PUBLIC_ROOT = Path("data/public")
DEFAULT_OUTPUT_ROOT = Path("competition")
ANSWER_COUNT = 100
SUPPORTED_TASKS = {"TKIS", "VKIS"}


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
        if args.resume and metadata_path.exists() and report_path.exists():
            print(f"[{number}/{len(corpus)}] skip {video.filename}: artifacts already exist")
            continue
        print(f"[{number}/{len(corpus)}] extracting {video.filename}")
        extract_keyframes_for_video(
            video_path=video_path,
            output_dir=output_root / "keyframes",
            metadata_path=metadata_path,
            report_path=report_path,
            phash_threshold=args.phash_threshold,
            phash_window_sec=args.phash_window_sec,
            jpeg_quality=args.jpeg_quality,
            shot_threshold=args.shot_threshold,
            shot_device=args.device,
            short_shot_max_sec=args.short_shot_max_sec,
            regular_shot_max_sec=args.regular_shot_max_sec,
            long_shot_interval_sec=args.long_shot_interval_sec,
        )


def embed_command(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.public_root)
    output_root = args.output_root
    resolved_device = choose_device(args.device)
    model, processor = load_siglip2_model_processor(
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=resolved_device,
        model_cache_dir=args.model_cache_dir,
    )

    for number, video in enumerate(corpus, start=1):
        metadata_path = output_root / "metadata" / f"keyframes_{video.video_id}.jsonl"
        embeddings_path = output_root / "embeddings" / f"{ARTIFACT_TAG}_{video.video_id}.npy"
        embedding_metadata_path = (
            output_root / "metadata" / f"{ARTIFACT_TAG}_embeddings_{video.video_id}.jsonl"
        )
        if args.resume and embeddings_path.exists() and embedding_metadata_path.exists():
            print(f"[{number}/{len(corpus)}] skip {video.filename}: embeddings already exist")
            continue
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Keyframe metadata missing for {video.filename}: {metadata_path}"
            )

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
        artifact_report["status"] = "passed"
        write_json(
            artifact_report,
            output_root
            / "metadata"
            / f"{ARTIFACT_TAG}_artifacts_{video.video_id}_validation.json",
        )


def index_command(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.public_root)
    sources: list[tuple[Path, Path, str]] = []
    for video in corpus:
        embeddings_path = (
            args.output_root / "embeddings" / f"{ARTIFACT_TAG}_{video.video_id}.npy"
        )
        metadata_path = (
            args.output_root
            / "metadata"
            / f"{ARTIFACT_TAG}_embeddings_{video.video_id}.jsonl"
        )
        if not embeddings_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Embedding artifacts missing for {video.filename}: "
                f"{embeddings_path}, {metadata_path}"
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
    print(f"FAISS index: {paths['index']} vectors={result['index'].ntotal}")


def competition_index_paths(output_root: Path) -> dict[str, Path]:
    return {
        "index": output_root / "indexes" / f"{ARTIFACT_TAG}_flat_ip.faiss",
        "index_metadata": output_root / "metadata" / f"{ARTIFACT_TAG}_faiss_metadata.jsonl",
        "frame_map": output_root / "metadata" / f"{ARTIFACT_TAG}_frame_map.json",
        "manifest": output_root / "metadata" / f"{ARTIFACT_TAG}_faiss_manifest.json",
        "report": output_root / "metadata" / f"{ARTIFACT_TAG}_index_report.json",
        "neighbors": output_root / "metadata" / "neighbors_all.jsonl",
        "segments": output_root / "metadata" / "segments_all.jsonl",
        "text_index": output_root / "indexes" / "retrieval_text_index.json",
    }


def _keyframe_metadata_path(output_root: Path, video: CorpusVideo) -> Path:
    return output_root / "metadata" / f"keyframes_{video.video_id}.jsonl"


def enrich_command(args: argparse.Namespace) -> None:
    """Run the four existing multimodal ingestion pipelines over the corpus."""
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
        print("[caption] loading one shared BLIP backend for the corpus")
        backend = BlipCaptionBackend(
            model_name=args.caption_model_name,
            device=resolved_device,
            cache_dir=args.model_cache_root / "caption",
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
            )

    if "ocr" in args.modalities:
        print("[ocr] loading one shared EasyOCR backend for the corpus")
        backend = EasyOcrBackend(device=resolved_device, languages=("vi", "en"))
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
            )

    if "objects" in args.modalities:
        print("[objects] loading one shared YOLO backend for the corpus")
        object_cache = args.model_cache_root / "objects"
        backend = YoloBackend(
            model_name=args.object_model_name,
            device=resolved_device,
            conf_threshold=args.object_conf_threshold,
            iou_threshold=args.object_iou_threshold,
            cache_dir=object_cache,
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
                model_cache_dir=object_cache,
            )

    if "asr" in args.modalities:
        print("[asr] loading one shared Whisper backend for videos with audio")
        backend = WhisperBackend(
            model_size=args.asr_model_size,
            device=resolved_device,
            backend=args.asr_backend,
            cache_dir=args.model_cache_root / "asr",
            vad_filter=not args.no_vad,
        )
        for number, video in enumerate(corpus, start=1):
            print(f"[asr {number}/{len(corpus)}] {video.filename}")
            run_asr_file(
                args.public_root / video.relative_path,
                metadata_path=_keyframe_metadata_path(args.output_root, video),
                output_dir=metadata_dir,
                device=resolved_device,
                model_size=args.asr_model_size,
                backend_name=args.asr_backend,
                overwrite=args.overwrite,
                vad_filter=not args.no_vad,
                backend=backend,
            )


def neighbors_command(args: argparse.Namespace) -> None:
    paths = competition_index_paths(args.output_root)
    result = build_neighbor_index(
        args.output_root / "metadata",
        paths["neighbors"],
        window_seconds=args.window_seconds,
        fps=25.0,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _require_multimodal_artifacts(
    corpus: Sequence[CorpusVideo],
    metadata_dir: Path,
) -> None:
    missing: list[Path] = []
    prefixes = ("captions", "ocr", "objects", "asr")
    for video in corpus:
        for prefix in prefixes:
            path = metadata_dir / f"{prefix}_{video.video_id}.jsonl"
            if not path.exists():
                missing.append(path)
    if missing:
        preview = ", ".join(path.as_posix() for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} multimodal artifacts; run enrich first. "
            f"First entries: {preview}"
        )


def segments_command(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.public_root)
    metadata_dir = args.output_root / "metadata"
    _require_multimodal_artifacts(corpus, metadata_dir)
    paths = competition_index_paths(args.output_root)
    result = build_segment_metadata(
        metadata_dir,
        paths["segments"],
        captions_path=metadata_dir,
        ocr_path=metadata_dir,
        asr_path=metadata_dir,
        objects_path=metadata_dir,
        strategy=args.strategy,
        fixed_duration_seconds=args.fixed_duration_seconds,
        fps=25.0,
        caption_similarity_threshold=args.caption_similarity_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def text_index_command(args: argparse.Namespace) -> None:
    paths = competition_index_paths(args.output_root)
    if not paths["segments"].exists():
        raise FileNotFoundError(
            f"Segment metadata not found: {paths['segments']}; run segments first"
        )
    summary = write_text_index(
        load_text_records(paths["segments"]),
        paths["text_index"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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


def build_competition_hybrid_engine(
    visual_engine: VisualSearchEngine,
    *,
    text_index_path: Path,
    retrieval_config_path: Path,
    search_depth: int,
) -> HybridSearchEngine:
    """Build the same visual+caption+OCR+ASR+object retrieval stack as the repo."""
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
        "asr": AsrSearchEngine(
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
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        for question in questions:
            answers = predictions.get(question.query_id)
            if answers is None or len(answers) != ANSWER_COUNT:
                raise ValueError(f"Missing 100 answers for {question.query_id}")
            writer.writerow([question.query_id, *answers])


def predict_command(args: argparse.Namespace) -> None:
    public_root = args.public_root
    output_root = args.output_root
    submission_path = (
        args.submission_path
        if args.submission_path is not None
        else output_root / "results" / "submission.csv"
    )
    corpus = load_corpus(public_root)
    questions = load_questions(public_root)
    columns = submission_columns(public_root)
    paths = competition_index_paths(output_root)
    runtime = load_retrieval_runtime_config(args.retrieval_config)

    contract = load_encoder_contract(paths["manifest"])
    resolved_device = choose_device(args.device)
    model, processor = load_siglip2_model_processor(
        model_name=contract.model_name,
        model_revision=contract.model_revision,
        device=resolved_device,
        model_cache_dir=args.model_cache_dir,
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

    predictions: dict[str, list[str]] = {}
    for number, question in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] {question.query_id} {question.task}")
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

    write_submission(
        submission_path,
        columns=columns,
        questions=questions,
        predictions=predictions,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_input_parser = subparsers.add_parser(
        "validate-input",
        description="Validate the public CSV/file contract without processing videos.",
    )
    validate_input_parser.add_argument(
        "--public-root",
        type=Path,
        default=DEFAULT_PUBLIC_ROOT,
    )

    extract_parser = subparsers.add_parser(
        "extract", description="Extract competition keyframes with the existing extractor."
    )
    _add_common_paths(extract_parser)
    extract_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    extract_parser.add_argument(
        "--phash-threshold", type=int, default=DEFAULT_PHASH_THRESHOLD
    )
    extract_parser.add_argument(
        "--phash-window-sec",
        type=float,
        default=DEFAULT_DEDUP_TEMPORAL_WINDOW_SEC,
    )
    extract_parser.add_argument("--jpeg-quality", type=int, default=95)
    extract_parser.add_argument("--shot-threshold", type=float, default=0.5)
    extract_parser.add_argument(
        "--short-shot-max-sec", type=float, default=DEFAULT_SHORT_SHOT_MAX_SEC
    )
    extract_parser.add_argument(
        "--regular-shot-max-sec", type=float, default=DEFAULT_REGULAR_SHOT_MAX_SEC
    )
    extract_parser.add_argument(
        "--long-shot-interval-sec", type=float, default=DEFAULT_LONG_SHOT_INTERVAL_SEC
    )
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
        description="Run caption, OCR, object, and ASR ingestion from the original system.",
    )
    _add_common_paths(enrich_parser)
    enrich_parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("caption", "ocr", "objects", "asr"),
        default=["caption", "ocr", "objects", "asr"],
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
    enrich_parser.add_argument("--caption-batch-size", type=int, default=4)
    enrich_parser.add_argument("--no-segment-caption", action="store_true")
    enrich_parser.add_argument("--ocr-batch-size", type=int, default=4)
    enrich_parser.add_argument("--ocr-conf-threshold", type=float, default=0.3)
    enrich_parser.add_argument(
        "--object-model-name",
        default=DEFAULT_OBJECT_MODEL_NAME,
    )
    enrich_parser.add_argument("--object-batch-size", type=int, default=8)
    enrich_parser.add_argument("--object-conf-threshold", type=float, default=0.25)
    enrich_parser.add_argument("--object-iou-threshold", type=float, default=0.7)
    enrich_parser.add_argument("--asr-model-size", default=DEFAULT_ASR_MODEL_SIZE)
    enrich_parser.add_argument(
        "--asr-backend",
        choices=("auto", "faster-whisper", "whisper"),
        default="auto",
    )
    enrich_parser.add_argument("--no-vad", action="store_true")
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
        description="Build the original caption/OCR/ASR/object lexical index.",
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
    if args.command == "validate-input":
        print(json.dumps(validate_input(args.public_root), ensure_ascii=False, indent=2))
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
        predict_command(args)
    elif args.command == "validate-submission":
        report = validate_submission(args.submission_path, args.public_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
