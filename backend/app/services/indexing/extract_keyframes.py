from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .keyframe_candidates import (
        DEFAULT_BOUNDARY_GUARD_SEC,
        DEFAULT_INTERVAL_SEC,
        DEFAULT_TINY_SHOT_MAX_SEC,
        KeyframeCandidate as DenseKeyframeCandidate,
        generate_keyframe_candidates,
    )
    from .keyframe_selection import (
        DEFAULT_MAX_GAP_SECONDS,
        PHASE_COVERAGE,
        PHASE_PROTECTED,
        SelectedCandidate,
        SelectionCandidate,
        SelectionConfig,
        SelectionResult,
        select_keyframes,
    )
else:  # Preserve the documented direct-script CLI invocation.
    repository_root = Path(__file__).resolve().parents[4]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from backend.app.services.indexing.keyframe_candidates import (
        DEFAULT_BOUNDARY_GUARD_SEC,
        DEFAULT_INTERVAL_SEC,
        DEFAULT_TINY_SHOT_MAX_SEC,
        KeyframeCandidate as DenseKeyframeCandidate,
        generate_keyframe_candidates,
    )
    from backend.app.services.indexing.keyframe_selection import (
        DEFAULT_MAX_GAP_SECONDS,
        PHASE_COVERAGE,
        PHASE_PROTECTED,
        SelectedCandidate,
        SelectionCandidate,
        SelectionConfig,
        SelectionResult,
        select_keyframes,
    )


KEYFRAME_STRATEGY_LEGACY = "legacy"
KEYFRAME_STRATEGY_DENSE_COVERAGE = "dense_coverage"
KEYFRAME_STRATEGIES = (
    KEYFRAME_STRATEGY_LEGACY,
    KEYFRAME_STRATEGY_DENSE_COVERAGE,
)


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class Shot:
    shot_index: int
    start_frame: int
    end_frame: int
    fps: float

    @property
    def start_sec(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_sec(self) -> float:
        return (self.end_frame + 1) / self.fps

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class KeyframeCandidate:
    candidate_index: int
    shot: Shot
    frame_index: int
    timestamp: float
    selection_reason: str


class ClipDeduper:
    """Optional near-duplicate filter using OpenCLIP image embeddings."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str,
        similarity_threshold: float,
        temporal_window_sec: float,
    ) -> None:
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "open_clip, torch, and Pillow are required for --enable-clip-dedup."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._torch = torch
        self._image_cls = Image
        self._device = device
        self.similarity_threshold = similarity_threshold
        self.temporal_window_sec = temporal_window_sec
        self._kept: list[tuple[np.ndarray, str, float]] = []

    def encode(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._image_cls.fromarray(rgb)
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy()[0].astype("float32")

    def find_duplicate(
        self,
        embedding: np.ndarray,
        timestamp: float,
    ) -> tuple[str, float] | None:
        best_frame_id = ""
        best_similarity = -1.0
        for kept_embedding, kept_frame_id, kept_timestamp in self._kept:
            if abs(timestamp - kept_timestamp) > self.temporal_window_sec:
                continue
            similarity = float(np.dot(embedding, kept_embedding))
            if similarity > best_similarity:
                best_frame_id = kept_frame_id
                best_similarity = similarity
        if best_similarity >= self.similarity_threshold:
            return best_frame_id, best_similarity
        return None

    def add(self, embedding: np.ndarray, frame_id: str, timestamp: float) -> None:
        self._kept.append((embedding, frame_id, timestamp))


def read_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Cannot read FPS from video: {video_path}")
    if frame_count <= 0:
        raise ValueError(f"Cannot read frame count from video: {video_path}")
    return VideoInfo(video_id=video_path.stem, fps=fps, frame_count=frame_count)


def detect_shots_transnetv2(
    video_path: Path,
    info: VideoInfo,
    threshold: float = 0.5,
    device: str = "auto",
) -> tuple[list[Shot], str]:
    """Detect shots with TransNetV2.

    The preferred dependency is the PyPI package `transnetv2-pytorch`, which
    bundles weights and exposes `transnetv2_pytorch.TransNetV2`. The older
    TensorFlow-style wrapper is still supported if a teammate installed it.
    """
    try:
        from transnetv2_pytorch import TransNetV2  # type: ignore

        model = TransNetV2(device=device)
        scenes = model.detect_scenes(str(video_path), threshold=threshold)
        scene_ranges = [
            (int(scene["start_frame"]), int(scene["end_frame"]))
            for scene in scenes
        ]
        shots = scenes_to_shots(scene_ranges, info)
        return shots, "transnetv2_pytorch"
    except ImportError:
        pass

    try:
        from transnetv2 import TransNetV2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TransNetV2 is required for official shot detection. Install "
            "transnetv2-pytorch from requirements.txt. If the error mentions "
            "ffmpeg, install FFmpeg and add it to PATH."
        ) from exc

    model = TransNetV2()
    _, single_frame_predictions, _ = model.predict_video(str(video_path))
    scenes = model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
    shots = scenes_to_shots(scenes, info)
    return shots, "transnetv2"


def scenes_to_shots(scenes: object, info: VideoInfo) -> list[Shot]:
    """Normalize detector ranges to sorted, disjoint inclusive video frames.

    Some TransNetV2 wrappers emit a trailing ``(frame_count, frame_count)``
    scene or share a transition frame between adjacent scenes.  Clamping such
    ranges independently creates a fake overlapping final shot, so ranges
    fully outside the video are discarded and later overlaps are trimmed.
    """

    normalized_ranges: list[tuple[int, int]] = []
    last_frame = info.frame_count - 1
    for scene in scenes:
        raw_start = int(scene[0])
        raw_end = int(scene[1])
        if raw_end < 0 or raw_start > last_frame:
            continue
        start_frame = max(0, raw_start)
        end_frame = min(last_frame, raw_end)
        if end_frame < start_frame:
            continue
        normalized_ranges.append((start_frame, end_frame))
    normalized_ranges.sort()

    shots: list[Shot] = []
    previous_end = -1
    for raw_start, end_frame in normalized_ranges:
        start_frame = max(raw_start, previous_end + 1)
        if start_frame > end_frame:
            continue
        shots.append(
            Shot(
                shot_index=len(shots) + 1,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=info.fps,
            )
        )
        previous_end = end_frame
    if not shots:
        shots.append(Shot(shot_index=1, start_frame=0, end_frame=info.frame_count - 1, fps=info.fps))
    return shots


def select_frame_indices(shot: Shot, min_boundary_margin_sec: float = 0.2) -> list[tuple[int, str]]:
    duration = shot.duration
    if duration < 4.0:
        offsets = [(duration / 2.0, "midpoint_lt_4s")]
    elif duration <= 8.0:
        offsets = [
            (duration / 3.0, "two_frames_4_to_8s"),
            (2.0 * duration / 3.0, "two_frames_4_to_8s"),
        ]
    else:
        first = min(2.0, duration / 2.0)
        offsets = []
        current = first
        while current < duration:
            offsets.append((current, "every_4s_gt_8s"))
            current += 4.0
        if not offsets:
            offsets = [(duration / 2.0, "midpoint_gt_8s")]

    margin_frames = max(0, int(round(min_boundary_margin_sec * shot.fps)))
    lo = min(shot.end_frame, shot.start_frame + margin_frames)
    hi = max(lo, shot.end_frame - margin_frames)
    selected: list[tuple[int, str]] = []
    seen: set[int] = set()
    for offset, reason in offsets:
        frame_index = int(round((shot.start_sec + offset) * shot.fps))
        frame_index = max(lo, min(frame_index, hi))
        if frame_index not in seen:
            selected.append((frame_index, reason))
            seen.add(frame_index)
    return selected


def build_keyframe_candidates(shots: list[Shot], fps: float) -> list[KeyframeCandidate]:
    candidates: list[KeyframeCandidate] = []
    for shot in shots:
        for frame_index, reason in select_frame_indices(shot):
            candidates.append(
                KeyframeCandidate(
                    candidate_index=len(candidates) + 1,
                    shot=shot,
                    frame_index=frame_index,
                    timestamp=frame_index / fps,
                    selection_reason=reason,
                )
            )
    return candidates


def ffmpeg_quality_from_jpeg_quality(jpeg_quality: int) -> int:
    """Map OpenCV-style JPEG quality 1-100 to FFmpeg mjpeg qscale 31-2."""
    jpeg_quality = max(1, min(int(jpeg_quality), 100))
    return max(2, min(31, round(31 - ((jpeg_quality - 1) * 29 / 99))))


def format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def extract_frame_ffmpeg(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    jpeg_quality: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        str(ffmpeg_quality_from_jpeg_quality(jpeg_quality)),
        str(output_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg executable was not found\n"
            f"command: {format_command(command)}\n"
            "stderr: <ffmpeg.exe is not available in PATH>"
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            "FFmpeg frame extraction failed\n"
            f"command: {format_command(command)}\n"
            f"stderr: {stderr or '<empty>'}"
        )


def load_extracted_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise ValueError(f"Extracted frame is missing: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Extracted frame is empty: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Extracted frame is not readable as an image: {path}")
    return frame


def remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def perceptual_hash(frame_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    low_freq = dct[:8, :8]
    median = float(np.median(low_freq[1:, 1:]))
    bits = low_freq > median
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _dense_frame_id(candidate_id: str) -> str:
    prefix = "CANDIDATE_"
    if not candidate_id.startswith(prefix):
        raise ValueError(f"Unexpected dense candidate_id: {candidate_id}")
    return f"FRAME_{candidate_id[len(prefix):]}"


def _dense_candidate_report_record(
    candidate: DenseKeyframeCandidate,
    *,
    candidate_index: int,
    selected: SelectedCandidate | None,
    failed: bool,
) -> dict:
    if failed:
        extraction_status = "failed"
    elif selected is not None:
        extraction_status = "selected"
    else:
        extraction_status = "not_selected"
    return {
        "candidate_index": candidate_index,
        "candidate_id": candidate.candidate_id,
        "shot_index": candidate.shot_index,
        "shot_id": f"SHOT_{candidate.video_id}_{candidate.shot_index:06d}",
        "frame_index": candidate.frame_index,
        "timestamp": round(candidate.timestamp_sec, 3),
        "selection_reason": candidate.reasons[0],
        "candidate_reasons": list(candidate.reasons),
        "selected": selected is not None,
        "extraction_status": extraction_status,
        "selection_phase": selected.selection_phase if selected is not None else None,
        "selection_rank": selected.selection_rank if selected is not None else None,
        "selection_reasons": (
            list(selected.selection_reasons) if selected is not None else []
        ),
        "covered_event_ids": (
            list(selected.covered_event_ids) if selected is not None else []
        ),
    }


def _extract_dense_coverage_for_video(
    *,
    video_path: Path,
    info: VideoInfo,
    shots: list[Shot],
    detector_name: str,
    output_dir: Path,
    metadata_path: Path,
    report_path: Path,
    phash_threshold: int,
    phash_window_sec: float,
    jpeg_quality: int,
    shot_threshold: float,
    shot_device: str,
    enable_clip_dedup: bool,
    clip_similarity_threshold: float,
    clip_window_sec: float,
    clip_model_name: str,
    clip_pretrained: str,
    clip_device: str,
    candidate_interval_sec: float,
    boundary_guard_sec: float,
    tiny_shot_max_sec: float,
    max_gap_seconds: float,
    gap_tolerance_seconds: float,
    target_keyframes: int | None,
    hard_max_keyframes: int | None,
) -> dict:
    video_output_dir = output_dir / info.video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_keyframe_candidates(
        info.video_id,
        shots,
        info.fps,
        interval_sec=candidate_interval_sec,
        boundary_guard_sec=boundary_guard_sec,
        tiny_shot_max_sec=tiny_shot_max_sec,
        frame_count=info.frame_count,
    )
    generated_by_id = {candidate.candidate_id: candidate for candidate in generated}
    candidate_index_by_id = {
        candidate.candidate_id: index
        for index, candidate in enumerate(generated, start=1)
    }
    shot_by_index = {shot.shot_index: shot for shot in shots}
    selection_config = SelectionConfig(
        max_gap_seconds=max_gap_seconds,
        target_keyframes=target_keyframes,
        hard_max_keyframes=hard_max_keyframes,
        gap_tolerance_seconds=gap_tolerance_seconds,
        protect_each_shot=True,
    )

    failed_ids: set[str] = set()
    extracted_paths: dict[str, Path] = {}
    skipped: list[dict] = []
    selection_result: SelectionResult
    while True:
        selectable = tuple(
            SelectionCandidate.from_generated_candidate(candidate)
            for candidate in generated
            if candidate.candidate_id not in failed_ids
        )
        selection_result = select_keyframes(
            selectable,
            (),
            video_duration=info.duration,
            config=selection_config,
        )
        newly_failed = False
        for selected in selection_result.selected:
            candidate = generated_by_id[selected.candidate.candidate_id]
            if candidate.candidate_id in extracted_paths:
                continue
            frame_id = _dense_frame_id(candidate.candidate_id)
            output_path = video_output_dir / f"{frame_id}.jpg"
            try:
                extract_frame_ffmpeg(
                    video_path=video_path,
                    timestamp=candidate.timestamp_sec,
                    output_path=output_path,
                    jpeg_quality=jpeg_quality,
                )
                load_extracted_image(output_path)
            except Exception as exc:  # noqa: BLE001 - blacklist and reselect another candidate.
                remove_file_if_exists(output_path)
                failed_ids.add(candidate.candidate_id)
                skipped.append(
                    {
                        "frame_id": frame_id,
                        "candidate_id": candidate.candidate_id,
                        "video_id": info.video_id,
                        "shot_index": candidate.shot_index,
                        "frame_index": candidate.frame_index,
                        "timestamp": round(candidate.timestamp_sec, 3),
                        "selection_phase": selected.selection_phase,
                        "selection_rank": selected.selection_rank,
                        "reason": "ffmpeg_extract_failed",
                        "error": str(exc),
                        "output_path": output_path.as_posix(),
                    }
                )
                newly_failed = True
                continue
            extracted_paths[candidate.candidate_id] = output_path
        if not newly_failed:
            break

    final_selected_by_id = {
        selected.candidate.candidate_id: selected
        for selected in selection_result.selected
    }
    deselected_cached_count = 0
    for candidate_id, output_path in extracted_paths.items():
        if candidate_id not in final_selected_by_id:
            remove_file_if_exists(output_path)
            deselected_cached_count += 1

    kept_phashes: list[tuple[int, str, float]] = []
    duplicate_retained: list[dict] = []
    clip_deduper = (
        ClipDeduper(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=clip_device,
            similarity_threshold=clip_similarity_threshold,
            temporal_window_sec=clip_window_sec,
        )
        if enable_clip_dedup and selection_result.selected
        else None
    )
    records: list[dict] = []
    for selected in selection_result.selected:
        candidate = generated_by_id[selected.candidate.candidate_id]
        shot = shot_by_index[candidate.shot_index]
        frame_id = _dense_frame_id(candidate.candidate_id)
        output_path = extracted_paths[candidate.candidate_id]
        frame = load_extracted_image(output_path)
        phash = perceptual_hash(frame)
        timestamp = candidate.timestamp_sec
        phash_duplicate = next(
            (
                (existing_hash, existing_frame_id)
                for existing_hash, existing_frame_id, existing_timestamp in kept_phashes
                if abs(timestamp - existing_timestamp) <= phash_window_sec
                if hamming_distance(phash, existing_hash) <= phash_threshold
            ),
            None,
        )
        duplicate_fields: dict[str, object] = {}
        if phash_duplicate is not None:
            distance = hamming_distance(phash, phash_duplicate[0])
            duplicate_fields.update(
                {
                    "phash_duplicate_of": phash_duplicate[1],
                    "phash_hamming_distance": distance,
                }
            )
            duplicate_retained.append(
                {
                    "frame_id": frame_id,
                    "candidate_id": candidate.candidate_id,
                    "reason": "phash_duplicate_retained",
                    "duplicate_of": phash_duplicate[1],
                    "hamming_distance": distance,
                }
            )
        kept_phashes.append((phash, frame_id, timestamp))

        clip_embedding = None
        if clip_deduper is not None:
            clip_embedding = clip_deduper.encode(frame)
            clip_duplicate = clip_deduper.find_duplicate(clip_embedding, timestamp)
            if clip_duplicate is not None:
                duplicate_frame_id, similarity = clip_duplicate
                duplicate_fields.update(
                    {
                        "clip_duplicate_of": duplicate_frame_id,
                        "clip_similarity": round(similarity, 6),
                    }
                )
                duplicate_retained.append(
                    {
                        "frame_id": frame_id,
                        "candidate_id": candidate.candidate_id,
                        "reason": "clip_duplicate_retained",
                        "duplicate_of": duplicate_frame_id,
                        "clip_similarity": round(similarity, 6),
                    }
                )
            clip_deduper.add(clip_embedding, frame_id, timestamp)

        normalized_path = output_path.as_posix()
        records.append(
            {
                "frame_id": frame_id,
                "video_id": info.video_id,
                "shot_id": f"SHOT_{info.video_id}_{shot.shot_index:06d}",
                "segment_id": f"SHOT_{info.video_id}_{shot.shot_index:06d}",
                "shot_index": shot.shot_index,
                "candidate_index": candidate_index_by_id[candidate.candidate_id],
                "candidate_id": candidate.candidate_id,
                "shot_start": round(candidate.shot_start_sec, 3),
                "shot_end": round(candidate.shot_end_sec, 3),
                "shot_duration": round(candidate.shot_end_sec - candidate.shot_start_sec, 3),
                "timestamp": round(timestamp, 3),
                "timestamp_source": "video_fps",
                "timestamp_confidence": 1.0,
                "frame_index": candidate.frame_index,
                "keyframe_path": normalized_path,
                "frame_path": normalized_path,
                "thumbnail_path": normalized_path,
                "source_video_path": video_path.as_posix(),
                "video_path": video_path.as_posix(),
                "selection_reason": candidate.reasons[0],
                "candidate_reasons": list(candidate.reasons),
                "selection_phase": selected.selection_phase,
                "selection_rank": selected.selection_rank,
                "selection_reasons": list(selected.selection_reasons),
                "covered_event_ids": list(selected.covered_event_ids),
                "selection_score": selected.selection_score,
                "protected": selected.selection_phase == PHASE_PROTECTED,
                "coverage_added": selected.selection_phase == PHASE_COVERAGE,
                "keyframe_strategy": KEYFRAME_STRATEGY_DENSE_COVERAGE,
                "phash": f"{phash:016x}",
                "shot_detector": detector_name,
                **duplicate_fields,
            }
        )

    selected_by_id = {
        selected.candidate.candidate_id: selected
        for selected in selection_result.selected
    }
    report = {
        "video_id": info.video_id,
        "video_path": video_path.as_posix(),
        "fps": info.fps,
        "frame_count": info.frame_count,
        "duration": round(info.duration, 3),
        "shot_detector": detector_name,
        "shot_threshold": shot_threshold,
        "shot_device": shot_device,
        "shot_count": len(shots),
        "keyframe_count": len(records),
        "skipped_count": len(skipped),
        "phash_threshold": phash_threshold,
        "phash_window_sec": phash_window_sec,
        "jpeg_quality": jpeg_quality,
        "frame_extractor": "ffmpeg",
        "clip_dedup_enabled": enable_clip_dedup,
        "clip_similarity_threshold": clip_similarity_threshold if enable_clip_dedup else None,
        "clip_window_sec": clip_window_sec if enable_clip_dedup else None,
        "metadata_path": metadata_path.as_posix(),
        "output_dir": video_output_dir.as_posix(),
        "keyframe_strategy": KEYFRAME_STRATEGY_DENSE_COVERAGE,
        "status": selection_result.status,
        "constraints_satisfied": selection_result.constraints_satisfied,
        "coverage_satisfied": selection_result.coverage_satisfied,
        "candidate_count": len(generated),
        "planned_selected_count": len(selection_result.selected),
        "candidate_interval_sec": candidate_interval_sec,
        "boundary_guard_sec": boundary_guard_sec,
        "tiny_shot_max_sec": tiny_shot_max_sec,
        "selection_config": {
            "max_gap_seconds": max_gap_seconds,
            "gap_tolerance_seconds": gap_tolerance_seconds,
            "target_keyframes": target_keyframes,
            "hard_max_keyframes": hard_max_keyframes,
            "protect_each_shot": True,
        },
        "selection": selection_result.to_report(),
        "deduplication_mode": "annotate_only",
        "duplicate_retained_count": len(duplicate_retained),
        "duplicate_retained": duplicate_retained,
        "deselected_cached_count": deselected_cached_count,
        "skipped": skipped,
        "candidates": [
            _dense_candidate_report_record(
                candidate,
                candidate_index=candidate_index_by_id[candidate.candidate_id],
                selected=selected_by_id.get(candidate.candidate_id),
                failed=candidate.candidate_id in failed_ids,
            )
            for candidate in generated
        ],
        "shots": [
            {
                **asdict(shot),
                "start_sec": round(shot.start_sec, 3),
                "end_sec": round(shot.end_sec, 3),
                "duration": round(shot.duration, 3),
                "selected_frame_count": sum(
                    1 for record in records if record["shot_index"] == shot.shot_index
                ),
            }
            for shot in shots
        ],
    }
    write_jsonl(records, metadata_path)
    write_json(report, report_path)
    return report


def extract_keyframes_for_video(
    video_path: Path,
    output_dir: Path,
    metadata_path: Path,
    report_path: Path,
    phash_threshold: int = 6,
    phash_window_sec: float = 12.0,
    jpeg_quality: int = 95,
    shot_threshold: float = 0.5,
    shot_device: str = "auto",
    enable_clip_dedup: bool = False,
    clip_similarity_threshold: float = 0.985,
    clip_window_sec: float = 12.0,
    clip_model_name: str = "ViT-B-16",
    clip_pretrained: str = "laion2b_s34b_b88k",
    clip_device: str = "auto",
    strategy: str = KEYFRAME_STRATEGY_LEGACY,
    candidate_interval_sec: float = DEFAULT_INTERVAL_SEC,
    boundary_guard_sec: float = DEFAULT_BOUNDARY_GUARD_SEC,
    tiny_shot_max_sec: float = DEFAULT_TINY_SHOT_MAX_SEC,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    gap_tolerance_seconds: float = 0.0,
    target_keyframes: int | None = None,
    hard_max_keyframes: int | None = None,
) -> dict:
    if strategy not in KEYFRAME_STRATEGIES:
        raise ValueError(
            f"Unknown keyframe strategy {strategy!r}; expected one of {KEYFRAME_STRATEGIES}"
        )
    info = read_video_info(video_path)
    try:
        shots, detector_name = detect_shots_transnetv2(
            video_path,
            info,
            threshold=shot_threshold,
            device=shot_device,
        )
    except Exception as exc:  # noqa: BLE001 - normalize dependency/runtime errors.
        raise RuntimeError(
            "TransNetV2 shot detection failed. Install transnetv2-pytorch from "
            "requirements.txt and make sure ffmpeg.exe is available in PATH."
        ) from exc

    if strategy == KEYFRAME_STRATEGY_DENSE_COVERAGE:
        return _extract_dense_coverage_for_video(
            video_path=video_path,
            info=info,
            shots=shots,
            detector_name=detector_name,
            output_dir=output_dir,
            metadata_path=metadata_path,
            report_path=report_path,
            phash_threshold=phash_threshold,
            phash_window_sec=phash_window_sec,
            jpeg_quality=jpeg_quality,
            shot_threshold=shot_threshold,
            shot_device=shot_device,
            enable_clip_dedup=enable_clip_dedup,
            clip_similarity_threshold=clip_similarity_threshold,
            clip_window_sec=clip_window_sec,
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
            clip_device=clip_device,
            candidate_interval_sec=candidate_interval_sec,
            boundary_guard_sec=boundary_guard_sec,
            tiny_shot_max_sec=tiny_shot_max_sec,
            max_gap_seconds=max_gap_seconds,
            gap_tolerance_seconds=gap_tolerance_seconds,
            target_keyframes=target_keyframes,
            hard_max_keyframes=hard_max_keyframes,
        )

    video_output_dir = output_dir / info.video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_keyframe_candidates(shots, info.fps)
    kept_phashes: list[tuple[int, str, float]] = []
    records: list[dict] = []
    skipped: list[dict] = []
    clip_deduper = (
        ClipDeduper(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=clip_device,
            similarity_threshold=clip_similarity_threshold,
            temporal_window_sec=clip_window_sec,
        )
        if enable_clip_dedup
        else None
    )

    for candidate in candidates:
        shot = candidate.shot
        frame_index = candidate.frame_index
        timestamp = candidate.timestamp
        reason = candidate.selection_reason
        frame_id = f"FRAME_{info.video_id}_{candidate.candidate_index:06d}"
        shot_id = f"SHOT_{info.video_id}_{shot.shot_index:06d}"
        output_path = video_output_dir / f"{frame_id}.jpg"

        try:
            extract_frame_ffmpeg(
                video_path=video_path,
                timestamp=timestamp,
                output_path=output_path,
                jpeg_quality=jpeg_quality,
            )
            frame = load_extracted_image(output_path)
        except Exception as exc:  # noqa: BLE001 - report all extraction failures and continue.
            remove_file_if_exists(output_path)
            skipped.append(
                {
                    "frame_id": frame_id,
                    "video_id": info.video_id,
                    "shot_index": shot.shot_index,
                    "frame_index": frame_index,
                    "timestamp": round(timestamp, 3),
                    "reason": "ffmpeg_extract_failed",
                    "error": str(exc),
                    "output_path": output_path.as_posix(),
                }
            )
            continue

        phash = perceptual_hash(frame)
        duplicate = next(
            (
                (existing_hash, existing_frame_id)
                for existing_hash, existing_frame_id, existing_timestamp in kept_phashes
                if abs(timestamp - existing_timestamp) <= phash_window_sec
                if hamming_distance(phash, existing_hash) <= phash_threshold
            ),
            None,
        )
        if duplicate is not None:
            remove_file_if_exists(output_path)
            skipped.append(
                {
                    "frame_id": frame_id,
                    "video_id": info.video_id,
                    "shot_index": shot.shot_index,
                    "frame_index": frame_index,
                    "timestamp": round(timestamp, 3),
                    "reason": "phash_duplicate",
                    "duplicate_of": duplicate[1],
                    "hamming_distance": hamming_distance(phash, duplicate[0]),
                }
            )
            continue

        clip_embedding = None
        if clip_deduper is not None:
            clip_embedding = clip_deduper.encode(frame)
            clip_duplicate = clip_deduper.find_duplicate(clip_embedding, timestamp)
            if clip_duplicate is not None:
                duplicate_frame_id, similarity = clip_duplicate
                remove_file_if_exists(output_path)
                skipped.append(
                    {
                        "frame_id": frame_id,
                        "video_id": info.video_id,
                        "shot_index": shot.shot_index,
                        "frame_index": frame_index,
                        "timestamp": round(timestamp, 3),
                        "reason": "clip_duplicate",
                        "duplicate_of": duplicate_frame_id,
                        "clip_similarity": round(similarity, 6),
                    }
                )
                continue

        kept_phashes.append((phash, frame_id, timestamp))
        if clip_deduper is not None and clip_embedding is not None:
            clip_deduper.add(clip_embedding, frame_id, timestamp)
        normalized_path = output_path.as_posix()
        records.append(
            {
                "frame_id": frame_id,
                "video_id": info.video_id,
                "shot_id": shot_id,
                "segment_id": shot_id,
                "shot_index": shot.shot_index,
                "candidate_index": candidate.candidate_index,
                "shot_start": round(shot.start_sec, 3),
                "shot_end": round(shot.end_sec, 3),
                "shot_duration": round(shot.duration, 3),
                "timestamp": round(timestamp, 3),
                "timestamp_source": "video_fps",
                "timestamp_confidence": 1.0,
                "frame_index": frame_index,
                "keyframe_path": normalized_path,
                "frame_path": normalized_path,
                "thumbnail_path": normalized_path,
                "source_video_path": video_path.as_posix(),
                "video_path": video_path.as_posix(),
                "selection_reason": reason,
                "phash": f"{phash:016x}",
                "shot_detector": detector_name,
            }
        )

    report = {
        "video_id": info.video_id,
        "video_path": video_path.as_posix(),
        "keyframe_strategy": KEYFRAME_STRATEGY_LEGACY,
        "fps": info.fps,
        "frame_count": info.frame_count,
        "duration": round(info.duration, 3),
        "shot_detector": detector_name,
        "shot_threshold": shot_threshold,
        "shot_device": shot_device,
        "shot_count": len(shots),
        "keyframe_count": len(records),
        "skipped_count": len(skipped),
        "phash_threshold": phash_threshold,
        "phash_window_sec": phash_window_sec,
        "jpeg_quality": jpeg_quality,
        "frame_extractor": "ffmpeg",
        "clip_dedup_enabled": enable_clip_dedup,
        "clip_similarity_threshold": clip_similarity_threshold if enable_clip_dedup else None,
        "clip_window_sec": clip_window_sec if enable_clip_dedup else None,
        "metadata_path": metadata_path.as_posix(),
        "output_dir": video_output_dir.as_posix(),
        "skipped": skipped,
        "candidates": [
            {
                "candidate_index": candidate.candidate_index,
                "shot_index": candidate.shot.shot_index,
                "shot_id": f"SHOT_{info.video_id}_{candidate.shot.shot_index:06d}",
                "frame_index": candidate.frame_index,
                "timestamp": round(candidate.timestamp, 3),
                "selection_reason": candidate.selection_reason,
            }
            for candidate in candidates
        ],
        "shots": [
            {
                **asdict(shot),
                "start_sec": round(shot.start_sec, 3),
                "end_sec": round(shot.end_sec, 3),
                "duration": round(shot.duration, 3),
                "selected_frame_count": sum(
                    1 for record in records if record["shot_index"] == shot.shot_index
                ),
            }
            for shot in shots
        ],
    }
    write_jsonl(records, metadata_path)
    write_json(report, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract TransNetV2 shot-aware keyframes with legacy dedup or "
            "dense hard-constraint coverage."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-path", type=Path)
    source.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-glob", default="*.mp4")
    parser.add_argument("--output-dir", type=Path, default=Path("data/keyframes"))
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--phash-window-sec", type=float, default=12.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--shot-threshold", type=float, default=0.5)
    parser.add_argument("--shot-device", default="auto")
    parser.add_argument("--enable-clip-dedup", action="store_true")
    parser.add_argument("--clip-similarity-threshold", type=float, default=0.985)
    parser.add_argument("--clip-window-sec", type=float, default=12.0)
    parser.add_argument("--clip-model-name", default="ViT-B-16")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument(
        "--strategy",
        "--keyframe-strategy",
        dest="strategy",
        choices=KEYFRAME_STRATEGIES,
        default=KEYFRAME_STRATEGY_LEGACY,
    )
    parser.add_argument("--candidate-interval-sec", type=float, default=DEFAULT_INTERVAL_SEC)
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
    parser.add_argument("--hard-max-keyframes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video_path is not None:
        video_paths = [args.video_path]
    else:
        if not args.video_dir.exists():
            raise SystemExit(f"Video directory does not exist: {args.video_dir}")
        video_paths = sorted(path for path in args.video_dir.glob(args.video_glob) if path.is_file())
        if not video_paths:
            raise SystemExit(f"No videos matched {args.video_glob!r} in {args.video_dir}")
        if args.metadata_path is not None or args.report_path is not None:
            raise SystemExit("--metadata-path/--report-path are only supported with --video-path")

    total_keyframes = 0
    total_shots = 0
    for video_path in video_paths:
        if not video_path.exists():
            raise SystemExit(f"Video does not exist: {video_path}")
        video_id = video_path.stem
        metadata_path = args.metadata_path or Path("data/metadata") / f"keyframes_{video_id}.jsonl"
        report_path = args.report_path or Path("data/metadata") / f"keyframes_{video_id}_extract_report.json"
        report = extract_keyframes_for_video(
            video_path=video_path,
            output_dir=args.output_dir,
            metadata_path=metadata_path,
            report_path=report_path,
            phash_threshold=args.phash_threshold,
            phash_window_sec=args.phash_window_sec,
            jpeg_quality=args.jpeg_quality,
            shot_threshold=args.shot_threshold,
            shot_device=args.shot_device,
            enable_clip_dedup=args.enable_clip_dedup,
            clip_similarity_threshold=args.clip_similarity_threshold,
            clip_window_sec=args.clip_window_sec,
            clip_model_name=args.clip_model_name,
            clip_pretrained=args.clip_pretrained,
            clip_device=args.clip_device,
            strategy=args.strategy,
            candidate_interval_sec=args.candidate_interval_sec,
            boundary_guard_sec=args.boundary_guard_sec,
            tiny_shot_max_sec=args.tiny_shot_max_sec,
            max_gap_seconds=args.max_gap_seconds,
            gap_tolerance_seconds=args.gap_tolerance_seconds,
            target_keyframes=args.target_keyframes,
            hard_max_keyframes=args.hard_max_keyframes,
        )
        total_keyframes += int(report["keyframe_count"])
        total_shots += int(report["shot_count"])
        print(
            f"{video_id}: {report['keyframe_count']} keyframes from "
            f"{report['shot_count']} shots ({report['shot_detector']}); "
            f"skipped={report['skipped_count']}"
        )
        print(f"  metadata: {metadata_path}")
        print(f"  report: {report_path}")

    if len(video_paths) > 1:
        print(
            f"Done: {len(video_paths)} videos, {total_shots} shots, "
            f"{total_keyframes} kept keyframes"
        )


if __name__ == "__main__":
    main()
