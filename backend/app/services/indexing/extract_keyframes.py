from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


KEYFRAME_STRATEGY = "adaptive_shot_sampling_v3"
DEFAULT_SHORT_SHOT_MAX_SEC = 1.0
DEFAULT_REGULAR_SHOT_MAX_SEC = 4.0
DEFAULT_LONG_SHOT_INTERVAL_SEC = 2.0
DEFAULT_PHASH_THRESHOLD = 6
DEFAULT_DEDUP_TEMPORAL_WINDOW_SEC = 2.0
DEFAULT_CLIP_SIMILARITY_THRESHOLD = 0.985


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


class PerceptualHashDeduper:
    """Conservative pHash dedup restricted to nearby frames in one shot."""

    def __init__(self, hamming_threshold: int, temporal_window_sec: float) -> None:
        self.hamming_threshold = hamming_threshold
        self.temporal_window_sec = temporal_window_sec
        self._kept_by_shot: dict[int, list[tuple[int, str, float]]] = {}

    def find_duplicate(
        self,
        phash: int,
        shot_index: int,
        timestamp: float,
    ) -> tuple[str, int] | None:
        for kept_hash, kept_frame_id, kept_timestamp in self._kept_by_shot.get(
            shot_index, []
        ):
            if abs(timestamp - kept_timestamp) > self.temporal_window_sec:
                continue
            distance = hamming_distance(phash, kept_hash)
            if distance <= self.hamming_threshold:
                return kept_frame_id, distance
        return None

    def add(self, phash: int, frame_id: str, shot_index: int, timestamp: float) -> None:
        self._kept_by_shot.setdefault(shot_index, []).append(
            (phash, frame_id, timestamp)
        )


class ClipDeduper:
    """Optional conservative OpenCLIP dedup within a shot and time window."""

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
        self._kept_by_shot: dict[int, list[tuple[np.ndarray, str, float]]] = {}

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
        shot_index: int = 0,
    ) -> tuple[str, float] | None:
        best_frame_id = ""
        best_similarity = -1.0
        for kept_embedding, kept_frame_id, kept_timestamp in self._kept_by_shot.get(
            shot_index, []
        ):
            if abs(timestamp - kept_timestamp) > self.temporal_window_sec:
                continue
            similarity = float(np.dot(embedding, kept_embedding))
            if similarity > best_similarity:
                best_frame_id = kept_frame_id
                best_similarity = similarity
        if best_similarity >= self.similarity_threshold:
            return best_frame_id, best_similarity
        return None

    def add(
        self,
        embedding: np.ndarray,
        frame_id: str,
        timestamp: float,
        shot_index: int = 0,
    ) -> None:
        self._kept_by_shot.setdefault(shot_index, []).append(
            (embedding, frame_id, timestamp)
        )


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
    shots: list[Shot] = []
    for shot_index, scene in enumerate(scenes, start=1):
        start_frame = int(scene[0])
        end_frame = int(scene[1])
        start_frame = max(0, min(start_frame, info.frame_count - 1))
        end_frame = max(start_frame, min(end_frame, info.frame_count - 1))
        shots.append(
            Shot(
                shot_index=shot_index,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=info.fps,
            )
        )
    if not shots:
        shots.append(Shot(shot_index=1, start_frame=0, end_frame=info.frame_count - 1, fps=info.fps))
    return shots


def select_frame_indices(
    shot: Shot,
    short_shot_max_sec: float = DEFAULT_SHORT_SHOT_MAX_SEC,
    regular_shot_max_sec: float = DEFAULT_REGULAR_SHOT_MAX_SEC,
    long_shot_interval_sec: float = DEFAULT_LONG_SHOT_INTERVAL_SEC,
) -> list[tuple[int, str]]:
    """Select in-shot frames with the high-recall adaptive strategy.

    Shot frame bounds are inclusive. Fractional locations are computed from
    the exact shot duration in frames, while long-shot offsets are measured
    from the shot start timestamp.
    Every computed index is clamped to the shot as a final safety invariant.
    """
    validate_sampling_parameters(
        short_shot_max_sec=short_shot_max_sec,
        regular_shot_max_sec=regular_shot_max_sec,
        long_shot_interval_sec=long_shot_interval_sec,
    )
    duration = shot.duration
    if duration <= short_shot_max_sec:
        positions = [(0.5, "short_shot_midpoint")]
        selected = [
            (
                shot.start_frame
                + int(round((shot.end_frame - shot.start_frame + 1) * fraction)),
                reason,
            )
            for fraction, reason in positions
        ]
    elif duration <= regular_shot_max_sec:
        selected = [
            (
                shot.start_frame
                + int(round((shot.end_frame - shot.start_frame + 1) * fraction)),
                "regular_shot_one_third_two_thirds",
            )
            for fraction in (1.0 / 3.0, 2.0 / 3.0)
        ]
    else:
        selected = []
        offset_sec = long_shot_interval_sec / 2.0
        while offset_sec < duration:
            selected.append(
                (
                    shot.start_frame + int(round(offset_sec * shot.fps)),
                    "long_shot_centered_interval",
                )
            )
            offset_sec += long_shot_interval_sec
        if not selected:
            selected = [
                (
                    shot.start_frame
                    + int(round((shot.end_frame - shot.start_frame + 1) * 0.5)),
                    "long_shot_centered_interval",
                )
            ]

    bounded: list[tuple[int, str]] = []
    seen: set[int] = set()
    for frame_index, reason in selected:
        frame_index = max(shot.start_frame, min(frame_index, shot.end_frame))
        if frame_index not in seen:
            bounded.append((frame_index, reason))
            seen.add(frame_index)
    return bounded


def validate_sampling_parameters(
    short_shot_max_sec: float,
    regular_shot_max_sec: float,
    long_shot_interval_sec: float,
) -> None:
    if short_shot_max_sec <= 0:
        raise ValueError("short_shot_max_sec must be > 0")
    if regular_shot_max_sec < short_shot_max_sec:
        raise ValueError("regular_shot_max_sec must be >= short_shot_max_sec")
    if long_shot_interval_sec <= 0:
        raise ValueError("long_shot_interval_sec must be > 0")


def build_keyframe_candidates(
    shots: list[Shot],
    fps: float,
    short_shot_max_sec: float = DEFAULT_SHORT_SHOT_MAX_SEC,
    regular_shot_max_sec: float = DEFAULT_REGULAR_SHOT_MAX_SEC,
    long_shot_interval_sec: float = DEFAULT_LONG_SHOT_INTERVAL_SEC,
) -> list[KeyframeCandidate]:
    candidates: list[KeyframeCandidate] = []
    seen_frame_indices: set[int] = set()
    for shot in shots:
        selected = select_frame_indices(
            shot,
            short_shot_max_sec=short_shot_max_sec,
            regular_shot_max_sec=regular_shot_max_sec,
            long_shot_interval_sec=long_shot_interval_sec,
        )
        for frame_index, reason in selected:
            if frame_index in seen_frame_indices:
                continue
            candidates.append(
                KeyframeCandidate(
                    candidate_index=len(candidates) + 1,
                    shot=shot,
                    frame_index=frame_index,
                    timestamp=frame_index / fps,
                    selection_reason=reason,
                )
            )
            seen_frame_indices.add(frame_index)
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


def extract_keyframes_for_video(
    video_path: Path,
    output_dir: Path,
    metadata_path: Path,
    report_path: Path,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
    phash_window_sec: float = DEFAULT_DEDUP_TEMPORAL_WINDOW_SEC,
    jpeg_quality: int = 95,
    shot_threshold: float = 0.5,
    shot_device: str = "auto",
    enable_clip_dedup: bool = False,
    clip_similarity_threshold: float = DEFAULT_CLIP_SIMILARITY_THRESHOLD,
    clip_window_sec: float = DEFAULT_DEDUP_TEMPORAL_WINDOW_SEC,
    clip_model_name: str = "ViT-B-16",
    clip_pretrained: str = "laion2b_s34b_b88k",
    clip_device: str = "auto",
    short_shot_max_sec: float = DEFAULT_SHORT_SHOT_MAX_SEC,
    regular_shot_max_sec: float = DEFAULT_REGULAR_SHOT_MAX_SEC,
    long_shot_interval_sec: float = DEFAULT_LONG_SHOT_INTERVAL_SEC,
) -> dict:
    validate_sampling_parameters(
        short_shot_max_sec=short_shot_max_sec,
        regular_shot_max_sec=regular_shot_max_sec,
        long_shot_interval_sec=long_shot_interval_sec,
    )
    if not 0 <= phash_threshold <= 64:
        raise ValueError("phash_threshold must be between 0 and 64")
    if phash_window_sec < 0:
        raise ValueError("phash_window_sec must be >= 0")
    if clip_window_sec < 0:
        raise ValueError("clip_window_sec must be >= 0")

    info = read_video_info(video_path)
    try:
        shots, detector_name = detect_shots_transnetv2(
            video_path,
            info,
            threshold=shot_threshold,
            device=shot_device,
        )
    except Exception as exc:  # noqa: BLE001 - convert dependency/runtime errors to a clear CLI message.
        raise RuntimeError(
            "TransNetV2 shot detection failed. Install transnetv2-pytorch from "
            "requirements.txt and make sure ffmpeg.exe is available in PATH."
        ) from exc

    video_output_dir = output_dir / info.video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_keyframe_candidates(
        shots,
        info.fps,
        short_shot_max_sec=short_shot_max_sec,
        regular_shot_max_sec=regular_shot_max_sec,
        long_shot_interval_sec=long_shot_interval_sec,
    )
    phash_deduper = PerceptualHashDeduper(
        hamming_threshold=phash_threshold,
        temporal_window_sec=phash_window_sec,
    )
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
        duplicate = phash_deduper.find_duplicate(
            phash=phash,
            shot_index=shot.shot_index,
            timestamp=timestamp,
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
                    "duplicate_of": duplicate[0],
                    "hamming_distance": duplicate[1],
                }
            )
            continue

        clip_embedding = None
        if clip_deduper is not None:
            clip_embedding = clip_deduper.encode(frame)
            clip_duplicate = clip_deduper.find_duplicate(
                clip_embedding,
                shot_index=shot.shot_index,
                timestamp=timestamp,
            )
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

        phash_deduper.add(phash, frame_id, shot.shot_index, timestamp)
        if clip_deduper is not None and clip_embedding is not None:
            clip_deduper.add(
                clip_embedding,
                frame_id,
                shot_index=shot.shot_index,
                timestamp=timestamp,
            )
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
                "keyframe_strategy": KEYFRAME_STRATEGY,
                "selection_reason": reason,
                "phash": f"{phash:016x}",
                "shot_detector": detector_name,
            }
        )

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
        "candidate_count": len(candidates),
        "keyframe_count": len(records),
        "skipped_count": len(skipped),
        "keyframe_strategy": KEYFRAME_STRATEGY,
        "short_shot_max_sec": short_shot_max_sec,
        "regular_shot_max_sec": regular_shot_max_sec,
        "long_shot_interval_sec": long_shot_interval_sec,
        "phash_threshold": phash_threshold,
        "phash_window_sec": phash_window_sec,
        "dedup_scope": "within_shot",
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
                "candidate_frame_count": sum(
                    1
                    for candidate in candidates
                    if candidate.shot.shot_index == shot.shot_index
                ),
                "selected_frame_count": sum(
                    1 for record in records if record["shot_index"] == shot.shot_index
                ),
                "skipped_frame_count": sum(
                    1 for item in skipped if item["shot_index"] == shot.shot_index
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
            "Extract TransNetV2 shot-aware keyframes with adaptive sampling "
            "and conservative within-shot dedup."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-path", type=Path)
    source.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-glob", default="*.mp4")
    parser.add_argument("--output-dir", type=Path, default=Path("data/keyframes"))
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--phash-threshold", type=int, default=DEFAULT_PHASH_THRESHOLD)
    parser.add_argument(
        "--phash-window-sec",
        type=float,
        default=DEFAULT_DEDUP_TEMPORAL_WINDOW_SEC,
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--shot-threshold", type=float, default=0.5)
    parser.add_argument("--shot-device", default="auto")
    parser.add_argument(
        "--short-shot-max-sec", type=float, default=DEFAULT_SHORT_SHOT_MAX_SEC
    )
    parser.add_argument(
        "--regular-shot-max-sec", type=float, default=DEFAULT_REGULAR_SHOT_MAX_SEC
    )
    parser.add_argument(
        "--long-shot-interval-sec", type=float, default=DEFAULT_LONG_SHOT_INTERVAL_SEC
    )
    parser.add_argument("--enable-clip-dedup", action="store_true")
    parser.add_argument(
        "--clip-similarity-threshold",
        type=float,
        default=DEFAULT_CLIP_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "--clip-window-sec",
        type=float,
        default=DEFAULT_DEDUP_TEMPORAL_WINDOW_SEC,
    )
    parser.add_argument("--clip-model-name", default="ViT-B-16")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--clip-device", default="auto")
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
            short_shot_max_sec=args.short_shot_max_sec,
            regular_shot_max_sec=args.regular_shot_max_sec,
            long_shot_interval_sec=args.long_shot_interval_sec,
            enable_clip_dedup=args.enable_clip_dedup,
            clip_similarity_threshold=args.clip_similarity_threshold,
            clip_window_sec=args.clip_window_sec,
            clip_model_name=args.clip_model_name,
            clip_pretrained=args.clip_pretrained,
            clip_device=args.clip_device,
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
