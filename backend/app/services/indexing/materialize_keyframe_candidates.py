"""Materialize the complete deterministic dense keyframe candidate pool.

This stage deliberately performs no keyframe selection.  Every candidate
produced by :func:`generate_keyframe_candidates` is decoded exactly once and
kept when its JPEG can be verified.  Perceptual-hash matches are annotations,
never reasons to remove a candidate.  A downstream Phase 3 orchestrator can
therefore fail closed when this report says the pool is incomplete, then run
multimodal feature extraction and protected-event selection over the exact
dense pool.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .extract_keyframes import (
    KEYFRAME_STRATEGY_DENSE_COVERAGE,
    Shot,
    VideoInfo,
    detect_shots_transnetv2,
    hamming_distance,
    load_extracted_image,
    perceptual_hash,
    read_video_info,
    remove_file_if_exists,
    write_json,
    write_jsonl,
)
from .keyframe_candidates import (
    DEFAULT_BOUNDARY_GUARD_SEC,
    DEFAULT_INTERVAL_SEC,
    DEFAULT_TINY_SHOT_MAX_SEC,
    KeyframeCandidate,
    generate_keyframe_candidates,
)


MATERIALIZATION_MODE = "full_dense_candidate_pool"
FRAME_EXTRACTOR = "opencv_sequential_grab_retrieve"


@dataclass(frozen=True)
class CandidateFrameDecode:
    """One target frame yielded by the single sequential decode pass."""

    candidate: KeyframeCandidate
    frame: np.ndarray | None
    error: str | None = None

    def __post_init__(self) -> None:
        if (self.frame is None) == (self.error is None):
            raise ValueError("exactly one of frame or error must be supplied")


def decode_candidate_frames_sequential(
    video_path: Path,
    candidates: Sequence[KeyframeCandidate],
) -> Iterator[CandidateFrameDecode]:
    """Yield exact candidate frames from one forward-only OpenCV pass.

    ``grab`` advances the stream one frame at a time, while ``retrieve`` only
    converts the requested candidate frames to arrays.  A target-specific
    retrieve failure does not stop later targets.  A terminal stream failure
    is reported separately for every remaining candidate so callers can write
    a complete partial-materialization report.
    """

    ordered = tuple(candidates)
    previous_frame_index = -1
    for position, candidate in enumerate(ordered):
        if candidate.frame_index <= previous_frame_index:
            raise ValueError(
                "candidate frame_index values must be strictly increasing: "
                f"position {position} has {candidate.frame_index} after "
                f"{previous_frame_index}"
            )
        previous_frame_index = candidate.frame_index
    if not ordered:
        return

    try:
        capture = cv2.VideoCapture(str(video_path))
    except Exception as exc:  # noqa: BLE001 - report every candidate as unavailable.
        error = f"OpenCV cannot create sequential decoder for {video_path}: {exc}"
        for candidate in ordered:
            yield CandidateFrameDecode(candidate=candidate, frame=None, error=error)
        return
    try:
        try:
            capture_opened = capture.isOpened()
        except Exception as exc:  # noqa: BLE001 - normalize backend-specific errors.
            capture_opened = False
            open_error = str(exc)
        else:
            open_error = ""
        if not capture_opened:
            error = f"OpenCV cannot open video for sequential decode: {video_path}"
            if open_error:
                error = f"{error}: {open_error}"
            for candidate in ordered:
                yield CandidateFrameDecode(candidate=candidate, frame=None, error=error)
            return

        next_frame_index = 0
        terminal_error: str | None = None
        for candidate in ordered:
            if terminal_error is not None:
                yield CandidateFrameDecode(
                    candidate=candidate,
                    frame=None,
                    error=terminal_error,
                )
                continue

            decoded_frame: np.ndarray | None = None
            retrieve_error: str | None = None
            while next_frame_index <= candidate.frame_index:
                try:
                    grabbed = capture.grab()
                except Exception as exc:  # noqa: BLE001 - terminal stream failure.
                    grabbed = False
                    grab_error = str(exc)
                else:
                    grab_error = ""
                if not grabbed:
                    terminal_error = (
                        "OpenCV sequential decode stopped before target frame "
                        f"{candidate.frame_index} (next frame was {next_frame_index})"
                    )
                    if grab_error:
                        terminal_error = f"{terminal_error}: {grab_error}"
                    break
                if next_frame_index == candidate.frame_index:
                    try:
                        retrieved, frame = capture.retrieve()
                    except Exception as exc:  # noqa: BLE001 - target-specific failure.
                        retrieved, frame = False, None
                        retrieve_detail = str(exc)
                    else:
                        retrieve_detail = ""
                    if not retrieved or frame is None:
                        retrieve_error = (
                            "OpenCV could not retrieve target frame "
                            f"{candidate.frame_index}"
                        )
                        if retrieve_detail:
                            retrieve_error = f"{retrieve_error}: {retrieve_detail}"
                    else:
                        decoded_frame = frame
                next_frame_index += 1

            if terminal_error is not None:
                yield CandidateFrameDecode(
                    candidate=candidate,
                    frame=None,
                    error=terminal_error,
                )
            elif retrieve_error is not None:
                yield CandidateFrameDecode(
                    candidate=candidate,
                    frame=None,
                    error=retrieve_error,
                )
            elif decoded_frame is None:  # Defensive guard for decoder contract drift.
                yield CandidateFrameDecode(
                    candidate=candidate,
                    frame=None,
                    error=(
                        "OpenCV sequential decoder produced no result for target frame "
                        f"{candidate.frame_index}"
                    ),
                )
            else:
                yield CandidateFrameDecode(
                    candidate=candidate,
                    frame=decoded_frame,
                )
    finally:
        try:
            capture.release()
        except Exception:  # noqa: BLE001 - do not mask materialization results.
            pass


def _frame_id(candidate_id: str) -> str:
    prefix = "CANDIDATE_"
    if not candidate_id.startswith(prefix):
        raise ValueError(f"Unexpected dense candidate_id: {candidate_id}")
    return f"FRAME_{candidate_id[len(prefix):]}"


def _candidate_report_record(
    candidate: KeyframeCandidate,
    *,
    candidate_index: int,
    materialized: bool,
) -> dict[str, object]:
    return {
        "candidate_index": candidate_index,
        "candidate_id": candidate.candidate_id,
        "frame_id": _frame_id(candidate.candidate_id),
        "video_id": candidate.video_id,
        "shot_index": candidate.shot_index,
        "shot_id": f"SHOT_{candidate.video_id}_{candidate.shot_index:06d}",
        "frame_index": candidate.frame_index,
        "timestamp": round(candidate.timestamp_sec, 3),
        "selection_reason": candidate.reasons[0],
        "candidate_reasons": list(candidate.reasons),
        "materialized": materialized,
        "extraction_status": "materialized" if materialized else "failed",
    }


def _materialize_candidates(
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
    candidate_interval_sec: float,
    boundary_guard_sec: float,
    tiny_shot_max_sec: float,
    include_video_endpoints: bool,
    candidates: Sequence[KeyframeCandidate] | None = None,
) -> dict[str, object]:
    video_output_dir = output_dir / info.video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    generated = (
        tuple(candidates)
        if candidates is not None
        else generate_keyframe_candidates(
            info.video_id,
            shots,
            info.fps,
            interval_sec=candidate_interval_sec,
            boundary_guard_sec=boundary_guard_sec,
            tiny_shot_max_sec=tiny_shot_max_sec,
            frame_count=info.frame_count,
            include_video_endpoints=include_video_endpoints,
        )
    )
    shot_by_index = {shot.shot_index: shot for shot in shots}
    kept_phashes: deque[tuple[int, str, float]] = deque()
    records: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    duplicate_retained: list[dict[str, object]] = []
    materialized_ids: set[str] = set()

    decode_results = iter(decode_candidate_frames_sequential(video_path, generated))
    for candidate_index, candidate in enumerate(generated, start=1):
        frame_id = _frame_id(candidate.candidate_id)
        shot = shot_by_index[candidate.shot_index]
        timestamp = candidate.timestamp_sec
        output_path = video_output_dir / f"{frame_id}.jpg"
        try:
            decoded = next(decode_results)
        except StopIteration:
            decoded = CandidateFrameDecode(
                candidate=candidate,
                frame=None,
                error="OpenCV sequential decoder omitted this candidate",
            )
        if decoded.candidate.candidate_id != candidate.candidate_id:
            raise RuntimeError(
                "OpenCV sequential decoder changed candidate order: "
                f"expected {candidate.candidate_id}, got "
                f"{decoded.candidate.candidate_id}"
            )
        if decoded.error is not None:
            remove_file_if_exists(output_path)
            skipped.append(
                {
                    "frame_id": frame_id,
                    "candidate_id": candidate.candidate_id,
                    "video_id": info.video_id,
                    "shot_index": candidate.shot_index,
                    "frame_index": candidate.frame_index,
                    "timestamp": round(timestamp, 3),
                    "reason": "opencv_decode_failed",
                    "error": decoded.error,
                    "output_path": output_path.as_posix(),
                }
            )
            continue

        try:
            assert decoded.frame is not None
            wrote_jpeg = cv2.imwrite(
                str(output_path),
                decoded.frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    max(1, min(int(jpeg_quality), 100)),
                ],
            )
            if not wrote_jpeg:
                raise RuntimeError(f"OpenCV failed to write candidate JPEG: {output_path}")
            frame = load_extracted_image(output_path)
        except Exception as exc:  # noqa: BLE001 - record failure and keep decoding.
            remove_file_if_exists(output_path)
            skipped.append(
                {
                    "frame_id": frame_id,
                    "candidate_id": candidate.candidate_id,
                    "video_id": info.video_id,
                    "shot_index": candidate.shot_index,
                    "frame_index": candidate.frame_index,
                    "timestamp": round(timestamp, 3),
                    "reason": "jpeg_write_or_verify_failed",
                    "error": str(exc),
                    "output_path": output_path.as_posix(),
                }
            )
            continue

        phash = perceptual_hash(frame)
        while (
            kept_phashes
            and timestamp - kept_phashes[0][2] > phash_window_sec
        ):
            kept_phashes.popleft()
        duplicate = next(
            (
                (existing_hash, existing_frame_id)
                for existing_hash, existing_frame_id, existing_timestamp in kept_phashes
                if abs(timestamp - existing_timestamp) <= phash_window_sec
                if hamming_distance(phash, existing_hash) <= phash_threshold
            ),
            None,
        )
        duplicate_fields: dict[str, object] = {}
        if duplicate is not None:
            distance = hamming_distance(phash, duplicate[0])
            duplicate_fields = {
                "phash_duplicate_of": duplicate[1],
                "phash_hamming_distance": distance,
            }
            duplicate_retained.append(
                {
                    "frame_id": frame_id,
                    "candidate_id": candidate.candidate_id,
                    "reason": "phash_duplicate_retained",
                    "duplicate_of": duplicate[1],
                    "hamming_distance": distance,
                }
            )
        kept_phashes.append((phash, frame_id, timestamp))

        normalized_path = output_path.as_posix()
        records.append(
            {
                "frame_id": frame_id,
                "candidate_id": candidate.candidate_id,
                "candidate_index": candidate_index,
                "video_id": info.video_id,
                "shot_id": f"SHOT_{info.video_id}_{shot.shot_index:06d}",
                "segment_id": f"SHOT_{info.video_id}_{shot.shot_index:06d}",
                "shot_index": shot.shot_index,
                "shot_start": round(candidate.shot_start_sec, 3),
                "shot_end": round(candidate.shot_end_sec, 3),
                "shot_duration": round(
                    candidate.shot_end_sec - candidate.shot_start_sec,
                    3,
                ),
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
                "keyframe_strategy": KEYFRAME_STRATEGY_DENSE_COVERAGE,
                "materialization_mode": MATERIALIZATION_MODE,
                "phash": f"{phash:016x}",
                "shot_detector": detector_name,
                **duplicate_fields,
            }
        )
        materialized_ids.add(candidate.candidate_id)

    try:
        unexpected = next(decode_results)
    except StopIteration:
        unexpected = None
    if unexpected is not None:
        raise RuntimeError(
            "OpenCV sequential decoder yielded an unexpected extra candidate: "
            f"{unexpected.candidate.candidate_id}"
        )

    candidate_count = len(generated)
    complete = (
        candidate_count > 0
        and len(records) == candidate_count
        and not skipped
    )
    candidate_config = {
        "candidate_interval_sec": candidate_interval_sec,
        "boundary_guard_sec": boundary_guard_sec,
        "tiny_shot_max_sec": tiny_shot_max_sec,
    }
    if include_video_endpoints:
        candidate_config["include_video_endpoints"] = True
    materialized_count_by_shot = Counter(
        int(record["shot_index"]) for record in records
    )
    report: dict[str, object] = {
        "video_id": info.video_id,
        "video_path": video_path.as_posix(),
        "fps": info.fps,
        "frame_count": info.frame_count,
        "duration": round(info.duration, 3),
        "shot_detector": detector_name,
        "shot_threshold": shot_threshold,
        "shot_device": shot_device,
        "shot_count": len(shots),
        "candidate_count": candidate_count,
        "keyframe_count": len(records),
        "skipped_count": len(skipped),
        "phash_threshold": phash_threshold,
        "phash_window_sec": phash_window_sec,
        "jpeg_quality": jpeg_quality,
        "frame_extractor": FRAME_EXTRACTOR,
        "sequential_decode_passes": 1,
        "metadata_path": metadata_path.as_posix(),
        "output_dir": video_output_dir.as_posix(),
        "keyframe_strategy": KEYFRAME_STRATEGY_DENSE_COVERAGE,
        "materialization_mode": MATERIALIZATION_MODE,
        "selection_applied": False,
        "status": "satisfied" if complete else "partial",
        "constraints_satisfied": complete,
        "coverage_satisfied": complete,
        "candidate_interval_sec": candidate_interval_sec,
        "boundary_guard_sec": boundary_guard_sec,
        "tiny_shot_max_sec": tiny_shot_max_sec,
        "include_video_endpoints": include_video_endpoints,
        "candidate_config": candidate_config,
        "deduplication_mode": "annotate_only",
        "duplicate_retained_count": len(duplicate_retained),
        "duplicate_retained": duplicate_retained,
        "skipped": skipped,
        "candidates": [
            _candidate_report_record(
                candidate,
                candidate_index=candidate_index,
                materialized=candidate.candidate_id in materialized_ids,
            )
            for candidate_index, candidate in enumerate(generated, start=1)
        ],
        "shots": [
            {
                **asdict(shot),
                "start_sec": round(shot.start_sec, 3),
                "end_sec": round(shot.end_sec, 3),
                "duration": round(shot.duration, 3),
                "materialized_candidate_count": materialized_count_by_shot.get(
                    shot.shot_index,
                    0,
                ),
            }
            for shot in shots
        ],
    }
    write_jsonl(records, metadata_path)
    write_json(report, report_path)
    return report


def materialize_keyframe_candidates_for_video(
    video_path: Path,
    output_dir: Path,
    metadata_path: Path,
    report_path: Path,
    phash_threshold: int = 6,
    phash_window_sec: float = 12.0,
    jpeg_quality: int = 95,
    shot_threshold: float = 0.5,
    shot_device: str = "auto",
    candidate_interval_sec: float = DEFAULT_INTERVAL_SEC,
    boundary_guard_sec: float = DEFAULT_BOUNDARY_GUARD_SEC,
    tiny_shot_max_sec: float = DEFAULT_TINY_SHOT_MAX_SEC,
    include_video_endpoints: bool = False,
) -> dict[str, object]:
    """Decode and verify all deterministic dense candidates for one video.

    Selector-only options such as ``max_gap_seconds``, ``target_keyframes``,
    and ``hard_max_keyframes`` are intentionally absent: this stage never runs
    protected-event, coverage, or MMR selection.
    """

    info = read_video_info(video_path)
    try:
        shots, detector_name = detect_shots_transnetv2(
            video_path,
            info,
            threshold=shot_threshold,
            device=shot_device,
        )
    except Exception as exc:  # noqa: BLE001 - normalize dependency/runtime errors.
        cuda_hint = (
            " The requested CUDA device requires a CUDA-enabled PyTorch build; "
            "verify torch.version.cuda and torch.cuda.is_available()."
            if shot_device == "cuda"
            else ""
        )
        raise RuntimeError(
            "TransNetV2 shot detection failed. Install transnetv2-pytorch from "
            "requirements.txt, make sure ffmpeg.exe is available in PATH, and "
            f"inspect the root cause: {type(exc).__name__}: {exc}.{cuda_hint}"
        ) from exc

    candidates = generate_keyframe_candidates(
        info.video_id,
        shots,
        info.fps,
        interval_sec=candidate_interval_sec,
        boundary_guard_sec=boundary_guard_sec,
        tiny_shot_max_sec=tiny_shot_max_sec,
        frame_count=info.frame_count,
        include_video_endpoints=include_video_endpoints,
    )
    return materialize_generated_keyframe_candidates_for_video(
        video_path=video_path,
        info=info,
        shots=shots,
        detector_name=detector_name,
        candidates=candidates,
        output_dir=output_dir,
        metadata_path=metadata_path,
        report_path=report_path,
        phash_threshold=phash_threshold,
        phash_window_sec=phash_window_sec,
        jpeg_quality=jpeg_quality,
        shot_threshold=shot_threshold,
        shot_device=shot_device,
        candidate_interval_sec=candidate_interval_sec,
        boundary_guard_sec=boundary_guard_sec,
        tiny_shot_max_sec=tiny_shot_max_sec,
        include_video_endpoints=include_video_endpoints,
    )


def materialize_generated_keyframe_candidates_for_video(
    *,
    video_path: Path,
    info: VideoInfo,
    shots: Sequence[Shot],
    detector_name: str,
    candidates: Sequence[KeyframeCandidate],
    output_dir: Path,
    metadata_path: Path,
    report_path: Path,
    phash_threshold: int = 6,
    phash_window_sec: float = 12.0,
    jpeg_quality: int = 95,
    shot_threshold: float = 0.5,
    shot_device: str = "auto",
    candidate_interval_sec: float = DEFAULT_INTERVAL_SEC,
    boundary_guard_sec: float = DEFAULT_BOUNDARY_GUARD_SEC,
    tiny_shot_max_sec: float = DEFAULT_TINY_SHOT_MAX_SEC,
    include_video_endpoints: bool = False,
) -> dict[str, object]:
    """Materialize an already-generated dense pool without re-running earlier stages.

    The canonical offline orchestrator checkpoints shot detection and candidate
    generation independently, then passes that exact candidate sequence here.
    The legacy convenience API above keeps its existing one-call behavior.
    """

    ordered = tuple(candidates)
    if not ordered:
        raise ValueError("candidates must contain at least one dense candidate")
    if any(candidate.video_id != info.video_id for candidate in ordered):
        raise ValueError("all candidates must match info.video_id")
    if len({candidate.candidate_id for candidate in ordered}) != len(ordered):
        raise ValueError("candidate_id values must be unique")

    return _materialize_candidates(
        video_path=video_path,
        info=info,
        shots=list(shots),
        detector_name=detector_name,
        output_dir=output_dir,
        metadata_path=metadata_path,
        report_path=report_path,
        phash_threshold=phash_threshold,
        phash_window_sec=phash_window_sec,
        jpeg_quality=jpeg_quality,
        shot_threshold=shot_threshold,
        shot_device=shot_device,
        candidate_interval_sec=candidate_interval_sec,
        boundary_guard_sec=boundary_guard_sec,
        tiny_shot_max_sec=tiny_shot_max_sec,
        include_video_endpoints=include_video_endpoints,
        candidates=ordered,
    )


__all__ = [
    "CandidateFrameDecode",
    "FRAME_EXTRACTOR",
    "MATERIALIZATION_MODE",
    "decode_candidate_frames_sequential",
    "materialize_generated_keyframe_candidates_for_video",
    "materialize_keyframe_candidates_for_video",
]
