from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from backend.app.services.ingestion.common import (
    Timer,
    append_jsonl,
    choose_device,
    existing_ids,
    identity,
    json_log,
    package_version,
    processing_fields,
    read_jsonl,
    report,
    utc_now,
    write_json,
)


DEFAULT_MODEL_SIZE = "small"


class AsrBackend(Protocol):
    model_name: str
    model_version: str

    def transcribe(self, video_path: Path) -> tuple[list[dict[str, Any]], str]: ...


def probe_audio_stream(video_path: Path) -> bool:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError(
            "ffprobe was not found in PATH. Install FFmpeg and ensure ffprobe is available."
        )
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed for {video_path}: {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {video_path}") from exc
    return bool(value.get("streams"))


class WhisperBackend:
    """Lazy faster-whisper backend with an openai-whisper fallback."""

    def __init__(
        self,
        *,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = "cpu",
        backend: str = "auto",
        cache_dir: Path | None = Path("data/model_cache/asr"),
        vad_filter: bool = True,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.requested_backend = backend
        self.cache_dir = cache_dir
        self.vad_filter = vad_filter
        self.model_name = f"{backend}/{model_size}"
        self.model_version = "unknown"
        self._model: Any | None = None
        self._active_backend: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.requested_backend in {"auto", "faster-whisper"}:
            try:
                from faster_whisper import WhisperModel

                compute_type = "float16" if self.device == "cuda" else "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=compute_type,
                    download_root=str(self.cache_dir) if self.cache_dir else None,
                )
                self._active_backend = "faster-whisper"
                self.model_name = f"faster-whisper/{self.model_size}"
                self.model_version = package_version("faster-whisper")
                return
            except ImportError:
                if self.requested_backend == "faster-whisper":
                    raise RuntimeError(
                        "faster-whisper is not installed. Install requirements.txt "
                        "or select --backend whisper."
                    )
        if self.requested_backend in {"auto", "whisper"}:
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "No ASR backend is installed. Install faster-whisper (preferred) "
                    "or openai-whisper."
                ) from exc
            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=str(self.cache_dir) if self.cache_dir else None,
            )
            self._active_backend = "whisper"
            self.model_name = f"openai-whisper/{self.model_size}"
            self.model_version = package_version("openai-whisper")
            return
        raise ValueError("--backend must be one of: auto, faster-whisper, whisper")

    def transcribe(self, video_path: Path) -> tuple[list[dict[str, Any]], str]:
        self._load()
        if self._active_backend == "faster-whisper":
            raw_segments, info = self._model.transcribe(
                str(video_path),
                language=None,
                vad_filter=self.vad_filter,
                beam_size=5,
            )
            language = str(info.language)
            values = []
            for segment in raw_segments:
                avg_logprob = float(segment.avg_logprob)
                values.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": str(segment.text).strip(),
                        "language": language,
                        "confidence": min(1.0, math.exp(avg_logprob)),
                        "avg_logprob": avg_logprob,
                        "no_speech_probability": float(segment.no_speech_prob),
                    }
                )
            return values, language

        result = self._model.transcribe(str(video_path), language=None, fp16=self.device == "cuda")
        language = str(result.get("language") or "unknown")
        values = []
        for segment in result.get("segments", []):
            avg_logprob = float(segment.get("avg_logprob", 0.0))
            values.append(
                {
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": str(segment["text"]).strip(),
                    "language": language,
                    "confidence": min(1.0, math.exp(avg_logprob)),
                    "avg_logprob": avg_logprob,
                    "no_speech_probability": float(segment.get("no_speech_prob", 0.0)),
                }
            )
        return values, language


def map_transcript_to_segments(
    keyframes: Sequence[dict[str, Any]],
    transcript: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in keyframes:
        key = str(frame.get("segment_id") or frame.get("shot_id") or frame.get("frame_id"))
        grouped[key].append(frame)

    mapped: list[dict[str, Any]] = []
    for group_id, frames in grouped.items():
        first = frames[0]
        starts = [float(frame["shot_start"]) for frame in frames if frame.get("shot_start") is not None]
        ends = [float(frame["shot_end"]) for frame in frames if frame.get("shot_end") is not None]
        if starts and ends:
            start, end = min(starts), max(ends)
        else:
            timestamps = [float(frame.get("timestamp", 0.0)) for frame in frames]
            start = end = timestamps[0] if timestamps else 0.0
        overlaps = [
            dict(segment)
            for segment in transcript
            if float(segment["end"]) > start and float(segment["start"]) < end
        ]
        mapped.append(
            {
                **identity(first),
                "segment_id": first.get("segment_id") or group_id,
                "shot_id": first.get("shot_id"),
                "frame_ids": [frame.get("frame_id") for frame in frames],
                "segment_start": start,
                "segment_end": end,
                "transcript_text": " ".join(
                    str(segment.get("text", "")).strip() for segment in overlaps
                ).strip(),
                "transcript_segments": overlaps,
            }
        )
    return mapped


def run_asr_file(
    video_path: Path,
    *,
    metadata_path: Path | None = None,
    output_dir: Path = Path("data/metadata"),
    output_path: Path | None = None,
    segments_output_path: Path | None = None,
    report_path: Path | None = None,
    device: str = "auto",
    model_size: str = DEFAULT_MODEL_SIZE,
    backend_name: str = "auto",
    overwrite: bool = False,
    vad_filter: bool = True,
    backend: AsrBackend | None = None,
    audio_probe: Callable[[Path], bool] = probe_audio_stream,
) -> dict[str, Any]:
    timer = Timer()
    video_id = video_path.stem
    output_path = output_path or output_dir / f"asr_{video_id}.jsonl"
    segments_output_path = segments_output_path or output_dir / f"asr_segments_{video_id}.jsonl"
    report_path = report_path or output_dir / f"asr_{video_id}_report.json"
    selected_device = choose_device(device)
    backend = backend or WhisperBackend(
        model_size=model_size,
        device=selected_device,
        backend=backend_name,
        vad_filter=vad_filter,
    )
    processed = set() if overwrite else existing_ids(output_path, "video_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments_output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        segments_output_path.write_text("", encoding="utf-8")
    run_at = utc_now()
    success_count = skipped_count = error_count = 0

    if video_id in processed:
        skipped_count = 1
    else:
        mode = "w" if overwrite else "a"
        try:
            has_audio = audio_probe(video_path)
            if not has_audio:
                with output_path.open(mode, encoding="utf-8") as handle:
                    append_jsonl(
                        handle,
                        {
                            "video_id": video_id,
                            "source_video_path": str(video_path),
                            **processing_fields(
                                pipeline="asr",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                status="skipped",
                                run_at=run_at,
                                skip_reason="no_audio_stream",
                            ),
                        },
                    )
                segments_output_path.write_text("", encoding="utf-8")
                skipped_count = 1
            else:
                if metadata_path and not metadata_path.exists():
                    raise FileNotFoundError(f"Metadata does not exist: {metadata_path}")
                keyframes = read_jsonl(metadata_path) if metadata_path else []
                metadata_video_ids = {
                    str(frame["video_id"])
                    for frame in keyframes
                    if frame.get("video_id") is not None
                }
                if metadata_video_ids and metadata_video_ids != {video_id}:
                    raise ValueError(
                        f"Metadata video_id {sorted(metadata_video_ids)} does not match {video_id}"
                    )
                transcript, language = backend.transcribe(video_path)
                with output_path.open(mode, encoding="utf-8") as handle:
                    if not transcript:
                        append_jsonl(
                            handle,
                            {
                                "video_id": video_id,
                                "source_video_path": str(video_path),
                                **processing_fields(
                                    pipeline="asr",
                                    model_name=backend.model_name,
                                    model_version=backend.model_version,
                                    status="success",
                                    run_at=run_at,
                                ),
                                "start": 0.0,
                                "end": 0.0,
                                "text": "",
                                "language": language,
                            },
                        )
                    for segment_index, segment in enumerate(transcript):
                        append_jsonl(
                            handle,
                            {
                                "video_id": video_id,
                                "source_video_path": str(video_path),
                                "transcript_segment_id": f"ASR_{video_id}_{segment_index:06d}",
                                **processing_fields(
                                    pipeline="asr",
                                    model_name=backend.model_name,
                                    model_version=backend.model_version,
                                    status="success",
                                    run_at=run_at,
                                ),
                                **segment,
                            },
                        )
                mapped = map_transcript_to_segments(keyframes, transcript)
                with segments_output_path.open("w", encoding="utf-8") as handle:
                    for value in mapped:
                        append_jsonl(
                            handle,
                            {
                                **value,
                                **processing_fields(
                                    pipeline="asr_segment_mapping",
                                    model_name=backend.model_name,
                                    model_version=backend.model_version,
                                    status="success",
                                    run_at=run_at,
                                ),
                            },
                        )
                success_count = len(transcript) if transcript else 1
        except Exception as exc:
            with output_path.open(mode, encoding="utf-8") as handle:
                append_jsonl(
                    handle,
                    {
                        "video_id": video_id,
                        "source_video_path": str(video_path),
                        **processing_fields(
                            pipeline="asr",
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            status="error",
                            run_at=run_at,
                            error=str(exc),
                        ),
                    },
                )
            error_count = 1

    result = report(
        pipeline="asr",
        input_path=video_path,
        output_path=output_path,
        model_name=backend.model_name,
        model_version=backend.model_version,
        device=selected_device,
        started_at=timer.started_at,
        elapsed=timer.elapsed,
        input_count=1,
        success_count=success_count,
        skipped_count=skipped_count,
        error_count=error_count,
    )
    result.update(
        {
            "model_size": model_size,
            "backend": backend_name,
            "vad_filter": vad_filter,
            "segments_output_path": str(segments_output_path),
            "metadata_path": str(metadata_path) if metadata_path else None,
        }
    )
    write_json(report_path, result)
    json_log("ingestion.asr", "completed", latency=timer.elapsed, **result)
    return result
