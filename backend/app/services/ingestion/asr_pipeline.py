"""asr_pipeline — trích xuất lời thoại (transcript) theo segment thời gian.

Khác caption/ocr/objects (theo frame), ASR chạy trên audio của cả video và sinh
ra các segment. Mỗi segment nhận `segment_id = SEG_{video_id}_{N:06d}`.

Backend pluggable:
- StubAsrBackend: trả về 0 segment (deterministic) — chạy được khi chưa có model.
- WhisperAsrBackend: cắm faster-whisper thật (lazy import).

Output JSONL: {segment_id, video_id, transcript, language, start_time, end_time,
               asr_confidence, asr_model}.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.app.models.metadata import ASR, make_segment_id
from backend.app.services.ingestion._common import PipelineReport, write_json, write_jsonl
from backend.app.services.ingestion.scheme_validator import validate_asr

logger = logging.getLogger(__name__)


def infer_video_id(video_path: str | Path) -> str:
    return Path(video_path).stem


class StubAsrBackend:
    """ASR giả lập — không phát sinh transcript. Không cần model/audio."""

    name = "stub-asr-v1"

    def transcribe(self, video_path: str | Path) -> list[dict]:
        return []


class WhisperAsrBackend:
    """ASR thật dùng faster-whisper (lazy import)."""

    def __init__(
        self,
        model_id: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        language: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.name = f"whisper-{model_id}"
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # type: ignore

        device = self.device
        if device == "auto":
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute = self.compute_type if device == "cuda" else "int8"
        logger.info("Loading Whisper %s on %s (%s)", self.model_id, device, compute)
        self._model = WhisperModel(self.model_id, device=device, compute_type=compute)

    def transcribe(self, video_path: str | Path) -> list[dict]:
        self._ensure_loaded()
        segments, info = self._model.transcribe(str(video_path), language=self.language)
        out: list[dict] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            # avg_logprob -> xấp xỉ confidence trong [0,1]
            conf = 0.0
            if getattr(seg, "avg_logprob", None) is not None:
                import math

                conf = round(float(math.exp(seg.avg_logprob)), 4)
                conf = max(0.0, min(1.0, conf))
            out.append(
                {
                    "start_time": round(float(seg.start), 3),
                    "end_time": round(float(seg.end), 3),
                    "transcript": text,
                    "language": info.language,
                    "asr_confidence": conf,
                }
            )
        return out


def build_backend(backend: str = "stub", **kwargs):
    if backend in ("stub", "dummy", "none"):
        return StubAsrBackend()
    if backend in ("whisper", "faster-whisper", "faster_whisper"):
        return WhisperAsrBackend(**kwargs)
    raise ValueError(f"asr backend không hỗ trợ: {backend!r}")


def run_asr_pipeline(
    video_path: str | Path,
    output_path: str | Path,
    *,
    backend: str = "stub",
    video_id: str | None = None,
    report_path: str | Path | None = None,
    backend_kwargs: dict | None = None,
) -> PipelineReport:
    """Chạy ASR trên 1 video, ghi segment transcript ra JSONL."""
    vid = video_id or infer_video_id(video_path)
    be = build_backend(backend, **(backend_kwargs or {}))

    report = PipelineReport(
        pipeline="asr",
        model=getattr(be, "name", "unknown"),
        input_path=str(video_path),
        output_path=str(output_path),
    )

    raw_segments = be.transcribe(video_path)
    report.total_input = len(raw_segments)

    outputs: list[dict] = []
    for i, seg in enumerate(raw_segments, start=1):
        record = ASR(
            segment_id=make_segment_id(vid, i),
            video_id=vid,
            transcript=seg.get("transcript", ""),
            language=seg.get("language", ""),
            start_time=seg.get("start_time"),
            end_time=seg.get("end_time"),
            asr_confidence=float(seg.get("asr_confidence") or 0.0),
            asr_model=be.name,
        ).to_dict()

        result = validate_asr(record)
        if not result.valid:
            report.total_skipped += 1
            report.errors.append(f"{record['segment_id']}: {result.errors}")
            continue
        if not record["transcript"].strip():
            report.total_empty += 1
        outputs.append(record)

    report.total_written = write_jsonl(output_path, outputs)
    report.extra = {"video_id": vid}

    if report_path is not None:
        write_json(report_path, report.to_dict())

    logger.info(
        "[asr] video=%s segments=%d written=%d model=%s",
        vid,
        report.total_input,
        report.total_written,
        report.model,
    )
    return report


__all__ = [
    "StubAsrBackend",
    "WhisperAsrBackend",
    "build_backend",
    "run_asr_pipeline",
    "infer_video_id",
    "ASR",
]
