"""ocr_pipeline — trích xuất text trên keyframe.

Backend pluggable:
- StubOcrBackend: trả rỗng (deterministic) — giữ luồng chạy khi chưa có model.
- EasyOcrBackend: cắm EasyOCR thật (lazy import). Hỗ trợ nhiều ngôn ngữ.

Output JSONL: {frame_id, ocr_text, ocr_confidence, ocr_model}.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.app.models.metadata import OCR
from backend.app.services.ingestion._base_pipeline import run_frame_pipeline
from backend.app.services.ingestion.scheme_validator import validate_ocr

logger = logging.getLogger(__name__)


class StubOcrBackend:
    """OCR giả lập — trả text rỗng, confidence 0. Không cần model."""

    name = "stub-ocr-v1"

    def process(self, image, record: dict) -> dict:
        return {"ocr_text": "", "ocr_confidence": 0.0, "ocr_model": self.name}


class EasyOcrBackend:
    """OCR thật dùng EasyOCR (lazy import).

    Ghép các đoạn text phát hiện được thành một chuỗi; ocr_confidence là trung
    bình các confidence của từng đoạn.
    """

    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        self.languages = languages or ["en", "vi"]
        self.gpu = gpu
        self.name = "easyocr"
        self._reader = None

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return
        import easyocr  # type: ignore

        logger.info("Loading EasyOCR reader langs=%s gpu=%s", self.languages, self.gpu)
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu)

    def process(self, image, record: dict) -> dict:
        if image is None:
            return {"ocr_text": "", "ocr_confidence": 0.0, "ocr_model": self.name}
        self._ensure_loaded()
        detections = self._reader.readtext(image)  # [(bbox, text, conf), ...]
        texts: list[str] = []
        confs: list[float] = []
        for _bbox, text, conf in detections:
            text = (text or "").strip()
            if text:
                texts.append(text)
                confs.append(float(conf))
        joined = " ".join(texts)
        avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
        return {"ocr_text": joined, "ocr_confidence": avg_conf, "ocr_model": self.name}


def build_backend(backend: str = "stub", **kwargs):
    if backend in ("stub", "dummy", "none"):
        return StubOcrBackend()
    if backend in ("easyocr", "easy"):
        return EasyOcrBackend(**kwargs)
    raise ValueError(f"ocr backend không hỗ trợ: {backend!r}")


def run_ocr_pipeline(
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    backend: str = "stub",
    image_base_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    limit: int | None = None,
    require_image: bool = False,
    backend_kwargs: dict | None = None,
):
    be = build_backend(backend, **(backend_kwargs or {}))
    return run_frame_pipeline(
        pipeline_name="ocr",
        metadata_path=metadata_path,
        output_path=output_path,
        backend=be,
        validate_fn=validate_ocr,
        image_base_dir=image_base_dir,
        require_image=require_image,
        report_path=report_path,
        limit=limit,
    )


__all__ = [
    "StubOcrBackend",
    "EasyOcrBackend",
    "build_backend",
    "run_ocr_pipeline",
    "OCR",
]
