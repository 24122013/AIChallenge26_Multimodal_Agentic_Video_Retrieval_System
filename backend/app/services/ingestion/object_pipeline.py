"""object_pipeline — phát hiện object trong keyframe.

Backend pluggable:
- StubObjectBackend: trả list rỗng (deterministic).
- YoloObjectBackend: cắm Ultralytics YOLO thật (lazy import).

Output JSONL: {frame_id, objects: [{label, confidence, bbox?}], object_model}.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.app.models.metadata import DetectedObject, ObjectAnnotation
from backend.app.services.ingestion._base_pipeline import run_frame_pipeline
from backend.app.services.ingestion.scheme_validator import validate_objects

logger = logging.getLogger(__name__)


class StubObjectBackend:
    """Object detector giả lập — trả rỗng. Không cần model."""

    name = "stub-object-v1"

    def process(self, image, record: dict) -> dict:
        return {"objects": [], "object_model": self.name}


class YoloObjectBackend:
    """Object detection thật dùng Ultralytics YOLO (lazy import)."""

    def __init__(
        self,
        model_id: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: str = "auto",
        max_objects: int = 30,
    ) -> None:
        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.device = None if device == "auto" else device
        self.max_objects = max_objects
        self.name = Path(model_id).stem
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO  # type: ignore

        logger.info("Loading YOLO model %s", self.model_id)
        self._model = YOLO(self.model_id)

    def process(self, image, record: dict) -> dict:
        if image is None:
            return {"objects": [], "object_model": self.name}
        self._ensure_loaded()
        results = self._model.predict(
            image, conf=self.conf_threshold, device=self.device, verbose=False
        )
        objects: list[dict] = []
        for result in results:
            names = result.names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                objects.append(
                    DetectedObject(
                        label=str(names.get(cls_id, cls_id)),
                        confidence=round(conf, 4),
                        bbox=xyxy,
                    ).to_dict()
                )
        objects = objects[: self.max_objects]
        return {"objects": objects, "object_model": self.name}


def build_backend(backend: str = "stub", **kwargs):
    if backend in ("stub", "dummy", "none"):
        return StubObjectBackend()
    if backend in ("yolo", "yolov8", "ultralytics"):
        return YoloObjectBackend(**kwargs)
    raise ValueError(f"object backend không hỗ trợ: {backend!r}")


def run_object_pipeline(
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
        pipeline_name="objects",
        metadata_path=metadata_path,
        output_path=output_path,
        backend=be,
        validate_fn=validate_objects,
        image_base_dir=image_base_dir,
        require_image=require_image,
        report_path=report_path,
        limit=limit,
    )


__all__ = [
    "StubObjectBackend",
    "YoloObjectBackend",
    "build_backend",
    "run_object_pipeline",
    "ObjectAnnotation",
    "DetectedObject",
]
