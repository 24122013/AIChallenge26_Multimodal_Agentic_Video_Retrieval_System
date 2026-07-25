from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Protocol, Sequence

from backend.app.services.ingestion.common import (
    Timer,
    append_jsonl,
    chunks,
    choose_device,
    existing_ids,
    identity,
    iter_progress,
    json_log,
    package_version,
    processing_fields,
    read_jsonl,
    report,
    resolve_image_path,
    safe_infer,
    utc_now,
    verify_image,
    video_id_from_records,
    write_json,
)


DEFAULT_MODEL_NAME = "yolo11n.pt"


class ObjectBackend(Protocol):
    model_name: str
    model_version: str

    def infer(self, paths: Sequence[Path]) -> Sequence[Any]: ...


class YoloBackend:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        cache_dir: Path = Path("data/model_cache/objects"),
    ) -> None:
        self.model_name = model_name
        self.model_version = package_version("ultralytics")
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.cache_dir = cache_dir
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO, settings
        except ImportError as exc:
            raise RuntimeError(
                "Object detection requires Ultralytics. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        settings.update({"weights_dir": str(self.cache_dir.resolve())})
        model_path = Path(self.model_name)
        if not model_path.is_absolute() and model_path.parent == Path("."):
            model_path = self.cache_dir / model_path
        self._model = YOLO(str(model_path))

    def infer(self, paths: Sequence[Path]) -> Sequence[Any]:
        self._load()
        return self._model.predict(
            [str(path) for path in paths],
            device=0 if self.device == "cuda" else "cpu",
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )


def normalize_detections(raw: Any) -> tuple[list[dict[str, Any]], list[int]]:
    if isinstance(raw, list):
        detections = raw
        image_size: list[int] = []
    elif isinstance(raw, dict):
        detections = raw.get("objects", raw.get("detections", []))
        image_size = list(raw.get("image_size", []))
    else:
        height, width = (int(value) for value in raw.orig_shape)
        image_size = [width, height]
        names = raw.names
        detections = []
        if raw.boxes is not None:
            for xyxy, confidence, class_id in zip(
                raw.boxes.xyxy.cpu().tolist(),
                raw.boxes.conf.cpu().tolist(),
                raw.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                class_id = int(class_id)
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": str(names[class_id]),
                        "confidence": float(confidence),
                        "bbox_xyxy": [float(value) for value in xyxy],
                    }
                )
    normalized: list[dict[str, Any]] = []
    for item in detections:
        value = dict(item)
        bbox = value.get("bbox_xyxy") or value.get("bbox")
        normalized.append(
            {
                "class_id": int(value["class_id"]),
                "class_name": str(value["class_name"]),
                "confidence": float(value["confidence"]),
                "bbox_xyxy": [float(number) for number in bbox],
            }
        )
    return normalized, image_size


def run_object_file(
    metadata_path: Path,
    *,
    output_dir: Path = Path("data/metadata"),
    output_path: Path | None = None,
    report_path: Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    overwrite: bool = False,
    backend: ObjectBackend | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    model_cache_dir: Path = Path("data/model_cache/objects"),
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be between 0 and 1")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    timer = Timer()
    records = read_jsonl(metadata_path)
    video_id = video_id_from_records(records, metadata_path.stem)
    output_path = output_path or output_dir / f"objects_{video_id}.jsonl"
    report_path = report_path or output_dir / f"objects_{video_id}_report.json"
    selected_device = choose_device(device)
    backend = backend or YoloBackend(
        model_name=model_name,
        device=selected_device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        cache_dir=model_cache_dir,
    )
    processed = set() if overwrite else existing_ids(output_path, "frame_id")
    pending = [record for record in records if str(record.get("frame_id")) not in processed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = error_count = 0
    run_at = utc_now()
    batches = list(chunks(pending, batch_size))
    with output_path.open("w" if overwrite else "a", encoding="utf-8") as handle:
        for batch in iter_progress(batches, total=len(batches), description=f"objects {video_id}"):
            batch_values: dict[int, dict[str, Any]] = {}
            valid_positions: list[int] = []
            valid_records: list[dict[str, Any]] = []
            valid_paths: list[Path] = []
            for position, record in enumerate(batch):
                try:
                    path = resolve_image_path(record, metadata_path)
                    verify_image(path)
                    valid_positions.append(position)
                    valid_records.append(record)
                    valid_paths.append(path)
                except Exception as exc:
                    batch_values[position] = {
                        **identity(record),
                        **processing_fields(
                            pipeline="objects",
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            status="error",
                            run_at=run_at,
                            error=str(exc),
                        ),
                        "objects": [],
                        "object_classes": [],
                        "class_counts": {},
                    }
                    error_count += 1
            for position, record, (raw, error) in zip(
                valid_positions,
                valid_records,
                safe_infer(valid_paths, backend.infer),
                strict=True,
            ):
                if error:
                    value = {
                        **identity(record),
                        **processing_fields(
                            pipeline="objects",
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            status="error",
                            run_at=run_at,
                            error=str(error),
                        ),
                        "objects": [],
                        "object_classes": [],
                        "class_counts": {},
                    }
                    error_count += 1
                else:
                    try:
                        detections, image_size = normalize_detections(raw)
                        detections = [
                            item
                            for item in detections
                            if item["confidence"] >= conf_threshold
                        ]
                        counts = Counter(item["class_name"] for item in detections)
                        value = {
                            **identity(record),
                            **processing_fields(
                                pipeline="objects",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                status="success",
                                run_at=run_at,
                            ),
                            "objects": detections,
                            "object_classes": sorted(counts),
                            "class_counts": dict(sorted(counts.items())),
                            "image_size": image_size,
                            "confidence_threshold": conf_threshold,
                            "iou_threshold": iou_threshold,
                        }
                        success_count += 1
                    except Exception as exc:
                        value = {
                            **identity(record),
                            **processing_fields(
                                pipeline="objects",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                status="error",
                                run_at=run_at,
                                error=str(exc),
                            ),
                            "objects": [],
                            "object_classes": [],
                            "class_counts": {},
                        }
                        error_count += 1
                batch_values[position] = value
            for position in range(len(batch)):
                value = batch_values[position]
                append_jsonl(handle, value)

    result = report(
        pipeline="objects",
        input_path=metadata_path,
        output_path=output_path,
        model_name=backend.model_name,
        model_version=backend.model_version,
        device=selected_device,
        started_at=timer.started_at,
        elapsed=timer.elapsed,
        input_count=len(records),
        success_count=success_count,
        skipped_count=len(records) - len(pending),
        error_count=error_count,
    )
    result.update(
        {
            "batch_size": batch_size,
            "confidence_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
        }
    )
    write_json(report_path, result)
    json_log("ingestion.objects", "completed", latency=timer.elapsed, **result)
    return result
