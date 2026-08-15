from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from backend.app.services.ingestion.common import (
    Timer,
    append_jsonl,
    chunks,
    choose_device,
    identity,
    iter_progress,
    json_log,
    package_version,
    processing_fields,
    read_jsonl,
    report,
    resolve_image_path,
    resumable_ids,
    safe_infer,
    utc_now,
    verify_image,
    video_id_from_records,
    write_json,
)


DEFAULT_MODEL_NAME = "yoloe-26l-seg.pt"
DEFAULT_MODEL_REVISION = "ultralytics-official"
DEFAULT_VOCABULARY = (
    "person",
    "face",
    "clothing",
    "bag",
    "phone",
    "computer",
    "screen",
    "book",
    "bottle",
    "cup",
    "food",
    "table",
    "chair",
    "vehicle",
    "car",
    "motorcycle",
    "bicycle",
    "bus",
    "sign",
    "animal",
)


class ObjectBackend(Protocol):
    model_name: str
    model_version: str
    model_revision: str | None

    def infer(self, paths: Sequence[Path]) -> Sequence[Any]: ...


def deterministic_class_id(class_name: str) -> int:
    normalized = " ".join(str(class_name).casefold().split())
    return int.from_bytes(hashlib.sha256(normalized.encode("utf-8")).digest()[:4], "big")


class YoloEBackend:
    """Lazy Ultralytics YOLOE text-prompt open-vocabulary backend."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        cache_dir: Path = Path("data/model_cache/objects"),
        vocabulary: Sequence[str] = DEFAULT_VOCABULARY,
        prompt_mode: str = "text",
    ) -> None:
        if prompt_mode not in {"text", "internal"}:
            raise ValueError("prompt_mode must be text or internal")
        cleaned = tuple(dict.fromkeys(" ".join(value.split()) for value in vocabulary if value.strip()))
        if prompt_mode == "text" and not cleaned:
            raise ValueError("text prompt mode requires at least one vocabulary class")
        self.model_name = model_name
        self.model_version = package_version("ultralytics")
        self.model_revision = revision
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.cache_dir = cache_dir
        self.vocabulary = cleaned
        self.prompt_mode = prompt_mode
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOE, settings
        except ImportError as exc:
            raise RuntimeError(
                "YOLOE requires a current Ultralytics package. Install requirements.txt."
            ) from exc
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        settings.update({"weights_dir": str(self.cache_dir.resolve())})
        model_path = Path(self.model_name)
        cached_path = (self.cache_dir / model_path.name).resolve()
        # Pass the intended cache filename even on the first run. Ultralytics'
        # asset downloader otherwise downloads a missing bare checkpoint into
        # the process working directory, which dirties the repository and
        # invalidates immutable-run fingerprints.
        model_ref = str(model_path.resolve() if model_path.is_file() else cached_path)
        self._model = YOLOE(model_ref)
        if self.prompt_mode == "text":
            self._model.set_classes(list(self.vocabulary))

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
    elif isinstance(raw, Mapping):
        detections = raw.get("objects", raw.get("detections", []))
        image_size = list(raw.get("image_size", []))
    else:
        height, width = (int(value) for value in raw.orig_shape)
        image_size = [width, height]
        names = raw.names
        detections = []
        if raw.boxes is not None:
            for xyxy, confidence, raw_class_id in zip(
                raw.boxes.xyxy.cpu().tolist(),
                raw.boxes.conf.cpu().tolist(),
                raw.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                raw_class_id = int(raw_class_id)
                detections.append(
                    {
                        "class_id": raw_class_id,
                        "class_name": str(names[raw_class_id]),
                        "confidence": float(confidence),
                        "bbox_xyxy": [float(value) for value in xyxy],
                    }
                )
    normalized: list[dict[str, Any]] = []
    for item in detections:
        value = dict(item)
        bbox = value.get("bbox_xyxy")
        if bbox is None:
            bbox = value.get("bbox")
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise ValueError("object bbox_xyxy must contain exactly four coordinates")
        class_name = " ".join(str(value["class_name"]).split())
        class_id = value.get("class_id")
        normalized.append(
            {
                "class_id": (
                    int(class_id)
                    if class_id is not None
                    else deterministic_class_id(class_name)
                ),
                "class_name": class_name,
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
    revision: str = DEFAULT_MODEL_REVISION,
    model_cache_dir: Path = Path("data/model_cache/objects"),
    vocabulary: Sequence[str] = DEFAULT_VOCABULARY,
    prompt_mode: str = "text",
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
    backend = backend or YoloEBackend(
        model_name=model_name,
        revision=revision,
        device=selected_device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        cache_dir=model_cache_dir,
        vocabulary=vocabulary,
        prompt_mode=prompt_mode,
    )
    requested_revision = getattr(backend, "model_revision", None)
    processed, stale = resumable_ids(
        output_path,
        "frame_id",
        model_name=backend.model_name,
        model_revision=requested_revision,
    )
    if overwrite:
        processed, stale = set(), True
    pending = [record for record in records if str(record.get("frame_id")) not in processed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = error_count = 0
    run_at = utc_now()
    batches = list(chunks(pending, batch_size))
    with output_path.open("w" if stale else "a", encoding="utf-8") as handle:
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
                            model_revision=requested_revision,
                            status="error",
                            run_at=run_at,
                            error=str(exc),
                        ),
                        "objects": [],
                        "object_classes": [],
                        "object_counts": {},
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
                            model_revision=requested_revision,
                            status="error",
                            run_at=run_at,
                            error=str(error),
                        ),
                        "objects": [],
                        "object_classes": [],
                        "object_counts": {},
                        "class_counts": {},
                    }
                    error_count += 1
                else:
                    try:
                        detections, image_size = normalize_detections(raw)
                        detections = [
                            item for item in detections if item["confidence"] >= conf_threshold
                        ]
                        counts = Counter(item["class_name"] for item in detections)
                        object_counts = dict(sorted(counts.items()))
                        value = {
                            **identity(record),
                            **processing_fields(
                                pipeline="objects",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                model_revision=requested_revision,
                                status="success",
                                run_at=run_at,
                            ),
                            "objects": detections,
                            "object_classes": sorted(counts),
                            "object_counts": object_counts,
                            # Retain the legacy alias for existing indexers.
                            "class_counts": object_counts,
                            "image_size": image_size,
                            "confidence_threshold": conf_threshold,
                            "iou_threshold": iou_threshold,
                            "open_vocabulary_mode": getattr(backend, "prompt_mode", prompt_mode),
                            "vocabulary": list(getattr(backend, "vocabulary", vocabulary)),
                            "evidence_only": True,
                        }
                        success_count += 1
                    except Exception as exc:
                        value = {
                            **identity(record),
                            **processing_fields(
                                pipeline="objects",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                model_revision=requested_revision,
                                status="error",
                                run_at=run_at,
                                error=str(exc),
                            ),
                            "objects": [],
                            "object_classes": [],
                            "object_counts": {},
                            "class_counts": {},
                        }
                        error_count += 1
                batch_values[position] = value
            for position in range(len(batch)):
                append_jsonl(handle, batch_values[position])

    result = report(
        pipeline="objects",
        input_path=metadata_path,
        output_path=output_path,
        model_name=backend.model_name,
        model_version=backend.model_version,
        model_revision=requested_revision,
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
            "open_vocabulary_mode": getattr(backend, "prompt_mode", prompt_mode),
            "vocabulary": list(getattr(backend, "vocabulary", vocabulary)),
            "model_cache_dir": str(getattr(backend, "cache_dir", model_cache_dir)),
            "evidence_only": True,
        }
    )
    write_json(report_path, result)
    json_log("ingestion.objects", "completed", latency=timer.elapsed, **result)
    return result
