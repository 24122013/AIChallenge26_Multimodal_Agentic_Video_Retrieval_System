from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image

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


DEFAULT_MODEL_NAME = "EasyOCR"


class OcrBackend(Protocol):
    model_name: str
    model_version: str

    def infer(self, paths: Sequence[Path]) -> Sequence[Sequence[Any]]: ...


class EasyOcrBackend:
    def __init__(self, *, device: str = "cpu", languages: Sequence[str] = ("vi", "en")) -> None:
        self.model_name = DEFAULT_MODEL_NAME
        self.model_version = package_version("easyocr")
        self.device = device
        self.languages = list(languages)
        self._reader: Any | None = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "OCR requires EasyOCR. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc
        self._reader = easyocr.Reader(self.languages, gpu=self.device == "cuda")

    def infer(self, paths: Sequence[Path]) -> Sequence[Sequence[Any]]:
        self._load()
        return [self._reader.readtext(str(path), detail=1, paragraph=False) for path in paths]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value, flags=re.UNICODE).strip()


def normalize_regions(raw: Sequence[Any], threshold: float) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            polygon = item.get("polygon") or item.get("bbox")
            text = item.get("text", "")
            confidence = float(item.get("confidence", item.get("score", 0.0)))
            language = item.get("language")
        else:
            polygon, text, confidence = item[:3]
            confidence = float(confidence)
            language = None
        if confidence < threshold:
            continue
        region: dict[str, Any] = {
            "text": normalize_text(str(text)),
            "polygon": [[float(point[0]), float(point[1])] for point in polygon],
            "confidence": confidence,
        }
        if language:
            region["language"] = str(language)
        regions.append(region)
    return regions


def run_ocr_file(
    metadata_path: Path,
    *,
    output_dir: Path = Path("data/metadata"),
    output_path: Path | None = None,
    report_path: Path | None = None,
    device: str = "auto",
    batch_size: int = 4,
    conf_threshold: float = 0.3,
    overwrite: bool = False,
    backend: OcrBackend | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be between 0 and 1")
    timer = Timer()
    records = read_jsonl(metadata_path)
    video_id = video_id_from_records(records, metadata_path.stem)
    output_path = output_path or output_dir / f"ocr_{video_id}.jsonl"
    report_path = report_path or output_dir / f"ocr_{video_id}_report.json"
    selected_device = choose_device(device)
    backend = backend or EasyOcrBackend(device=selected_device)
    processed = set() if overwrite else existing_ids(output_path, "frame_id")
    pending = [record for record in records if str(record.get("frame_id")) not in processed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = error_count = 0
    run_at = utc_now()
    batches = list(chunks(pending, batch_size))
    with output_path.open("w" if overwrite else "a", encoding="utf-8") as handle:
        for batch in iter_progress(batches, total=len(batches), description=f"ocr {video_id}"):
            batch_values: dict[int, dict[str, Any]] = {}
            valid_positions: list[int] = []
            valid_records: list[dict[str, Any]] = []
            valid_paths: list[Path] = []
            valid_image_sizes: list[list[int]] = []
            for position, record in enumerate(batch):
                try:
                    path = resolve_image_path(record, metadata_path)
                    verify_image(path)
                    with Image.open(path) as image:
                        image_size = [int(image.width), int(image.height)]
                    valid_positions.append(position)
                    valid_records.append(record)
                    valid_paths.append(path)
                    valid_image_sizes.append(image_size)
                except Exception as exc:
                    batch_values[position] = {
                        **identity(record),
                        **processing_fields(
                            pipeline="ocr",
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            status="error",
                            run_at=run_at,
                            error=str(exc),
                        ),
                        "ocr_text": "",
                        "text_regions": [],
                    }
                    error_count += 1
            for position, record, image_size, (raw, error) in zip(
                valid_positions,
                valid_records,
                valid_image_sizes,
                safe_infer(valid_paths, backend.infer),
                strict=True,
            ):
                if error:
                    value = {
                        **identity(record),
                        **processing_fields(
                            pipeline="ocr",
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            status="error",
                            run_at=run_at,
                            error=str(error),
                        ),
                        "ocr_text": "",
                        "text_regions": [],
                        "image_size": image_size,
                    }
                    error_count += 1
                else:
                    try:
                        regions = normalize_regions(raw or [], conf_threshold)
                        value = {
                            **identity(record),
                            **processing_fields(
                                pipeline="ocr",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                status="success",
                                run_at=run_at,
                            ),
                            "ocr_text": normalize_text(
                                " ".join(item["text"] for item in regions)
                            ),
                            "text_regions": regions,
                            "image_size": image_size,
                            "languages": ["vi", "en"],
                            "confidence_threshold": conf_threshold,
                        }
                        success_count += 1
                    except Exception as exc:
                        value = {
                            **identity(record),
                            **processing_fields(
                                pipeline="ocr",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                status="error",
                                run_at=run_at,
                                error=str(exc),
                            ),
                            "ocr_text": "",
                            "text_regions": [],
                            "image_size": image_size,
                        }
                        error_count += 1
                batch_values[position] = value
            for position in range(len(batch)):
                value = batch_values[position]
                append_jsonl(handle, value)

    result = report(
        pipeline="ocr",
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
    result.update({"batch_size": batch_size, "confidence_threshold": conf_threshold})
    write_json(report_path, result)
    json_log("ingestion.ocr", "completed", latency=timer.elapsed, **result)
    return result
