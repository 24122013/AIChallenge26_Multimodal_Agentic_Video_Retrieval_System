from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from PIL import Image

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


DEFAULT_DETECTION_MODEL = "PP-OCRv5_server_det"
DEFAULT_RECOGNITION_MODEL = "latin_PP-OCRv5_mobile_rec"
DEFAULT_MODEL_NAME = f"{DEFAULT_DETECTION_MODEL}+{DEFAULT_RECOGNITION_MODEL}"
DEFAULT_MODEL_REVISION = "PP-OCRv5"
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
_VIETNAMESE_MARKS = frozenset("ăâđêôơưĂÂĐÊÔƠƯ")


class OcrBackend(Protocol):
    model_name: str
    model_version: str
    model_revision: str | None

    def infer(self, paths: Sequence[Path]) -> Sequence[Any]: ...


class PaddleOcrBackend:
    """Lazy local PP-OCRv5 detector + Latin recognizer pipeline."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        detection_model: str = DEFAULT_DETECTION_MODEL,
        recognition_model: str = DEFAULT_RECOGNITION_MODEL,
        revision: str = DEFAULT_MODEL_REVISION,
        cache_dir: Path = Path("data/model_cache/ocr"),
    ) -> None:
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.model_name = f"{detection_model}+{recognition_model}"
        self.model_version = package_version("paddleocr")
        self.model_revision = revision
        self.device = device
        self.cache_dir = cache_dir
        self._pipeline: Any | None = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Set cache roots before importing PaddleOCR/PaddleX so their module
        # initialization observes the cache_dir explicitly selected by this
        # run instead of writing a second copy under a stale process setting.
        cache_root = str(self.cache_dir.resolve())
        os.environ["PADDLE_PDX_CACHE_HOME"] = cache_root
        os.environ["PADDLE_HOME"] = cache_root
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PP-OCRv5 requires paddleocr and a compatible PaddlePaddle "
                "inference wheel. See the repository installation notes."
            ) from exc
        self._pipeline = PaddleOCR(
            text_detection_model_name=self.detection_model,
            text_recognition_model_name=self.recognition_model,
            device="gpu:0" if self.device == "cuda" else "cpu",
            # PaddlePaddle 3.3.x has a PIR -> oneDNN regression that fails on
            # PP-OCRv5 ArrayAttribute<DoubleAttribute> values.  PaddleX may
            # still route individual operators through its CPU predictor even
            # when the pipeline device is CUDA, so make the safe backend choice
            # explicit instead of relying on a process environment flag.
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def infer(self, paths: Sequence[Path]) -> Sequence[Any]:
        self._load()
        # The official general OCR pipeline invokes recognition only for the
        # regions returned by its detector, so empty detections do not run the
        # recognition model.
        return list(self._pipeline.predict([str(path) for path in paths]))


def choose_ocr_device(requested: str) -> str:
    """Resolve the OCR device without importing Torch in an isolated worker."""

    if os.environ.get("OCR_ISOLATED_PADDLE_RUNTIME") != "1":
        return choose_device(requested)
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"

    import paddle

    available = bool(
        paddle.device.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    )
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested, but Paddle cannot access a CUDA device.")
    return "cuda" if available else "cpu"


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", str(value))
    normalized = "".join(
        character
        for character in normalized
        if character in "\t\n\r" or not unicodedata.category(character).startswith("C")
    )
    return _WHITESPACE.sub(" ", normalized).strip()


def unaccent_text(value: str) -> str:
    value = normalize_text(value).replace("đ", "d").replace("Đ", "D")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )


def infer_language(value: str) -> str | None:
    normalized = unicodedata.normalize("NFD", value)
    if any(character in _VIETNAMESE_MARKS for character in value) or any(
        unicodedata.category(character) == "Mn" for character in normalized
    ):
        return "vi"
    if any(character.isalpha() and character.isascii() for character in value):
        return "en"
    return None


def _result_mapping(raw: Any) -> Mapping[str, Any] | None:
    if isinstance(raw, Mapping):
        value: Any = raw
    else:
        value = getattr(raw, "json", None)
        if callable(value):
            value = value()
        if value is None:
            value = getattr(raw, "res", None)
    if not isinstance(value, Mapping):
        return None
    nested = value.get("res")
    return nested if isinstance(nested, Mapping) else value


def _paddle_items(raw: Any) -> list[dict[str, Any]] | None:
    value = _result_mapping(raw)
    if value is None:
        return None
    polygons = value.get("rec_polys")
    if polygons is None:
        polygons = value.get("dt_polys")
    if polygons is None:
        polygons = []
    texts = value.get("rec_texts")
    if texts is None:
        texts = []
    scores = value.get("rec_scores")
    if scores is None:
        scores = []
    if hasattr(polygons, "tolist"):
        polygons = polygons.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return [
        {"polygon": polygon, "text": text, "confidence": score}
        for polygon, text, score in zip(polygons, texts, scores, strict=True)
    ]


def normalize_regions(raw: Any, threshold: float) -> list[dict[str, Any]]:
    paddle_items = _paddle_items(raw)
    items: Sequence[Any]
    if paddle_items is not None:
        items = paddle_items
    elif raw is None:
        items = ()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = raw
    else:
        raise TypeError("unsupported PP-OCRv5 result payload")

    regions: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            polygon = item.get("polygon")
            if polygon is None:
                polygon = item.get("bbox")
            raw_text = str(item.get("raw_text", item.get("text", "")))
            confidence = float(item.get("confidence", item.get("score", 0.0)))
            language = item.get("language")
        else:
            polygon, raw_text, confidence = item[:3]
            raw_text = str(raw_text)
            confidence = float(confidence)
            language = None
        if hasattr(polygon, "tolist"):
            polygon = polygon.tolist()
        if confidence < threshold:
            continue
        if not isinstance(polygon, Sequence) or len(polygon) < 4:
            raise ValueError("OCR polygon must contain at least four points")
        text = normalize_text(raw_text)
        if not text:
            continue
        region: dict[str, Any] = {
            "text": text,
            "normalized_text": text,
            "unaccented_text": unaccent_text(text),
            "raw_text": raw_text,
            "polygon": [
                [float(point[0]), float(point[1])]
                for point in polygon
            ],
            "confidence": confidence,
        }
        resolved_language = str(language) if language else infer_language(text)
        if resolved_language:
            region["language"] = resolved_language
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
    detection_model: str = DEFAULT_DETECTION_MODEL,
    recognition_model: str = DEFAULT_RECOGNITION_MODEL,
    revision: str = DEFAULT_MODEL_REVISION,
    model_cache_dir: Path = Path("data/model_cache/ocr"),
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
    selected_device = choose_ocr_device(device)
    backend = backend or PaddleOcrBackend(
        device=selected_device,
        detection_model=detection_model,
        recognition_model=recognition_model,
        revision=revision,
        cache_dir=model_cache_dir,
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
                            model_revision=requested_revision,
                            status="error",
                            run_at=run_at,
                            error=str(exc),
                        ),
                        "ocr_text": "",
                        "ocr_text_unaccented": "",
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
                            model_revision=requested_revision,
                            status="error",
                            run_at=run_at,
                            error=str(error),
                        ),
                        "ocr_text": "",
                        "ocr_text_unaccented": "",
                        "text_regions": [],
                        "image_size": image_size,
                    }
                    error_count += 1
                else:
                    try:
                        regions = normalize_regions(raw, conf_threshold)
                        ocr_text = normalize_text(" ".join(item["text"] for item in regions))
                        value = {
                            **identity(record),
                            **processing_fields(
                                pipeline="ocr",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                model_revision=requested_revision,
                                status="success",
                                run_at=run_at,
                            ),
                            "ocr_text": ocr_text,
                            "ocr_text_normalized": ocr_text,
                            "ocr_text_unaccented": unaccent_text(ocr_text),
                            "raw_ocr_text": " ".join(item["raw_text"] for item in regions),
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
                                model_revision=requested_revision,
                                status="error",
                                run_at=run_at,
                                error=str(exc),
                            ),
                            "ocr_text": "",
                            "ocr_text_unaccented": "",
                            "text_regions": [],
                            "image_size": image_size,
                        }
                        error_count += 1
                batch_values[position] = value
            for position in range(len(batch)):
                append_jsonl(handle, batch_values[position])

    result = report(
        pipeline="ocr",
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
            "detection_model": getattr(backend, "detection_model", detection_model),
            "recognition_model": getattr(backend, "recognition_model", recognition_model),
            "model_cache_dir": str(getattr(backend, "cache_dir", model_cache_dir)),
            "languages": ["vi", "en"],
        }
    )
    write_json(report_path, result)
    json_log("ingestion.ocr", "completed", latency=timer.elapsed, **result)
    return result
