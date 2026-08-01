from __future__ import annotations

from collections import defaultdict
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


DEFAULT_MODEL_NAME = "Salesforce/blip-image-captioning-base"


class CaptionBackend(Protocol):
    model_name: str
    model_version: str

    def infer(self, paths: Sequence[Path]) -> Sequence[str]: ...


class BlipCaptionBackend:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        cache_dir: Path | None = Path("data/model_cache/caption"),
        max_new_tokens: int = 60,
    ) -> None:
        self.model_name = model_name
        self.model_version = package_version("transformers")
        self.device = device
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Captioning requires torch, Pillow and transformers. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(self.cache_dir)
        self._processor = BlipProcessor.from_pretrained(self.model_name, **kwargs)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = BlipForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=dtype, **kwargs
        ).to(self.device)
        self._model.eval()

    def infer(self, paths: Sequence[Path]) -> Sequence[str]:
        self._load()
        import torch

        images: list[Image.Image] = []
        try:
            for path in paths:
                images.append(Image.open(path).convert("RGB"))
            # BLIP base is an image-captioning model, not an instruction-following
            # model. Supplying a long text prompt makes the decoder reproduce that
            # prompt instead of describing the image.
            inputs = self._processor(
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            if self.device == "cuda" and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.inference_mode():
                tokens = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            values = self._processor.batch_decode(tokens, skip_special_tokens=True)
            return [value.strip() for value in values]
        finally:
            for image in images:
                image.close()


def _segment_caption(captions: Sequence[str]) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for caption in captions:
        normalized = " ".join(caption.lower().split()).rstrip(".")
        if normalized and normalized not in seen:
            unique.append(caption.strip().rstrip("."))
            seen.add(normalized)
    return ". ".join(unique) + ("." if unique else "")


def run_caption_file(
    metadata_path: Path,
    *,
    output_dir: Path = Path("data/metadata"),
    output_path: Path | None = None,
    report_path: Path | None = None,
    device: str = "auto",
    batch_size: int = 4,
    overwrite: bool = False,
    include_segment_caption: bool = False,
    backend: CaptionBackend | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    timer = Timer()
    records = read_jsonl(metadata_path)
    video_id = video_id_from_records(records, metadata_path.stem)
    output_path = output_path or output_dir / f"captions_{video_id}.jsonl"
    report_path = report_path or output_dir / f"captions_{video_id}_report.json"
    selected_device = choose_device(device)
    backend = backend or BlipCaptionBackend(model_name=model_name, device=selected_device)
    processed = set() if overwrite else existing_ids(output_path, "frame_id")
    pending = [record for record in records if str(record.get("frame_id")) not in processed]
    mode = "w" if overwrite else "a"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = error_count = 0
    generated: list[dict[str, Any]] = []
    run_at = utc_now()
    batches = list(chunks(pending, batch_size))
    with output_path.open(mode, encoding="utf-8") as handle:
        for batch in iter_progress(batches, total=len(batches), description=f"caption {video_id}"):
            batch_values: dict[int, dict[str, Any]] = {}
            valid_positions: list[int] = []
            valid_records: list[dict[str, Any]] = []
            valid_paths: list[Path] = []
            for position, record in enumerate(batch):
                try:
                    image_path = resolve_image_path(record, metadata_path)
                    verify_image(image_path)
                    valid_positions.append(position)
                    valid_records.append(record)
                    valid_paths.append(image_path)
                except Exception as exc:
                    batch_values[position] = {
                        **identity(record),
                        **processing_fields(
                            pipeline="caption",
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            status="error",
                            run_at=run_at,
                            error=str(exc),
                        ),
                        "caption": "",
                    }
                    error_count += 1
            for position, record, (caption, error) in zip(
                valid_positions,
                valid_records,
                safe_infer(valid_paths, backend.infer),
                strict=True,
            ):
                if error:
                    value = {
                        **identity(record),
                        **processing_fields(
                            pipeline="caption",
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            status="error",
                            run_at=run_at,
                            error=str(error),
                        ),
                        "caption": "",
                    }
                    error_count += 1
                else:
                    value = {
                        **identity(record),
                        **processing_fields(
                            pipeline="caption",
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            status="success",
                            run_at=run_at,
                        ),
                        "caption": str(caption).strip(),
                        "caption_language": "en",
                    }
                    success_count += 1
                batch_values[position] = value
            for position in range(len(batch)):
                value = batch_values[position]
                append_jsonl(handle, value)
                generated.append(value)

    if include_segment_caption and generated:
        # Rewrite only this video's separate caption artifact; source keyframe metadata is untouched.
        all_output = read_jsonl(output_path)
        grouped: dict[str, list[str]] = defaultdict(list)
        for value in all_output:
            group = str(value.get("segment_id") or value.get("shot_id") or "")
            if value.get("status") == "success" and value.get("caption"):
                grouped[group].append(str(value["caption"]))
        for value in all_output:
            group = str(value.get("segment_id") or value.get("shot_id") or "")
            value["segment_caption"] = _segment_caption(grouped.get(group, []))
        with output_path.open("w", encoding="utf-8") as handle:
            for value in all_output:
                append_jsonl(handle, value)

    result = report(
        pipeline="caption",
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
    result["batch_size"] = batch_size
    result["segment_caption_enabled"] = include_segment_caption
    write_json(report_path, result)
    json_log("ingestion.caption", "completed", latency=timer.elapsed, **result)
    return result
