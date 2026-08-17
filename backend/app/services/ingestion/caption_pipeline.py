from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

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


DEFAULT_MODEL_NAME = "florence-community/Florence-2-base-ft"
DEFAULT_MODEL_REVISION = "0b03b6f15a4a211370fb204aee4e7dd48887ea37"
DEFAULT_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


class CaptionBackend(Protocol):
    model_name: str
    model_version: str
    model_revision: str | None

    def infer(self, paths: Sequence[Path]) -> Sequence[Any]: ...


@dataclass(frozen=True)
class FlorenceCaptionOutput:
    """A normalized Florence caption plus its original decoded generation."""

    caption: str
    raw_output: str


class FlorenceCaptionBackend:
    """Lazy local Florence-2 task-prompt image caption backend."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str | None = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
        cache_dir: Path | None = Path("data/model_cache/caption"),
        max_new_tokens: int = 256,
        dtype: str = "auto",
        quantization: str = "none",
        task_prompt: str = DEFAULT_TASK_PROMPT,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be auto, bfloat16, float16, or float32")
        if quantization not in {"none", "8bit", "4bit"}:
            raise ValueError("quantization must be none, 8bit, or 4bit")
        if quantization != "none":
            raise ValueError(
                "Florence-2 4/8-bit quantization is not supported or tested by "
                "this caption pipeline; use quantization='none'."
            )
        task_prompt = str(task_prompt).strip()
        if not task_prompt:
            raise ValueError("task_prompt must not be empty")
        self.model_name = model_name
        self.model_version = package_version("transformers")
        self.requested_model_revision = revision
        self.model_revision = revision
        self.device = device
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.quantization = quantization
        self.task_prompt = task_prompt
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch_dtype: Any | None = None

    def _resolve_dtype(self, torch: Any) -> Any:
        if self.dtype == "float32" or self.device == "cpu":
            return torch.float32
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("bfloat16 was requested but this CUDA device does not support it")
            return torch.bfloat16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Florence-2 captioning requires torch and a Transformers release "
                "with native Florence-2 support. Install requirements.txt."
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(self.cache_dir)
        if self.model_revision:
            kwargs["revision"] = self.model_revision
        # transformers>=5.2 has native Florence-2 support, so executing model-repo
        # Python is unnecessary. Keep this identical for processor and model.
        kwargs["trust_remote_code"] = False
        self._processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        self._torch_dtype = self._resolve_dtype(torch)
        model_kwargs = dict(kwargs)
        model_kwargs["dtype"] = self._torch_dtype
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self._model = self._model.to(self.device)
        self._model.eval()
        resolved = getattr(self._model.config, "_commit_hash", None)
        if resolved:
            self.model_revision = str(resolved)

    def infer(self, paths: Sequence[Path]) -> Sequence[FlorenceCaptionOutput]:
        self._load()
        import torch

        images: list[Image.Image] = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                images.append(image)
            image_sizes = [image.size for image in images]
            inputs = self._processor(
                text=[self.task_prompt] * len(images),
                images=images,
                return_tensors="pt",
                padding=True,
            )
            model_device = getattr(self._model, "device", torch.device(self.device))
            inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            for key, value in list(inputs.items()):
                if (
                    self.device == "cuda"
                    and hasattr(value, "is_floating_point")
                    and value.is_floating_point()
                ):
                    inputs[key] = value.to(dtype=self._torch_dtype)
            with torch.inference_mode():
                tokens = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=3,
                )
            decoded = self._processor.batch_decode(
                tokens,
                skip_special_tokens=False,
            )
            if len(decoded) != len(images):
                raise RuntimeError(
                    f"Florence-2 decoded {len(decoded)} outputs for {len(images)} images."
                )
            outputs: list[FlorenceCaptionOutput] = []
            for raw_output, image_size in zip(decoded, image_sizes, strict=True):
                processed = self._processor.post_process_generation(
                    raw_output,
                    task=self.task_prompt,
                    image_size=image_size,
                )
                if not isinstance(processed, dict) or self.task_prompt not in processed:
                    raise ValueError(
                        "Florence-2 post-processing did not return the configured "
                        f"task key {self.task_prompt!r}."
                    )
                caption = " ".join(str(processed[self.task_prompt]).split()).strip()
                if not caption:
                    raise ValueError("Florence-2 returned an empty English caption")
                outputs.append(
                    FlorenceCaptionOutput(
                        caption=caption,
                        raw_output=str(raw_output).strip(),
                    )
                )
            return outputs
        finally:
            for image in images:
                image.close()


def parse_caption_output(raw: Any) -> dict[str, Any]:
    """Adapt Florence task output to the stable downstream caption schema."""
    if isinstance(raw, FlorenceCaptionOutput):
        caption = " ".join(raw.caption.split()).strip()
        raw_output = raw.raw_output.strip()
    else:
        # A plain-string path keeps injected/test backends backward compatible.
        caption = " ".join(str(raw).split()).strip()
        raw_output = str(raw).strip()
    if not caption:
        raise ValueError("caption must not be empty")
    return {
        "caption": caption,
        "structured_caption": None,
        "caption_parse_status": "success",
        "raw_caption_output": raw_output,
    }


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
    batch_size: int = 2,
    overwrite: bool = False,
    include_segment_caption: bool = False,
    backend: CaptionBackend | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str | None = DEFAULT_MODEL_REVISION,
    max_new_tokens: int = 256,
    dtype: str = "auto",
    quantization: str = "none",
    model_cache_dir: Path = Path("data/model_cache/caption"),
    task_prompt: str = DEFAULT_TASK_PROMPT,
    prompt: str | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    timer = Timer()
    records = read_jsonl(metadata_path)
    video_id = video_id_from_records(records, metadata_path.stem)
    output_path = output_path or output_dir / f"captions_{video_id}.jsonl"
    report_path = report_path or output_dir / f"captions_{video_id}_report.json"
    selected_device = choose_device(device)
    backend_was_supplied = backend is not None
    if prompt is not None:
        task_prompt = prompt
    backend = backend or FlorenceCaptionBackend(
        model_name=model_name,
        revision=revision,
        device=selected_device,
        cache_dir=model_cache_dir,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        quantization=quantization,
        task_prompt=task_prompt,
    )
    requested_revision = getattr(backend, "requested_model_revision", None)
    if requested_revision is None:
        requested_revision = (
            getattr(backend, "model_revision", None)
            if backend_was_supplied
            else revision
        )
    processed, stale = resumable_ids(
        output_path,
        "frame_id",
        model_name=backend.model_name,
        requested_model_revision=requested_revision,
    )
    if overwrite:
        processed, stale = set(), True
    pending = [record for record in records if str(record.get("frame_id")) not in processed]
    mode = "w" if stale else "a"
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
                            model_revision=getattr(backend, "model_revision", None),
                            requested_model_revision=requested_revision,
                            status="error",
                            run_at=run_at,
                            error=str(exc),
                        ),
                        "caption": "",
                        "structured_caption": None,
                        "caption_parse_status": "error",
                        "raw_caption_output": "",
                        "caption_language": "en",
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
                            pipeline="caption",
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            model_revision=getattr(backend, "model_revision", None),
                            requested_model_revision=requested_revision,
                            status="error",
                            run_at=run_at,
                            error=str(error),
                        ),
                        "caption": "",
                        "structured_caption": None,
                        "caption_parse_status": "error",
                        "raw_caption_output": "",
                        "caption_language": "en",
                    }
                    error_count += 1
                else:
                    try:
                        parsed = parse_caption_output(raw)
                    except (TypeError, ValueError) as exc:
                        value = {
                            **identity(record),
                            **processing_fields(
                                pipeline="caption",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                model_revision=getattr(backend, "model_revision", None),
                                requested_model_revision=requested_revision,
                                status="error",
                                run_at=run_at,
                                error=str(exc),
                            ),
                            "caption": "",
                            "structured_caption": None,
                            "caption_parse_status": "error",
                            "raw_caption_output": str(raw).strip(),
                            "caption_language": "en",
                        }
                        error_count += 1
                    else:
                        value = {
                            **identity(record),
                            **processing_fields(
                                pipeline="caption",
                                model_name=backend.model_name,
                                model_version=backend.model_version,
                                model_revision=getattr(backend, "model_revision", None),
                                requested_model_revision=requested_revision,
                                status="success",
                                run_at=run_at,
                            ),
                            **parsed,
                            "caption_language": "en",
                        }
                        success_count += 1
                batch_values[position] = value
            for position in range(len(batch)):
                value = batch_values[position]
                append_jsonl(handle, value)
                generated.append(value)

    if include_segment_caption and generated:
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
        model_revision=getattr(backend, "model_revision", None),
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
            "requested_model_revision": requested_revision,
            "resolved_model_revision": getattr(backend, "model_revision", None),
            "model_cache_dir": str(getattr(backend, "cache_dir", model_cache_dir)),
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "dtype": dtype,
            "quantization": quantization,
            "segment_caption_enabled": include_segment_caption,
            "task_prompt": task_prompt,
            "prompt": task_prompt,
        }
    )
    write_json(report_path, result)
    json_log("ingestion.caption", "completed", latency=timer.elapsed, **result)
    return result
