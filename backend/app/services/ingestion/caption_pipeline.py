from __future__ import annotations

import json
import re
from collections import defaultdict
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


DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_REVISION = "c202236"
DEFAULT_PROMPT = """Describe this video keyframe specifically for multimedia retrieval.

Focus only on visually observable information.

Extract:
1. Scene/environment
2. People
3. Clothing and visual attributes
4. Main objects
5. Actions
6. Spatial relationships
7. Important colors
8. Visible text only when clearly readable
9. One concise retrieval-oriented caption

Do not infer identity, intention, emotion, location, or events that are not visibly supported.
Use exactly this schema:
{"scene":"string","people":[{"type":"string","attributes":["string"]}],"objects":["string"],"actions":["string"],"relationships":["string"],"colors":["string"],"visible_text":["string"],"caption":"string"}
Return valid JSON only."""

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
_STRUCTURED_KEYS = (
    "scene",
    "people",
    "objects",
    "actions",
    "relationships",
    "colors",
    "visible_text",
    "caption",
)


class CaptionBackend(Protocol):
    model_name: str
    model_version: str
    model_revision: str | None

    def infer(self, paths: Sequence[Path]) -> Sequence[str]: ...


class QwenCaptionBackend:
    """Lazy local Qwen3.5 multimodal caption backend."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str | None = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
        cache_dir: Path | None = Path("data/model_cache/caption"),
        max_new_tokens: int = 384,
        dtype: str = "auto",
        quantization: str = "none",
        prompt: str = DEFAULT_PROMPT,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be auto, bfloat16, float16, or float32")
        if quantization not in {"none", "8bit", "4bit"}:
            raise ValueError("quantization must be none, 8bit, or 4bit")
        if device == "cpu" and quantization != "none":
            raise ValueError("bitsandbytes quantization requires CUDA")
        self.model_name = model_name
        self.model_version = package_version("transformers")
        self.requested_model_revision = revision
        self.model_revision = revision
        self.device = device
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.quantization = quantization
        self.prompt = prompt
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
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.5 captioning requires torch and a Transformers release with "
                "AutoModelForMultimodalLM support. Install requirements.txt."
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(self.cache_dir)
        if self.model_revision:
            kwargs["revision"] = self.model_revision
        self._processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        self._torch_dtype = self._resolve_dtype(torch)
        model_kwargs = dict(kwargs)
        model_kwargs["torch_dtype"] = self._torch_dtype
        if self.quantization != "none":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("4/8-bit captioning requires bitsandbytes support") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.quantization == "4bit",
                load_in_8bit=self.quantization == "8bit",
                bnb_4bit_compute_dtype=self._torch_dtype,
            )
            model_kwargs["device_map"] = "auto"
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        if self.quantization == "none":
            self._model = self._model.to(self.device)
        self._model.eval()
        resolved = getattr(self._model.config, "_commit_hash", None)
        if resolved:
            self.model_revision = str(resolved)

    def infer(self, paths: Sequence[Path]) -> Sequence[str]:
        self._load()
        import torch

        images: list[Image.Image] = []
        try:
            messages: list[list[dict[str, Any]]] = []
            for path in paths:
                image = Image.open(path).convert("RGB")
                images.append(image)
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": self.prompt},
                            ],
                        }
                    ]
                )
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
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
            input_length = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                tokens = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generated = tokens[:, input_length:]
            return [
                value.strip()
                for value in self._processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
            ]
        finally:
            for image in images:
                image.close()


def _clean_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return " ".join(value.split()).strip()


def _clean_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return [cleaned for item in value if (cleaned := " ".join(item.split()).strip())]


def _validate_structured_caption(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("caption output must be a JSON object")
    missing = [key for key in _STRUCTURED_KEYS if key not in value]
    if missing:
        raise ValueError("missing structured caption fields: " + ", ".join(missing))
    people_raw = value["people"]
    if not isinstance(people_raw, list):
        raise TypeError("people must be a list")
    people: list[dict[str, Any]] = []
    for index, person in enumerate(people_raw):
        if not isinstance(person, dict):
            raise TypeError(f"people[{index}] must be an object")
        people.append(
            {
                "type": _clean_string(person.get("type", ""), f"people[{index}].type"),
                "attributes": _clean_string_list(
                    person.get("attributes", []),
                    f"people[{index}].attributes",
                ),
            }
        )
    structured = {
        "scene": _clean_string(value["scene"], "scene"),
        "people": people,
        "objects": _clean_string_list(value["objects"], "objects"),
        "actions": _clean_string_list(value["actions"], "actions"),
        "relationships": _clean_string_list(value["relationships"], "relationships"),
        "colors": _clean_string_list(value["colors"], "colors"),
        "visible_text": _clean_string_list(value["visible_text"], "visible_text"),
        "caption": _clean_string(value["caption"], "caption"),
    }
    if not structured["caption"]:
        raise ValueError("caption must not be empty")
    return structured


def _strip_markdown_fence(text: str) -> str:
    match = _FENCE.match(text)
    return match.group(1).strip() if match else text.strip()


def _fallback_caption(cleaned: str) -> str:
    match = re.search(r'''["']caption["']\s*:\s*["']([^"']+)''', cleaned, re.IGNORECASE)
    if match:
        return " ".join(match.group(1).split())
    return " ".join(cleaned.split())


def parse_caption_output(raw: str) -> dict[str, Any]:
    """Parse Qwen JSON without losing useful text on malformed generations."""
    cleaned = _strip_markdown_fence(str(raw))
    try:
        structured = _validate_structured_caption(json.loads(cleaned))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "caption": _fallback_caption(cleaned),
            "structured_caption": None,
            "caption_parse_status": "fallback",
            "caption_parse_error": str(exc),
            "raw_caption_output": str(raw).strip(),
        }
    return {
        "caption": structured["caption"],
        "structured_caption": structured,
        "caption_parse_status": "success",
        "raw_caption_output": str(raw).strip(),
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
    max_new_tokens: int = 384,
    dtype: str = "auto",
    quantization: str = "none",
    model_cache_dir: Path = Path("data/model_cache/caption"),
    prompt: str = DEFAULT_PROMPT,
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
    backend = backend or QwenCaptionBackend(
        model_name=model_name,
        revision=revision,
        device=selected_device,
        cache_dir=model_cache_dir,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        quantization=quantization,
        prompt=prompt,
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
                    }
                    error_count += 1
                else:
                    parsed = parse_caption_output(str(raw))
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
            "prompt": prompt,
        }
    )
    write_json(report_path, result)
    json_log("ingestion.caption", "completed", latency=timer.elapsed, **result)
    return result
