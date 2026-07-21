"""caption_pipeline — sinh caption cho mỗi keyframe.

Kiến trúc backend pluggable:
- StubCaptionBackend: caption xác định (deterministic) từ record — chạy được ngay
  cả khi chưa có model, phục vụ nguyên tắc "End-to-End trước, tối ưu sau".
- Qwen2VLCaptionBackend: cắm model thật (Qwen2.5-VL) qua lazy import, dùng khi
  chạy trên máy có GPU + transformers.

Output JSONL mỗi dòng: {frame_id, caption, caption_model}.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.app.models.metadata import Caption
from backend.app.services.ingestion._base_pipeline import run_frame_pipeline
from backend.app.services.ingestion.scheme_validator import validate_caption

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class StubCaptionBackend:
    """Caption giả lập, xác định — không cần model.

    Sinh mô tả dựa trên metadata sẵn có (video_id, shot, timestamp). Hữu ích để
    validate luồng end-to-end và test; máy GPU sẽ thay bằng backend thật.
    """

    name = "stub-caption-v1"

    def process(self, image, record: dict) -> dict:
        video_id = record.get("video_id", "unknown")
        ts = record.get("timestamp", 0.0)
        shot = record.get("shot_id", "")
        shape_hint = ""
        if image is not None:
            try:
                h, w = image.shape[:2]
                shape_hint = f", frame {w}x{h}"
            except Exception:
                shape_hint = ""
        caption = f"keyframe from {video_id} at {float(ts):.2f}s (shot {shot}){shape_hint}"
        return {"caption": caption, "caption_model": self.name}


class Qwen2VLCaptionBackend:
    """Backend caption thật dùng Qwen2.5-VL (lazy import transformers/torch).

    Chỉ import model khi khởi tạo => môi trường không có GPU/transformers vẫn
    import được module này. Prompt mặc định yêu cầu mô tả ngắn gọn 1 câu.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        prompt: str = "Describe this image in one concise English sentence.",
        device: str = "auto",
        max_new_tokens: int = 64,
    ) -> None:
        self.name = model_id.split("/")[-1].lower()
        self.model_id = model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._device = device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        logger.info("Loading caption model %s on %s", self.model_id, device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=device if device == "cuda" else None,
        )
        if device == "cpu":
            self._model = self._model.to("cpu")

    def process(self, image, record: dict) -> dict:
        self._ensure_loaded()
        from PIL import Image  # type: ignore

        if image is None:
            return {"caption": "", "caption_model": self.name}
        pil = Image.fromarray(image[:, :, ::-1])  # BGR->RGB
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[pil], return_tensors="pt").to(
            self._device
        )
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        caption = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
        return {"caption": caption, "caption_model": self.name}


def build_backend(backend: str = "stub", **kwargs):
    """Factory chọn backend theo tên."""
    if backend in ("stub", "dummy", "none"):
        return StubCaptionBackend()
    if backend in ("qwen", "qwen2.5-vl", "qwen2_5_vl"):
        return Qwen2VLCaptionBackend(**kwargs)
    raise ValueError(f"caption backend không hỗ trợ: {backend!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_caption_pipeline(
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
    """Chạy caption pipeline. Trả về PipelineReport."""
    be = build_backend(backend, **(backend_kwargs or {}))
    return run_frame_pipeline(
        pipeline_name="caption",
        metadata_path=metadata_path,
        output_path=output_path,
        backend=be,
        validate_fn=validate_caption,
        image_base_dir=image_base_dir,
        require_image=require_image,
        report_path=report_path,
        limit=limit,
    )


__all__ = [
    "StubCaptionBackend",
    "Qwen2VLCaptionBackend",
    "build_backend",
    "run_caption_pipeline",
    "Caption",
]
