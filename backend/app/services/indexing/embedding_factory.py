"""embedding_factory — chọn & nạp model embedding ảnh/text (Team P2).

Gói các backend embedding sau một interface chung để `build_clip_index.py` và
`build_sigclip_index.py` dùng lại cùng một luồng encode. Tất cả backend hiện đùng
thư viện `open_clip` (hỗ trợ OpenCLIP, CLIP OpenAI, và SigLIP) nên chỉ khác nhau ở
`model_name` / `pretrained`.

Model được nạp *lazy* (chỉ import torch/open_clip khi encode lần đầu) nên import
module này không kéo theo thư viện nặng.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Registry: tên thân thiện -> (model_name, pretrained) của open_clip
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # OpenCLIP (baseline hiện tại)
    "openclip": ("ViT-B-16", "laion2b_s34b_b88k"),
    "openclip-vit-b16": ("ViT-B-16", "laion2b_s34b_b88k"),
    "openclip-vit-l14": ("ViT-L-14", "laion2b_s32b_b82k"),
    # CLIP gốc của OpenAI (qua open_clip)
    "clip": ("ViT-B-16", "openai"),
    "clip-vit-b32": ("ViT-B-32", "openai"),
    # SigLIP
    "sigclip": ("ViT-B-16-SigLIP", "webli"),
    "siglip": ("ViT-B-16-SigLIP", "webli"),
    "siglip-l16": ("ViT-L-16-SigLIP-256", "webli"),
}


def resolve_model_spec(
    kind: str,
    model_name: str | None = None,
    pretrained: str | None = None,
) -> tuple[str, str]:
    """Trả về (model_name, pretrained). Ưu tiên tham số truyền thẳng, sau đó registry."""
    if model_name and pretrained:
        return model_name, pretrained
    key = kind.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"kind embedding không hỗ trợ: {kind!r}. "
            f"Chọn một trong {sorted(MODEL_REGISTRY)} hoặc truyền --model-name/--pretrained."
        )
    reg_name, reg_pretrained = MODEL_REGISTRY[key]
    return model_name or reg_name, pretrained or reg_pretrained


# ---------------------------------------------------------------------------
# Backend embedding
# ---------------------------------------------------------------------------

class OpenClipEmbeddingModel:
    """Wrapper open_clip cho cả image & text embedding (lazy import)."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str = "auto",
        model_cache_dir: Path | None = None,
        use_autocast: bool = True,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.model_tag = f"{model_name}_{pretrained}".lower().replace("/", "-")
        self.requested_device = device
        self.model_cache_dir = model_cache_dir
        self.use_autocast = use_autocast
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._torch = None
        self._device = ""
        self._vector_dim: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip  # type: ignore
        import torch  # type: ignore

        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=device,
            cache_dir=self.model_cache_dir.as_posix() if self.model_cache_dir else None,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._torch = torch
        self._device = device

    @property
    def device(self) -> str:
        self._load()
        return self._device

    @property
    def vector_dim(self) -> int:
        if self._vector_dim is None:
            # encode 1 ảnh giả để biết dim, hoặc suy từ text
            self._load()
            tokens = self._tokenizer(["probe"]).to(self._device)
            with self._torch.no_grad():
                feats = self._model.encode_text(tokens)
            self._vector_dim = int(feats.shape[1])
        return self._vector_dim

    def _autocast(self):
        from contextlib import nullcontext

        if self.use_autocast and self._device.startswith("cuda"):
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return nullcontext()

    def encode_images(self, images: Iterable) -> np.ndarray:
        """Encode list ảnh PIL -> mảng float32 đã L2-normalize (N, D)."""
        self._load()
        tensors = [self._preprocess(img.convert("RGB")) for img in images]
        if not tensors:
            return np.empty((0, self.vector_dim), dtype="float32")
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.no_grad(), self._autocast():
            feats = self._model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.detach().cpu().numpy().astype("float32")
        self._vector_dim = int(arr.shape[1])
        return arr

    def encode_text(self, queries: list[str]) -> np.ndarray:
        self._load()
        tokens = self._tokenizer(queries).to(self._device)
        with self._torch.no_grad():
            feats = self._model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype("float32")


def create_embedding_model(
    kind: str = "openclip",
    *,
    model_name: str | None = None,
    pretrained: str | None = None,
    device: str = "auto",
    model_cache_dir: Path | None = None,
    use_autocast: bool = True,
) -> OpenClipEmbeddingModel:
    """Factory tạo model embedding theo `kind` hoặc model_name/pretrained tường minh."""
    resolved_name, resolved_pretrained = resolve_model_spec(kind, model_name, pretrained)
    return OpenClipEmbeddingModel(
        model_name=resolved_name,
        pretrained=resolved_pretrained,
        device=device,
        model_cache_dir=model_cache_dir,
        use_autocast=use_autocast,
    )


# ---------------------------------------------------------------------------
# Encode keyframes -> artifacts (dùng chung cho build_clip / build_sigclip)
# ---------------------------------------------------------------------------

@dataclass
class EncodeArtifacts:
    embeddings: np.ndarray
    embedding_records: list[dict]
    skipped_records: list[dict]
    benchmark: dict


def encode_keyframe_records(
    records: list[dict],
    model: OpenClipEmbeddingModel,
    *,
    batch_size: int = 32,
    output_model_tag: str | None = None,
) -> EncodeArtifacts:
    """Encode list keyframe record (mỗi record có keyframe_path) thành embeddings.

    Định dạng embedding_records khớp với build_openclip_index để build_faiss_index
    dùng lại được ngay.
    """
    from PIL import Image  # type: ignore

    started = time.perf_counter()
    model_tag = output_model_tag or model.model_tag
    batches: list[np.ndarray] = []
    embedding_records: list[dict] = []
    skipped: list[dict] = []
    image_load_sec = 0.0
    inference_sec = 0.0

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        images = []
        valid = []
        t0 = time.perf_counter()
        for record in chunk:
            try:
                with Image.open(record["keyframe_path"]) as im:
                    images.append(im.convert("RGB").copy())
                valid.append(record)
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    {
                        "frame_id": record.get("frame_id", ""),
                        "video_id": record.get("video_id", ""),
                        "keyframe_path": record.get("keyframe_path", ""),
                        "skip_reason": "image_load_error",
                        "error": str(exc),
                    }
                )
        image_load_sec += time.perf_counter() - t0
        if not images:
            continue

        t1 = time.perf_counter()
        feats = model.encode_images(images)
        inference_sec += time.perf_counter() - t1
        batches.append(feats)

        for record in valid:
            idx = len(embedding_records)
            embedding_records.append(
                {
                    "embedding_id": f"EMB_{record['frame_id']}",
                    "frame_id": record["frame_id"],
                    "video_id": record["video_id"],
                    "shot_id": record.get("shot_id", ""),
                    "segment_id": record.get("segment_id", ""),
                    "shot_index": record.get("shot_index"),
                    "shot_start": record.get("shot_start"),
                    "shot_end": record.get("shot_end"),
                    "timestamp": record["timestamp"],
                    "timestamp_source": record.get("timestamp_source"),
                    "timestamp_confidence": record.get("timestamp_confidence"),
                    "frame_index": record.get("frame_index"),
                    "keyframe_path": record["keyframe_path"],
                    "thumbnail_path": record.get("thumbnail_path", record["keyframe_path"]),
                    "source_video_path": record.get("source_video_path") or record.get("video_path"),
                    "video_path": record.get("video_path") or record.get("source_video_path"),
                    "selection_reason": record.get("selection_reason"),
                    "model_name": model_tag,
                    "vector_dim": int(feats.shape[1]),
                    "embedding_index": idx,
                }
            )

    if not batches:
        raise ValueError("Không tạo được embedding nào (tất cả ảnh lỗi?).")

    embeddings = np.concatenate(batches, axis=0)
    runtime = time.perf_counter() - started
    benchmark = {
        "model_name": model_tag,
        "device": model.device,
        "input_record_count": len(records),
        "encoded_count": len(embedding_records),
        "skipped_count": len(skipped),
        "embedding_shape": list(embeddings.shape),
        "vector_dim": int(embeddings.shape[1]),
        "batch_size": batch_size,
        "runtime_sec": round(runtime, 3),
        "image_load_sec": round(image_load_sec, 3),
        "inference_sec": round(inference_sec, 3),
        "throughput_img_per_sec": round(len(embedding_records) / max(runtime, 1e-9), 3),
    }
    return EncodeArtifacts(embeddings, embedding_records, skipped, benchmark)


def save_encode_artifacts(
    artifacts: EncodeArtifacts,
    *,
    embeddings_path: Path,
    embedding_metadata_path: Path,
    skipped_path: Path | None = None,
    benchmark_path: Path | None = None,
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, artifacts.embeddings)
    _write_jsonl(embedding_metadata_path, artifacts.embedding_records)
    if skipped_path is not None:
        _write_jsonl(skipped_path, artifacts.skipped_records)
    if benchmark_path is not None:
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        with benchmark_path.open("w", encoding="utf-8") as f:
            json.dump(artifacts.benchmark, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = [
    "MODEL_REGISTRY",
    "resolve_model_spec",
    "OpenClipEmbeddingModel",
    "create_embedding_model",
    "EncodeArtifacts",
    "encode_keyframe_records",
    "save_encode_artifacts",
]
