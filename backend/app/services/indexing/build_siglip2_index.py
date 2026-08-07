from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_MODEL_NAME = "google/siglip2-so400m-patch16-384"
ARTIFACT_TAG = "siglip2_so400m_patch16_384"
DEFAULT_MODEL_CACHE_DIR = Path("data/model_cache/siglip2")
DEFAULT_CPU_BATCH_SIZE = 4


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    return records


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def choose_device(requested_device: str) -> str:
    if requested_device not in {"auto", "cuda", "cpu"}:
        raise ValueError("--device must be one of: auto, cuda, cpu")
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return requested_device


def compute_dtype_for(device: str, use_autocast: bool) -> torch.dtype:
    if not device.startswith("cuda") or not use_autocast:
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def autocast_context(device: str, enabled: bool, compute_dtype: torch.dtype):
    if not enabled or not device.startswith("cuda"):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=compute_dtype)


def synchronize_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return (
        isinstance(exc, (RuntimeError, oom_type))
        and "out of memory" in str(exc).lower()
    )


def clear_cuda_cache(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def load_siglip2_model_processor(
    model_name: str,
    model_revision: str | None,
    device: str,
    model_cache_dir: Path | None,
):
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "Transformers with SigLIP2 support is required. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    common_kwargs: dict[str, Any] = {}
    if model_revision:
        common_kwargs["revision"] = model_revision
    if model_cache_dir:
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        common_kwargs["cache_dir"] = model_cache_dir.as_posix()

    model = AutoModel.from_pretrained(model_name, **common_kwargs)
    processor = AutoProcessor.from_pretrained(model_name, **common_kwargs)
    model.to(device)
    model.eval()
    return model, processor


def resolve_model_revision(model: Any, requested_revision: str | None) -> str:
    config = getattr(model, "config", None)
    resolved = getattr(config, "_commit_hash", None)
    return str(resolved or requested_revision or "main")


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


def expected_projection_dimensions(model: Any) -> set[int]:
    config = getattr(model, "config", None)
    candidates: list[Any] = []
    for owner in (config, getattr(config, "vision_config", None)):
        if owner is None:
            continue
        candidates.extend(
            [
                getattr(owner, "projection_dim", None),
                getattr(owner, "projection_size", None),
            ]
        )
    return {value for candidate in candidates if (value := _positive_int(candidate))}


def validate_vector_dim(model: Any, vector_dim: int) -> None:
    expected_dims = expected_projection_dimensions(model)
    if expected_dims and vector_dim not in expected_dims:
        raise ValueError(
            "SigLIP2 output dimension does not match model config: "
            f"features.shape[-1]={vector_dim}, configured projection dimensions="
            f"{sorted(expected_dims)}"
        )


def infer_input_resolution(processor: Any, model: Any) -> int | None:
    image_processor = getattr(processor, "image_processor", processor)
    size = getattr(image_processor, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(size, dict):
        for key in ("height", "width", "shortest_edge"):
            value = _positive_int(size.get(key))
            if value:
                return value
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    return _positive_int(getattr(vision_config, "image_size", None))


def _move_inputs_to_device(inputs: Any, device: str) -> dict[str, Any]:
    items = inputs.items() if hasattr(inputs, "items") else dict(inputs).items()
    return {
        key: value.to(device, non_blocking=device.startswith("cuda"))
        if hasattr(value, "to")
        else value
        for key, value in items
    }


def _feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(
        "model.get_image_features() returned an unsupported value: "
        f"{type(output).__name__}"
    )


def run_image_batch(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    device: str,
    use_autocast: bool,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if not images:
        raise ValueError("Cannot encode an empty image batch")
    if any(image.mode != "RGB" for image in images):
        raise ValueError("SigLIP2 processor must receive PIL RGB images")

    inputs = processor(images=images, return_tensors="pt")
    model_inputs = _move_inputs_to_device(inputs, device)
    with torch.inference_mode():
        with autocast_context(device, use_autocast, compute_dtype):
            features = _feature_tensor(model.get_image_features(**model_inputs))
    if features.ndim != 2 or features.shape[0] != len(images):
        raise ValueError(
            "Unexpected SigLIP2 image feature shape: "
            f"{tuple(features.shape)} for batch size {len(images)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError("SigLIP2 produced NaN or Inf image features")
    norms = features.float().norm(dim=-1, keepdim=True)
    if torch.any(norms <= 0):
        raise ValueError("SigLIP2 produced one or more zero vectors")
    return features.float() / norms


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


class KeyframeDataset(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        path = Path(record.get("keyframe_path") or "")
        try:
            image = load_rgb_image(path)
            return {"record": record, "image": image, "error": None}
        except Exception as exc:  # noqa: BLE001 - bad images are an expected skip case.
            return {"record": record, "image": None, "error": str(exc)}


def collate_keyframes(items: list[dict]) -> list[dict]:
    return items


def iter_loaded_batches(
    records: list[dict],
    batch_size: int,
    num_workers: int,
    device: str,
    prefetch_factor: int,
):
    loader_kwargs: dict[str, Any] = {
        "dataset": KeyframeDataset(records),
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_keyframes,
        "pin_memory": device.startswith("cuda"),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    yield from DataLoader(**loader_kwargs)


def sample_real_images(records: list[dict], sample_count: int = 8) -> list[Image.Image]:
    images: list[Image.Image] = []
    for record in records:
        try:
            images.append(load_rgb_image(Path(record.get("keyframe_path") or "")))
        except Exception:  # noqa: BLE001 - the main pass records bad images in skipped JSONL.
            continue
        if len(images) >= sample_count:
            break
    return images


def tune_batch_size(
    model: Any,
    processor: Any,
    sample_images: list[Image.Image],
    input_record_count: int,
    device: str,
    use_autocast: bool,
    compute_dtype: torch.dtype,
) -> tuple[int, list[dict]]:
    if not sample_images:
        raise ValueError("Cannot auto-tune batch size because no valid sample image was found")
    if input_record_count <= 0:
        raise ValueError("input_record_count must be positive")

    if not device.startswith("cuda"):
        selected = min(DEFAULT_CPU_BATCH_SIZE, input_record_count)
        return selected, [
            {
                "batch_size": selected,
                "status": "selected",
                "reason": "CPU auto mode uses a conservative fixed batch size",
            }
        ]

    results: list[dict] = []
    stable_results: list[dict] = []
    candidate = 1
    while candidate <= input_record_count:
        images = [sample_images[index % len(sample_images)] for index in range(candidate)]
        try:
            # Warm-up the exact candidate before timing it.
            warmup_features = run_image_batch(
                model, processor, images, device, use_autocast, compute_dtype
            )
            del warmup_features
            synchronize_cuda(device)

            started_at = time.perf_counter()
            features = run_image_batch(
                model, processor, images, device, use_autocast, compute_dtype
            )
            synchronize_cuda(device)
            elapsed_sec = time.perf_counter() - started_at
            del features
            result = {
                "batch_size": candidate,
                "status": "ok",
                "elapsed_sec": round(elapsed_sec, 6),
                "throughput_img_per_sec": round(candidate / max(elapsed_sec, 1e-9), 3),
            }
            results.append(result)
            stable_results.append(result)
        except BaseException as exc:
            if not is_cuda_oom(exc):
                raise
            results.append(
                {
                    "batch_size": candidate,
                    "status": "cuda_oom",
                    "error": str(exc),
                }
            )
            clear_cuda_cache(device)
            break
        finally:
            del images

        candidate *= 2

    if not stable_results:
        raise RuntimeError("CUDA OOM occurred even at batch size 1")
    selected_result = max(
        stable_results,
        key=lambda item: (item["throughput_img_per_sec"], -item["batch_size"]),
    )
    selected_result["selected"] = True
    return int(selected_result["batch_size"]), results


def _embedding_record(
    record: dict,
    embedding_index: int,
    contract: dict,
) -> dict:
    keyframe_path = record["keyframe_path"]
    return {
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
        "keyframe_path": keyframe_path,
        "thumbnail_path": record.get("thumbnail_path", keyframe_path),
        "source_video_path": record.get("source_video_path") or record.get("video_path"),
        "video_path": record.get("video_path") or record.get("source_video_path"),
        "keyframe_strategy": record.get("keyframe_strategy"),
        "selection_reason": record.get("selection_reason"),
        "embedding_index": embedding_index,
        **contract,
    }


def validate_embedding_artifacts(
    embeddings: np.ndarray,
    records: list[dict],
) -> dict:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape={embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise ValueError(f"Embeddings must be float32, got dtype={embeddings.dtype}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Embeddings contain a zero vector")
    if embeddings.shape[0] != len(records):
        raise ValueError(
            f"Embedding/metadata count mismatch: {embeddings.shape[0]} != {len(records)}"
        )
    for offset, record in enumerate(records):
        if record.get("embedding_index") != offset:
            raise ValueError(
                f"embedding_index mismatch at metadata offset {offset}: "
                f"{record.get('embedding_index')}"
            )
        if record.get("vector_dim") != embeddings.shape[1]:
            raise ValueError(
                f"vector_dim mismatch at metadata offset {offset}: "
                f"{record.get('vector_dim')} != {embeddings.shape[1]}"
            )
        if record.get("normalized") is not True:
            raise ValueError(f"normalized must be true at metadata offset {offset}")
    return {
        "vector_count": int(embeddings.shape[0]),
        "vector_dim": int(embeddings.shape[1]),
        "norm_min": float(norms.min()) if len(norms) else None,
        "norm_max": float(norms.max()) if len(norms) else None,
    }


def encode_keyframes(
    records: list[dict],
    model_name: str = DEFAULT_MODEL_NAME,
    model_revision: str | None = None,
    batch_size: str | int = "auto",
    num_workers: int = 0,
    device: str = "auto",
    use_autocast: bool = True,
    model_cache_dir: Path | None = DEFAULT_MODEL_CACHE_DIR,
    prefetch_factor: int = 2,
    model: Any | None = None,
    processor: Any | None = None,
) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    if not records:
        raise ValueError("No keyframe records were supplied")
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")

    started_at = time.perf_counter()
    resolved_device = choose_device(device)
    compute_dtype = compute_dtype_for(resolved_device, use_autocast)
    if model is None or processor is None:
        model, processor = load_siglip2_model_processor(
            model_name=model_name,
            model_revision=model_revision,
            device=resolved_device,
            model_cache_dir=model_cache_dir,
        )
    else:
        model.to(resolved_device)
        model.eval()

    resolved_revision = resolve_model_revision(model, model_revision)
    input_resolution = infer_input_resolution(processor, model)
    requested_batch_size: str | int = batch_size
    if batch_size == "auto":
        tuning_images = sample_real_images(records)
        selected_batch_size, tuning_results = tune_batch_size(
            model=model,
            processor=processor,
            sample_images=tuning_images,
            input_record_count=len(records),
            device=resolved_device,
            use_autocast=use_autocast,
            compute_dtype=compute_dtype,
        )
    else:
        selected_batch_size = int(batch_size)
        if selected_batch_size <= 0:
            raise ValueError("--batch-size must be a positive integer or 'auto'")
        selected_batch_size = min(selected_batch_size, len(records))
        tuning_results = [
            {
                "batch_size": selected_batch_size,
                "status": "selected",
                "reason": "explicit batch size requested",
            }
        ]

    if resolved_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    embedding_batches: list[np.ndarray] = []
    embedding_records: list[dict] = []
    skipped_records: list[dict] = []
    image_load_sec = 0.0
    inference_sec = 0.0
    vector_dim: int | None = None

    loaded_batch_iterator = iter(
        iter_loaded_batches(
            records=records,
            batch_size=selected_batch_size,
            num_workers=num_workers,
            device=resolved_device,
            prefetch_factor=prefetch_factor,
        )
    )
    batch_number = 0
    while True:
        image_load_started_at = time.perf_counter()
        try:
            loaded_items = next(loaded_batch_iterator)
        except StopIteration:
            break
        image_load_sec += time.perf_counter() - image_load_started_at
        batch_number += 1
        valid_images: list[Image.Image] = []
        valid_records: list[dict] = []
        for item in loaded_items:
            record = item["record"]
            if item["error"] is not None:
                skipped_records.append(
                    {
                        "frame_id": record.get("frame_id", ""),
                        "video_id": record.get("video_id", ""),
                        "keyframe_path": record.get("keyframe_path", ""),
                        "skip_reason": "image_load_error",
                        "error": item["error"],
                    }
                )
                continue
            valid_images.append(item["image"])
            valid_records.append(record)
        if not valid_images:
            continue

        inference_started_at = time.perf_counter()
        features = run_image_batch(
            model=model,
            processor=processor,
            images=valid_images,
            device=resolved_device,
            use_autocast=use_autocast,
            compute_dtype=compute_dtype,
        )
        synchronize_cuda(resolved_device)
        inference_sec += time.perf_counter() - inference_started_at
        features_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
        del features

        batch_vector_dim = int(features_np.shape[-1])
        if vector_dim is None:
            vector_dim = batch_vector_dim
            validate_vector_dim(model, vector_dim)
        elif batch_vector_dim != vector_dim:
            raise ValueError(
                f"SigLIP2 vector dimension changed between batches: "
                f"{batch_vector_dim} != {vector_dim}"
            )

        contract = {
            "model_family": "siglip2",
            "model_name": model_name,
            "model_revision": resolved_revision,
            "processor_name": model_name,
            "vector_dim": vector_dim,
            "input_resolution": input_resolution,
            "normalized": True,
            "similarity": "cosine",
            "output_dtype": "float32",
        }
        embedding_batches.append(features_np)
        for record in valid_records:
            embedding_records.append(
                _embedding_record(record, len(embedding_records), contract)
            )
        print(
            f"Batch {batch_number}: encoded={len(embedding_records)} "
            f"skipped={len(skipped_records)}"
        )

    if not embedding_batches or vector_dim is None:
        raise ValueError("No embeddings were generated")

    embeddings = np.concatenate(embedding_batches, axis=0).astype(np.float32, copy=False)
    validation = validate_embedding_artifacts(embeddings, embedding_records)
    runtime_sec = time.perf_counter() - started_at
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if resolved_device.startswith("cuda")
        else 0.0
    )
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:  # pragma: no cover - fake-only test environment.
        transformers_version = "unavailable"

    benchmark = {
        "model_family": "siglip2",
        "model_name": model_name,
        "model_revision": resolved_revision,
        "processor_name": model_name,
        "vector_dim": vector_dim,
        "input_resolution": input_resolution,
        "device": resolved_device,
        "compute_dtype": dtype_name(compute_dtype),
        "output_dtype": "float32",
        "normalized": True,
        "requested_batch_size": requested_batch_size,
        "selected_batch_size": selected_batch_size,
        "batch_tuning_results": tuning_results,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "input_record_count": len(records),
        "encoded_count": len(embedding_records),
        "skipped_count": len(skipped_records),
        "embedding_shape": list(embeddings.shape),
        "runtime_sec": round(runtime_sec, 3),
        "image_load_sec": round(image_load_sec, 3),
        "inference_sec": round(inference_sec, 3),
        "throughput_img_per_sec": round(
            len(embedding_records) / max(runtime_sec, 1e-9), 3
        ),
        "peak_gpu_memory_mb": round(float(peak_gpu_memory_mb), 3),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "validation": validation,
    }
    return embeddings, embedding_records, skipped_records, benchmark


def parse_batch_size(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        batch_size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'") from exc
    if batch_size <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return batch_size


def default_artifact_paths(metadata_path: Path, records: list[dict]) -> dict[str, Path]:
    video_ids = {str(record.get("video_id") or "") for record in records}
    if len(video_ids) != 1 or not next(iter(video_ids)):
        raise ValueError(
            "Cannot infer output paths: metadata must contain exactly one non-empty video_id"
        )
    video_id = next(iter(video_ids))
    return {
        "embeddings_path": Path("data/embeddings") / f"{ARTIFACT_TAG}_{video_id}.npy",
        "embedding_metadata_path": Path("data/metadata")
        / f"{ARTIFACT_TAG}_embeddings_{video_id}.jsonl",
        "skipped_path": Path("data/metadata") / f"{ARTIFACT_TAG}_skipped_{video_id}.jsonl",
        "benchmark_path": Path("data/metadata")
        / f"{ARTIFACT_TAG}_benchmark_{video_id}.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode keyframes with SigLIP2 and write synchronized artifacts."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/metadata/keyframes_L26_V200.jsonl"),
    )
    parser.add_argument("--embeddings-path", type=Path, default=None)
    parser.add_argument("--embedding-metadata-path", type=Path, default=None)
    parser.add_argument("--skipped-path", type=Path, default=None)
    parser.add_argument("--benchmark-path", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--batch-size", type=parse_batch_size, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.metadata_path.exists():
        raise SystemExit(f"Metadata file does not exist: {args.metadata_path}")
    records = load_jsonl(args.metadata_path)
    if not records:
        raise SystemExit(f"No records found in {args.metadata_path}")
    defaults = default_artifact_paths(args.metadata_path, records)
    embeddings_path = args.embeddings_path or defaults["embeddings_path"]
    embedding_metadata_path = (
        args.embedding_metadata_path or defaults["embedding_metadata_path"]
    )
    skipped_path = args.skipped_path or defaults["skipped_path"]
    benchmark_path = args.benchmark_path or defaults["benchmark_path"]

    embeddings, embedding_records, skipped_records, benchmark = encode_keyframes(
        records=records,
        model_name=args.model_name,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        use_autocast=not args.no_autocast,
        model_cache_dir=args.model_cache_dir,
        prefetch_factor=args.prefetch_factor,
    )
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    write_jsonl(embedding_records, embedding_metadata_path)
    write_jsonl(skipped_records, skipped_path)
    write_json(benchmark, benchmark_path)

    print(f"Saved embeddings: {embeddings_path} shape={embeddings.shape}")
    print(f"Saved embedding metadata: {embedding_metadata_path}")
    print(f"Saved skipped log: {skipped_path} ({len(skipped_records)} records)")
    print(f"Saved benchmark: {benchmark_path}")


if __name__ == "__main__":
    main()
