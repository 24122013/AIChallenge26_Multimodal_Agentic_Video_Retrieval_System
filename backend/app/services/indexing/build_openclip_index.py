from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

try:
    import open_clip
    import torch
    from PIL import Image
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit(
        "OpenCLIP, PyTorch, Pillow, and NumPy are required. Install them with: "
        "pip install -r requirements.txt"
    ) from exc


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def choose_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def batched(records: list[dict], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield start, records[start : start + batch_size]


def load_image(record: dict, preprocess):
    image_path = Path(record["keyframe_path"])
    with Image.open(image_path) as image:
        return preprocess(image.convert("RGB"))


def get_autocast_context(device: str, enabled: bool):
    if not enabled or device == "cpu":
        return nullcontext()
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def encode_keyframes(
    records: list[dict],
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: str,
    use_autocast: bool,
    model_cache_dir: Path | None = None,
) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    started_at = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
        cache_dir=model_cache_dir.as_posix() if model_cache_dir else None,
    )
    model.eval()

    embedding_batches = []
    embedding_records: list[dict] = []
    skipped_records: list[dict] = []
    model_tag = f"openclip_{model_name}_{pretrained}"
    image_load_sec = 0.0
    inference_sec = 0.0

    with torch.no_grad():
        for start_index, batch in batched(records, batch_size):
            images = []
            valid_records = []
            for record in batch:
                image_started_at = time.perf_counter()
                try:
                    images.append(load_image(record, preprocess))
                    valid_records.append(record)
                except Exception as exc:  # noqa: BLE001 - encode should continue after bad images.
                    skipped_records.append(
                        {
                            "frame_id": record.get("frame_id", ""),
                            "video_id": record.get("video_id", ""),
                            "keyframe_path": record.get("keyframe_path", ""),
                            "skip_reason": "image_load_error",
                            "error": str(exc),
                        }
                    )
                finally:
                    image_load_sec += time.perf_counter() - image_started_at

            if not images:
                print(
                    f"Skipped batch {start_index}-{start_index + len(batch)}: no valid images"
                )
                continue

            image_tensor = torch.stack(images).to(device)
            inference_started_at = time.perf_counter()
            with get_autocast_context(device, enabled=use_autocast):
                features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            inference_sec += time.perf_counter() - inference_started_at
            features_np = features.detach().cpu().numpy().astype("float32")
            embedding_batches.append(features_np)

            for offset, record in enumerate(valid_records):
                embedding_index = len(embedding_records)
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
                        "vector_dim": int(features_np.shape[1]),
                        "embedding_index": embedding_index,
                    }
                )

            processed_count = min(start_index + len(batch), len(records))
            print(
                f"Processed {processed_count}/{len(records)} | "
                f"encoded {len(embedding_records)} | skipped {len(skipped_records)}"
            )

    if not embedding_batches:
        raise ValueError("No embeddings were generated.")

    embeddings = np.concatenate(embedding_batches, axis=0)
    assert embeddings.shape[0] == len(embedding_records)
    runtime_sec = time.perf_counter() - started_at
    benchmark = {
        "model_name": model_tag,
        "device": device,
        "autocast_enabled": bool(use_autocast and device != "cpu"),
        "input_record_count": len(records),
        "encoded_count": len(embedding_records),
        "skipped_count": len(skipped_records),
        "embedding_shape": list(embeddings.shape),
        "vector_dim": int(embeddings.shape[1]),
        "batch_size": batch_size,
        "runtime_sec": round(runtime_sec, 3),
        "image_load_sec": round(image_load_sec, 3),
        "inference_sec": round(inference_sec, 3),
        "throughput_img_per_sec": round(len(embedding_records) / max(runtime_sec, 1e-9), 3),
    }

    return embeddings, embedding_records, skipped_records, benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode keyframes with OpenCLIP ViT-B/16 and write embedding artifacts."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/metadata/keyframes_L26_V200.jsonl"),
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("data/embeddings/openclip_vit_b16_L26_V200.npy"),
    )
    parser.add_argument(
        "--embedding-metadata-path",
        type=Path,
        default=Path("data/metadata/openclip_vit_b16_embeddings_L26_V200.jsonl"),
    )
    parser.add_argument(
        "--skipped-path",
        type=Path,
        default=Path("data/metadata/openclip_vit_b16_skipped_L26_V200.jsonl"),
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/metadata/openclip_vit_b16_benchmark_L26_V200.json"),
    )
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("data/model_cache/openclip"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.metadata_path)
    if not records:
        raise SystemExit(f"No records found in {args.metadata_path}")

    device = choose_device(args.device)
    embeddings, embedding_records, skipped_records, benchmark = encode_keyframes(
        records=records,
        model_name=args.model_name,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        device=device,
        use_autocast=not args.no_autocast,
        model_cache_dir=args.model_cache_dir,
    )

    args.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings_path, embeddings)
    write_jsonl(embedding_records, args.embedding_metadata_path)
    write_jsonl(skipped_records, args.skipped_path)
    args.benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with args.benchmark_path.open("w", encoding="utf-8") as file:
        json.dump(benchmark, file, ensure_ascii=False, indent=2)

    print(f"Saved embeddings: {args.embeddings_path} shape={embeddings.shape}")
    print(f"Saved embedding metadata: {args.embedding_metadata_path}")
    print(f"Saved skipped log: {args.skipped_path} ({len(skipped_records)} records)")
    print(f"Saved benchmark: {args.benchmark_path}")


if __name__ == "__main__":
    main()
