from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .build_siglip2_index import (
        ARTIFACT_TAG,
        DEFAULT_MODEL_CACHE_DIR,
        DEFAULT_MODEL_NAME,
        encode_keyframes,
        parse_batch_size,
        validate_embedding_artifacts,
        write_jsonl,
    )
    from .normalize_keyframe_metadata import (
        build_metadata_records,
        infer_video_id,
        write_jsonl as write_metadata_jsonl,
    )
    from .validate_keyframes import validate_records
except ImportError:  # pragma: no cover - supports direct script execution.
    from build_siglip2_index import (
        ARTIFACT_TAG,
        DEFAULT_MODEL_CACHE_DIR,
        DEFAULT_MODEL_NAME,
        encode_keyframes,
        parse_batch_size,
        validate_embedding_artifacts,
        write_jsonl,
    )
    from normalize_keyframe_metadata import (
        build_metadata_records,
        infer_video_id,
        write_jsonl as write_metadata_jsonl,
    )
    from validate_keyframes import validate_records


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def default_paths(output_root: Path, video_id: str) -> dict[str, Path]:
    metadata_dir = output_root / "metadata"
    return {
        "metadata_path": metadata_dir / f"keyframes_{video_id}.jsonl",
        "validation_report_path": metadata_dir / f"keyframes_{video_id}_validation.json",
        "embeddings_path": output_root
        / "embeddings"
        / f"{ARTIFACT_TAG}_{video_id}.npy",
        "embedding_metadata_path": metadata_dir
        / f"{ARTIFACT_TAG}_embeddings_{video_id}.jsonl",
        "skipped_path": metadata_dir / f"{ARTIFACT_TAG}_skipped_{video_id}.jsonl",
        "benchmark_path": metadata_dir / f"{ARTIFACT_TAG}_benchmark_{video_id}.json",
        "artifact_validation_path": metadata_dir
        / f"{ARTIFACT_TAG}_artifacts_{video_id}_validation.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize, validate, and encode one keyframe folder with SigLIP2."
    )
    parser.add_argument("--keyframe-dir", type=Path, required=True)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--timestamp-interval-sec", type=float, default=2.0)
    parser.add_argument("--search-window-sec", type=float, default=12.0)
    parser.add_argument("--min-width", type=int, default=16)
    parser.add_argument("--min-height", type=int, default=16)
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
    if not args.keyframe_dir.exists():
        raise SystemExit(f"Keyframe directory does not exist: {args.keyframe_dir}")

    video_id = infer_video_id(args.keyframe_dir, args.video_id)
    paths = default_paths(args.output_root, video_id)

    print(f"[1/4] Normalizing metadata for {video_id}")
    records = build_metadata_records(
        keyframe_dir=args.keyframe_dir,
        video_id=video_id,
        timestamp_interval_sec=args.timestamp_interval_sec,
        video_path=args.video_path,
        search_window_sec=args.search_window_sec,
    )
    if not records:
        raise SystemExit(f"No keyframe images found in: {args.keyframe_dir}")
    write_metadata_jsonl(records, paths["metadata_path"])

    print("[2/4] Validating keyframes")
    validation_report = validate_records(
        records=records,
        min_width=args.min_width,
        min_height=args.min_height,
    )
    write_json(validation_report, paths["validation_report_path"])
    if not validation_report["valid"]:
        raise SystemExit(
            "Keyframe validation failed. See "
            f"{paths['validation_report_path']} before encoding."
        )

    print("[3/4] Auto-tuning and encoding SigLIP2 embeddings")
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
    paths["embeddings_path"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["embeddings_path"], embeddings)
    write_jsonl(embedding_records, paths["embedding_metadata_path"])
    write_jsonl(skipped_records, paths["skipped_path"])
    write_json(benchmark, paths["benchmark_path"])

    print("[4/4] Validating synchronized embedding artifacts")
    artifact_report = validate_embedding_artifacts(embeddings, embedding_records)
    artifact_report["status"] = "passed"
    artifact_report["embeddings_path"] = paths["embeddings_path"].as_posix()
    artifact_report["embedding_metadata_path"] = paths[
        "embedding_metadata_path"
    ].as_posix()
    write_json(artifact_report, paths["artifact_validation_path"])

    print(f"Embeddings: {paths['embeddings_path']} shape={embeddings.shape}")
    print(f"Embedding metadata: {paths['embedding_metadata_path']}")
    print(f"Skipped: {paths['skipped_path']} ({len(skipped_records)} records)")
    print(f"Benchmark: {paths['benchmark_path']}")
    print(f"Artifact validation: {paths['artifact_validation_path']}")


if __name__ == "__main__":
    main()
