from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit(
        "FAISS is required. Install dependencies with: pip install -r requirements.txt"
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


def write_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def infer_video_id(path: Path, prefix: str, suffix: str) -> str:
    name = path.stem
    if not name.startswith(prefix):
        raise ValueError(f"{path} does not start with expected prefix {prefix!r}")
    if suffix and not name.endswith(suffix):
        raise ValueError(f"{path} does not end with expected suffix {suffix!r}")

    start = len(prefix)
    end = len(name) - len(suffix) if suffix else len(name)
    return name[start:end]


def resolve_embedding_sources(
    embeddings_glob: str,
    embedding_metadata_template: str,
    embeddings_prefix: str,
    embeddings_suffix: str,
) -> list[tuple[Path, Path, str]]:
    sources = []
    matched_paths = sorted(Path(path) for path in glob.glob(embeddings_glob))
    for embeddings_path in matched_paths:
        video_id = infer_video_id(embeddings_path, embeddings_prefix, embeddings_suffix)
        metadata_path = Path(embedding_metadata_template.format(video_id=video_id))
        sources.append((embeddings_path, metadata_path, video_id))
    return sources


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def summarize_norms(vectors: np.ndarray) -> dict:
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "norm_mean": float(np.mean(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
    }


def build_index(vectors: np.ndarray, metric: str):
    dim = int(vectors.shape[1])
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    index.add(vectors)
    return index


def infer_timestamp_source(record: dict) -> str:
    if record.get("timestamp_source"):
        return record["timestamp_source"]
    if record.get("frame_index") is not None:
        return "matched_frame"
    if record.get("timestamp") is not None:
        return "interval"
    return "unknown"


def infer_timestamp_confidence(record: dict) -> float:
    if record.get("timestamp_confidence") is not None:
        return float(record["timestamp_confidence"])
    source = infer_timestamp_source(record)
    return {
        "matched_frame": 0.9,
        "video_fps": 1.0,
        "interval": 0.5,
        "unknown": 0.0,
    }.get(source, 0.5)


def checks_passed(checklist: dict[str, bool | None]) -> bool:
    return all(value is not False for value in checklist.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from OpenCLIP embedding .npy artifacts."
    )
    parser.add_argument(
        "--embeddings-glob",
        default="data/embeddings/openclip_vit_b16_*.npy",
        help="Glob for embedding .npy files to combine into one FAISS index.",
    )
    parser.add_argument(
        "--embedding-metadata-template",
        default="data/metadata/openclip_vit_b16_embeddings_{video_id}.jsonl",
        help="Metadata JSONL path template. Use {video_id} from the embeddings filename.",
    )
    parser.add_argument(
        "--embeddings-prefix",
        default="openclip_vit_b16_",
        help="Filename stem prefix used to infer video_id.",
    )
    parser.add_argument(
        "--embeddings-suffix",
        default="",
        help="Filename stem suffix used to infer video_id.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/indexes/openclip_vit_b16_flat_ip.faiss"),
    )
    parser.add_argument(
        "--index-metadata-path",
        type=Path,
        default=Path("data/metadata/openclip_vit_b16_faiss_metadata.jsonl"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/metadata/openclip_vit_b16_faiss_manifest.json"),
    )
    parser.add_argument(
        "--frame-map-path",
        type=Path,
        default=Path("data/metadata/openclip_vit_b16_frame_map.json"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/metadata/openclip_vit_b16_index_report.json"),
    )
    parser.add_argument(
        "--metric",
        choices=("ip", "l2"),
        default="ip",
        help="FAISS metric. Use ip for cosine search with normalized OpenCLIP vectors.",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Do not L2-normalize vectors before adding them to FAISS.",
    )
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    sources = resolve_embedding_sources(
        embeddings_glob=args.embeddings_glob,
        embedding_metadata_template=args.embedding_metadata_template,
        embeddings_prefix=args.embeddings_prefix,
        embeddings_suffix=args.embeddings_suffix,
    )
    if not sources:
        raise SystemExit(f"No embeddings found for glob: {args.embeddings_glob}")

    vector_batches = []
    index_records = []
    source_summaries = []
    vector_dim: int | None = None
    checklist = {
        "loaded_embeddings_npy": False,
        "loaded_embedding_metadata_jsonl": False,
        "metadata_shape_matched": False,
        "vectors_normalized_for_index": False,
        "built_expected_faiss_index": False,
        "saved_faiss_index": False,
        "saved_frame_map_json": False,
        "saved_index_report_json": False,
        "faiss_ntotal_equals_vector_count": False,
        "embedding_index_equals_offset": False,
        "vectors_are_finite": False,
    }

    for embeddings_path, metadata_path, video_id in sources:
        if not metadata_path.exists():
            raise SystemExit(f"Missing metadata for {embeddings_path}: {metadata_path}")

        vectors = np.load(embeddings_path).astype("float32", copy=False)
        checklist["loaded_embeddings_npy"] = True
        if vectors.ndim != 2:
            raise SystemExit(f"Expected 2D embeddings in {embeddings_path}, got {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise SystemExit(f"Found NaN or Inf values in {embeddings_path}")
        checklist["vectors_are_finite"] = True
        if vector_dim is None:
            vector_dim = int(vectors.shape[1])
        elif vectors.shape[1] != vector_dim:
            raise SystemExit(
                f"Dimension mismatch for {embeddings_path}: {vectors.shape[1]} != {vector_dim}"
            )

        records = load_jsonl(metadata_path)
        checklist["loaded_embedding_metadata_jsonl"] = True
        if len(records) != vectors.shape[0]:
            raise SystemExit(
                f"Count mismatch for {video_id}: {vectors.shape[0]} vectors vs "
                f"{len(records)} metadata records"
            )
        checklist["metadata_shape_matched"] = True

        base_index = len(index_records)
        for offset, record in enumerate(records):
            embedding_index = record.get("embedding_index")
            if embedding_index != offset:
                raise SystemExit(
                    f"embedding_index mismatch in {metadata_path}: "
                    f"record offset {offset} has embedding_index={embedding_index}"
                )
            faiss_index = base_index + offset
            next_record = dict(record)
            next_record["faiss_index"] = faiss_index
            index_records.append(next_record)
        checklist["embedding_index_equals_offset"] = True

        vector_batches.append(vectors)
        source_norms = summarize_norms(vectors)
        source_summaries.append(
            {
                "video_id": video_id,
                "embeddings_path": embeddings_path.as_posix(),
                "embedding_metadata_path": metadata_path.as_posix(),
                "vector_count": int(vectors.shape[0]),
                **source_norms,
            }
        )
        print(f"Loaded {video_id}: {vectors.shape[0]} vectors from {embeddings_path}")

    all_vectors = np.concatenate(vector_batches, axis=0)
    input_norms = summarize_norms(all_vectors)
    if not args.skip_normalize:
        all_vectors = l2_normalize(all_vectors).astype("float32", copy=False)
    index_norms = summarize_norms(all_vectors)
    if args.skip_normalize:
        checklist["vectors_normalized_for_index"] = None
    else:
        checklist["vectors_normalized_for_index"] = (
            abs(index_norms["norm_mean"] - 1.0) <= 1e-5
            and abs(index_norms["norm_min"] - 1.0) <= 1e-4
            and abs(index_norms["norm_max"] - 1.0) <= 1e-4
        )

    index = build_index(all_vectors, args.metric)
    expected_index_type = "IndexFlatIP" if args.metric == "ip" else "IndexFlatL2"
    checklist["built_expected_faiss_index"] = type(index).__name__ == expected_index_type
    checklist["faiss_ntotal_equals_vector_count"] = int(index.ntotal) == int(all_vectors.shape[0])
    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, args.index_path.as_posix())
    checklist["saved_faiss_index"] = args.index_path.exists()
    write_jsonl(index_records, args.index_metadata_path)

    frame_map = {
        str(record["faiss_index"]): {
            "frame_id": record.get("frame_id"),
            "video_id": record.get("video_id"),
            "shot_id": record.get("shot_id", ""),
            "segment_id": record.get("segment_id", ""),
            "shot_index": record.get("shot_index"),
            "shot_start": record.get("shot_start"),
            "shot_end": record.get("shot_end"),
            "timestamp": record.get("timestamp"),
            "timestamp_source": infer_timestamp_source(record),
            "timestamp_confidence": infer_timestamp_confidence(record),
            "frame_index": record.get("frame_index"),
            "keyframe_path": record.get("keyframe_path"),
            "thumbnail_path": record.get("thumbnail_path", record.get("keyframe_path")),
            "source_video_path": record.get("source_video_path") or record.get("video_path"),
            "video_path": record.get("video_path") or record.get("source_video_path"),
            "embedding_id": record.get("embedding_id"),
            "embedding_index": record.get("embedding_index"),
            "selection_reason": record.get("selection_reason"),
        }
        for record in index_records
    }
    write_json(frame_map, args.frame_map_path)
    checklist["saved_frame_map_json"] = args.frame_map_path.exists()

    runtime_sec = time.perf_counter() - started_at
    index_file_size_mb = args.index_path.stat().st_size / (1024 * 1024)
    manifest = {
        "index_type": type(index).__name__,
        "metric": args.metric,
        "normalized": not args.skip_normalize,
        "index_path": args.index_path.as_posix(),
        "index_metadata_path": args.index_metadata_path.as_posix(),
        "frame_map_path": args.frame_map_path.as_posix(),
        "report_path": args.report_path.as_posix(),
        "vector_count": int(index.ntotal),
        "metadata_record_count": len(index_records),
        "vector_dim": int(all_vectors.shape[1]),
        "input_norm_mean": input_norms["norm_mean"],
        "input_norm_min": input_norms["norm_min"],
        "input_norm_max": input_norms["norm_max"],
        "norm_mean": index_norms["norm_mean"],
        "norm_min": index_norms["norm_min"],
        "norm_max": index_norms["norm_max"],
        "runtime_sec": round(runtime_sec, 3),
        "index_file_size_mb": round(index_file_size_mb, 3),
        "sources": source_summaries,
    }
    write_json(manifest, args.manifest_path)

    report = {
        "status": "passed" if checks_passed(checklist) else "warning",
        "checks": checklist,
        "faiss_ntotal": int(index.ntotal),
        "vector_count": int(all_vectors.shape[0]),
        "metadata_record_count": len(index_records),
        "manifest_path": args.manifest_path.as_posix(),
        "index_path": args.index_path.as_posix(),
        "frame_map_path": args.frame_map_path.as_posix(),
    }
    write_json(report, args.report_path)
    checklist["saved_index_report_json"] = args.report_path.exists()
    report["status"] = "passed" if checks_passed(checklist) else "warning"
    report["checks"] = checklist
    write_json(report, args.report_path)

    print(f"Saved FAISS index: {args.index_path} vectors={index.ntotal}")
    print(f"Saved FAISS metadata: {args.index_metadata_path}")
    print(f"Saved frame map: {args.frame_map_path}")
    print(f"Saved index report: {args.report_path}")
    print(f"Saved manifest: {args.manifest_path}")


if __name__ == "__main__":
    main()
