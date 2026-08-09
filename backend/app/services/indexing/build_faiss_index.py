from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - handled when index operations are requested.
    faiss = None


ARTIFACT_TAG = "siglip2_so400m_patch16_384"
CONTRACT_FIELDS = (
    "model_family",
    "model_name",
    "model_revision",
    "processor_name",
    "vector_dim",
    "input_resolution",
    "normalized",
    "similarity",
    "output_dtype",
)


def require_faiss():
    if faiss is None:
        raise RuntimeError(
            "FAISS is required. Install dependencies with: pip install -r requirements.txt"
        )
    return faiss


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
                    f"Expected object in {path} at line {line_number}, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def write_json(data: dict | list, path: Path, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        if compact:
            json.dump(data, file, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(data, file, ensure_ascii=False, indent=2)


def infer_video_id(path: Path, prefix: str, suffix: str) -> str:
    name = path.stem
    if not name.startswith(prefix):
        raise ValueError(f"{path} does not start with expected prefix {prefix!r}")
    if suffix and not name.endswith(suffix):
        raise ValueError(f"{path} does not end with expected suffix {suffix!r}")
    start = len(prefix)
    end = len(name) - len(suffix) if suffix else len(name)
    video_id = name[start:end]
    if not video_id:
        raise ValueError(f"Could not infer video_id from {path}")
    return video_id


def resolve_embedding_sources(
    embeddings_glob: str,
    embedding_metadata_template: str,
    embeddings_prefix: str,
    embeddings_suffix: str,
) -> list[tuple[Path, Path, str]]:
    sources: list[tuple[Path, Path, str]] = []
    for embeddings_path in sorted(Path(path) for path in glob.glob(embeddings_glob)):
        video_id = infer_video_id(embeddings_path, embeddings_prefix, embeddings_suffix)
        metadata_path = Path(embedding_metadata_template.format(video_id=video_id))
        sources.append((embeddings_path, metadata_path, video_id))
    return sources


def summarize_norms(vectors: np.ndarray) -> dict:
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "norm_mean": float(np.mean(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
    }


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Cannot normalize embeddings containing zero vectors")
    return vectors / norms


def build_index(vectors: np.ndarray, metric: str):
    faiss_module = require_faiss()
    dim = int(vectors.shape[1])
    if metric == "ip":
        index = faiss_module.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss_module.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    index.add(vectors)
    return index


def infer_timestamp_source(record: dict) -> str:
    if record.get("timestamp_source"):
        return str(record["timestamp_source"])
    if record.get("frame_index") is not None:
        return "matched_frame"
    if record.get("timestamp") is not None:
        return "interval"
    return "unknown"


def infer_timestamp_confidence(record: dict) -> float:
    if record.get("timestamp_confidence") is not None:
        return float(record["timestamp_confidence"])
    return {
        "matched_frame": 0.9,
        "video_fps": 1.0,
        "interval": 0.5,
        "unknown": 0.0,
    }.get(infer_timestamp_source(record), 0.5)


def contract_from_record(record: dict, metadata_path: Path, offset: int) -> dict:
    missing = [field for field in CONTRACT_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"Missing encoder contract fields in {metadata_path} record {offset}: {missing}"
        )
    contract = {field: record[field] for field in CONTRACT_FIELDS}
    if not contract["model_name"]:
        raise ValueError(f"Empty model_name in {metadata_path} record {offset}")
    if not contract["model_revision"]:
        raise ValueError(f"Empty model_revision in {metadata_path} record {offset}")
    if not contract["model_family"]:
        raise ValueError(f"Empty model_family in {metadata_path} record {offset}")
    if contract["normalized"] is not True:
        raise ValueError(
            f"normalized must be true in {metadata_path} record {offset}, "
            f"got {contract['normalized']!r}"
        )
    if contract["similarity"] != "cosine":
        raise ValueError(
            f"similarity must be 'cosine' in {metadata_path} record {offset}, "
            f"got {contract['similarity']!r}"
        )
    if contract["output_dtype"] != "float32":
        raise ValueError(
            f"output_dtype must be 'float32' in {metadata_path} record {offset}, "
            f"got {contract['output_dtype']!r}"
        )
    try:
        contract["vector_dim"] = int(contract["vector_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid vector_dim in {metadata_path} record {offset}: "
            f"{contract['vector_dim']!r}"
        ) from exc
    return contract


def validate_embedding_source(
    embeddings_path: Path,
    metadata_path: Path,
    video_id: str,
) -> tuple[np.ndarray, list[dict], dict, dict]:
    if not embeddings_path.exists():
        raise ValueError(f"Missing embedding file: {embeddings_path}")
    if not metadata_path.exists():
        raise ValueError(f"Missing metadata for {embeddings_path}: {metadata_path}")

    vectors = np.load(embeddings_path)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embeddings in {embeddings_path}, got {vectors.shape}")
    if vectors.shape[0] == 0:
        raise ValueError(f"Embedding source is empty: {embeddings_path}")
    try:
        vectors = vectors.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Embeddings are not numeric float vectors: {embeddings_path}") from exc
    if not np.isfinite(vectors).all():
        raise ValueError(f"Found NaN or Inf values in {embeddings_path}")
    norms = np.linalg.norm(vectors, axis=1)
    zero_offsets = np.flatnonzero(norms <= 0)
    if len(zero_offsets):
        raise ValueError(
            f"Found zero vector in {embeddings_path} at offset {int(zero_offsets[0])}"
        )

    records = load_jsonl(metadata_path)
    if len(records) != vectors.shape[0]:
        raise ValueError(
            f"Count mismatch for {video_id}: {vectors.shape[0]} vectors in "
            f"{embeddings_path} vs {len(records)} records in {metadata_path}"
        )

    source_contract: dict | None = None
    for offset, record in enumerate(records):
        if record.get("embedding_index") != offset:
            raise ValueError(
                f"embedding_index mismatch in {metadata_path} record {offset}: "
                f"expected {offset}, got {record.get('embedding_index')!r}"
            )
        record_contract = contract_from_record(record, metadata_path, offset)
        if record_contract["vector_dim"] != vectors.shape[1]:
            raise ValueError(
                f"vector_dim mismatch in {metadata_path} record {offset}: metadata "
                f"{record_contract['vector_dim']} != {embeddings_path} shape[1] "
                f"{vectors.shape[1]}"
            )
        if source_contract is None:
            source_contract = record_contract
        elif record_contract != source_contract:
            differing = [
                field
                for field in CONTRACT_FIELDS
                if record_contract[field] != source_contract[field]
            ]
            raise ValueError(
                f"Inconsistent encoder contract in {metadata_path} record {offset}; "
                f"differing fields: {differing}"
            )

    assert source_contract is not None
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
        bad_offset = int(np.flatnonzero(~np.isclose(norms, 1.0, atol=1e-4, rtol=1e-4))[0])
        raise ValueError(
            f"Metadata says normalized=true but vector norm is {norms[bad_offset]:.8f} "
            f"in {embeddings_path} at offset {bad_offset}"
        )

    summary = {
        "video_id": video_id,
        "embeddings_path": embeddings_path.as_posix(),
        "embedding_metadata_path": metadata_path.as_posix(),
        "vector_count": int(vectors.shape[0]),
        "encoder": source_contract,
        **summarize_norms(vectors),
    }
    return vectors, records, source_contract, summary


def validate_global_contract(
    expected: dict | None,
    current: dict,
    metadata_path: Path,
) -> dict:
    if expected is None:
        return dict(current)
    differing = [
        field for field in CONTRACT_FIELDS if expected[field] != current[field]
    ]
    if differing:
        details = ", ".join(
            f"{field}: {expected[field]!r} != {current[field]!r}"
            for field in differing
        )
        raise ValueError(
            f"Encoder contract mismatch for {metadata_path}; refusing to mix sources: "
            f"{details}"
        )
    return expected


def frame_map_record(record: dict) -> dict:
    value = {
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
        "model_name": record.get("model_name"),
        "model_revision": record.get("model_revision"),
        "vector_dim": record.get("vector_dim"),
    }
    for field in (
        "candidate_index",
        "candidate_id",
        "candidate_reasons",
        "keyframe_strategy",
        "selection_phase",
        "selection_rank",
        "selection_reasons",
        "covered_event_ids",
        "selection_score",
        "protected",
        "coverage_added",
        "importance_score",
        "semantic_novelty",
        "component_scores",
        "available_modalities",
        "protected_event_ids",
        "selection_provenance",
    ):
        if field in record:
            value[field] = record[field]
    return value


def build_faiss_artifacts(
    sources: list[tuple[Path, Path, str]],
    index_path: Path,
    index_metadata_path: Path,
    frame_map_path: Path,
    manifest_path: Path,
    report_path: Path,
    metric: str = "ip",
    normalize_for_index: bool = True,
) -> dict:
    if not sources:
        raise ValueError("No embedding sources were supplied")
    started_at = time.perf_counter()
    vector_batches: list[np.ndarray] = []
    index_records: list[dict] = []
    source_summaries: list[dict] = []
    encoder_contract: dict | None = None

    # Validate every source and the cross-source contract before writing anything.
    for embeddings_path, metadata_path, video_id in sources:
        vectors, records, source_contract, source_summary = validate_embedding_source(
            embeddings_path, metadata_path, video_id
        )
        encoder_contract = validate_global_contract(
            encoder_contract, source_contract, metadata_path
        )
        base_index = len(index_records)
        for offset, record in enumerate(records):
            indexed_record = dict(record)
            indexed_record["faiss_index"] = base_index + offset
            index_records.append(indexed_record)
        vector_batches.append(vectors)
        source_summaries.append(source_summary)

    assert encoder_contract is not None
    all_vectors = np.concatenate(vector_batches, axis=0).astype(np.float32, copy=False)
    input_norms = summarize_norms(all_vectors)
    if normalize_for_index:
        all_vectors = l2_normalize(all_vectors).astype(np.float32, copy=False)
    index_norms = summarize_norms(all_vectors)
    index = build_index(all_vectors, metric)
    expected_type = "IndexFlatIP" if metric == "ip" else "IndexFlatL2"
    if type(index).__name__ != expected_type:
        raise RuntimeError(
            f"Built unexpected FAISS index type: {type(index).__name__} != {expected_type}"
        )
    if int(index.ntotal) != len(index_records):
        raise RuntimeError(
            f"FAISS ntotal mismatch: {index.ntotal} != {len(index_records)}"
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    require_faiss().write_index(index, index_path.as_posix())
    write_jsonl(index_records, index_metadata_path)
    frame_map = {
        str(record["faiss_index"]): frame_map_record(record) for record in index_records
    }
    # frame_map is a production artifact and can be large. Whitespace is not
    # part of its contract, so compact JSON is backward compatible.
    write_json(frame_map, frame_map_path, compact=True)

    runtime_sec = time.perf_counter() - started_at
    encoder_manifest = {
        field: encoder_contract[field] for field in CONTRACT_FIELDS
    }
    manifest = {
        "schema_version": "1.2",
        "encoder": encoder_manifest,
        "index_type": type(index).__name__,
        "metric": metric,
        "vector_count": int(index.ntotal),
        "metadata_record_count": len(index_records),
        "index_path": index_path.as_posix(),
        "index_metadata_path": index_metadata_path.as_posix(),
        "frame_map_path": frame_map_path.as_posix(),
        "report_path": report_path.as_posix(),
        "input_norm_mean": input_norms["norm_mean"],
        "input_norm_min": input_norms["norm_min"],
        "input_norm_max": input_norms["norm_max"],
        "norm_mean": index_norms["norm_mean"],
        "norm_min": index_norms["norm_min"],
        "norm_max": index_norms["norm_max"],
        "runtime_sec": round(runtime_sec, 3),
        "index_file_size_mb": round(index_path.stat().st_size / (1024 * 1024), 3),
        "sources": source_summaries,
    }
    write_json(manifest, manifest_path)

    report = {
        "status": "passed",
        "checks": {
            "all_sources_are_2d": True,
            "vectors_are_finite": True,
            "no_zero_vectors": True,
            "metadata_counts_match": True,
            "embedding_indices_match_offsets": True,
            "encoder_contract_consistent": True,
            "metadata_vector_dim_matches_shape": True,
            "source_vectors_normalized": True,
            "faiss_ntotal_equals_vector_count": True,
            "frame_map_keys_match_faiss_indices": list(frame_map)
            == [str(index) for index in range(int(index.ntotal))],
        },
        "faiss_ntotal": int(index.ntotal),
        "vector_count": int(all_vectors.shape[0]),
        "metadata_record_count": len(index_records),
        "encoder": encoder_manifest,
        "manifest_path": manifest_path.as_posix(),
        "index_path": index_path.as_posix(),
        "frame_map_path": frame_map_path.as_posix(),
    }
    write_json(report, report_path)
    return {
        "index": index,
        "frame_map": frame_map,
        "manifest": manifest,
        "report": report,
        "index_records": index_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a contract-validated FAISS index from embedding artifacts."
    )
    parser.add_argument(
        "--embeddings-glob",
        default=f"data/embeddings/{ARTIFACT_TAG}_*.npy",
    )
    parser.add_argument(
        "--embedding-metadata-template",
        default=f"data/metadata/{ARTIFACT_TAG}_embeddings_{{video_id}}.jsonl",
    )
    parser.add_argument("--embeddings-prefix", default=f"{ARTIFACT_TAG}_")
    parser.add_argument("--embeddings-suffix", default="")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(f"data/indexes/{ARTIFACT_TAG}_flat_ip.faiss"),
    )
    parser.add_argument(
        "--index-metadata-path",
        type=Path,
        default=Path(f"data/metadata/{ARTIFACT_TAG}_faiss_metadata.jsonl"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(f"data/metadata/{ARTIFACT_TAG}_faiss_manifest.json"),
    )
    parser.add_argument(
        "--frame-map-path",
        type=Path,
        default=Path(f"data/metadata/{ARTIFACT_TAG}_frame_map.json"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(f"data/metadata/{ARTIFACT_TAG}_index_report.json"),
    )
    parser.add_argument("--metric", choices=("ip", "l2"), default="ip")
    parser.add_argument("--skip-normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = resolve_embedding_sources(
        embeddings_glob=args.embeddings_glob,
        embedding_metadata_template=args.embedding_metadata_template,
        embeddings_prefix=args.embeddings_prefix,
        embeddings_suffix=args.embeddings_suffix,
    )
    if not sources:
        raise SystemExit(f"No embeddings found for glob: {args.embeddings_glob}")
    try:
        result = build_faiss_artifacts(
            sources=sources,
            index_path=args.index_path,
            index_metadata_path=args.index_metadata_path,
            frame_map_path=args.frame_map_path,
            manifest_path=args.manifest_path,
            report_path=args.report_path,
            metric=args.metric,
            normalize_for_index=not args.skip_normalize,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Saved FAISS index: {args.index_path} vectors={result['index'].ntotal}")
    print(f"Saved FAISS metadata: {args.index_metadata_path}")
    print(f"Saved frame map: {args.frame_map_path}")
    print(f"Saved manifest: {args.manifest_path}")
    print(f"Saved index report: {args.report_path}")


if __name__ == "__main__":
    main()
