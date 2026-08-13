"""Build and validate the queryable dense-candidate safety index.

The Phase-3 pipeline already computed one normalized SigLIP2 shard per video.
This module publishes those shards as one immutable, row-addressable index while
retaining strict lineage back to every cached candidate pool.  Candidate JPEGs
are referenced in place and are never copied into a run directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DENSE_INDEX_VERSION = 1
DENSE_DIRNAME = "dense"
DENSE_INDEX_NAME = "dense_flat_ip.faiss"
DENSE_VECTORS_NAME = "dense_vectors.npy"
DENSE_FRAME_MAP_NAME = "dense_frame_map.jsonl"
DENSE_MANIFEST_NAME = "dense_manifest.json"


@dataclass(frozen=True)
class DenseIndexPaths:
    root: Path
    index: Path
    vectors: Path
    frame_map: Path
    manifest: Path

    @classmethod
    def from_run_root(cls, run_root: Path) -> "DenseIndexPaths":
        root = run_root.resolve() / DENSE_DIRNAME
        return cls(
            root=root,
            index=root / DENSE_INDEX_NAME,
            vectors=root / DENSE_VECTORS_NAME,
            frame_map=root / DENSE_FRAME_MAP_NAME,
            manifest=root / DENSE_MANIFEST_NAME,
        )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_save_npy(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_faiss(path: Path, vectors: np.ndarray) -> None:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency preflight
        raise RuntimeError("faiss is required to build the dense safety index") from exc

    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temp_path = Path(raw_temp)
    try:
        faiss.write_index(index, str(temp_path))
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _relative_reference(path: Path, run_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), run_root.resolve())).as_posix()


def resolve_run_reference(run_root: Path, reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        raise ValueError(f"Dense manifest path must be run-root-relative: {reference}")
    return (run_root.resolve() / path).resolve()


def _records_by_candidate(
    path: Path,
    *,
    allow_video_level: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    video_level: list[dict[str, Any]] = []
    for record in _read_jsonl(path):
        candidate_id = str(record.get("candidate_id") or "")
        if candidate_id:
            if candidate_id in by_candidate:
                raise ValueError(f"Duplicate candidate_id {candidate_id!r} in {path}")
            by_candidate[candidate_id] = record
        elif allow_video_level:
            video_level.append(record)
    return by_candidate, video_level


def _object_labels(record: Mapping[str, Any]) -> list[str]:
    labels = record.get("object_classes")
    if isinstance(labels, list):
        return [str(value) for value in labels if str(value)]
    objects = record.get("objects")
    if not isinstance(objects, list):
        return []
    result: list[str] = []
    for value in objects:
        if isinstance(value, Mapping):
            label = value.get("class_name") or value.get("label") or value.get("name")
            if label:
                result.append(str(label))
        elif value:
            result.append(str(value))
    return list(dict.fromkeys(result))


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _select_phase3_manifests(source_output_root: Path) -> list[Path]:
    manifests = sorted(
        (source_output_root.resolve() / "metadata").glob(
            "keyframes_*_phase3_manifest.json"
        )
    )
    if not manifests:
        raise FileNotFoundError(
            f"No Phase-3 manifests found under {source_output_root / 'metadata'}"
        )
    return manifests


def build_dense_index(
    *,
    run_root: Path,
    source_workspace: Path,
    source_output_root: Path,
) -> dict[str, Any]:
    """Publish cached Phase-3 dense shards as one validated FlatIP index."""
    run_root = run_root.resolve()
    source_workspace = source_workspace.resolve()
    source_output_root = source_output_root.resolve()
    paths = DenseIndexPaths.from_run_root(run_root)
    manifests = _select_phase3_manifests(source_output_root)

    vector_shards: list[np.ndarray] = []
    dense_records: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    expected_dim: int | None = None
    expected_model: tuple[str, str] | None = None
    expected_offline_config: str | None = None
    row_offset = 0

    for phase3_path in manifests:
        phase3 = _read_json(phase3_path)
        video_id = str(phase3.get("video_id") or "")
        run_id = str(phase3.get("candidate_pool_run_id") or "")
        if not video_id or not run_id:
            raise ValueError(f"Invalid Phase-3 lineage in {phase3_path}")
        if phase3.get("status") != "passed" or phase3.get("degraded") is True:
            raise ValueError(f"Dense index refuses non-passed/degraded run: {phase3_path}")

        workspace = source_workspace / video_id / run_id
        feature_path = workspace / "feature_manifest.json"
        vector_path = workspace / "siglip2.npy"
        metadata_path = workspace / "siglip2_metadata.jsonl"
        if not feature_path.exists() or not vector_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Incomplete dense workspace: {workspace}")
        feature_sha = sha256_file(feature_path)
        if feature_sha != str(phase3.get("feature_manifest_sha256") or ""):
            raise ValueError(f"Stale feature manifest for {video_id}")
        feature = _read_json(feature_path)
        if feature.get("status") != "passed" or not feature.get("hard_feature_complete"):
            raise ValueError(f"Incomplete hard features for {video_id}")

        vectors = np.load(vector_path, allow_pickle=False)
        metadata = _read_jsonl(metadata_path)
        candidate_count = int(phase3.get("candidate_count", -1))
        if vectors.ndim != 2 or len(vectors) != len(metadata) or len(vectors) != candidate_count:
            raise ValueError(
                f"Dense row mismatch for {video_id}: vectors={len(vectors)}, "
                f"metadata={len(metadata)}, candidates={candidate_count}"
            )
        dim = int(vectors.shape[1])
        if expected_dim is None:
            expected_dim = dim
        elif dim != expected_dim:
            raise ValueError(f"Embedding dimension mismatch for {video_id}: {dim} != {expected_dim}")

        siglip_config = feature.get("feature_config", {}).get("siglip2", {})
        model_contract = (
            str(siglip_config.get("model_name") or ""),
            str(siglip_config.get("resolved_model_revision") or ""),
        )
        if not all(model_contract):
            raise ValueError(f"Missing resolved SigLIP2 contract for {video_id}")
        if expected_model is None:
            expected_model = model_contract
        elif model_contract != expected_model:
            raise ValueError(f"Encoder lineage mismatch for {video_id}")

        offline_config = json.dumps(
            {
                "adapter_config": phase3.get("adapter_config"),
                "selection_config": phase3.get("selection_config"),
                "feature_config": phase3.get("feature_config"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        offline_config_sha = hashlib.sha256(offline_config.encode("utf-8")).hexdigest()
        if expected_offline_config is None:
            expected_offline_config = offline_config_sha
        elif offline_config_sha != expected_offline_config:
            raise ValueError(f"Offline config lineage mismatch for {video_id}")

        captions, _ = _records_by_candidate(workspace / "captions.jsonl")
        ocr, _ = _records_by_candidate(workspace / "ocr.jsonl")
        objects, _ = _records_by_candidate(workspace / "objects.jsonl")
        # A cached-feature reselect keeps embeddings/modalities in the original
        # workspace but publishes its selection/event ledgers under the new run.
        # Prefer those run-local ledgers so the dense index cannot silently mix
        # a new selector with the baseline protection map.
        selection_workspace = (
            source_output_root / "work" / "keyframe_v3" / video_id / run_id
        )
        score_path = selection_workspace / "candidate_scores.jsonl"
        event_path = selection_workspace / "protected_events.jsonl"
        if not score_path.is_file():
            score_path = workspace / "candidate_scores.jsonl"
        if not event_path.is_file():
            event_path = workspace / "protected_events.jsonl"
        expected_score_sha = str(phase3.get("candidate_scores_sha256") or "")
        expected_event_sha = str(phase3.get("protected_events_sha256") or "")
        if expected_score_sha and sha256_file(score_path) != expected_score_sha:
            raise ValueError(f"Selection score lineage mismatch for {video_id}")
        if expected_event_sha and sha256_file(event_path) != expected_event_sha:
            raise ValueError(f"Protected-event lineage mismatch for {video_id}")
        scores, _ = _records_by_candidate(score_path)
        protected_events = _read_jsonl(event_path)

        previous_embedding_index = -1
        seen_candidate_ids: set[str] = set()
        for local_row, metadata_record in enumerate(metadata):
            embedding_index = int(metadata_record.get("embedding_index", -1))
            candidate_id = str(metadata_record.get("candidate_id") or "")
            if embedding_index != local_row or embedding_index <= previous_embedding_index:
                raise ValueError(f"Non-canonical embedding order for {video_id} row {local_row}")
            if not candidate_id or candidate_id in seen_candidate_ids:
                raise ValueError(f"Invalid/duplicate candidate id for {video_id} row {local_row}")
            previous_embedding_index = embedding_index
            seen_candidate_ids.add(candidate_id)

            timestamp = float(metadata_record.get("timestamp", 0.0))
            score_record = scores.get(candidate_id, {})
            keyframe_path = Path(str(metadata_record.get("keyframe_path") or ""))
            source_video_path = Path(str(metadata_record.get("source_video_path") or ""))
            event_ids = [
                str(event.get("event_id"))
                for event in protected_events
                if candidate_id in set(event.get("candidate_ids") or [])
            ]
            dense_records.append(
                {
                    "dense_row": row_offset + local_row,
                    "candidate_id": candidate_id,
                    "frame_id": str(metadata_record.get("frame_id") or ""),
                    "video_id": video_id,
                    "shot_id": str(metadata_record.get("shot_id") or ""),
                    "segment_id": str(
                        metadata_record.get("segment_id")
                        or metadata_record.get("shot_id")
                        or ""
                    ),
                    "shot_index": metadata_record.get("shot_index"),
                    "shot_start": metadata_record.get("shot_start"),
                    "shot_end": metadata_record.get("shot_end"),
                    "timestamp": timestamp,
                    "frame_index": int(metadata_record.get("frame_index", 0)),
                    "candidate_pool_run_id": run_id,
                    "candidate_image": _relative_reference(keyframe_path, run_root),
                    "source_video": _relative_reference(source_video_path, run_root),
                    "caption": str(captions.get(candidate_id, {}).get("caption") or ""),
                    "ocr_text": str(ocr.get(candidate_id, {}).get("ocr_text") or ""),
                    "objects": _object_labels(objects.get(candidate_id, {})),
                    "importance_score": _safe_float(score_record.get("importance_score")),
                    "semantic_novelty": _safe_float(score_record.get("semantic_novelty")),
                    "component_scores": score_record.get("component_scores", {}),
                    "protected_event_ids": event_ids,
                    "selected_offline": bool(score_record.get("selected", False)),
                }
            )

        vectors32 = np.ascontiguousarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors32, axis=1)
        if not np.all(np.isfinite(vectors32)) or not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"Dense shard is not finite/L2-normalized for {video_id}")
        vector_shards.append(vectors32)
        source_entries.append(
            {
                "video_id": video_id,
                "candidate_pool_run_id": run_id,
                "phase3_manifest": _relative_reference(phase3_path, run_root),
                "phase3_manifest_sha256": sha256_file(phase3_path),
                "feature_manifest": _relative_reference(feature_path, run_root),
                "feature_manifest_sha256": feature_sha,
                "embedding_shard": _relative_reference(vector_path, run_root),
                "embedding_shard_sha256": sha256_file(vector_path),
                "embedding_metadata": _relative_reference(metadata_path, run_root),
                "embedding_metadata_sha256": sha256_file(metadata_path),
                "selection_scores": _relative_reference(score_path, run_root),
                "selection_scores_sha256": sha256_file(score_path),
                "protected_events": _relative_reference(event_path, run_root),
                "protected_events_sha256": sha256_file(event_path),
                "row_start": row_offset,
                "row_end": row_offset + len(vectors),
                "row_count": len(vectors),
            }
        )
        row_offset += len(vectors)

    if not vector_shards or expected_dim is None or expected_model is None:
        raise ValueError("Dense index cannot be empty")
    all_vectors = np.ascontiguousarray(np.concatenate(vector_shards, axis=0), dtype=np.float32)
    if len(all_vectors) != len(dense_records):
        raise AssertionError("Dense vectors and frame map diverged before publication")

    paths.root.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(paths.vectors, all_vectors)
    frame_map_content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in dense_records
    )
    _atomic_write_text(paths.frame_map, frame_map_content)
    _atomic_write_faiss(paths.index, all_vectors)

    manifest: dict[str, Any] = {
        "version": DENSE_INDEX_VERSION,
        "index_type": "IndexFlatIP",
        "similarity": "cosine",
        "normalized": True,
        "candidate_count": len(dense_records),
        "vector_dim": expected_dim,
        "encoder": {
            "model_name": expected_model[0],
            "resolved_model_revision": expected_model[1],
        },
        "offline_config_sha256": expected_offline_config,
        "artifacts": {
            "index": {
                "path": _relative_reference(paths.index, run_root),
                "sha256": sha256_file(paths.index),
            },
            "vectors": {
                "path": _relative_reference(paths.vectors, run_root),
                "sha256": sha256_file(paths.vectors),
            },
            "frame_map": {
                "path": _relative_reference(paths.frame_map, run_root),
                "sha256": sha256_file(paths.frame_map),
            },
        },
        "sources": source_entries,
    }
    _atomic_write_text(
        paths.manifest,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def validate_dense_index(run_root: Path, *, verify_sources: bool = True) -> dict[str, Any]:
    """Validate row order, checksums, encoder lineage and optional source shards."""
    run_root = run_root.resolve()
    paths = DenseIndexPaths.from_run_root(run_root)
    manifest = _read_json(paths.manifest)
    if int(manifest.get("version", -1)) != DENSE_INDEX_VERSION:
        raise ValueError("Unsupported dense manifest version")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Dense manifest is missing artifacts")
    resolved_artifacts: dict[str, Path] = {}
    for name in ("index", "vectors", "frame_map"):
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Dense manifest is missing artifact {name}")
        path = resolve_run_reference(run_root, str(record.get("path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"Dense artifact not found: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != str(record.get("sha256") or ""):
            raise ValueError(f"Dense {name} checksum mismatch")
        resolved_artifacts[name] = path

    vectors = np.load(resolved_artifacts["vectors"], mmap_mode="r", allow_pickle=False)
    records = _read_jsonl(resolved_artifacts["frame_map"])
    candidate_count = int(manifest.get("candidate_count", -1))
    vector_dim = int(manifest.get("vector_dim", -1))
    if vectors.ndim != 2 or vectors.shape != (candidate_count, vector_dim):
        raise ValueError(
            f"Dense vector shape mismatch: {vectors.shape} != {(candidate_count, vector_dim)}"
        )
    if len(records) != candidate_count:
        raise ValueError("Dense frame-map row count mismatch")
    candidate_ids: set[str] = set()
    for row, record in enumerate(records):
        if int(record.get("dense_row", -1)) != row:
            raise ValueError(f"Dense row order mismatch at row {row}")
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"Invalid/duplicate candidate id at dense row {row}")
        candidate_ids.add(candidate_id)
        for path_key in ("candidate_image", "source_video"):
            resolve_run_reference(run_root, str(record.get(path_key) or ""))

    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency preflight
        raise RuntimeError("faiss is required to validate the dense index") from exc
    index = faiss.read_index(str(resolved_artifacts["index"]))
    if int(index.ntotal) != candidate_count or int(index.d) != vector_dim:
        raise ValueError("Dense FAISS row/dimension mismatch")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Dense manifest has no source lineage")
    expected_row = 0
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("Invalid dense source entry")
        row_start = int(source.get("row_start", -1))
        row_end = int(source.get("row_end", -1))
        row_count = int(source.get("row_count", -1))
        if row_start != expected_row or row_end - row_start != row_count:
            raise ValueError("Dense per-video offsets are not contiguous")
        expected_row = row_end
        if verify_sources:
            source_keys = [
                "phase3_manifest",
                "feature_manifest",
                "embedding_shard",
                "embedding_metadata",
            ]
            source_keys.extend(
                key
                for key in ("selection_scores", "protected_events")
                if source.get(key)
            )
            for key in source_keys:
                source_path = resolve_run_reference(run_root, str(source.get(key) or ""))
                if not source_path.is_file():
                    raise FileNotFoundError(f"Dense source not found: {source_path}")
                expected_sha = str(source.get(f"{key}_sha256") or "")
                if sha256_file(source_path) != expected_sha:
                    raise ValueError(f"Dense source checksum mismatch: {key}")
    if expected_row != candidate_count:
        raise ValueError("Dense source offsets do not cover every row")

    return {
        "status": "passed",
        "run_root": run_root.as_posix(),
        "candidate_count": candidate_count,
        "vector_dim": vector_dim,
        "faiss_ntotal": int(index.ntotal),
        "source_video_count": len(sources),
        "sources_verified": bool(verify_sources),
    }


class DenseCandidateIndex:
    """Memory-mapped vectors, metadata and FAISS search for advanced retrieval."""

    def __init__(self, run_root: Path, *, verify_sources: bool = False) -> None:
        validate_dense_index(run_root, verify_sources=verify_sources)
        self.run_root = run_root.resolve()
        self.paths = DenseIndexPaths.from_run_root(self.run_root)
        self.manifest = _read_json(self.paths.manifest)
        artifact_paths = self.manifest["artifacts"]
        self.vectors = np.load(
            resolve_run_reference(self.run_root, artifact_paths["vectors"]["path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.records = _read_jsonl(
            resolve_run_reference(self.run_root, artifact_paths["frame_map"]["path"])
        )
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faiss is required to search the dense index") from exc
        self.index = faiss.read_index(
            str(resolve_run_reference(self.run_root, artifact_paths["index"]["path"]))
        )
        self.rows_by_clip: dict[tuple[str, str], list[int]] = {}
        for row, record in enumerate(self.records):
            clip_id = str(record.get("segment_id") or record.get("shot_id") or "")
            self.rows_by_clip.setdefault((str(record["video_id"]), clip_id), []).append(row)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Dense query vector must have a positive finite norm")
        vector /= norm
        limit = min(max(0, int(top_k)), len(self.records))
        if limit == 0:
            return []
        scores, indices = self.index.search(np.ascontiguousarray(vector), limit)
        return [
            (int(row), float(score))
            for row, score in zip(indices[0], scores[0])
            if int(row) >= 0
        ]

    def rows_for_clips(self, clips: Iterable[tuple[str, str]]) -> list[int]:
        rows: list[int] = []
        for clip in clips:
            rows.extend(self.rows_by_clip.get(clip, []))
        return list(dict.fromkeys(rows))
