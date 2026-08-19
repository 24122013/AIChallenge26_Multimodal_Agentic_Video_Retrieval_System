"""Validated corpus-wide SigLIP2 dense-candidate index.

The selected-keyframe visual index remains the coarse retrieval index.  This
module owns the separate full dense-candidate pool used for global rescue and
per-clip CSES.  Online queries always use FAISS; ``vectors`` is exposed only so
the bounded CSES/rerank stages can score rows that FAISS has already narrowed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend.app.services.indexing.build_faiss_index import ARTIFACT_TAG, frame_map_record


DENSE_ARTIFACT_TAG = f"{ARTIFACT_TAG}_dense"
DENSE_INDEX_NAME = f"{DENSE_ARTIFACT_TAG}_flat_ip.faiss"
DENSE_METADATA_NAME = f"{DENSE_ARTIFACT_TAG}_faiss_metadata.jsonl"
DENSE_FRAME_MAP_NAME = f"{DENSE_ARTIFACT_TAG}_frame_map.json"
DENSE_MANIFEST_NAME = f"{DENSE_ARTIFACT_TAG}_faiss_manifest.json"
DENSE_REPORT_NAME = f"{DENSE_ARTIFACT_TAG}_index_report.json"
DENSE_MANIFEST_SCHEMA_VERSION = "1.2"
DENSE_ARTIFACT_ROLE = "dense_candidate_index"


@dataclass(frozen=True)
class DenseCandidateIndexConfig:
    """Paths for one atomically published full dense-candidate bundle."""

    index_path: Path = Path("data/indexes") / DENSE_INDEX_NAME
    metadata_path: Path = Path("data/metadata") / DENSE_METADATA_NAME
    frame_map_path: Path = Path("data/metadata") / DENSE_FRAME_MAP_NAME
    manifest_path: Path = Path("data/metadata") / DENSE_MANIFEST_NAME
    report_path: Path = Path("data/metadata") / DENSE_REPORT_NAME

    def __post_init__(self) -> None:
        for field_name in (
            "index_path",
            "metadata_path",
            "frame_map_path",
            "manifest_path",
            "report_path",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))


@dataclass(frozen=True)
class ValidatedDenseCandidateArtifacts:
    config: DenseCandidateIndexConfig
    manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    rows_by_clip: Mapping[tuple[str, str], tuple[int, ...]]
    index: Any
    vectors: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_generation(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(hashes),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Dense artifact must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Dense metadata row must be an object: {path}:{line_number}"
                )
            records.append(value)
    return records


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "faiss is required for dense-candidate retrieval. "
            "Install project requirements first."
        ) from exc
    return faiss


def _verify_declared_artifact(
    path: Path,
    declaration: object,
    *,
    label: str,
) -> str:
    if not isinstance(declaration, dict):
        raise ValueError(f"Dense manifest is missing the {label} artifact declaration")
    if declaration.get("filename") != path.name:
        raise ValueError(f"Dense manifest has invalid {label} filename lineage")
    declared_sha256 = declaration.get("sha256")
    if not isinstance(declared_sha256, str) or not declared_sha256:
        raise ValueError(f"Dense manifest has invalid {label} checksum")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != declared_sha256:
        raise ValueError(f"Dense {label} checksum does not match manifest")
    return actual_sha256


def _validate_encoder_contract(manifest: Mapping[str, Any]) -> int:
    encoder = manifest.get("encoder")
    if not isinstance(encoder, dict):
        raise ValueError("Dense manifest is missing the SigLIP2 encoder contract")
    if encoder.get("model_family") != "siglip2":
        raise ValueError("Dense candidate index must use SigLIP2 vectors")
    if encoder.get("normalized") is not True:
        raise ValueError("Dense candidate vectors must be normalized")
    if encoder.get("similarity") != "cosine" or encoder.get("output_dtype") != "float32":
        raise ValueError("Dense candidate index requires cosine float32 vectors")
    for field_name in ("model_name", "model_revision", "processor_name"):
        if not isinstance(encoder.get(field_name), str) or not encoder[field_name]:
            raise ValueError(f"Dense encoder contract has invalid {field_name}")
    try:
        vector_dim = int(encoder["vector_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Dense encoder contract has invalid vector_dim") from exc
    if vector_dim <= 0:
        raise ValueError("Dense encoder vector_dim must be positive")
    return vector_dim


def _validate_records(
    records: list[dict[str, Any]],
    *,
    vector_dim: int,
) -> dict[tuple[str, str], tuple[int, ...]]:
    if not records:
        raise ValueError("Dense candidate metadata must not be empty")
    candidate_ids: set[str] = set()
    mutable_rows: dict[tuple[str, str], list[int]] = {}
    for row, record in enumerate(records):
        if record.get("faiss_index") != row:
            raise ValueError(f"Dense metadata faiss_index mismatch at row {row}")
        candidate_id = str(record.get("candidate_id") or "")
        video_id = str(record.get("video_id") or "")
        clip_id = str(record.get("segment_id") or record.get("shot_id") or "")
        if not candidate_id or not video_id or not clip_id:
            raise ValueError(
                f"Dense metadata row {row} requires candidate_id, video_id, and clip id"
            )
        if candidate_id in candidate_ids:
            raise ValueError(f"Duplicate dense candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        if record.get("vector_dim") != vector_dim:
            raise ValueError(f"Dense metadata vector_dim mismatch at row {row}")
        if record.get("normalized") is not True:
            raise ValueError(f"Dense metadata row {row} is not normalized")
        if not isinstance(record.get("caption"), str):
            raise ValueError(f"Dense metadata row {row} has invalid caption evidence")
        if not isinstance(record.get("ocr_text"), str):
            raise ValueError(f"Dense metadata row {row} has invalid OCR evidence")
        if not isinstance(record.get("objects"), list) or any(
            not isinstance(value, str) for value in record["objects"]
        ):
            raise ValueError(f"Dense metadata row {row} has invalid object evidence")
        if not isinstance(record.get("protected_event_ids"), list) or any(
            not isinstance(value, str) for value in record["protected_event_ids"]
        ):
            raise ValueError(f"Dense metadata row {row} has invalid protected events")
        mutable_rows.setdefault((video_id, clip_id), []).append(row)
    return {key: tuple(rows) for key, rows in mutable_rows.items()}


def _index_vectors(index: Any, faiss_module: Any, count: int, dim: int) -> np.ndarray:
    """Expose IndexFlat vectors, preferring a zero-copy FAISS view."""

    try:
        pointer = index.get_xb()
        flat = faiss_module.rev_swig_ptr(pointer, count * dim)
        vectors = np.asarray(flat, dtype=np.float32).reshape(count, dim)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        vectors = np.empty((count, dim), dtype=np.float32)
        index.reconstruct_n(0, count, vectors)
    if vectors.shape != (count, dim) or vectors.dtype != np.float32:
        raise ValueError("Dense FAISS vectors have an invalid shape or dtype")
    if not np.isfinite(vectors).all():
        raise ValueError("Dense FAISS vectors contain NaN or Inf")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0) or not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("Dense FAISS vectors are not unit-normalized")
    vectors.setflags(write=False)
    return vectors


def validate_dense_candidate_artifacts(
    config: DenseCandidateIndexConfig | None = None,
) -> ValidatedDenseCandidateArtifacts:
    """Load one complete dense bundle and fail closed on any lineage drift."""

    config = config or DenseCandidateIndexConfig()
    paths = (
        config.index_path,
        config.metadata_path,
        config.frame_map_path,
        config.manifest_path,
        config.report_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing dense-candidate artifact: {path}")

    manifest_sha256 = _sha256_file(config.manifest_path)
    manifest = _read_json_object(config.manifest_path)
    if manifest.get("schema_version") != DENSE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported dense-candidate manifest schema")
    if manifest.get("artifact_role") != DENSE_ARTIFACT_ROLE:
        raise ValueError("Manifest is not a full dense-candidate index")
    if manifest.get("index_type") != "IndexFlatIP" or manifest.get("metric") != "ip":
        raise ValueError("Dense candidate retrieval requires IndexFlatIP with IP metric")
    vector_dim = _validate_encoder_contract(manifest)

    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("Dense manifest is missing bundle artifact checksums")
    artifact_paths = {
        "index": config.index_path,
        "metadata": config.metadata_path,
        "frame_map": config.frame_map_path,
        "report": config.report_path,
    }
    hashes = {
        label: _verify_declared_artifact(path, declared.get(label), label=label)
        for label, path in artifact_paths.items()
    }
    if manifest.get("bundle_generation") != _bundle_generation(hashes):
        raise ValueError("Dense bundle generation does not match artifact checksums")

    records = _read_jsonl(config.metadata_path)
    rows_by_clip = _validate_records(records, vector_dim=vector_dim)
    raw_frame_map = _read_json_object(config.frame_map_path)
    expected_frame_map = {
        str(row): frame_map_record(record) for row, record in enumerate(records)
    }
    if raw_frame_map != expected_frame_map:
        raise ValueError("Dense frame map does not match dense metadata order")

    report = _read_json_object(config.report_path)
    if report.get("status") != "passed":
        raise ValueError("Dense candidate index report is not passed")
    declared_count = int(manifest.get("vector_count", -1))
    if (
        declared_count != len(records)
        or int(manifest.get("metadata_record_count", -1)) != len(records)
        or int(report.get("vector_count", -1)) != len(records)
        or int(report.get("metadata_record_count", -1)) != len(records)
    ):
        raise ValueError("Dense manifest/report/metadata counts do not match")
    if int(manifest.get("clip_count", -1)) != len(rows_by_clip):
        raise ValueError("Dense manifest clip count does not match metadata")

    faiss_module = _require_faiss()
    index = faiss_module.read_index(config.index_path.as_posix())
    if type(index).__name__ != "IndexFlatIP":
        raise ValueError("Dense FAISS artifact is not IndexFlatIP")
    if int(index.d) != vector_dim or int(index.ntotal) != len(records):
        raise ValueError("Dense FAISS count or dimension does not match manifest")
    vectors = _index_vectors(index, faiss_module, len(records), vector_dim)

    # Close the checksum/read TOCTOU window after FAISS and metadata are loaded.
    for label, path in artifact_paths.items():
        if _sha256_file(path) != hashes[label]:
            raise ValueError(f"Dense {label} changed while artifacts were loading")
    if _sha256_file(config.manifest_path) != manifest_sha256:
        raise ValueError("Dense manifest changed while artifacts were loading")

    return ValidatedDenseCandidateArtifacts(
        config=config,
        manifest=manifest,
        records=tuple(records),
        rows_by_clip=rows_by_clip,
        index=index,
        vectors=vectors,
    )


class FaissDenseCandidateIndex:
    """Production ``DenseCandidateIndex`` backed by one validated FAISS index."""

    def __init__(self, config: DenseCandidateIndexConfig | None = None) -> None:
        self.artifacts = validate_dense_candidate_artifacts(config)
        self.records = self.artifacts.records
        self.vectors = self.artifacts.vectors
        self.rows_by_clip = self.artifacts.rows_by_clip
        self.encoder_contract = self.artifacts.manifest["encoder"]

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        requested = int(top_k)
        if requested <= 0:
            raise ValueError("dense top_k must be positive")
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("Dense query vector dimension does not match the index")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Dense query vector must have a positive finite norm")
        query = np.ascontiguousarray((query / norm).reshape(1, -1), dtype=np.float32)
        limit = min(requested, len(self.records))
        scores, rows = self.artifacts.index.search(query, limit)
        return [
            (int(row), float(score))
            for score, row in zip(scores[0].tolist(), rows[0].tolist())
            if int(row) >= 0
        ]


# A semantic alias for callers that do not need to name the FAISS backend.
DenseCandidateIndex = FaissDenseCandidateIndex


__all__ = [
    "DENSE_ARTIFACT_ROLE",
    "DENSE_ARTIFACT_TAG",
    "DENSE_FRAME_MAP_NAME",
    "DENSE_INDEX_NAME",
    "DENSE_MANIFEST_NAME",
    "DENSE_MANIFEST_SCHEMA_VERSION",
    "DENSE_METADATA_NAME",
    "DENSE_REPORT_NAME",
    "DenseCandidateIndex",
    "DenseCandidateIndexConfig",
    "FaissDenseCandidateIndex",
    "ValidatedDenseCandidateArtifacts",
    "validate_dense_candidate_artifacts",
]
