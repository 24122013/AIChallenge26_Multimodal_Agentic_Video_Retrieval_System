"""Dense-only BGE-M3 indexing and retrieval with auditable artifacts.

This module deliberately uses only the 1024-dimensional dense representation
from ``BAAI/bge-m3``.  BM25 remains a separate retriever; BGE-M3 sparse and
ColBERT outputs are outside this contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from backend.app.models.retrieval import RetrievalResult


DEFAULT_BGE_M3_MODEL = "BAAI/bge-m3"
DEFAULT_BGE_M3_REVISION = "main"
BGE_M3_DIMENSION = 1024
BGE_M3_INDEX_NAME = "bge_m3_flat_ip.faiss"
BGE_M3_FRAME_MAP_NAME = "bge_m3_frame_map.json"
BGE_M3_MANIFEST_NAME = "bge_m3_manifest.json"
BGE_M3_SCHEMA_VERSION = "1.0"
ORDERING_POLICY = "canonical_video_timestamp_frame_v1"


class TextEncoder(Protocol):
    """Small injection surface used by builders, searchers, and unit tests."""

    def __call__(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class BgeM3ArtifactPaths:
    root: Path
    index: Path
    frame_map: Path
    manifest: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "BgeM3ArtifactPaths":
        value = Path(root)
        return cls(
            root=value,
            index=value / BGE_M3_INDEX_NAME,
            frame_map=value / BGE_M3_FRAME_MAP_NAME,
            manifest=value / BGE_M3_MANIFEST_NAME,
        )


@dataclass(frozen=True)
class BgeM3BuildReport:
    artifact_root: str
    model_name: str
    model_revision: str
    vector_count: int
    dimension: int
    source_records_sha256: str
    documents_sha256: str
    source_kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_root": self.artifact_root,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "source_records_sha256": self.source_records_sha256,
            "documents_sha256": self.documents_sha256,
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True)
class ValidatedBgeM3Artifacts:
    paths: BgeM3ArtifactPaths
    manifest: Mapping[str, object]
    frame_records: tuple[Mapping[str, object], ...]
    index: Any


def field_tagged_document(record: Mapping[str, object]) -> str:
    """Build the exact caption/OCR/object text passed to BGE components.

    ASR is intentionally not inspected, even if a legacy metadata record still
    contains an ASR-shaped key.
    """

    caption = _first_text(
        record,
        ("caption", "captions_aggregated", "caption_text", "description"),
    )
    ocr = _ocr_text(record.get("ocr_text")) or _ocr_text(record.get("ocr"))
    objects = _object_text(record.get("objects"))
    return "\n".join(
        (
            f"[CAPTION] {caption}",
            f"[OCR] {ocr}",
            f"[OBJECTS] {objects}",
        )
    ).strip()


def has_retrievable_text(record: Mapping[str, object]) -> bool:
    document = field_tagged_document(record)
    return any(
        line.partition("]")[2].strip()
        for line in document.splitlines()
        if "]" in line
    )


def _source_kind(
    records: Sequence[Mapping[str, object]],
    *,
    canonical_only: bool,
) -> str:
    selected = all(
        str(record.get("artifact_role") or "").casefold()
        == "selected_keyframe"
        and bool(str(record.get("video_id") or "").strip())
        and bool(str(record.get("frame_id") or "").strip())
        and bool(
            str(
                record.get("keyframe_path")
                or record.get("frame_path")
                or record.get("thumbnail_path")
                or ""
            ).strip()
        )
        for record in records
    )
    segments = all(
        bool(str(record.get("video_id") or "").strip())
        and bool(str(record.get("segment_id") or record.get("shot_id") or "").strip())
        and isinstance(record.get("keyframe_selection"), list)
        and bool(record.get("keyframe_selection"))
        for record in records
    )
    if selected:
        return "selected_keyframes"
    if segments:
        return "canonical_segments"
    if canonical_only:
        raise ValueError(
            "canonical-only BGE-M3 indexing requires selected_keyframe records "
            "or Phase-4 segments with non-empty keyframe_selection lineage"
        )
    return "legacy_metadata"


def build_bge_m3_index(
    records: Sequence[Mapping[str, object]],
    output_root: str | Path,
    *,
    encoder: TextEncoder | None = None,
    model_name: str = DEFAULT_BGE_M3_MODEL,
    model_revision: str = DEFAULT_BGE_M3_REVISION,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    canonical_only: bool = False,
) -> BgeM3BuildReport:
    """Build three deterministic artifacts from existing metadata records."""

    if not records:
        raise ValueError("BGE-M3 indexing requires at least one metadata record")
    dense_candidate_count = sum(
        str(record.get("artifact_role") or "").casefold() == "dense_candidate"
        for record in records
    )
    if dense_candidate_count:
        raise ValueError(
            "BGE-M3 index refuses dense_candidate metadata; build it from "
            "canonical selected-keyframe/segment metadata only"
        )
    source_kind = _source_kind(records, canonical_only=canonical_only)
    if not str(model_name).strip() or not str(model_revision).strip():
        raise ValueError("BGE-M3 model name and revision must be non-empty")
    paths = BgeM3ArtifactPaths.from_root(output_root)
    paths.root.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        (_json_safe_mapping(record) for record in records),
        key=_canonical_order_key,
    )
    documents = [field_tagged_document(record) for record in ordered]
    scorer = encoder or LazyBgeM3Encoder(
        model_name=model_name,
        model_revision=model_revision,
        batch_size=batch_size,
        device=device,
        cache_dir=Path(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )
    vectors = _encode_and_normalize(scorer, documents, expected_rows=len(ordered))
    resolved_revision = str(getattr(scorer, "resolved_revision", model_revision))

    faiss = _import_faiss()
    index = faiss.IndexFlatIP(BGE_M3_DIMENSION)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    source_hashes = [_canonical_sha256(record) for record in ordered]
    document_hashes = [_sha256_text(document) for document in documents]
    frame_records = [
        {
            "row": row,
            "source_sha256": source_hashes[row],
            "document_sha256": document_hashes[row],
            "metadata": ordered[row],
        }
        for row in range(len(ordered))
    ]
    frame_map_payload = {
        "schema_version": BGE_M3_SCHEMA_VERSION,
        "ordering": {
            "policy": ORDERING_POLICY,
            "keys": ["video_id", "timestamp", "frame_id", "segment_id", "shot_id"],
        },
        "records": frame_records,
    }

    _atomic_write_faiss(index, paths.index, faiss)
    _atomic_write_json(frame_map_payload, paths.frame_map)
    source_records_sha256 = _ordered_digest(source_hashes)
    documents_sha256 = _ordered_digest(document_hashes)
    manifest = {
        "schema_version": BGE_M3_SCHEMA_VERSION,
        "model": {
            "name": model_name,
            "revision": resolved_revision,
            "representation": "dense",
            "dimension": BGE_M3_DIMENSION,
            "normalized": True,
            "pooling": "cls",
            "output_dtype": "float32",
        },
        "index_type": "IndexFlatIP",
        "metric": "ip",
        "vector_count": len(ordered),
        "ordering": frame_map_payload["ordering"],
        "source_hashes": {
            "records_sha256": source_records_sha256,
            "documents_sha256": documents_sha256,
        },
        "source_contract": {
            "canonical_only": bool(canonical_only),
            "source_kind": source_kind,
            "dense_candidates_rejected": True,
        },
        "artifacts": {
            "index": {
                "filename": paths.index.name,
                "sha256": _sha256_file(paths.index),
            },
            "frame_map": {
                "filename": paths.frame_map.name,
                "sha256": _sha256_file(paths.frame_map),
            },
        },
    }
    _atomic_write_json(manifest, paths.manifest)
    validate_bge_m3_artifacts(paths.root)
    return BgeM3BuildReport(
        artifact_root=paths.root.as_posix(),
        model_name=model_name,
        model_revision=resolved_revision,
        vector_count=len(ordered),
        dimension=BGE_M3_DIMENSION,
        source_records_sha256=source_records_sha256,
        documents_sha256=documents_sha256,
        source_kind=source_kind,
    )


def validate_bge_m3_artifacts(
    root: str | Path,
    *,
    expected_model_name: str | None = DEFAULT_BGE_M3_MODEL,
    expected_model_revision: str | None = None,
) -> ValidatedBgeM3Artifacts:
    """Load and fully validate BGE-M3 artifacts before they can be searched."""

    paths = BgeM3ArtifactPaths.from_root(root)
    for path in (paths.index, paths.frame_map, paths.manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Missing BGE-M3 artifact: {path}")
    manifest = _read_json_object(paths.manifest)
    if manifest.get("schema_version") != BGE_M3_SCHEMA_VERSION:
        raise ValueError("Unsupported BGE-M3 manifest schema")
    if manifest.get("index_type") != "IndexFlatIP" or manifest.get("metric") != "ip":
        raise ValueError("BGE-M3 dense retrieval requires IndexFlatIP with IP metric")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("BGE-M3 manifest is missing the model contract")
    if model.get("representation") != "dense":
        raise ValueError("BGE-M3 manifest must declare dense-only representation")
    if model.get("dimension") != BGE_M3_DIMENSION:
        raise ValueError(f"BGE-M3 manifest dimension must be {BGE_M3_DIMENSION}")
    if model.get("normalized") is not True or model.get("output_dtype") != "float32":
        raise ValueError("BGE-M3 manifest must declare normalized float32 vectors")
    if expected_model_name is not None and model.get("name") != expected_model_name:
        raise ValueError(
            f"BGE-M3 model mismatch: {model.get('name')!r} != {expected_model_name!r}"
        )
    if expected_model_revision is not None and model.get("revision") != expected_model_revision:
        raise ValueError("BGE-M3 model revision does not match the requested revision")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("BGE-M3 manifest is missing artifact checksums")
    _verify_artifact(paths.index, artifacts.get("index"), BGE_M3_INDEX_NAME)
    _verify_artifact(paths.frame_map, artifacts.get("frame_map"), BGE_M3_FRAME_MAP_NAME)

    frame_map = _read_json_object(paths.frame_map)
    if frame_map.get("schema_version") != BGE_M3_SCHEMA_VERSION:
        raise ValueError("BGE-M3 frame-map schema does not match the manifest")
    ordering = manifest.get("ordering")
    if frame_map.get("ordering") != ordering:
        raise ValueError("BGE-M3 frame-map ordering contract does not match manifest")
    if not isinstance(ordering, dict) or ordering.get("policy") != ORDERING_POLICY:
        raise ValueError("Unsupported BGE-M3 ordering policy")
    raw_records = frame_map.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("BGE-M3 frame map must contain at least one record")

    source_hashes: list[str] = []
    document_hashes: list[str] = []
    last_key: tuple[object, ...] | None = None
    frame_records: list[Mapping[str, object]] = []
    for expected_row, item in enumerate(raw_records):
        if not isinstance(item, dict) or item.get("row") != expected_row:
            raise ValueError("BGE-M3 frame-map rows are missing or out of order")
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"BGE-M3 frame-map row {expected_row} has no metadata")
        source_sha = _canonical_sha256(metadata)
        document_sha = _sha256_text(field_tagged_document(metadata))
        if item.get("source_sha256") != source_sha:
            raise ValueError(f"BGE-M3 source hash mismatch at row {expected_row}")
        if item.get("document_sha256") != document_sha:
            raise ValueError(f"BGE-M3 document hash mismatch at row {expected_row}")
        order_key = _canonical_order_key(metadata)
        if last_key is not None and order_key < last_key:
            raise ValueError("BGE-M3 frame-map metadata violates canonical ordering")
        last_key = order_key
        source_hashes.append(source_sha)
        document_hashes.append(document_sha)
        frame_records.append(item)

    source_contract = manifest.get("source_contract")
    if not isinstance(source_contract, dict):
        raise ValueError("BGE-M3 manifest is missing the source contract")
    canonical_only = source_contract.get("canonical_only")
    if not isinstance(canonical_only, bool):
        raise ValueError("BGE-M3 source contract has invalid canonical_only")
    if source_contract.get("dense_candidates_rejected") is not True:
        raise ValueError("BGE-M3 source contract must reject dense candidates")
    metadata_records = [
        item["metadata"]
        for item in frame_records
        if isinstance(item.get("metadata"), Mapping)
    ]
    computed_source_kind = _source_kind(
        metadata_records,
        canonical_only=canonical_only,
    )
    if source_contract.get("source_kind") != computed_source_kind:
        raise ValueError("BGE-M3 source kind does not match frame-map lineage")

    declared_sources = manifest.get("source_hashes")
    expected_sources = {
        "records_sha256": _ordered_digest(source_hashes),
        "documents_sha256": _ordered_digest(document_hashes),
    }
    if declared_sources != expected_sources:
        raise ValueError("BGE-M3 source lineage hashes do not match frame-map metadata")

    faiss = _import_faiss()
    index = faiss.read_index(paths.index.as_posix())
    if int(index.d) != BGE_M3_DIMENSION:
        raise ValueError("BGE-M3 FAISS dimension does not match the manifest")
    vector_count = int(manifest.get("vector_count", -1))
    if vector_count != len(frame_records) or int(index.ntotal) != vector_count:
        raise ValueError("BGE-M3 index, frame map, and manifest counts do not match")
    return ValidatedBgeM3Artifacts(
        paths=paths,
        manifest=manifest,
        frame_records=tuple(frame_records),
        index=index,
    )


class BgeM3DenseSearchEngine:
    """Validated IndexFlatIP search with a lazily allocated query encoder."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        encoder: TextEncoder | None = None,
        model_name: str = DEFAULT_BGE_M3_MODEL,
        model_revision: str | None = None,
        batch_size: int = 16,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.artifacts = validate_bge_m3_artifacts(
            artifact_root,
            expected_model_name=model_name,
            expected_model_revision=model_revision,
        )
        model_contract = self.artifacts.manifest["model"]
        assert isinstance(model_contract, dict)
        resolved_revision = str(model_contract["revision"])
        self.encoder = encoder or LazyBgeM3Encoder(
            model_name=model_name,
            model_revision=resolved_revision,
            batch_size=batch_size,
            device=device,
            cache_dir=Path(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
        )

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        normalized_query = " ".join(str(query).split())
        if not normalized_query:
            raise ValueError("BGE-M3 query must not be empty")
        if int(top_k) <= 0:
            raise ValueError("BGE-M3 top_k must be positive")
        query_vector = _encode_and_normalize(self.encoder, [normalized_query], expected_rows=1)
        limit = min(int(top_k), len(self.artifacts.frame_records))
        scores, rows = self.artifacts.index.search(
            np.ascontiguousarray(query_vector, dtype=np.float32),
            limit,
        )
        output: list[RetrievalResult] = []
        for score, row in zip(scores[0].tolist(), rows[0].tolist()):
            if row < 0:
                continue
            record = self.artifacts.frame_records[row]["metadata"]
            assert isinstance(record, dict)
            output.append(_retrieval_result(record, float(score), row))
        return output


class LazyBgeM3Encoder:
    """Lazy Transformers BGE-M3 CLS encoder for GPU or CPU execution."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_BGE_M3_MODEL,
        model_revision: str = DEFAULT_BGE_M3_REVISION,
        batch_size: int = 16,
        max_length: int = 1024,
        device: str = "auto",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("BGE-M3 batch size and max length must be positive")
        if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
            raise ValueError("BGE-M3 device must be auto, cpu, cuda, or cuda:<index>")
        self.model_name = model_name
        self.model_revision = model_revision
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.requested_device = device
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self.resolved_revision = model_revision

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, BGE_M3_DIMENSION), dtype=np.float32)
        self._load()
        assert self._torch is not None and self._model is not None and self._tokenizer is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [str(value) for value in texts[start : start + self.batch_size]]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                output = self._model(**inputs)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None or hidden.ndim != 3:
                raise RuntimeError("BGE-M3 model did not return last_hidden_state")
            embeddings = hidden[:, 0].float()
            batches.append(embeddings.cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("PyTorch and Transformers are required for BGE-M3") from exc
        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.requested_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for BGE-M3 but is unavailable")
        common: dict[str, object] = {
            "revision": self.model_revision,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            common["cache_dir"] = self.cache_dir.as_posix()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **common)
        model_kwargs = dict(common)
        if device.startswith("cuda"):
            model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        model.to(device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        config = getattr(model, "config", None)
        self.resolved_revision = str(
            getattr(config, "_commit_hash", None) or self.model_revision
        )


def _encode_and_normalize(
    encoder: TextEncoder,
    texts: Sequence[str],
    *,
    expected_rows: int,
) -> np.ndarray:
    vectors = np.asarray(encoder(texts), dtype=np.float32)
    if vectors.shape != (expected_rows, BGE_M3_DIMENSION):
        raise ValueError(
            "BGE-M3 dense encoder shape mismatch: "
            f"expected {(expected_rows, BGE_M3_DIMENSION)}, got {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("BGE-M3 dense encoder produced NaN or Inf")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("BGE-M3 dense encoder produced a zero vector")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _retrieval_result(record: Mapping[str, object], score: float, row: int) -> RetrievalResult:
    objects = [value for value in _object_values(record.get("objects")) if value]
    frame_id = str(
        record.get("frame_id")
        or record.get("start_keyframe")
        or record.get("keyframe_id")
        or f"BGE_ROW_{row:08d}"
    )
    timestamp = _safe_float(
        record.get("timestamp", record.get("start_time", record.get("shot_start", 0.0)))
    )
    return RetrievalResult(
        video_id=str(record.get("video_id") or ""),
        frame_id=frame_id,
        timestamp=timestamp,
        score=score,
        segment_id=str(record.get("segment_id") or ""),
        shot_id=str(record.get("shot_id") or record.get("segment_id") or ""),
        faiss_index=row,
        frame_index=_optional_int(record.get("frame_index")),
        keyframe_path=str(record.get("keyframe_path") or record.get("image_path") or ""),
        thumbnail_path=str(record.get("thumbnail_path") or ""),
        timestamp_source=str(record.get("timestamp_source") or "metadata"),
        timestamp_confidence=_safe_float(record.get("timestamp_confidence", 1.0)),
        caption=_first_text(record, ("caption", "captions_aggregated", "caption_text")),
        ocr_text=_ocr_text(record.get("ocr_text")) or _ocr_text(record.get("ocr")),
        objects=objects,
        modality_scores={"dense_text": score},
    )


def _first_text(record: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _ocr_text(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if not isinstance(value, (list, tuple)):
        return ""
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item
        elif isinstance(item, Mapping):
            text = str(item.get("text") or item.get("value") or "")
        else:
            text = ""
        normalized = " ".join(text.split())
        if normalized and normalized not in output:
            output.append(normalized)
    return " | ".join(output)


def _object_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            label = item
        elif isinstance(item, Mapping):
            label = str(
                item.get("label")
                or item.get("class_name")
                or item.get("name")
                or ""
            )
        else:
            label = ""
        label = " ".join(label.split())
        if label and label not in output:
            output.append(label)
    return output


def _object_text(value: object) -> str:
    return " | ".join(_object_values(value))


def _canonical_order_key(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(record.get("video_id") or ""),
        _safe_float(record.get("timestamp", record.get("start_time", 0.0))),
        str(record.get("frame_id") or record.get("start_keyframe") or ""),
        str(record.get("segment_id") or ""),
        str(record.get("shot_id") or ""),
        _canonical_sha256(record),
    )


def _json_safe_mapping(record: Mapping[str, object]) -> dict[str, object]:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
    value = json.loads(encoded)
    if not isinstance(value, dict):  # pragma: no cover - Mapping always serializes as object
        raise TypeError("Metadata record must serialize to a JSON object")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in BGE-M3 artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"BGE-M3 artifact must contain a JSON object: {path}")
    return value


def _verify_artifact(path: Path, contract: object, expected_name: str) -> None:
    if not isinstance(contract, dict):
        raise ValueError(f"Missing BGE-M3 checksum contract for {expected_name}")
    if contract.get("filename") != expected_name:
        raise ValueError(f"Unexpected BGE-M3 artifact filename for {expected_name}")
    if contract.get("sha256") != _sha256_file(path):
        raise ValueError(f"BGE-M3 artifact checksum mismatch: {path}")


def _atomic_write_json(value: Mapping[str, object], path: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_faiss(index: Any, path: Path, faiss: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        faiss.write_index(index, temporary.as_posix())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("faiss-cpu or faiss-gpu is required for BGE-M3 indexing") from exc
    return faiss


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
