"""Visual text-to-keyframe retrieval backed by SigLIP2 and FAISS.

The FAISS manifest is the source of truth for the SigLIP2 model, revision,
vector dimension, normalization, and similarity contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from backend.app.models.retrieval import NeighborFrame, RetrievalResult, VisualSearchResponse
from backend.app.services.metadata.metadata_store import FrameRecord, MetadataStore


class TextEncoder(Protocol):
    def encode(self, query: str) -> np.ndarray:
        ...

    def encode_texts(self, queries: Sequence[str]) -> np.ndarray:
        ...


class VectorSearcher(Protocol):
    def search(self, vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class VisualSearchConfig:
    index_path: Path = Path(
        "data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss"
    )
    frame_map_path: Path = Path(
        "data/metadata/siglip2_so400m_patch16_384_frame_map.json"
    )
    manifest_path: Path = Path(
        "data/metadata/siglip2_so400m_patch16_384_faiss_manifest.json"
    )
    device: str = "auto"
    model_cache_dir: Path = Path("data/model_cache/siglip2")
    no_autocast: bool = False
    local_files_only: bool = False
    default_top_k: int = 20
    max_top_k: int = 200
    min_score: float | None = None


@dataclass(frozen=True)
class VisualBundleIntegrity:
    generation: str
    index_sha256: str
    frame_map_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_generation(hashes: dict[str, str]) -> str:
    payload = json.dumps(
        hashes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_visual_bundle_integrity(
    config: VisualSearchConfig,
) -> VisualBundleIntegrity | None:
    """Validate generation-bound artifacts when a canonical manifest provides them.

    Older schema-1.2 manifests did not include bundle hashes, so they remain
    readable.  The canonical offline publisher always emits this stronger
    contract and therefore fails closed across interrupted multi-file updates.
    """

    if not config.manifest_path.is_file():
        raise FileNotFoundError(f"FAISS manifest not found: {config.manifest_path}")
    with config.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"FAISS manifest must be an object: {config.manifest_path}")
    declared = manifest.get("artifacts")
    generation = manifest.get("bundle_generation")
    if declared is None and generation is None:
        # Snapshot legacy schema-1.2 artifacts too.  They lack a manifest-level
        # generation, but pinning both hashes prevents a lazy index load from
        # mixing a newly published index with the frame map loaded at startup.
        if not config.index_path.is_file() or not config.frame_map_path.is_file():
            raise FileNotFoundError("Legacy FAISS bundle is missing index or frame map")
        index_sha256 = _sha256_file(config.index_path)
        frame_map_sha256 = _sha256_file(config.frame_map_path)
        return VisualBundleIntegrity(
            generation="legacy-" + _bundle_generation(
                {
                    "index": index_sha256,
                    "frame_map": frame_map_sha256,
                }
            ),
            index_sha256=index_sha256,
            frame_map_sha256=frame_map_sha256,
        )
    if not isinstance(declared, dict) or not isinstance(generation, str) or not generation:
        raise ValueError("FAISS manifest has an incomplete bundle-integrity contract")
    declared_hashes: dict[str, str] = {}
    for label in ("index", "metadata", "frame_map", "report"):
        item = declared.get(label)
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"FAISS manifest is missing the {label} checksum")
        declared_hashes[label] = str(item["sha256"])
    if _bundle_generation(declared_hashes) != generation:
        raise ValueError("FAISS bundle generation does not match declared checksums")
    for label, path in (
        ("index", config.index_path),
        ("frame_map", config.frame_map_path),
    ):
        item = declared[label]
        if item.get("filename") != path.name:
            raise ValueError(f"FAISS {label} filename does not match the manifest")
        if not path.is_file() or _sha256_file(path) != declared_hashes[label]:
            raise ValueError(
                f"FAISS {label} does not belong to manifest generation {generation}"
            )
    return VisualBundleIntegrity(
        generation=generation,
        index_sha256=declared_hashes["index"],
        frame_map_sha256=declared_hashes["frame_map"],
    )


@dataclass(frozen=True)
class EncoderContract:
    model_family: str
    model_name: str
    model_revision: str
    processor_name: str
    vector_dim: int
    input_resolution: int | None
    normalized: bool
    similarity: str
    output_dtype: str


def load_encoder_contract(manifest_path: Path) -> EncoderContract:
    if not manifest_path.exists():
        raise FileNotFoundError(f"FAISS manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError(f"FAISS manifest must be an object: {manifest_path}")
    if manifest.get("schema_version") != "1.2":
        raise ValueError(
            f"Unsupported FAISS manifest schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}; expected '1.2'"
        )
    if manifest.get("index_type") != "IndexFlatIP" or manifest.get("metric") != "ip":
        raise ValueError(
            f"SigLIP2 retrieval requires IndexFlatIP/metric=ip in {manifest_path}"
        )

    encoder = manifest.get("encoder")
    if not isinstance(encoder, dict):
        raise ValueError(f"Missing encoder contract in {manifest_path}")
    required_fields = {
        "model_family",
        "model_name",
        "model_revision",
        "processor_name",
        "vector_dim",
        "input_resolution",
        "normalized",
        "similarity",
        "output_dtype",
    }
    missing = sorted(required_fields - encoder.keys())
    if missing:
        raise ValueError(f"Missing encoder fields in {manifest_path}: {missing}")
    if encoder["model_family"] != "siglip2":
        raise ValueError(
            f"Retrieval expected model_family='siglip2', got "
            f"{encoder['model_family']!r} in {manifest_path}"
        )
    if encoder["normalized"] is not True:
        raise ValueError(f"SigLIP2 manifest must declare normalized=true: {manifest_path}")
    if encoder["similarity"] != "cosine":
        raise ValueError(f"SigLIP2 manifest must declare similarity='cosine': {manifest_path}")
    if encoder["output_dtype"] != "float32":
        raise ValueError(f"SigLIP2 manifest must declare output_dtype='float32': {manifest_path}")
    for field in ("model_name", "model_revision", "processor_name"):
        if not encoder[field]:
            raise ValueError(f"encoder.{field} must not be empty in {manifest_path}")
    try:
        vector_dim = int(encoder["vector_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid encoder vector_dim in {manifest_path}: {encoder['vector_dim']!r}"
        ) from exc
    if vector_dim <= 0:
        raise ValueError(f"encoder.vector_dim must be positive in {manifest_path}")

    return EncoderContract(
        model_family=str(encoder["model_family"]),
        model_name=str(encoder["model_name"]),
        model_revision=str(encoder["model_revision"]),
        processor_name=str(encoder["processor_name"]),
        vector_dim=vector_dim,
        input_resolution=(
            int(encoder["input_resolution"])
            if encoder["input_resolution"] is not None
            else None
        ),
        normalized=True,
        similarity="cosine",
        output_dtype="float32",
    )


def normalize_query_vector(vector: np.ndarray) -> np.ndarray:
    """Return a 2D float32 L2-normalized query vector."""
    vector = np.asarray(vector, dtype="float32")
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    if vector.ndim != 2 or vector.shape[0] != 1:
        raise ValueError(f"Expected one query vector, got shape={vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("Query vector contains NaN or Inf")

    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("Query vector must not be zero")
    return np.ascontiguousarray(vector / norm, dtype="float32")


class Siglip2TextEncoder:
    """Lazy SigLIP2 encoder configured from the visual FAISS manifest.

    Text encoding remains the public retrieval contract.  ``encode_images`` is
    a small companion surface for bounded TRAKE local refinement, allowing the
    refiner to share this exact lazy model/processor instead of loading a second
    SigLIP2 checkpoint or introducing a VLM.
    """

    def __init__(
        self,
        contract: EncoderContract,
        device: str = "auto",
        model_cache_dir: Path | None = None,
        no_autocast: bool = False,
        local_files_only: bool = False,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.contract = contract
        self.requested_device = device
        self.model_cache_dir = model_cache_dir
        self.no_autocast = no_autocast
        self.local_files_only = bool(local_files_only)
        self._model = model
        self._processor = processor
        self._torch = None
        self._device = ""
        self._compute_dtype = None
        self._model_load_count = 0
        self._last_model_load_ms = 0.0

    @property
    def model_load_count(self) -> int:
        return self._model_load_count

    @property
    def last_model_load_ms(self) -> float:
        return self._last_model_load_ms

    def _load(self) -> None:
        if self._device:
            return

        if self.local_files_only:
            # Set before importing Transformers/Hugging Face so a cached-only
            # runtime performs no metadata/network probe.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError(
                "torch is required for SigLIP2 visual search. "
                "Install project requirements before running retrieval."
            ) from exc

        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cuda", "cpu"}:
            raise ValueError("RETRIEVAL_DEVICE must be one of: auto, cuda, cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for retrieval, but it is unavailable")

        if self._model is None or self._processor is None:
            load_started = time.perf_counter()
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError as exc:  # pragma: no cover - depends on local environment.
                raise RuntimeError(
                    "Transformers with SigLIP2 support is required for visual search. "
                    "Install project requirements first."
                ) from exc
            kwargs: dict[str, Any] = {"revision": self.contract.model_revision}
            kwargs["local_files_only"] = self.local_files_only
            if self.model_cache_dir:
                self.model_cache_dir.mkdir(parents=True, exist_ok=True)
                kwargs["cache_dir"] = self.model_cache_dir.as_posix()
            self._model = AutoModel.from_pretrained(
                self.contract.model_name,
                **kwargs,
            )
            self._processor = AutoProcessor.from_pretrained(
                self.contract.processor_name,
                **kwargs,
            )
            self._model_load_count += 1
            self._last_model_load_ms = round(
                (time.perf_counter() - load_started) * 1000.0,
                3,
            )

        self._model.to(device)
        self._model.eval()
        self._torch = torch
        self._device = device
        if device == "cuda" and not self.no_autocast:
            self._compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self._compute_dtype = torch.float32

    def encode(self, query: str) -> np.ndarray:
        return self.encode_texts([query])

    def encode_texts(self, queries: Sequence[str]) -> np.ndarray:
        values = [str(query).strip() for query in queries]
        if not values or any(not query for query in values):
            raise ValueError("queries must not be empty")

        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        assert self._compute_dtype is not None

        inputs = self._processor(
            text=values,
            padding="max_length",
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(
                self._device,
                non_blocking=self._device == "cuda",
            )
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }
        autocast = (
            self._torch.autocast(
                device_type="cuda",
                dtype=self._compute_dtype,
            )
            if self._device == "cuda" and not self.no_autocast
            else nullcontext()
        )
        with self._torch.inference_mode():
            with autocast:
                output = self._model.get_text_features(**model_inputs)
        features = _feature_tensor(output, self._torch)
        if features.ndim != 2 or features.shape[0] != len(values):
            raise ValueError(
                f"Unexpected SigLIP2 text feature shape: {tuple(features.shape)}"
            )
        if int(features.shape[-1]) != self.contract.vector_dim:
            raise ValueError(
                "SigLIP2 query dimension does not match FAISS manifest: "
                f"{int(features.shape[-1])} != {self.contract.vector_dim}"
            )
        if not self._torch.isfinite(features).all():
            raise ValueError("SigLIP2 produced NaN or Inf text features")
        norm = features.float().norm(dim=-1, keepdim=True)
        if self._torch.any(norm <= 0):
            raise ValueError("SigLIP2 produced a zero text vector")
        features = features.float() / norm
        return features.detach().cpu().numpy().astype("float32", copy=False)

    def encode_images(
        self,
        images: Sequence[Any],
        *,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return normalized SigLIP2 features for a bounded RGB image batch.

        Callers own color conversion; this method intentionally accepts the
        same RGB PIL/numpy values supported by the Hugging Face processor.  It
        batches inference to keep TRAKE's local frame windows memory-bounded.
        """

        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("SigLIP2 image batch_size must be positive")
        values = list(images)
        if not values:
            return np.empty((0, self.contract.vector_dim), dtype="float32")

        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        assert self._compute_dtype is not None

        batches: list[np.ndarray] = []
        for start in range(0, len(values), int(batch_size)):
            image_batch = values[start : start + int(batch_size)]
            inputs = self._processor(
                images=image_batch,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(
                    self._device,
                    non_blocking=self._device == "cuda",
                )
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }
            autocast = (
                self._torch.autocast(
                    device_type="cuda",
                    dtype=self._compute_dtype,
                )
                if self._device == "cuda" and not self.no_autocast
                else nullcontext()
            )
            with self._torch.inference_mode():
                with autocast:
                    output = self._model.get_image_features(**model_inputs)
            features = _feature_tensor(output, self._torch)
            if features.ndim != 2 or features.shape[0] != len(image_batch):
                raise ValueError(
                    "Unexpected SigLIP2 image feature shape: "
                    f"{tuple(features.shape)} for batch size {len(image_batch)}"
                )
            if int(features.shape[-1]) != self.contract.vector_dim:
                raise ValueError(
                    "SigLIP2 image dimension does not match FAISS manifest: "
                    f"{int(features.shape[-1])} != {self.contract.vector_dim}"
                )
            if not self._torch.isfinite(features).all():
                raise ValueError("SigLIP2 produced NaN or Inf image features")
            norm = features.float().norm(dim=-1, keepdim=True)
            if self._torch.any(norm <= 0):
                raise ValueError("SigLIP2 produced a zero image vector")
            normalized = features.float() / norm
            batches.append(
                normalized.detach().cpu().numpy().astype("float32", copy=False)
            )
        return np.ascontiguousarray(np.concatenate(batches, axis=0), dtype="float32")


def _feature_tensor(output: Any, torch_module: Any):
    if isinstance(output, torch_module.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(
        "model.get_text_features() returned an unsupported value: "
        f"{type(output).__name__}"
    )


class FaissVectorSearcher:
    """Thin wrapper around a FAISS index."""

    def __init__(
        self,
        index_path: Path,
        expected_dim: int | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        self.index_path = index_path
        self.expected_dim = expected_dim
        self.expected_sha256 = expected_sha256
        self._index = None

    def _load(self) -> None:
        if self._index is not None:
            return
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if (
            self.expected_sha256 is not None
            and _sha256_file(self.index_path) != self.expected_sha256
        ):
            raise ValueError("FAISS index changed after its manifest was validated")
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError(
                "faiss is required for visual search. Install project requirements first."
            ) from exc
        loaded_index = faiss.read_index(self.index_path.as_posix())
        if (
            self.expected_sha256 is not None
            and _sha256_file(self.index_path) != self.expected_sha256
        ):
            raise ValueError("FAISS index changed while it was being loaded")
        if self.expected_dim is not None and int(loaded_index.d) != self.expected_dim:
            raise ValueError(
                f"FAISS dimension does not match manifest: "
                f"{int(loaded_index.d)} != {self.expected_dim}"
            )
        self._index = loaded_index

    def search(self, vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        self._load()
        assert self._index is not None
        return self._index.search(vector, top_k)


class VisualSearchEngine:
    """Coordinates text encoding, vector search, and metadata mapping."""

    def __init__(
        self,
        config: VisualSearchConfig,
        encoder: TextEncoder | None = None,
        searcher: VectorSearcher | None = None,
        metadata_store: MetadataStore | None = None,
    ) -> None:
        self.config = config
        self.encoder_contract: EncoderContract | None = None
        bundle_integrity: VisualBundleIntegrity | None = None
        if encoder is None:
            bundle_integrity = validate_visual_bundle_integrity(config)
            self.encoder_contract = load_encoder_contract(config.manifest_path)
            self.encoder = Siglip2TextEncoder(
                contract=self.encoder_contract,
                device=config.device,
                model_cache_dir=config.model_cache_dir,
                no_autocast=config.no_autocast,
                local_files_only=config.local_files_only,
            )
        else:
            self.encoder = encoder
        self.searcher = searcher or FaissVectorSearcher(
            config.index_path,
            expected_dim=(
                self.encoder_contract.vector_dim if self.encoder_contract else None
            ),
            expected_sha256=(
                bundle_integrity.index_sha256 if bundle_integrity else None
            ),
        )
        self.metadata_store = metadata_store or MetadataStore.from_frame_map(config.frame_map_path)
        if bundle_integrity is not None:
            verified_again = validate_visual_bundle_integrity(config)
            if verified_again != bundle_integrity:
                raise ValueError("FAISS bundle changed while visual search was loading")

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        requested_top_k = top_k if top_k is not None else self.config.default_top_k
        bounded_top_k = max(1, min(int(requested_top_k), self.config.max_top_k))
        return self._search_bounded(query, bounded_top_k)

    def search_by_vector(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int | None = None,
    ) -> VisualSearchResponse:
        """Search selected keyframes with an already encoded query vector.

        The canonical coarse-to-dense path uses this method for the original
        query so selected-keyframe FAISS and dense-candidate FAISS share the
        exact same SigLIP2 embedding without a second model forward pass.
        """

        requested_top_k = top_k if top_k is not None else self.config.default_top_k
        bounded_top_k = max(1, min(int(requested_top_k), self.config.max_top_k))
        return self._search_vector_bounded(query, query_vector, bounded_top_k)

    def search_pool(self, query: str, top_k: int) -> VisualSearchResponse:
        """Retrieve a validated internal candidate pool beyond the public cap.

        Public visual responses remain limited by ``config.max_top_k``.  TRAKE
        needs a wider pre-alignment pool and applies its own validated bound;
        this explicit method avoids silently turning ``event_top_k=300`` into
        200 while retaining a defensive service-level ceiling.
        """

        requested = int(top_k)
        if not 1 <= requested <= 10_000:
            raise ValueError("internal visual candidate pool must be between 1 and 10000")
        return self._search_bounded(query, requested)

    def search_many_pool(
        self,
        queries: Sequence[str],
        top_k: int,
    ) -> list[VisualSearchResponse]:
        """Batch text encoding and FAISS lookup for TRAKE event queries."""

        requested = int(top_k)
        if not 1 <= requested <= 10_000:
            raise ValueError("internal visual candidate pool must be between 1 and 10000")
        values = [str(query) for query in queries]
        if not values:
            return []
        encode_many = getattr(self.encoder, "encode_texts", None)
        if not callable(encode_many):
            return [self._search_bounded(query, requested) for query in values]
        started_at = time.perf_counter()
        vectors = np.asarray(encode_many(values), dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(values) or not np.isfinite(vectors).all():
            raise ValueError("batched SigLIP2 query vectors must be finite and aligned")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("batched SigLIP2 query vectors must not be zero")
        vectors = np.ascontiguousarray(vectors / norms, dtype="float32")
        scores, indices = self.searcher.search(vectors, requested)
        latency = round((time.perf_counter() - started_at) * 1000, 3)
        return [
            VisualSearchResponse(
                query=query,
                top_k=requested,
                latency_ms=latency,
                results=self._to_results(scores=row_scores, indices=row_indices),
            )
            for query, row_scores, row_indices in zip(values, scores, indices)
        ]

    def _search_bounded(self, query: str, bounded_top_k: int) -> VisualSearchResponse:
        started_at = time.perf_counter()
        query_vector = normalize_query_vector(self.encoder.encode(query))
        return self._search_vector_bounded(
            query,
            query_vector,
            bounded_top_k,
            started_at=started_at,
        )

    def _search_vector_bounded(
        self,
        query: str,
        query_vector: np.ndarray,
        bounded_top_k: int,
        *,
        started_at: float | None = None,
    ) -> VisualSearchResponse:
        started_at = time.perf_counter() if started_at is None else started_at
        query_vector = normalize_query_vector(query_vector)
        if (
            self.encoder_contract is not None
            and query_vector.shape[1] != self.encoder_contract.vector_dim
        ):
            raise ValueError(
                f"Query vector dimension does not match manifest: "
                f"{query_vector.shape[1]} != {self.encoder_contract.vector_dim}"
            )
        scores, indices = self.searcher.search(query_vector, bounded_top_k)
        results = self._to_results(scores=scores[0], indices=indices[0])

        latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
        return VisualSearchResponse(
            query=query,
            top_k=bounded_top_k,
            latency_ms=latency_ms,
            results=results,
        )

    def _to_results(self, scores: np.ndarray, indices: np.ndarray) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for raw_score, raw_index in zip(scores, indices):
            score = float(raw_score)
            if self.config.min_score is not None and score < self.config.min_score:
                continue
            faiss_index = int(raw_index)
            if faiss_index < 0:
                continue
            record = self.metadata_store.get_by_faiss_index(faiss_index)
            if record is None:
                continue
            neighbors = self.metadata_store.get_same_shot_neighbors(
                faiss_index=faiss_index,
                max_neighbors=4,
            )
            results.append(frame_record_to_result(record, score, neighbors))
        return results


def frame_record_to_neighbor(record: FrameRecord) -> NeighborFrame:
    return NeighborFrame(
        video_id=record.video_id,
        frame_id=record.frame_id,
        segment_id=record.segment_id,
        shot_id=record.shot_id,
        timestamp=record.timestamp,
        frame_index=record.frame_index,
        faiss_index=record.faiss_index,
        keyframe_path=record.keyframe_path,
        thumbnail_path=record.thumbnail_path,
    )


def frame_record_to_result(
    record: FrameRecord,
    score: float,
    neighbors: list[FrameRecord] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        video_id=record.video_id,
        frame_id=record.frame_id,
        segment_id=record.segment_id,
        shot_id=record.shot_id,
        timestamp=record.timestamp,
        timestamp_source=record.timestamp_source,
        timestamp_confidence=record.timestamp_confidence,
        frame_index=record.frame_index,
        faiss_index=record.faiss_index,
        score=round(score, 6),
        keyframe_path=record.keyframe_path,
        thumbnail_path=record.thumbnail_path,
        caption=record.caption,
        ocr_text=record.ocr_text,
        objects=list(record.objects),
        modality_scores={"visual": round(score, 6)},
        neighbors=[frame_record_to_neighbor(neighbor) for neighbor in neighbors or []],
    )
