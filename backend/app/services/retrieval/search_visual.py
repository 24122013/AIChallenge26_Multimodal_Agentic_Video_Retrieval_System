"""Visual text-to-keyframe retrieval backed by SigLIP2 and FAISS.

The FAISS manifest is the source of truth for the SigLIP2 model, revision,
vector dimension, normalization, and similarity contract.
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from backend.app.models.retrieval import NeighborFrame, RetrievalResult, VisualSearchResponse
from backend.app.services.metadata.metadata_store import FrameRecord, MetadataStore


class TextEncoder(Protocol):
    def encode(self, query: str) -> np.ndarray:
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
    default_top_k: int = 20
    max_top_k: int = 200
    min_score: float | None = None


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
    """Lazy SigLIP2 text encoder configured from the FAISS manifest."""

    def __init__(
        self,
        contract: EncoderContract,
        device: str = "auto",
        model_cache_dir: Path | None = None,
        no_autocast: bool = False,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.contract = contract
        self.requested_device = device
        self.model_cache_dir = model_cache_dir
        self.no_autocast = no_autocast
        self._model = model
        self._processor = processor
        self._torch = None
        self._device = ""
        self._compute_dtype = None

    def _load(self) -> None:
        if self._device:
            return

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
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError as exc:  # pragma: no cover - depends on local environment.
                raise RuntimeError(
                    "Transformers with SigLIP2 support is required for visual search. "
                    "Install project requirements first."
                ) from exc
            kwargs: dict[str, Any] = {"revision": self.contract.model_revision}
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
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        assert self._compute_dtype is not None

        inputs = self._processor(
            text=[query],
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
        if features.ndim != 2 or features.shape[0] != 1:
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

    def __init__(self, index_path: Path, expected_dim: int | None = None) -> None:
        self.index_path = index_path
        self.expected_dim = expected_dim
        self._index = None

    def _load(self) -> None:
        if self._index is not None:
            return
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError(
                "faiss is required for visual search. Install project requirements first."
            ) from exc
        self._index = faiss.read_index(self.index_path.as_posix())
        if self.expected_dim is not None and int(self._index.d) != self.expected_dim:
            raise ValueError(
                f"FAISS dimension does not match manifest: "
                f"{int(self._index.d)} != {self.expected_dim}"
            )

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
        if encoder is None:
            self.encoder_contract = load_encoder_contract(config.manifest_path)
            self.encoder = Siglip2TextEncoder(
                contract=self.encoder_contract,
                device=config.device,
                model_cache_dir=config.model_cache_dir,
                no_autocast=config.no_autocast,
            )
        else:
            self.encoder = encoder
        self.searcher = searcher or FaissVectorSearcher(
            config.index_path,
            expected_dim=(
                self.encoder_contract.vector_dim if self.encoder_contract else None
            ),
        )
        self.metadata_store = metadata_store or MetadataStore.from_frame_map(config.frame_map_path)

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        started_at = time.perf_counter()
        requested_top_k = top_k if top_k is not None else self.config.default_top_k
        bounded_top_k = max(1, min(int(requested_top_k), self.config.max_top_k))

        query_vector = normalize_query_vector(self.encoder.encode(query))
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
        asr_text=record.asr_text,
        objects=list(record.objects),
        modality_scores={"visual": round(score, 6)},
        neighbors=[frame_record_to_neighbor(neighbor) for neighbor in neighbors or []],
    )
