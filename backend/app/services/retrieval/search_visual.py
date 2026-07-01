"""Visual text-to-keyframe retrieval backed by OpenCLIP and FAISS.

Phase 1 target:
    text query -> text embedding -> FAISS top-k -> frame_map metadata -> results
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    index_path: Path = Path("data/indexes/openclip_vit_b16_flat_ip.faiss")
    frame_map_path: Path = Path("data/metadata/openclip_vit_b16_frame_map.json")
    model_name: str = "ViT-B-16"
    pretrained: str = "laion2b_s34b_b88k"
    device: str = "auto"
    model_cache_dir: Path = Path("data/model_cache/openclip")
    default_top_k: int = 20
    max_top_k: int = 200


def normalize_query_vector(vector: np.ndarray) -> np.ndarray:
    """Return a 2D float32 L2-normalized query vector."""
    vector = np.asarray(vector, dtype="float32")
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    if vector.ndim != 2 or vector.shape[0] != 1:
        raise ValueError(f"Expected one query vector, got shape={vector.shape}")

    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return vector / norm


class OpenCLIPTextEncoder:
    """Lazy OpenCLIP text encoder used by visual retrieval."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str = "auto",
        model_cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.requested_device = device
        self.model_cache_dir = model_cache_dir
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = ""

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError(
                "open_clip and torch are required for visual search. "
                "Install project requirements before running retrieval."
            ) from exc

        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=device,
            cache_dir=self.model_cache_dir.as_posix() if self.model_cache_dir else None,
        )
        model.eval()

        self._model = model
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._torch = torch
        self._device = device

    def encode(self, query: str) -> np.ndarray:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None

        tokens = self._tokenizer([query]).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().astype("float32")


class FaissVectorSearcher:
    """Thin wrapper around a FAISS index."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
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
        self.encoder = encoder or OpenCLIPTextEncoder(
            model_name=config.model_name,
            pretrained=config.pretrained,
            device=config.device,
            model_cache_dir=config.model_cache_dir,
        )
        self.searcher = searcher or FaissVectorSearcher(config.index_path)
        self.metadata_store = metadata_store or MetadataStore.from_frame_map(config.frame_map_path)

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        started_at = time.perf_counter()
        requested_top_k = top_k if top_k is not None else self.config.default_top_k
        bounded_top_k = max(1, min(int(requested_top_k), self.config.max_top_k))

        query_vector = normalize_query_vector(self.encoder.encode(query))
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
            results.append(frame_record_to_result(record, float(raw_score), neighbors))
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
        neighbors=[frame_record_to_neighbor(neighbor) for neighbor in neighbors or []],
    )
