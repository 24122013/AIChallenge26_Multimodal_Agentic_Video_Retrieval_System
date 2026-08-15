"""Retrieval data models shared by services and API endpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class NeighborFrame:
    """Frame shown as temporal context for a retrieval hit."""

    video_id: str
    frame_id: str
    timestamp: float
    shot_id: str = ""
    segment_id: str = ""
    faiss_index: int | None = None
    frame_index: int | None = None
    keyframe_path: str = ""
    thumbnail_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    """Unified response item returned by retrieval modules."""

    video_id: str
    frame_id: str
    timestamp: float
    score: float
    segment_id: str = ""
    shot_id: str = ""
    faiss_index: int | None = None
    frame_index: int | None = None
    keyframe_path: str = ""
    thumbnail_path: str = ""
    timestamp_source: str = "unknown"
    timestamp_confidence: float = 0.0
    caption: str = ""
    ocr_text: str = ""
    objects: list[str] = field(default_factory=list)
    modality_scores: dict[str, float] = field(default_factory=dict)
    neighbors: list[NeighborFrame] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["neighbors"] = [neighbor.to_dict() for neighbor in self.neighbors]
        return data


@dataclass(frozen=True)
class VisualSearchRequest:
    query: str
    top_k: int = 20


@dataclass(frozen=True)
class VisualSearchResponse:
    query: str
    top_k: int
    latency_ms: float
    results: list[RetrievalResult]
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "query": self.query,
            "top_k": self.top_k,
            "latency_ms": self.latency_ms,
            "results": [result.to_dict() for result in self.results],
        }
        if self.trace:
            payload["trace"] = self.trace
        return payload
