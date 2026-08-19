"""Retrieval data models shared by services and API endpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")

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
    asr_text: str = ""
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

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "latency_ms": self.latency_ms,
            "results": [result.to_dict() for result in self.results],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> VisualSearchResponse:
        return cls(
            query = data.get("query"),
            top_k = data.get("top_k"),
            latency_ms = data.get("latency_ms"),
            results = data.get("results"),
        )
    
@dataclass(frozen=True)
class QASearchResponse:
    question: str
    answer_target: str
    retrieval_queries: list[str]
    top_k: int
    answer_mode: str
    evidence_count: int
    results: list[RetrievalResult]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer_target": self.answer_target,
            "retrieval_queries": self.retrieval_queries,
            "top_k": self.top_k,
            "answer_mode": self.answer_mode,
            "evidence_count": self.evidence_count,
            "results": [result.to_dict() for result in self.results],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> QASearchResponse:
        return cls(
            question=data.get("question"),
            answer_target=data.get("answer_target"),
            retrieval_queries=data.get("retrieval_queries"),
            top_k=data.get("top_k"),
            answer_mode=data.get("answer_mode"),
            evidence_count=data.get("evidence_count"),
            results=data.get("results"),
        )

@dataclass(frozen=True)
class APIResponse(Generic[T]):
    data: T
    success: bool = True
    message: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
        }
