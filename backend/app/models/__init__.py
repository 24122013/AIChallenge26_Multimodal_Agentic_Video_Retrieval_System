"""Shared data models.

Retrieval models sống ở `retrieval.py` (do team retrieval dùng).
Metadata models (Team P3) sống ở `metadata.py`.
"""
from __future__ import annotations

from .metadata import (
    ASR,
    OCR,
    Caption,
    DetectedObject,
    EmbeddingMetadata,
    Keyframe,
    ObjectAnnotation,
    Segment,
    TIMESTAMP_CONFIDENCE_BY_SOURCE,
    TimestampSource,
    UnifiedMetadataRecord,
    Video,
    make_embedding_id,
    make_frame_id,
    make_segment_id,
    make_shot_id,
)
from .retrieval import (
    NeighborFrame,
    RetrievalResult,
    VisualSearchRequest,
    VisualSearchResponse,
)

__all__ = [
    # metadata
    "ASR",
    "OCR",
    "Caption",
    "DetectedObject",
    "EmbeddingMetadata",
    "Keyframe",
    "ObjectAnnotation",
    "Segment",
    "TIMESTAMP_CONFIDENCE_BY_SOURCE",
    "TimestampSource",
    "UnifiedMetadataRecord",
    "Video",
    "make_embedding_id",
    "make_frame_id",
    "make_segment_id",
    "make_shot_id",
    # retrieval
    "NeighborFrame",
    "RetrievalResult",
    "VisualSearchRequest",
    "VisualSearchResponse",
]
