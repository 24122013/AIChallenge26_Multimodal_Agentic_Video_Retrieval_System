"""Ingestion services (Team P3 - Metadata).

Sinh và gộp metadata: caption, ocr, asr, objects -> unified metadata / frame_map.
Không chứa search / rerank / agent logic (xem docs/service_boundaries.md).
"""
from __future__ import annotations

from .asr_pipeline import run_asr_pipeline
from .caption_pipeline import run_caption_pipeline
from .metadata_builder import (
    build_unified_metadata,
    enrich_frame_map,
    load_bundle,
)
from .object_pipeline import run_object_pipeline
from .ocr_pipeline import run_ocr_pipeline
from .scheme_validator import validate_record, validate_records

__all__ = [
    "run_caption_pipeline",
    "run_ocr_pipeline",
    "run_object_pipeline",
    "run_asr_pipeline",
    "build_unified_metadata",
    "enrich_frame_map",
    "load_bundle",
    "validate_record",
    "validate_records",
]
