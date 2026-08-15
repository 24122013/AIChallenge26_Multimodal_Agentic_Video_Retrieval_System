"""Backward-compatible adapter for the canonical segment metadata builder."""

from src.indexing.build_segment_metadata import (
    aggregate_captions,
    aggregate_objects,
    aggregate_ocr,
    build_parser,
    build_segment_metadata,
    build_segment_records,
    build_segments,
    main,
)

__all__ = [
    "aggregate_captions",
    "aggregate_objects",
    "aggregate_ocr",
    "build_parser",
    "build_segment_metadata",
    "build_segment_records",
    "build_segments",
    "main",
]


if __name__ == "__main__":
    main()
