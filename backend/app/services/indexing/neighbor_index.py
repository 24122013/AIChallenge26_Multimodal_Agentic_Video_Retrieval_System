"""Backward-compatible adapter for the canonical neighbor-index builder."""

from src.indexing.build_neighbor_index import (
    build_neighbor_index,
    build_parser,
    iter_neighbor_records,
    main,
)

__all__ = [
    "build_neighbor_index",
    "build_parser",
    "iter_neighbor_records",
    "main",
]


if __name__ == "__main__":
    main()
