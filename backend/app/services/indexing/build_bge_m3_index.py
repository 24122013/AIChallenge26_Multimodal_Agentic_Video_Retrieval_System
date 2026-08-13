"""CLI for building the dense-only BGE-M3 text index from metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from backend.app.services.indexing.build_text_index import load_many
    from backend.app.services.retrieval.bge_dense import (
        DEFAULT_BGE_M3_MODEL,
        DEFAULT_BGE_M3_REVISION,
        build_bge_m3_index,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from backend.app.services.indexing.build_text_index import load_many
    from backend.app.services.retrieval.bge_dense import (
        DEFAULT_BGE_M3_MODEL,
        DEFAULT_BGE_M3_REVISION,
        build_bge_m3_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized 1024-d BGE-M3 dense embeddings from existing "
            "caption/OCR/object metadata; video extraction is never invoked."
        )
    )
    parser.add_argument(
        "--metadata",
        nargs="+",
        default=["data/metadata"],
        help="Metadata JSON/JSONL file(s) or folder(s).",
    )
    parser.add_argument(
        "--output-root",
        default="data/indexes/bge_m3",
        help="Folder for bge_m3_flat_ip.faiss, frame map, and manifest.",
    )
    parser.add_argument("--model-name", default=DEFAULT_BGE_M3_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_BGE_M3_REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default="data/model_cache/bge_m3")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    report = build_bge_m3_index(
        load_many(args.metadata),
        args.output_root,
        model_name=args.model_name,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
