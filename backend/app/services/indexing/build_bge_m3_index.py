"""CLI for building the dense-only BGE-M3 text index from metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from backend.app.services.indexing.build_text_index import load_many
    from backend.app.services.retrieval.bge_dense import (
        DEFAULT_BGE_M3_MODEL,
        DEFAULT_BGE_M3_REVISION,
        build_bge_m3_index,
        has_retrievable_text,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from backend.app.services.indexing.build_text_index import load_many
    from backend.app.services.retrieval.bge_dense import (
        DEFAULT_BGE_M3_MODEL,
        DEFAULT_BGE_M3_REVISION,
        build_bge_m3_index,
        has_retrievable_text,
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
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help=(
            "Reject candidate-level metadata and require selected-keyframe or "
            "Phase-4 segment lineage. Competition/E2E runs must enable this."
        ),
    )
    args = parser.parse_args()

    records = (
        load_canonical_keyframe_records(Path(args.metadata[0]))
        if args.canonical_only
        and len(args.metadata) == 1
        and Path(args.metadata[0]).is_dir()
        else load_many(args.metadata)
    )

    report = build_bge_m3_index(
        records,
        args.output_root,
        model_name=args.model_name,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        canonical_only=args.canonical_only,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def load_canonical_keyframe_records(metadata_root: Path) -> list[dict[str, Any]]:
    """Join final selected keyframes with their text metadata by stable frame ID."""

    if not metadata_root.is_dir():
        raise FileNotFoundError(f"Canonical metadata root not found: {metadata_root}")
    keyframe_files = sorted(metadata_root.glob("keyframes_*.jsonl"))
    if not keyframe_files:
        raise FileNotFoundError(
            f"No canonical keyframes_*.jsonl artifacts found in {metadata_root}"
        )
    keyframes = load_many(keyframe_files)
    for index, record in enumerate(keyframes):
        if str(record.get("artifact_role") or "").casefold() != "selected_keyframe":
            raise ValueError(
                f"canonical keyframe row {index} is not artifact_role=selected_keyframe"
            )
        if not _frame_key(record) or not str(
            record.get("keyframe_path") or record.get("frame_path") or ""
        ).strip():
            raise ValueError(
                f"canonical keyframe row {index} lacks stable frame/image lineage"
            )

    caption_index = _metadata_index(metadata_root, "captions_*.jsonl")
    ocr_index = _metadata_index(metadata_root, "ocr_*.jsonl")
    object_index = _metadata_index(metadata_root, "objects_*.jsonl")
    joined: list[dict[str, Any]] = []
    for keyframe in keyframes:
        key = _frame_key(keyframe)
        assert key is not None
        record = dict(keyframe)
        caption = caption_index.get(key, {})
        ocr = ocr_index.get(key, {})
        objects = object_index.get(key, {})
        record["caption"] = str(
            caption.get("caption")
            or caption.get("captions_aggregated")
            or caption.get("caption_text")
            or ""
        )
        record["ocr_text"] = str(ocr.get("ocr_text") or "")
        record["objects"] = objects.get(
            "objects",
            objects.get("object_classes", []),
        )
        record["bge_source_kind"] = "canonical_selected_keyframe"
        if has_retrievable_text(record):
            joined.append(record)
    if not joined:
        raise ValueError("Canonical keyframes have no caption/OCR/object text to index")
    return joined


def _metadata_index(
    metadata_root: Path,
    pattern: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    files = sorted(metadata_root.glob(pattern))
    records = load_many(files) if files else []
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        key = _frame_key(record)
        if key is None:
            continue
        if key in output:
            raise ValueError(f"Duplicate metadata frame lineage for {pattern}: {key}")
        output[key] = record
    return output


def _frame_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    video_id = str(record.get("video_id") or "").strip()
    frame_id = str(record.get("frame_id") or "").strip()
    return (video_id, frame_id) if video_id and frame_id else None


if __name__ == "__main__":
    main()
