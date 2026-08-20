"""CLI for building the dense-only BGE-M3 text index from metadata."""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
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


def load_canonical_keyframe_records(
    metadata_root: Path,
    *,
    video_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Join final selected keyframes with their text metadata by stable frame ID."""

    if not metadata_root.is_dir():
        raise FileNotFoundError(f"Canonical metadata root not found: {metadata_root}")
    scoped_video_ids = _normalize_video_ids(video_ids)
    keyframe_files = _artifact_files(
        metadata_root,
        "keyframes_*.jsonl",
        scoped_video_ids,
    )
    if not keyframe_files:
        raise FileNotFoundError(
            f"No canonical keyframes_*.jsonl artifacts found in {metadata_root}"
        )
    if scoped_video_ids is not None:
        found_ids = {
            path.stem.removeprefix("keyframes_") for path in keyframe_files
        }
        missing_ids = sorted(set(scoped_video_ids) - found_ids)
        if missing_ids:
            raise FileNotFoundError(
                "Missing canonical selected-keyframe metadata for videos: "
                f"{missing_ids}"
            )
    keyframes = _load_artifact_records(
        keyframe_files,
        pattern="keyframes_*.jsonl",
    )
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

    caption_index = _metadata_index(
        metadata_root,
        "captions_*.jsonl",
        video_ids=scoped_video_ids,
    )
    ocr_index = _metadata_index(
        metadata_root,
        "ocr_*.jsonl",
        video_ids=scoped_video_ids,
    )
    object_index = _metadata_index(
        metadata_root,
        "objects_*.jsonl",
        video_ids=scoped_video_ids,
    )
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
    *,
    video_ids: tuple[str, ...] | None = None,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    files = _artifact_files(metadata_root, pattern, video_ids)
    records = (
        _load_artifact_records(files, pattern=pattern)
        if files
        else []
    )
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        key = _frame_key(record)
        if key is None:
            continue
        if key in output:
            raise ValueError(f"Duplicate metadata frame lineage for {pattern}: {key}")
        output[key] = record
    return output


def _normalize_video_ids(video_ids: Iterable[str] | None) -> tuple[str, ...] | None:
    if video_ids is None:
        return None
    supplied = tuple(str(video_id).strip() for video_id in video_ids)
    normalized = tuple(sorted(set(supplied)))
    if not normalized or any(not video_id for video_id in normalized):
        raise ValueError("video_ids must contain at least one non-empty video ID")
    if any(Path(video_id).name != video_id for video_id in normalized):
        raise ValueError("video_ids must not contain path separators")
    if len({video_id.casefold() for video_id in supplied}) != len(supplied):
        raise ValueError("video_ids must be unique on a case-insensitive filesystem")
    return normalized


def _artifact_files(
    metadata_root: Path,
    pattern: str,
    video_ids: tuple[str, ...] | None,
) -> list[Path]:
    if video_ids is None:
        return sorted(metadata_root.glob(pattern))
    if pattern.count("*") != 1:
        raise ValueError(f"scoped artifact pattern requires one wildcard: {pattern}")
    prefix, suffix = pattern.split("*", 1)
    return [
        path
        for video_id in video_ids
        if (path := metadata_root / f"{prefix}{video_id}{suffix}").is_file()
    ]


def _load_artifact_records(
    files: Iterable[Path],
    *,
    pattern: str,
) -> list[dict[str, Any]]:
    # Canonical filenames encode exactly one video ID.  Enforce that lineage
    # in both explicit/scoped loading and the legacy directory-wide CLI path;
    # otherwise a misnamed file can silently inject another video's records.
    prefix, suffix = pattern.split("*", 1)
    stem_suffix = Path(f"placeholder{suffix}").stem.removeprefix("placeholder")
    records: list[dict[str, Any]] = []
    for path in files:
        stem = path.stem
        if not stem.startswith(prefix) or (stem_suffix and not stem.endswith(stem_suffix)):
            raise ValueError(f"Artifact filename does not match {pattern}: {path}")
        end = len(stem) - len(stem_suffix) if stem_suffix else len(stem)
        expected_video_id = stem[len(prefix) : end]
        for record in load_many((path,)):
            if str(record.get("video_id") or "") != expected_video_id:
                raise ValueError(
                    f"Scoped artifact {path.name} contains video_id="
                    f"{record.get('video_id')!r}, expected {expected_video_id!r}"
                )
            records.append(record)
    return records


def _frame_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    video_id = str(record.get("video_id") or "").strip()
    frame_id = str(record.get("frame_id") or "").strip()
    return (video_id, frame_id) if video_id and frame_id else None


if __name__ == "__main__":
    main()
