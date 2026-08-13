"""Build lexical text indexes for Retrieval Phase 2."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:
    from backend.app.services.retrieval.text_index import build_text_index
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from backend.app.services.retrieval.text_index import build_text_index


_ARTIFACT_PREFIXES = ("captions_", "ocr_", "objects_", "segments_")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load one JSON/JSONL file or discover current metadata artifacts in a folder."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Metadata source not found: {source}")
    if source.is_dir():
        return _load_directory(source)
    return _load_file(source)


def load_many(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_records(path))
    return records


def write_text_index(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_text_index(records)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            temporary_path = Path(handle.name)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return {
        "output_path": output.as_posix(),
        "input_records": len(records),
        "modalities": {
            modality: data["stats"]
            for modality, data in payload.get("modalities", {}).items()
        },
    }


def _load_directory(source: Path) -> list[dict[str, Any]]:
    segments_all = source / "segments_all.jsonl"
    if segments_all.exists():
        return _load_file(segments_all)

    segment_files = sorted(
        path
        for path in source.glob("segments_*.jsonl")
        if path.name != "segments_all.jsonl"
    )
    if segment_files:
        return load_many(segment_files)

    artifact_files = sorted(
        path
        for path in source.glob("*.jsonl")
        if path.name.startswith(_ARTIFACT_PREFIXES)
        and "_report" not in path.stem
        and not path.name.startswith("keyframes_")
    )
    if not artifact_files:
        raise FileNotFoundError(
            f"No segment or multimodal JSONL artifacts found in {source}"
        )
    return load_many(artifact_files)


def _load_file(source: Path) -> list[dict[str, Any]]:
    if source.suffix.casefold() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {source} at line {line_number}: {exc}"
                ) from exc
            if isinstance(value, dict):
                records.append(value)
        return records

    raw = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        records = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            record = dict(value)
            record.setdefault(
                "faiss_index",
                int(key) if str(key).isdigit() else key,
            )
            records.append(record)
        return records
    if isinstance(raw, list):
        return [record for record in raw if isinstance(record, dict)]
    raise ValueError(f"Unsupported metadata format in {source}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Retrieval Phase 2 caption/OCR/object text index."
    )
    parser.add_argument(
        "--metadata",
        nargs="+",
        default=["data/metadata"],
        help=(
            "Metadata file(s) or folder(s). Folder mode prefers segments_all.jsonl, "
            "then segments_<video>.jsonl, then separate multimodal artifacts."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/indexes/retrieval_text_index.json",
        help="Output text index JSON path.",
    )
    args = parser.parse_args()
    summary = write_text_index(load_many(args.metadata), args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
