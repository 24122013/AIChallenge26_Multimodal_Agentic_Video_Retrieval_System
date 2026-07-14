"""Build lexical text indexes for Phase 2 multimodal retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.app.services.retrieval.text_index import build_text_index


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"metadata source not found: {source}")
    if source.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raw = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        records = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            record = dict(value)
            record.setdefault("faiss_index", int(key) if str(key).isdigit() else key)
            records.append(record)
        return records
    if isinstance(raw, list):
        return [record for record in raw if isinstance(record, dict)]
    raise ValueError(f"unsupported metadata format in {source}")


def write_text_index(records: list[dict[str, Any]], output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_text_index(records)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "output_path": output.as_posix(),
        "modalities": {
            modality: data["stats"]
            for modality, data in payload.get("modalities", {}).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Retrieval Phase 2 text index.")
    parser.add_argument(
        "--metadata",
        default="data/metadata/openclip_vit_b16_frame_map.json",
        help="Frame map JSON, metadata JSON, or metadata JSONL.",
    )
    parser.add_argument(
        "--output",
        default="data/indexes/retrieval_text_index.json",
        help="Output text index JSON path.",
    )
    args = parser.parse_args()
    summary = write_text_index(load_records(args.metadata), args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
