"""CLI: OCR keyframe.

Ví dụ (stub):
    python -B backend/app/services/ingestion/run_ocr.py \
        --metadata-path data/metadata/keyframes_L01_V001.jsonl \
        --output-path data/metadata/ocr_L01_V001.jsonl --backend stub

Model thật:
    ... --backend easyocr --languages en vi
"""
from __future__ import annotations

import argparse
import json
import logging

from backend.app.services.ingestion.ocr_pipeline import run_ocr_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR keyframes")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--backend", default="stub", help="stub | easyocr")
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--image-base-dir", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-image", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    backend_kwargs = {}
    if args.languages:
        backend_kwargs["languages"] = args.languages
    if args.no_gpu:
        backend_kwargs["gpu"] = False

    report = run_ocr_pipeline(
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        backend=args.backend,
        image_base_dir=args.image_base_dir,
        report_path=args.report_path,
        limit=args.limit,
        require_image=args.require_image,
        backend_kwargs=backend_kwargs,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
