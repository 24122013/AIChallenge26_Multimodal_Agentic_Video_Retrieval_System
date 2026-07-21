"""CLI: sinh caption cho keyframe.

Ví dụ (stub, chạy được ngay):
    python -B backend/app/services/ingestion/run_caption.py \
        --metadata-path data/metadata/keyframes_L01_V001.jsonl \
        --output-path data/metadata/captions_L01_V001.jsonl \
        --backend stub

Dùng model thật trên máy GPU:
    ... --backend qwen --model-id Qwen/Qwen2.5-VL-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import logging

from backend.app.services.ingestion.caption_pipeline import run_caption_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption keyframes")
    parser.add_argument("--metadata-path", required=True, help="Keyframe metadata JSONL")
    parser.add_argument("--output-path", required=True, help="Output caption JSONL")
    parser.add_argument("--backend", default="stub", help="stub | qwen")
    parser.add_argument("--model-id", default=None, help="HF model id cho backend thật")
    parser.add_argument("--image-base-dir", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-image", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    backend_kwargs = {}
    if args.model_id:
        backend_kwargs["model_id"] = args.model_id

    report = run_caption_pipeline(
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
