"""CLI: object detection cho keyframe.

Ví dụ (stub):
    python -B backend/app/services/ingestion/run_object_detection.py \
        --metadata-path data/metadata/keyframes_L01_V001.jsonl \
        --output-path data/metadata/objects_L01_V001.jsonl --backend stub

Model thật:
    ... --backend yolo --model-id yolov8n.pt --conf-threshold 0.25
"""
from __future__ import annotations

import argparse
import json
import logging

from backend.app.services.ingestion.object_pipeline import run_object_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect objects in keyframes")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--backend", default="stub", help="stub | yolo")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--conf-threshold", type=float, default=None)
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
    if args.conf_threshold is not None:
        backend_kwargs["conf_threshold"] = args.conf_threshold

    report = run_object_pipeline(
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
