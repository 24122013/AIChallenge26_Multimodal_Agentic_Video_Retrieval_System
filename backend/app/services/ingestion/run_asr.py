"""CLI: ASR (transcript) cho một video.

Ví dụ (stub):
    python -B backend/app/services/ingestion/run_asr.py \
        --video-path data/raw/video/L01_V001.mp4 \
        --output-path data/metadata/asr_L01_V001.jsonl --backend stub

Model thật:
    ... --backend whisper --model-id large-v3
"""
from __future__ import annotations

import argparse
import json
import logging

from backend.app.services.ingestion.asr_pipeline import run_asr_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR transcript for a video")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--backend", default="stub", help="stub | whisper")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    backend_kwargs = {}
    if args.model_id:
        backend_kwargs["model_id"] = args.model_id
    if args.language:
        backend_kwargs["language"] = args.language

    report = run_asr_pipeline(
        video_path=args.video_path,
        output_path=args.output_path,
        backend=args.backend,
        video_id=args.video_id,
        report_path=args.report_path,
        backend_kwargs=backend_kwargs,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
