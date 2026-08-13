"""Isolated one-video ASR worker used by the competition pipeline.

Keeping CTranslate2 in a child process lets the parent enforce a real timeout
and terminate a stuck CUDA kernel without losing other modality checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from backend.app.services.ingestion.asr_pipeline import WhisperBackend, run_asr_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe one competition video.")
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--segments-output-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument(
        "--backend",
        choices=("auto", "faster-whisper", "whisper"),
        default="auto",
    )
    parser.add_argument("--no-vad", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = WhisperBackend(
        model_size=args.model_size,
        device=args.device,
        backend=args.backend,
        cache_dir=args.model_cache_dir,
        vad_filter=not args.no_vad,
    )
    report = run_asr_file(
        args.video_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        segments_output_path=args.segments_output_path,
        report_path=args.report_path,
        device=args.device,
        model_size=args.model_size,
        backend_name=args.backend,
        overwrite=True,
        vad_filter=not args.no_vad,
        backend=backend,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report.get("error_count") == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
