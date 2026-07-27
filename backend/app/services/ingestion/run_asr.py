from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.app.services.ingestion.asr_pipeline import DEFAULT_MODEL_SIZE, run_asr_file
from backend.app.services.ingestion.common import configure_logging, discover_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe video audio and map it to shots.")
    parser.add_argument("--video-path", type=Path, required=True, help="Video file or directory.")
    parser.add_argument("--video-glob", default="*.mp4")
    parser.add_argument("--metadata-path", type=Path, help="Matching keyframe JSONL or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--output-path", type=Path, help="Only valid for one video.")
    parser.add_argument("--segments-output-path", type=Path, help="Only valid for one video.")
    parser.add_argument("--report-path", type=Path, help="Only valid for one video.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("auto", "faster-whisper", "whisper"), default="auto")
    parser.add_argument("--model-size", default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _metadata_for(video: Path, metadata_path: Path | None) -> Path | None:
    if metadata_path is None:
        candidate = Path("data/metadata") / f"keyframes_{video.stem}.jsonl"
        return candidate if candidate.exists() else None
    if metadata_path.is_file():
        return metadata_path
    candidate = metadata_path / f"keyframes_{video.stem}.jsonl"
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.metadata_path is not None and not args.metadata_path.exists():
        raise SystemExit(f"Metadata input does not exist: {args.metadata_path}")
    videos = discover_files(args.video_path, args.video_glob)
    if len(videos) > 1 and (args.output_path or args.segments_output_path or args.report_path):
        raise SystemExit("Explicit output paths require a single video.")
    for video in videos:
        run_asr_file(
            video,
            metadata_path=_metadata_for(video, args.metadata_path),
            output_dir=args.output_dir,
            output_path=args.output_path,
            segments_output_path=args.segments_output_path,
            report_path=args.report_path,
            device=args.device,
            backend_name=args.backend,
            model_size=args.model_size,
            vad_filter=not args.no_vad,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
