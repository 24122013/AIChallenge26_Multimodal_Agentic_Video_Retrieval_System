from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.app.services.ingestion.common import configure_logging, discover_files
from backend.app.services.ingestion.object_pipeline import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_VOCABULARY,
    run_object_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO object detection on keyframes.")
    parser.add_argument("--metadata-path", type=Path, required=True, help="JSONL file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--output-path", type=Path, help="Only valid for one input file.")
    parser.add_argument("--report-path", type=Path, help="Only valid for one input file.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--model-cache-dir", type=Path, default=Path("data/model_cache/objects"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--prompt-mode", choices=("text", "internal"), default="text")
    parser.add_argument(
        "--vocabulary",
        nargs="+",
        default=list(DEFAULT_VOCABULARY),
        help="Text-prompt classes used by YOLOE (space-separated).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    inputs = discover_files(args.metadata_path, "keyframes_*.jsonl")
    if len(inputs) > 1 and (args.output_path or args.report_path):
        raise SystemExit("--output-path/--report-path require a single metadata file.")
    for path in inputs:
        run_object_file(
            path,
            output_dir=args.output_dir,
            output_path=args.output_path,
            report_path=args.report_path,
            device=args.device,
            batch_size=args.batch_size,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            overwrite=args.overwrite,
            model_name=args.model_name,
            revision=args.model_revision,
            model_cache_dir=args.model_cache_dir,
            vocabulary=args.vocabulary,
            prompt_mode=args.prompt_mode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
