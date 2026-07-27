from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.app.services.ingestion.common import configure_logging, discover_files
from backend.app.services.ingestion.ocr_pipeline import run_ocr_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Vietnamese/English OCR on keyframes.")
    parser.add_argument("--metadata-path", type=Path, required=True, help="JSONL file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--output-path", type=Path, help="Only valid for one input file.")
    parser.add_argument("--report-path", type=Path, help="Only valid for one input file.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
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
        run_ocr_file(
            path,
            output_dir=args.output_dir,
            output_path=args.output_path,
            report_path=args.report_path,
            device=args.device,
            batch_size=args.batch_size,
            conf_threshold=args.conf_threshold,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
