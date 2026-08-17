from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.app.services.ingestion.common import configure_logging, discover_files
from backend.app.services.ingestion.ocr_pipeline import (
    DEFAULT_DETECTION_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_RECOGNITION_MODEL,
    run_ocr_file,
)


def prepare_isolated_paddle_runtime() -> None:
    """Load Paddle first and hide Torch from optional PaddleX dependencies.

    On Windows, the CUDA wheels for Paddle and Torch can bundle different
    cuDNN builds under identical DLL names.  Loading both frameworks in one
    process then fails with WinError 127.  The OCR worker does not use Torch,
    but ModelScope (an optional PaddleX model source) detects and imports it
    when it is installed.  Preloading Paddle and hiding Torch from optional
    dependency detection keeps this dedicated worker on one CUDA runtime.
    """

    if "torch" in sys.modules:
        raise RuntimeError(
            "The isolated OCR worker must start before Torch is imported."
        )

    # Import Paddle before PaddleOCR/PaddleX so its CUDA and cuDNN DLLs own the
    # process.  This worker exits after OCR, so the patch is intentionally
    # process-local and does not affect the parent pipeline.
    import paddle  # noqa: F401
    os.environ["OCR_ISOLATED_PADDLE_RUNTIME"] = "1"

    original_find_spec = importlib.util.find_spec

    def find_spec_without_torch(name: str, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            return None
        return original_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = find_spec_without_torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Vietnamese/English OCR on keyframes.")
    parser.add_argument("--metadata-path", type=Path, required=True, help="JSONL file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--output-path", type=Path, help="Only valid for one input file.")
    parser.add_argument("--report-path", type=Path, help="Only valid for one input file.")
    parser.add_argument("--detection-model", default=DEFAULT_DETECTION_MODEL)
    parser.add_argument("--recognition-model", default=DEFAULT_RECOGNITION_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--model-cache-dir", type=Path, default=Path("data/model_cache/ocr"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--isolate-paddle-runtime",
        action="store_true",
        help=(
            "Preload Paddle and prevent optional PaddleX dependencies from "
            "loading Torch/cuDNN in this dedicated OCR process."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.isolate_paddle_runtime:
        prepare_isolated_paddle_runtime()
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
            detection_model=args.detection_model,
            recognition_model=args.recognition_model,
            revision=args.model_revision,
            model_cache_dir=args.model_cache_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
