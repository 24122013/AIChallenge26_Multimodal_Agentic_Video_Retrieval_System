from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.app.core.environment import load_project_env
from backend.app.services.ingestion.caption_pipeline import (
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TASK_PROMPT,
    run_caption_file,
)
from backend.app.services.ingestion.common import configure_logging, discover_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate English captions for keyframe metadata.")
    parser.add_argument("--metadata-path", type=Path, required=True, help="JSONL file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--output-path", type=Path, help="Only valid for one input file.")
    parser.add_argument("--report-path", type=Path, help="Only valid for one input file.")
    parser.add_argument(
        "--model-name",
        default=os.getenv("CAPTION_MODEL", DEFAULT_MODEL_NAME),
        help="Florence-2 checkpoint name (default: CAPTION_MODEL or code default).",
    )
    parser.add_argument(
        "--model-revision",
        default=os.getenv("CAPTION_MODEL_REVISION", DEFAULT_MODEL_REVISION),
        help="Pinned commit (default: CAPTION_MODEL_REVISION or code default).",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(os.getenv("CAPTION_MODEL_CACHE_DIR", str(DEFAULT_MODEL_CACHE_DIR))),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=os.getenv("CAPTION_DEVICE", "auto"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "8bit", "4bit"),
        default="none",
        help="Florence-2 currently supports only 'none'; 4/8-bit fail explicitly.",
    )
    parser.add_argument(
        "--task-prompt",
        "--prompt",
        dest="task_prompt",
        default=os.getenv("CAPTION_TASK_PROMPT", DEFAULT_TASK_PROMPT),
        help="Florence-2 task token (legacy alias: --prompt).",
    )
    parser.add_argument("--segment-caption", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    inputs = discover_files(args.metadata_path, "keyframes_*.jsonl")
    if len(inputs) > 1 and (args.output_path or args.report_path):
        raise SystemExit("--output-path/--report-path require a single metadata file.")
    for path in inputs:
        run_caption_file(
            path,
            output_dir=args.output_dir,
            output_path=args.output_path,
            report_path=args.report_path,
            device=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            include_segment_caption=args.segment_caption,
            model_name=args.model_name,
            revision=args.model_revision,
            model_cache_dir=args.model_cache_dir,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            quantization=args.quantization,
            task_prompt=args.task_prompt,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
