"""Run the complete public TKIS/VKIS pipeline and validate its submission.

Each stage runs in a separate Python process.  This releases model/GPU memory
between stages and lets a failed long run restart from a named stage without
duplicating orchestration logic from :mod:`competition.pipeline`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from backend.app.services.indexing.keyframe_selection import DEFAULT_MAX_GAP_SECONDS
from competition.experiment_tracker import (
    DEFAULT_REPORT_PATH,
    append_experiment,
    build_runner_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_ROOT = REPO_ROOT / "data" / "public"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "competition"
DEFAULT_RETRIEVAL_CONFIG = REPO_ROOT / "configs" / "retrieval.yaml"
STAGES = (
    "validate-input",
    "keyframes",
    "index",
    "neighbors",
    "segments",
    "text-index",
    "predict",
    "validate-submission",
)
CUDA_OOM_EXIT_CODE = 75
RETRYABLE_KEYFRAME_EXIT_CODES = {
    3221225477,
    -1073741819,
    CUDA_OOM_EXIT_CODE,
}


def _batch_size(value: str) -> str:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch size must be 'auto' or an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return str(parsed)


def _score(value: str) -> float:
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise argparse.ArgumentTypeError("score must be within [0, 1]")
    return score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation, multimodal keyframe extraction, FAISS/text indexing, "
            "TKIS/VKIS prediction, and strict submission validation."
        )
    )
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--submission-path", type=Path, default=None)
    parser.add_argument("--experiment-report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--public-score", type=_score, default=None)
    parser.add_argument("--private-score", type=_score, default=None)
    parser.add_argument("--experiment-note", default=None)
    parser.add_argument(
        "--no-experiment-log",
        action="store_true",
        help="Do not append this run to reports/Experiment.md.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=_batch_size, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--candidate-interval-sec", type=float, default=0.5)
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=DEFAULT_MAX_GAP_SECONDS,
    )
    parser.add_argument("--neighbor-window-seconds", type=float, default=5.0)
    parser.add_argument("--search-depth", type=int, default=200)
    parser.add_argument(
        "--tkis-routing",
        choices=("hybrid", "auto-temporal"),
        default="auto-temporal",
    )
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG,
    )
    parser.add_argument("--vkis-refine-top-k", type=int, default=20)
    parser.add_argument("--vkis-refine-radius-frames", type=int, default=75)
    parser.add_argument("--start-at", choices=STAGES, default=STAGES[0])
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Do not pass --resume to the expensive keyframe stage.",
    )
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument(
        "--native-retries",
        type=int,
        default=5,
        help=(
            "Retry the resumable keyframe stage after a Windows native access "
            "violation or CUDA OOM (default: 5)."
        ),
    )
    parser.add_argument(
        "--allow-partial-features",
        action="store_true",
        help=(
            "Explicit degraded mode. Do not use for a final full-multimodal run "
            "unless missing OCR/object coverage is accepted."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser


def selected_stage_names(start_at: str, stop_after: str) -> tuple[str, ...]:
    start = STAGES.index(start_at)
    stop = STAGES.index(stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")
    return STAGES[start : stop + 1]


def _run_stage_with_native_retries(
    command: Sequence[str],
    *,
    stage: str,
    environment: dict[str, str],
    retries: int,
) -> None:
    attempt = 0
    while True:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode == 0:
            return
        retryable = (
            stage == "keyframes"
            and completed.returncode in RETRYABLE_KEYFRAME_EXIT_CODES
            and attempt < retries
        )
        if not retryable:
            raise subprocess.CalledProcessError(completed.returncode, command)
        attempt += 1
        reason = (
            "CUDA OOM"
            if completed.returncode == CUDA_OOM_EXIT_CODE
            else "Windows native access violation"
        )
        print(
            f"{reason} during keyframes; "
            f"restarting from durable checkpoints ({attempt}/{retries}).",
            flush=True,
        )


def runtime_preflight(
    *,
    device: str,
    require_ffmpeg: bool,
    torch_module: object | None = None,
) -> dict[str, object]:
    """Fail early for the environment errors that otherwise appear mid-video."""

    if require_ffmpeg:
        missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
        if missing:
            raise ValueError(
                "missing required executable(s) in PATH: " + ", ".join(missing)
            )
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise ValueError("PyTorch is not installed in the active environment") from exc

    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    version_info = getattr(torch_module, "version", None)
    cuda_build = getattr(version_info, "cuda", None)
    cuda_api = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda_api is not None and cuda_api.is_available())
    if device == "cuda" and not cuda_available:
        build_detail = (
            "CPU-only PyTorch build"
            if cuda_build is None
            else f"CUDA build {cuda_build!s} cannot access a GPU"
        )
        raise ValueError(
            f"--device cuda was requested, but torch {torch_version} reports "
            f"cuda_available=False ({build_detail}). Install a CUDA-enabled torch/"
            "torchvision wheel from https://pytorch.org/get-started/locally/ or "
            "rerun with --device cpu."
        )

    resolved_device = "cuda" if device in {"auto", "cuda"} and cuda_available else "cpu"
    device_name: str | None = None
    if resolved_device == "cuda":
        device_name = str(cuda_api.get_device_name(0))
    return {
        "torch_version": torch_version,
        "torch_cuda_build": cuda_build,
        "cuda_available": cuda_available,
        "resolved_device": resolved_device,
        "device_name": device_name,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "ffprobe_available": shutil.which("ffprobe") is not None,
    }


def build_stage_commands(
    args: argparse.Namespace,
    *,
    python_executable: str | Path = sys.executable,
) -> dict[str, list[str]]:
    public_root = args.public_root.resolve()
    output_root = args.output_root.resolve()
    submission_path = (
        args.submission_path.resolve()
        if args.submission_path is not None
        else output_root / "results" / "submission.csv"
    )
    retrieval_config = args.retrieval_config.resolve()
    python = str(python_executable)
    module = [python, "-m", "competition.pipeline"]
    common = [
        "--public-root",
        str(public_root),
        "--output-root",
        str(output_root),
    ]
    keyframes = [
        *module,
        "keyframes",
        *common,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--candidate-interval-sec",
        str(args.candidate_interval_sec),
        "--max-gap-seconds",
        str(args.max_gap_seconds),
    ]
    if not args.fresh:
        keyframes.append("--resume")
    if args.no_autocast:
        keyframes.append("--no-autocast")
    if args.allow_partial_features:
        keyframes.append("--allow-partial-features")

    predict = [
        *module,
        "predict",
        *common,
        "--submission-path",
        str(submission_path),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--search-depth",
        str(args.search_depth),
        "--tkis-routing",
        args.tkis_routing,
        "--retrieval-config",
        str(retrieval_config),
        "--vkis-refine-top-k",
        str(args.vkis_refine_top_k),
        "--vkis-refine-radius-frames",
        str(args.vkis_refine_radius_frames),
    ]
    if args.no_autocast:
        predict.append("--no-autocast")

    return {
        "validate-input": [
            *module,
            "validate-input",
            "--public-root",
            str(public_root),
        ],
        "keyframes": keyframes,
        "index": [*module, "index", *common],
        "neighbors": [
            *module,
            "neighbors",
            *common,
            "--window-seconds",
            str(args.neighbor_window_seconds),
        ],
        "segments": [*module, "segments", *common, "--strategy", "auto"],
        "text-index": [*module, "text-index", *common],
        "predict": predict,
        "validate-submission": [
            *module,
            "validate-submission",
            "--public-root",
            str(public_root),
            "--submission-path",
            str(submission_path),
        ],
    }


def run(args: argparse.Namespace) -> Path:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.candidate_interval_sec <= 0 or args.max_gap_seconds <= 0:
        raise ValueError("candidate interval and max gap must be positive")
    if args.neighbor_window_seconds < 0:
        raise ValueError("--neighbor-window-seconds must be non-negative")
    if args.search_depth < 100:
        raise ValueError("--search-depth must be at least 100")
    if args.vkis_refine_top_k < 0 or args.vkis_refine_radius_frames < 0:
        raise ValueError("VKIS refinement values must be non-negative")
    if args.native_retries < 0:
        raise ValueError("--native-retries must be non-negative")

    stages = selected_stage_names(args.start_at, args.stop_after)
    commands = build_stage_commands(args)
    submission_path = (
        args.submission_path.resolve()
        if args.submission_path is not None
        else args.output_root.resolve() / "results" / "submission.csv"
    )
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    run_started = time.perf_counter()
    stage_durations: dict[str, float] = {}
    final_status = "failed"
    failure: str | None = None
    try:
        model_stages = {"keyframes", "predict"}.intersection(stages)
        if model_stages and not args.dry_run:
            preflight = runtime_preflight(
                device=args.device,
                require_ffmpeg="keyframes" in stages,
            )
            print(
                "Runtime preflight:\n"
                + json.dumps(preflight, ensure_ascii=False, indent=2),
                flush=True,
            )
        for position, stage in enumerate(stages, start=1):
            command = commands[stage]
            print(f"\n[{position}/{len(stages)}] {stage}", flush=True)
            print(subprocess.list2cmdline(command), flush=True)
            stage_started = time.perf_counter()
            try:
                if not args.dry_run:
                    _run_stage_with_native_retries(
                        command,
                        stage=stage,
                        environment=environment,
                        retries=args.native_retries,
                    )
            finally:
                stage_durations[stage] = time.perf_counter() - stage_started

        final_status = "dry_run" if args.dry_run else "completed"
        print(
            json.dumps(
                {
                    "status": final_status,
                    "stages": list(stages),
                    "submission_path": submission_path.as_posix(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return submission_path
    except BaseException as exc:
        final_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not args.dry_run and not args.no_experiment_log:
            try:
                record = build_runner_record(
                    status=final_status,
                    public_root=args.public_root.resolve(),
                    output_root=args.output_root.resolve(),
                    submission_path=submission_path,
                    stages=stages,
                    stage_durations_seconds=stage_durations,
                    duration_seconds=time.perf_counter() - run_started,
                    device=args.device,
                    batch_size=str(args.batch_size),
                    candidate_interval_sec=args.candidate_interval_sec,
                    max_gap_seconds=args.max_gap_seconds,
                    public_score=args.public_score,
                    private_score=args.private_score,
                    note=args.experiment_note,
                    error=failure,
                )
                append_experiment(args.experiment_report.resolve(), record)
                print(
                    f"Experiment appended: {args.experiment_report.resolve()}",
                    flush=True,
                )
            except Exception as tracking_error:  # noqa: BLE001 - do not mask pipeline status.
                print(
                    f"warning: failed to update experiment report: {tracking_error}",
                    file=sys.stderr,
                    flush=True,
                )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"end-to-end pipeline failed: {exc}\n")


if __name__ == "__main__":
    main()
