"""Run the complete terminal-only retrieval v2 architecture.

This orchestrator intentionally contains no retrieval algorithm.  It launches
the durable CLI stages in independent processes so GPU/model memory is released
between stages and a stopped run can resume from its persisted ``run_root``.
Ground-truth evaluation is deliberately outside this runner; Experiment.md is
updated only after every architecture stage and strict submission validation
have passed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from competition.experiment_tracker import (
    DEFAULT_REPORT_PATH,
    append_experiment,
    build_runner_record,
)
from competition.run_end_to_end import (
    _batch_size,
    _run_stage_with_native_retries,
    runtime_preflight,
)
from competition.run_manifest import (
    dataset_fingerprint,
    git_fingerprint,
    initialize_run_manifest,
    read_run_manifest,
    sha256_file,
    update_run_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_ROOT = REPO_ROOT / "data" / "public"
DEFAULT_RETRIEVAL_CONFIG = REPO_ROOT / "configs" / "retrieval.yaml"
DEFAULT_MODEL_CACHE_ROOT = REPO_ROOT / "data" / "model_cache"
STAGES = (
    "validate-input",
    "keyframes",
    "index",
    "neighbors",
    "segments",
    "text-index",
    "bge-text-index",
    "dense-index",
    "predict",
    "validate-submission",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build offline multimodal artifacts, coarse+dense indexes, run "
            "advanced retrieval, and validate one competition submission."
        )
    )
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--submission-path", type=Path, default=None)
    parser.add_argument("--experiment-report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--experiment-note", default=None)
    parser.add_argument("--no-experiment-log", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=_batch_size, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--candidate-interval-sec", type=float, default=0.5)
    parser.add_argument("--max-gap-seconds", type=float, default=2.0)
    parser.add_argument("--target-density-per-second", type=float, default=0.5)
    parser.add_argument("--dedup-similarity-threshold", type=float, default=0.92)
    parser.add_argument(
        "--endpoint-protection",
        choices=("on", "off"),
        default="on",
    )
    parser.add_argument("--neighbor-window-seconds", type=float, default=5.0)
    parser.add_argument("--model-cache-root", type=Path, default=None)
    parser.add_argument("--offline-model-cache", action="store_true")
    parser.add_argument("--retrieval-config", type=Path, default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--search-depth", type=int, default=300)
    parser.add_argument("--coarse-top-n", type=int, default=50)
    parser.add_argument("--dense-global-top-k", type=int, default=300)
    parser.add_argument("--dense-rescue-clips", type=int, default=10)
    parser.add_argument("--max-candidate-clips", type=int, default=60)
    parser.add_argument("--dense-frames-per-clip", type=int, default=12)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--modality-hint-boost", type=float, default=1.5)
    parser.add_argument("--cses-similarity-threshold", type=float, default=0.92)
    parser.add_argument("--cses-temporal-window-seconds", type=float, default=2.0)
    parser.add_argument("--vkis-refine-top-k", type=int, default=20)
    parser.add_argument("--vkis-refine-radius-frames", type=int, default=75)
    parser.add_argument("--vlm-mode", choices=("off", "optional", "required"), default="off")
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--vlm-model-revision", default="main")
    parser.add_argument("--vlm-top-m", type=int, default=20)
    parser.add_argument("--vlm-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--bge-dense-mode",
        choices=("off", "optional", "required"),
        default="off",
    )
    parser.add_argument(
        "--bge-reranker-mode",
        choices=("off", "optional", "required"),
        default="off",
    )
    parser.add_argument("--bge-m3-model-name", default="BAAI/bge-m3")
    parser.add_argument("--bge-m3-model-revision", default="main")
    parser.add_argument(
        "--bge-reranker-model-name",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument("--bge-reranker-model-revision", default="main")
    parser.add_argument("--bge-reranker-alpha", type=float, default=0.5)
    parser.add_argument("--bge-batch-size", type=int, default=16)
    parser.add_argument("--start-at", choices=STAGES, default=STAGES[0])
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--native-retries", type=int, default=5)
    parser.add_argument("--allow-partial-features", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def selected_stage_names(start_at: str, stop_after: str) -> tuple[str, ...]:
    start = STAGES.index(start_at)
    stop = STAGES.index(stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")
    return STAGES[start : stop + 1]


def _resolved_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    public_root = args.public_root.resolve()
    run_root = args.run_root.resolve()
    submission_path = (
        args.submission_path.resolve()
        if args.submission_path is not None
        else run_root / "results" / "submission.csv"
    )
    cache_root = (
        args.model_cache_root.resolve()
        if args.model_cache_root is not None
        else DEFAULT_MODEL_CACHE_ROOT
    )
    return public_root, run_root, submission_path, cache_root


def build_stage_commands(
    args: argparse.Namespace,
    *,
    python_executable: str | Path = sys.executable,
) -> dict[str, list[str]]:
    public_root, run_root, submission_path, cache_root = _resolved_paths(args)
    python = str(python_executable)
    module = [python, "-m", "competition.pipeline"]
    common = ["--public-root", str(public_root), "--output-root", str(run_root)]
    model_cache_dir = cache_root / "huggingface"

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
        "--target-density-per-second",
        str(args.target_density_per_second),
        "--dedup-similarity-threshold",
        str(args.dedup_similarity_threshold),
        "--endpoint-protection",
        args.endpoint_protection,
        "--model-cache-dir",
        str(model_cache_dir),
        "--model-cache-root",
        str(cache_root),
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
        "--run-root",
        str(run_root),
        "--dense-run-root",
        str(run_root),
        "--submission-path",
        str(submission_path),
        "--retrieval-mode",
        "advanced",
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--model-cache-dir",
        str(model_cache_dir),
        "--retrieval-config",
        str(args.retrieval_config.resolve()),
        "--search-depth",
        str(args.search_depth),
        "--tkis-routing",
        "auto-temporal",
        "--coarse-top-n",
        str(args.coarse_top_n),
        "--dense-global-top-k",
        str(args.dense_global_top_k),
        "--dense-rescue-clips",
        str(args.dense_rescue_clips),
        "--max-candidate-clips",
        str(args.max_candidate_clips),
        "--dense-frames-per-clip",
        str(args.dense_frames_per_clip),
        "--rrf-k",
        str(args.rrf_k),
        "--modality-hint-boost",
        str(args.modality_hint_boost),
        "--cses-similarity-threshold",
        str(args.cses_similarity_threshold),
        "--cses-temporal-window-seconds",
        str(args.cses_temporal_window_seconds),
        "--vkis-refine-top-k",
        str(args.vkis_refine_top_k),
        "--vkis-refine-radius-frames",
        str(args.vkis_refine_radius_frames),
        "--vlm-mode",
        args.vlm_mode,
        "--vlm-model-name",
        args.vlm_model_name,
        "--vlm-model-revision",
        args.vlm_model_revision,
        "--vlm-top-m",
        str(args.vlm_top_m),
        "--vlm-timeout-seconds",
        str(args.vlm_timeout_seconds),
        "--bge-dense-mode",
        args.bge_dense_mode,
        "--bge-text-index-root",
        str(run_root / "indexes" / "bge_m3"),
        "--bge-m3-model-name",
        args.bge_m3_model_name,
        "--bge-m3-model-revision",
        args.bge_m3_model_revision,
        "--bge-reranker-mode",
        args.bge_reranker_mode,
        "--bge-reranker-model-name",
        args.bge_reranker_model_name,
        "--bge-reranker-model-revision",
        args.bge_reranker_model_revision,
        "--bge-reranker-alpha",
        str(args.bge_reranker_alpha),
        "--bge-batch-size",
        str(args.bge_batch_size),
        "--bge-model-cache-dir",
        str(cache_root / "bge_m3"),
    ]
    if args.offline_model_cache:
        predict.append("--offline-model-cache")
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
        "bge-text-index": [
            python,
            "-m",
            "backend.app.services.indexing.build_bge_m3_index",
            "--metadata",
            str(run_root / "metadata"),
            "--output-root",
            str(run_root / "indexes" / "bge_m3"),
            "--model-name",
            args.bge_m3_model_name,
            "--model-revision",
            args.bge_m3_model_revision,
            "--batch-size",
            str(args.bge_batch_size),
            "--device",
            args.device,
            "--cache-dir",
            str(cache_root / "bge_m3"),
        ],
        "dense-index": [
            *module,
            "dense-index",
            "--run-root",
            str(run_root),
            "--source-workspace",
            str(run_root / "work" / "keyframe_v3"),
            "--source-output-root",
            str(run_root),
            "--public-root",
            str(public_root),
        ],
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


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "candidate interval": args.candidate_interval_sec,
        "max gap": args.max_gap_seconds,
        "target density": args.target_density_per_second,
        "CSES temporal window": args.cses_temporal_window_seconds,
        "VLM timeout": args.vlm_timeout_seconds,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError("values must be positive: " + ", ".join(invalid))
    if args.num_workers < 0 or args.native_retries < 0:
        raise ValueError("workers and retry counts must be non-negative")
    if not 0.0 <= args.dedup_similarity_threshold <= 1.0:
        raise ValueError("dedup similarity threshold must be within [0, 1]")
    if args.search_depth < 100:
        raise ValueError("search depth must be at least 100")
    counts = (
        args.coarse_top_n,
        args.dense_global_top_k,
        args.dense_rescue_clips,
        args.max_candidate_clips,
        args.dense_frames_per_clip,
        args.rrf_k,
        args.vlm_top_m,
        args.bge_batch_size,
    )
    if any(value < 0 for value in counts):
        raise ValueError("retrieval cardinalities must be non-negative")
    if args.bge_batch_size < 1:
        raise ValueError("BGE batch size must be positive")
    if not 0.0 <= float(args.bge_reranker_alpha) <= 1.0:
        raise ValueError("BGE reranker alpha must be within [0, 1]")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _path_reference(path: Path, run_root: Path) -> dict[str, str]:
    """Describe managed paths relatively and explicit external dependencies safely."""

    try:
        relative = Path(os.path.relpath(path.resolve(), run_root.resolve())).as_posix()
    except ValueError:
        return {"scope": "external", "path": path.resolve().as_posix()}
    return {"scope": "run_relative", "path": relative}


def _collect_offline_lineage(run_root: Path) -> dict[str, Any]:
    manifest_paths = sorted(
        (run_root / "metadata").glob("keyframes_*_phase3_manifest.json")
    )
    if not manifest_paths:
        raise ValueError("No Phase-3 manifests were published by the keyframe stage")
    videos: list[dict[str, Any]] = []
    candidate_total = 0
    selected_total = 0
    degraded = 0
    model_revisions: set[str] = set()
    for path in manifest_paths:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "passed":
            raise ValueError(f"Phase-3 manifest is not passed: {path}")
        candidate_total += int(manifest.get("candidate_count", 0))
        selected_total += int(manifest.get("selected_count", 0))
        degraded += int(bool(manifest.get("degraded")))
        feature_config = manifest.get("feature_config")
        if isinstance(feature_config, Mapping):
            siglip = feature_config.get("siglip2")
            if isinstance(siglip, Mapping) and siglip.get("resolved_model_revision"):
                model_revisions.add(str(siglip["resolved_model_revision"]))
        videos.append(
            {
                "video_id": manifest.get("video_id"),
                "phase3_manifest": path.relative_to(run_root).as_posix(),
                "phase3_manifest_sha256": sha256_file(path),
                "candidate_pool_run_id": manifest.get("candidate_pool_run_id"),
                "feature_manifest_sha256": manifest.get("feature_manifest_sha256"),
                "selection_run_id": manifest.get("selection_run_id"),
                "candidate_count": int(manifest.get("candidate_count", 0)),
                "selected_count": int(manifest.get("selected_count", 0)),
            }
        )
    lineage = {
        "version": 1,
        "video_count": len(videos),
        "candidate_count": candidate_total,
        "selected_count": selected_total,
        "degraded_video_count": degraded,
        "siglip2_resolved_revisions": sorted(model_revisions),
        "videos": videos,
    }
    lineage_path = run_root / "lineage" / "offline_lineage.json"
    _atomic_json(lineage_path, lineage)
    update_run_manifest(
        run_root,
        offline={
            "candidate_count": candidate_total,
            "selected_count": selected_total,
            "degraded_video_count": degraded,
            "siglip2_resolved_revisions": sorted(model_revisions),
        },
        artifacts={
            "offline_lineage": {
                "path": lineage_path.relative_to(run_root).as_posix(),
                "sha256": sha256_file(lineage_path),
            }
        },
    )
    return lineage


def _all_stages_passed(run_root: Path) -> bool:
    manifest = read_run_manifest(run_root)
    states = manifest.get("stages")
    return isinstance(states, Mapping) and all(
        isinstance(states.get(stage), Mapping)
        and states[stage].get("status") == "passed"
        for stage in STAGES
    )


def _ensure_run_lineage(
    manifest: Mapping[str, Any],
    *,
    public_root: Path,
    offline_config: Mapping[str, Any],
) -> None:
    stored_git = manifest.get("git")
    current_git = git_fingerprint(REPO_ROOT)
    if not isinstance(stored_git, Mapping) or any(
        stored_git.get(key) != current_git.get(key)
        for key in ("sha", "dirty_diff_sha256")
    ):
        raise ValueError(
            "run_root was created by different source code; use a new run_root "
            "instead of mixing checkpoints"
        )
    stored_dataset = manifest.get("dataset")
    current_dataset = dataset_fingerprint(public_root)
    if not isinstance(stored_dataset, Mapping) or (
        stored_dataset.get("sha256") != current_dataset.get("sha256")
    ):
        raise ValueError(
            "run_root dataset fingerprint differs from --public-root; use a new run_root"
        )
    stored_offline = manifest.get("offline")
    if not isinstance(stored_offline, Mapping) or any(
        stored_offline.get(key) != value for key, value in offline_config.items()
    ):
        raise ValueError(
            "run_root offline configuration differs from this command; use a new run_root"
        )


def run(args: argparse.Namespace) -> Path:
    _validate_args(args)
    selected = selected_stage_names(args.start_at, args.stop_after)
    commands = build_stage_commands(args)
    public_root, run_root, submission_path, cache_root = _resolved_paths(args)
    if not args.dry_run:
        existing_manifest = (run_root / "run_manifest.json").is_file()
        offline_config = {
            "candidate_interval_sec": args.candidate_interval_sec,
            "max_gap_seconds": args.max_gap_seconds,
            "target_density_per_second": args.target_density_per_second,
            "dedup_similarity_threshold": args.dedup_similarity_threshold,
            "endpoint_protection": args.endpoint_protection == "on",
        }
        manifest = initialize_run_manifest(
            run_root=run_root,
            repo_root=REPO_ROOT,
            public_root=public_root,
            run_id=run_root.name,
            offline_config=offline_config,
        )
        if existing_manifest:
            _ensure_run_lineage(
                manifest,
                public_root=public_root,
                offline_config=offline_config,
            )
        update_run_manifest(
            run_root,
            orchestration={
                "runner": "competition.run_retrieval_v2",
                "architecture_stages": list(STAGES),
                "ground_truth_evaluation": "excluded",
                "model_cache": _path_reference(cache_root, run_root),
            },
            status="running",
        )

    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("HF_HOME", str(cache_root / "huggingface"))
    environment.setdefault("TORCH_HOME", str(cache_root / "torch"))
    environment.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
    if args.offline_model_cache:
        environment.setdefault("HF_HUB_OFFLINE", "1")
        environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    started = time.perf_counter()
    stage_durations: dict[str, float] = {}

    if not args.dry_run and {"keyframes", "predict"}.intersection(selected):
        preflight = runtime_preflight(
            device=args.device,
            require_ffmpeg="keyframes" in selected,
        )
        update_run_manifest(run_root, runtime=preflight)
        print("Runtime preflight:\n" + json.dumps(preflight, indent=2), flush=True)

    try:
        for position, stage in enumerate(selected, start=1):
            command = commands[stage]
            print(f"\n[{position}/{len(selected)}] {stage}", flush=True)
            if (
                stage == "bge-text-index"
                and args.bge_dense_mode == "off"
                and not args.dry_run
            ):
                print("BGE-M3 dense retrieval disabled; stage recorded as passed/skipped.")
                stage_durations[stage] = 0.0
                update_run_manifest(
                    run_root,
                    stages={
                        stage: {
                            "status": "passed",
                            "elapsed_seconds": 0.0,
                            "outcome": "disabled",
                            "command": command,
                        }
                    },
                )
                continue
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
                    if stage == "keyframes":
                        _collect_offline_lineage(run_root)
            except BaseException as exc:
                elapsed = time.perf_counter() - stage_started
                if (
                    stage == "bge-text-index"
                    and args.bge_dense_mode == "optional"
                    and not args.dry_run
                ):
                    stage_durations[stage] = elapsed
                    update_run_manifest(
                        run_root,
                        stages={
                            stage: {
                                "status": "passed",
                                "elapsed_seconds": round(elapsed, 3),
                                "outcome": "fallback",
                                "command": command,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        },
                    )
                    print(f"Optional BGE-M3 build failed; continuing: {exc}")
                    continue
                if not args.dry_run:
                    update_run_manifest(
                        run_root,
                        stages={
                            stage: {
                                "status": "failed",
                                "elapsed_seconds": round(elapsed, 3),
                                "command": command,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        },
                        status="failed",
                    )
                raise
            elapsed = time.perf_counter() - stage_started
            stage_durations[stage] = elapsed
            if not args.dry_run:
                update_run_manifest(
                    run_root,
                    stages={
                        stage: {
                            "status": "passed",
                            "elapsed_seconds": round(elapsed, 3),
                            "command": command,
                        }
                    },
                )

        if args.dry_run:
            print(json.dumps({"status": "dry_run", "stages": list(selected)}, indent=2))
            return submission_path

        completed = _all_stages_passed(run_root)
        update_run_manifest(
            run_root,
            status="architecture_complete" if completed else "partial",
            total_elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        if completed and not args.no_experiment_log:
            record = build_runner_record(
                status="completed",
                public_root=public_root,
                output_root=run_root,
                submission_path=submission_path,
                stages=STAGES,
                stage_durations_seconds={
                    stage: float(read_run_manifest(run_root)["stages"][stage]["elapsed_seconds"])
                    for stage in STAGES
                },
                duration_seconds=float(
                    read_run_manifest(run_root).get("total_elapsed_seconds", 0.0)
                ),
                device=args.device,
                batch_size=str(args.batch_size),
                candidate_interval_sec=args.candidate_interval_sec,
                max_gap_seconds=args.max_gap_seconds,
                public_score=None,
                private_score=None,
                note=args.experiment_note
                or "Retrieval v2 architecture E2E; no ground-truth metrics supplied.",
                error=None,
            )
            append_experiment(args.experiment_report.resolve(), record)
            update_run_manifest(
                run_root,
                experiment={
                    "report": _path_reference(
                        args.experiment_report.resolve(), run_root
                    ),
                    "experiment_id": record["experiment_id"],
                    "recorded_after_full_validation": True,
                },
            )
            print(f"Experiment appended: {args.experiment_report.resolve()}", flush=True)

        print(
            json.dumps(
                {
                    "status": "architecture_complete" if completed else "partial",
                    "run_root": run_root.as_posix(),
                    "submission_path": submission_path.as_posix(),
                    "experiment_logged": completed and not args.no_experiment_log,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return submission_path
    except KeyboardInterrupt:
        if not args.dry_run:
            update_run_manifest(run_root, status="interrupted")
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"retrieval v2 pipeline failed: {exc}\n")


if __name__ == "__main__":
    main()
