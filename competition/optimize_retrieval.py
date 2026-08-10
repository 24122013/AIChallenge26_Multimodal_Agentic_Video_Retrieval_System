"""Sequential retrieval ablation runner for terminal-only leaderboard work."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from competition.keyframe_phase3 import atomic_write_json, atomic_write_jsonl
from competition.keyframe_phase5 import load_split_manifest, write_split_manifest
from competition.retrieval_metrics import evaluate_submission


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.load(handle, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict) or int(value.get("version", -1)) != 1:
        raise ValueError("retrieval experiment config must be a version-1 mapping")
    experiments = value.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("retrieval experiment config needs a non-empty experiments list")
    ids: set[str] = set()
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("each experiment must be a mapping")
        experiment_id = str(experiment.get("id") or "")
        if not experiment_id or experiment_id in ids:
            raise ValueError(f"missing/duplicate experiment id: {experiment_id!r}")
        ids.add(experiment_id)
        if experiment.get("retrieval_mode") not in {"legacy", "advanced"}:
            raise ValueError(f"invalid retrieval_mode for {experiment_id}")
    return value


def initialize_labeling_set(
    *,
    output_root: Path,
    evaluation_root: Path,
    seed: int = 42,
) -> dict[str, Any]:
    video_ids, features = _video_features(output_root)
    if len(video_ids) < 16:
        raise ValueError(f"Need at least 16 processed videos, got {len(video_ids)}")
    selected = _farthest_point_sample(video_ids, features, count=16, seed=seed)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    split_path = evaluation_root / "split_manifest.json"
    split = write_split_manifest(split_path, selected, seed=seed)
    evidence_cycle = ("ocr", "asr", "objects", "transition")
    manual_events: list[dict[str, Any]] = []
    query_templates: list[dict[str, Any]] = []
    for video_position, video_id in enumerate(selected):
        for event_index in range(5):
            manual_events.append(
                {
                    "video_id": video_id,
                    "event_id": f"{video_id}_event_{event_index + 1:02d}",
                    "start": None,
                    "end": None,
                    "evidence_type": evidence_cycle[(video_position + event_index) % 4],
                    "description": "",
                    "review_status": "needs_human_label",
                }
            )
        query_templates.extend(
            [
                {
                    "query_id": f"LABEL_{video_id}_TKIS_SINGLE",
                    "video_id": video_id,
                    "task": "TKIS",
                    "query_kind": "single_event",
                    "query": "",
                    "relevant": [],
                    "review_status": "needs_human_label",
                },
                {
                    "query_id": f"LABEL_{video_id}_TKIS_TEMPORAL",
                    "video_id": video_id,
                    "task": "TKIS",
                    "query_kind": "temporal",
                    "query": "",
                    "relevant": [],
                    "temporal_chain": [],
                    "review_status": "needs_human_label",
                },
                {
                    "query_id": f"LABEL_{video_id}_VKIS",
                    "video_id": video_id,
                    "task": "VKIS",
                    "query_kind": "exact_frame",
                    "query_image": "",
                    "relevant": [
                        {"video": f"{video_id}.mp4", "frame": None, "tolerance": 12}
                    ],
                    "review_status": "needs_human_label",
                },
            ]
        )
    atomic_write_jsonl(evaluation_root / "manual_events_template.jsonl", manual_events)
    atomic_write_jsonl(evaluation_root / "retrieval_labels_template.jsonl", query_templates)
    report = {
        "status": "labeling_required",
        "seed": seed,
        "selection_method": "normalized_feature_farthest_point_sampling",
        "selected_video_ids": selected,
        "split": split,
        "manual_event_template_count": len(manual_events),
        "query_template_count": len(query_templates),
        "instructions": (
            "Human reviewers must fill five intervals and three queries per video, "
            "then publish retrieval_labels.jsonl before optimization."
        ),
    }
    atomic_write_json(evaluation_root / "labeling_manifest.json", report)
    return report


def run_optimization(args: argparse.Namespace) -> dict[str, Any]:
    config = load_experiment_config(args.experiment_config)
    labels_path = args.labels or Path(str(config.get("labels")))
    if not labels_path.is_file() and not args.dry_run:
        raise FileNotFoundError(
            f"Ground-truth labels not found: {labels_path}. Run --init-labeling, "
            "complete human review, and publish retrieval_labels.jsonl."
        )
    base_run_root = Path(str(config["base_run_root"]))
    evaluation_root = args.evaluation_root.resolve()
    evaluation_root.mkdir(parents=True, exist_ok=True)
    split_manifest = load_split_manifest(evaluation_root / "split_manifest.json")
    allowed_video_ids = set(split_manifest["splits"][args.split])
    if args.split == "test":
        marker = evaluation_root / "retrieval_locked_test_marker.json"
        if marker.exists():
            raise RuntimeError("Locked retrieval test has already been opened")
        if not args.confirm_locked_test:
            raise ValueError("--confirm-locked-test is required for the one-shot test split")

    reports: list[dict[str, Any]] = []
    for experiment in config["experiments"]:
        experiment_id = str(experiment["id"])
        run_root = Path("competition/runs") / f"retrieval-v2-{args.split}-{experiment_id}"
        submission = (
            Path(str(experiment["submission"]))
            if experiment.get("submission")
            else run_root / "results" / "submission.csv"
        )
        command = _experiment_command(
            experiment,
            defaults=config,
            python=args.python,
            public_root=args.public_root,
            output_root=args.output_root,
            run_root=run_root,
            dense_run_root=base_run_root,
            submission=submission,
        )
        if args.dry_run:
            reports.append({"id": experiment_id, "command": command})
            continue
        if experiment["retrieval_mode"] == "advanced":
            subprocess.run(command, check=True)
        if not submission.is_file():
            raise FileNotFoundError(f"Experiment submission missing: {submission}")
        metrics = evaluate_submission(
            submission_path=submission,
            labels_path=labels_path,
            allowed_video_ids=allowed_video_ids,
            trace_path=(
                run_root / "results" / "query_traces.jsonl"
                if (run_root / "results" / "query_traces.jsonl").is_file()
                else None
            ),
        )
        reports.append(
            {
                "id": experiment_id,
                "run_root": run_root.as_posix(),
                "submission": submission.as_posix(),
                "command": command,
                "metrics": metrics,
            }
        )

    result: dict[str, Any] = {
        "version": 1,
        "split": args.split,
        "dry_run": args.dry_run,
        "labels": labels_path.as_posix(),
        "baseline_score": float(config.get("baseline_score", 0.818)),
        "experiments": reports,
        "offline_grid": config.get("offline_grid", {}),
    }
    if not args.dry_run:
        result["promotion"] = _promotion_summary(reports, config)
        report_path = evaluation_root / f"retrieval_v2_{args.split}_report.json"
        atomic_write_json(report_path, result)
        result["report_path"] = report_path.as_posix()
        if args.split == "test":
            atomic_write_json(
                evaluation_root / "retrieval_locked_test_marker.json",
                {
                    "split": "test",
                    "report_path": report_path.as_posix(),
                    "config": args.experiment_config.as_posix(),
                },
            )
    return result


def _experiment_command(
    experiment: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
    python: Path,
    public_root: Path,
    output_root: Path,
    run_root: Path,
    dense_run_root: Path,
    submission: Path,
) -> list[str]:
    if experiment["retrieval_mode"] == "legacy":
        return []
    command = [
        str(python),
        "-m",
        "competition.pipeline",
        "predict",
        "--public-root",
        str(public_root),
        "--output-root",
        str(output_root),
        "--run-root",
        str(run_root),
        "--dense-run-root",
        str(dense_run_root),
        "--submission-path",
        str(submission),
        "--retrieval-mode",
        "advanced",
        "--offline-model-cache",
        "--vlm-mode",
        str(experiment.get("vlm_mode", "off")),
    ]
    defaults = defaults or {}
    retrieval_defaults = defaults.get("retrieval") or {}
    rrf_defaults = defaults.get("rrf") or {}
    dense_defaults = defaults.get("dense_recovery") or {}
    rerank_defaults = defaults.get("rerank") or {}
    flag_map = {
        "no_query_plan": "--no-query-plan",
        "no_rrf": "--no-rrf",
        "no_dense_rescue": "--no-dense-rescue",
        "no_cses": "--no-cses",
        "no_deterministic_rerank": "--no-deterministic-rerank",
    }
    flags = experiment.get("flags") or {}
    if not isinstance(flags, Mapping):
        raise ValueError(f"experiment {experiment['id']} flags must be a mapping")
    for name, cli_flag in flag_map.items():
        if flags.get(name) is True:
            command.append(cli_flag)
    for name, cli_flag in {
        "coarse_top_n": "--coarse-top-n",
        "dense_global_top_k": "--dense-global-top-k",
        "dense_frames_per_clip": "--dense-frames-per-clip",
        "rrf_k": "--rrf-k",
    }.items():
        if experiment.get(name) is not None:
            command.extend([cli_flag, str(experiment[name])])
    nested_options = {
        "--visual-top-k": (retrieval_defaults.get("visual") or {}).get("top_k"),
        "--caption-top-k": (retrieval_defaults.get("caption") or {}).get("top_k"),
        "--ocr-top-k": (retrieval_defaults.get("ocr") or {}).get("top_k"),
        "--object-top-k": (retrieval_defaults.get("objects") or {}).get("top_k"),
        "--asr-top-k": (retrieval_defaults.get("asr") or {}).get("top_k"),
        "--coarse-top-n": rrf_defaults.get("candidate_top_n"),
        "--rrf-k": rrf_defaults.get("k"),
        "--dense-frames-per-clip": dense_defaults.get("frames_per_segment"),
        "--dense-expansion-before-sec": dense_defaults.get("expansion_before_sec"),
        "--dense-expansion-after-sec": dense_defaults.get("expansion_after_sec"),
        "--rerank-top-n": rerank_defaults.get("top_n"),
    }
    for cli_flag, value in nested_options.items():
        if value is not None and cli_flag not in command:
            command.extend([cli_flag, str(value)])
    fusion_mode = experiment.get("fusion_mode")
    if fusion_mode is None:
        fusion_mode = (
            "adaptive_rrf"
            if bool(rrf_defaults.get("query_adaptive_weights", True))
            else "weighted_rrf"
        )
    command.extend(["--fusion-mode", str(fusion_mode)])
    if dense_defaults.get("enabled") is False and "--no-dense-rescue" not in command:
        command.append("--no-dense-rescue")
    if rerank_defaults.get("final_top_k") is not None:
        command.extend(["--final-top-k", str(rerank_defaults["final_top_k"])])
    if experiment.get("retrieval_modalities") is not None:
        command.extend(
            ["--retrieval-modalities", str(experiment["retrieval_modalities"])]
        )
    return command


def _promotion_summary(
    reports: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    measured = [report for report in reports if report.get("metrics")]
    if not measured:
        return {"status": "no_measured_experiments"}
    baseline = measured[0]
    baseline_metrics = baseline["metrics"]
    baseline_macro = baseline_metrics["macro"]
    gates = config.get("gates") or {}
    max_recall_drop = float(gates.get("max_recall100_drop_by_task", 0.01))
    minimum_delta = float(gates.get("min_ndcg20_delta", 0.0))
    deterministic = next(
        (
            report
            for report in measured
            if report.get("id") == "rrf_dense_final_reranker"
        ),
        None,
    )
    decisions: list[dict[str, Any]] = []
    eligible: list[Mapping[str, Any]] = []
    for report in measured[1:]:
        metrics = report["metrics"]
        macro = metrics["macro"]
        ndcg_delta = float(macro.get("nDCG@20", 0.0)) - float(
            baseline_macro.get("nDCG@20", 0.0)
        )
        recall_guardrails: dict[str, dict[str, float | bool]] = {}
        recall_passed = True
        for task in ("TKIS", "VKIS"):
            baseline_task = baseline_metrics.get("by_task", {}).get(task, {})
            candidate_task = metrics.get("by_task", {}).get(task, {})
            baseline_recall = float(baseline_task.get("Recall@100", 0.0))
            candidate_recall = float(candidate_task.get("Recall@100", 0.0))
            drop = baseline_recall - candidate_recall
            passed = drop <= max_recall_drop
            recall_guardrails[task] = {
                "baseline": baseline_recall,
                "candidate": candidate_recall,
                "drop": drop,
                "passed": passed,
            }
            recall_passed = recall_passed and passed
        temporal_passed = float(macro.get("temporal_Hit@20", 0.0)) >= float(
            baseline_macro.get("temporal_Hit@20", 0.0)
        )
        vkis_passed = float(macro.get("VKIS_Hit@100", 0.0)) >= float(
            baseline_macro.get("VKIS_Hit@100", 0.0)
        )
        vlm_passed = True
        vlm_delta: float | None = None
        if report.get("id") == "optional_vlm" and deterministic is not None:
            deterministic_ndcg = float(
                deterministic["metrics"]["macro"].get("nDCG@20", 0.0)
            )
            vlm_delta = float(macro.get("nDCG@20", 0.0)) - deterministic_ndcg
            vlm_passed = vlm_delta >= float(gates.get("min_vlm_ndcg20_delta", 0.01))
        passed = (
            ndcg_delta > minimum_delta
            and recall_passed
            and temporal_passed
            and vkis_passed
            and vlm_passed
        )
        decision = {
            "id": report["id"],
            "passed": passed,
            "ndcg20_delta": ndcg_delta,
            "recall100_by_task": recall_guardrails,
            "temporal_hit20_not_lower": temporal_passed,
            "vkis_hit100_not_lower": vkis_passed,
            "vlm_ndcg20_delta_vs_deterministic": vlm_delta,
            "vlm_gate_passed": vlm_passed,
        }
        decisions.append(decision)
        if passed:
            eligible.append(report)
    best = max(
        eligible,
        key=lambda report: float(report["metrics"]["macro"].get("nDCG@20", 0.0)),
        default=None,
    )
    return {
        "status": "eligible_for_validation" if best is not None else "rejected",
        "baseline_id": baseline["id"],
        "best_id": best["id"] if best is not None else None,
        "decisions": decisions,
    }


def _video_features(output_root: Path) -> tuple[list[str], np.ndarray]:
    manifests = sorted((output_root / "metadata").glob("keyframes_*_phase3_manifest.json"))
    video_ids: list[str] = []
    features: list[list[float]] = []
    for path in manifests:
        phase3 = json.loads(path.read_text(encoding="utf-8"))
        video_id = str(phase3.get("video_id") or "")
        report_path = output_root / "metadata" / f"keyframes_{video_id}_extract_report.json"
        if not video_id or not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        adapter = phase3.get("feature_adapter_report") or {}
        counts = adapter.get("modality_available_counts") or {}
        candidate_count = max(1, int(phase3.get("candidate_count", 1)))
        duration = max(1e-6, float(report.get("duration", 0.0)))
        features.append(
            [
                duration,
                float(report.get("shot_count", 0)),
                float(counts.get("ocr", 0)) / candidate_count,
                float(counts.get("asr", 0)) / candidate_count,
                float(counts.get("objects", 0)) / candidate_count,
                float(adapter.get("transition_boundary_count", 0)) / candidate_count,
            ]
        )
        video_ids.append(video_id)
    matrix = np.asarray(features, dtype=np.float64)
    if not len(matrix):
        return video_ids, matrix
    minimum = matrix.min(axis=0)
    span = matrix.max(axis=0) - minimum
    span[span == 0] = 1.0
    return video_ids, (matrix - minimum) / span


def _farthest_point_sample(
    video_ids: Sequence[str],
    features: np.ndarray,
    *,
    count: int,
    seed: int,
) -> list[str]:
    if len(video_ids) != len(features) or count > len(video_ids):
        raise ValueError("Invalid farthest-point sampling inputs")
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(0, len(video_ids)))]
    minimum_distance = np.linalg.norm(features - features[chosen[0]], axis=1)
    while len(chosen) < count:
        candidates = [index for index in range(len(video_ids)) if index not in chosen]
        next_index = min(
            candidates,
            key=lambda index: (-minimum_distance[index], video_ids[index]),
        )
        chosen.append(next_index)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(features - features[next_index], axis=1),
        )
    return [video_ids[index] for index in chosen]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "validation", "test"), default="dev")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/retrieval_v2.yaml"),
    )
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--public-root", type=Path, default=Path("data/public"))
    parser.add_argument("--output-root", type=Path, default=Path("competition"))
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("competition/evaluation"),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument("--init-labeling", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.init_labeling:
        result = initialize_labeling_set(
            output_root=args.output_root,
            evaluation_root=args.evaluation_root,
            seed=args.seed,
        )
    else:
        result = run_optimization(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
