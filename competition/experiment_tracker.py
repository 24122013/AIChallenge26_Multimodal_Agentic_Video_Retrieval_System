"""Append reproducible competition run snapshots to ``reports/Experiment.md``."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "Experiment.md"
SUMMARY_START = "<!-- scoreboard-summary:start -->"
SUMMARY_END = "<!-- scoreboard-summary:end -->"
LOG_MARKER = "<!-- experiment-log:append-below -->"
ENCODED_RECORD_PATTERN = re.compile(
    r"<!-- experiment-json-base64:([A-Za-z0-9_=-]+) -->"
)
PLAIN_RECORD_PATTERN = re.compile(r"<!-- experiment-json:(.*?) -->", re.DOTALL)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _jsonl_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except (OSError, UnicodeError):
        return 0


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return []


def _git_snapshot() -> dict[str, object]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    branch = run("branch", "--show-current")
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "branch": branch,
        "commit": commit,
        "dirty": bool(status),
        "changed_path_count": len(status.splitlines()) if status else 0,
    }


def collect_local_metrics(
    *,
    public_root: Path,
    output_root: Path,
    submission_path: Path,
) -> dict[str, object]:
    """Collect cheap artifact-backed metrics without loading ML dependencies."""

    corpus = _csv_rows(public_root / "corpus.csv")
    questions = _csv_rows(public_root / "questions.csv")
    tasks: dict[str, int] = {}
    for row in questions:
        task = str(row.get("task") or "unknown")
        tasks[task] = tasks.get(task, 0) + 1

    workspace = output_root / "work" / "keyframe_v3"
    candidate_files = sorted(workspace.glob("*/*/candidates.jsonl"))
    candidate_count = sum(_jsonl_count(path) for path in candidate_files)
    stage_names = {
        "siglip2_video_count": "siglip2.npy",
        "caption_video_count": "captions.jsonl",
        "ocr_video_count": "ocr.jsonl",
        "object_video_count": "objects.jsonl",
        "asr_video_count": "asr.jsonl",
        "feature_manifest_count": "feature_manifest.json",
    }
    workspace_metrics: dict[str, object] = {
        "candidate_pool_video_count": len(candidate_files),
        "candidate_count": candidate_count,
    }
    for key, filename in stage_names.items():
        workspace_metrics[key] = sum(
            1 for path in workspace.glob(f"*/*/{filename}") if path.is_file()
        )

    extract_reports = []
    for path in sorted(
        (output_root / "metadata").glob("keyframes_*_extract_report.json")
    ):
        value = _read_json(path)
        if value is not None:
            extract_reports.append(value)
    selected_count = sum(
        int(report.get("keyframe_count") or 0) for report in extract_reports
    )
    canonical_candidate_count = sum(
        int(report.get("candidate_count") or 0) for report in extract_reports
    )
    guarantees = [
        report.get("guarantees")
        for report in extract_reports
        if isinstance(report.get("guarantees"), Mapping)
    ]
    constraint_passes = sum(
        guarantee.get("constraints_satisfied") is True for guarantee in guarantees
    )
    event_passes = sum(
        guarantee.get("event_recall_satisfied") is True for guarantee in guarantees
    )
    observed_gaps = [
        float(guarantee["observed_max_gap_seconds"])
        for guarantee in guarantees
        if isinstance(guarantee.get("observed_max_gap_seconds"), (int, float))
    ]
    canonical = {
        "published_video_count": len(extract_reports),
        "candidate_count": canonical_candidate_count,
        "selected_keyframe_count": selected_count,
        "selection_ratio": (
            selected_count / canonical_candidate_count
            if canonical_candidate_count
            else None
        ),
        "constraint_pass_rate": (
            constraint_passes / len(guarantees) if guarantees else None
        ),
        "detected_event_coverage_pass_rate": (
            event_passes / len(guarantees) if guarantees else None
        ),
        "observed_max_gap_seconds": max(observed_gaps) if observed_gaps else None,
    }

    submission: dict[str, object] = {"exists": submission_path.is_file()}
    if submission_path.is_file():
        rows = _csv_rows(submission_path)
        try:
            with submission_path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
        except (OSError, UnicodeError, csv.Error):
            header = []
        submission.update(
            {
                "sha256": _sha256(submission_path),
                "size_bytes": submission_path.stat().st_size,
                "query_count": len(rows),
                "answers_per_query": max(0, len(header) - 1),
            }
        )

    manifest_path = (
        output_root / "metadata" / "siglip2_so400m_patch16_384_faiss_manifest.json"
    )
    index_manifest = _read_json(manifest_path) or {}
    index = {
        key: index_manifest.get(key)
        for key in (
            "index_type",
            "metric",
            "vector_count",
            "runtime_sec",
            "index_file_size_mb",
        )
    }
    index["exists"] = bool(index_manifest)

    phase5: dict[str, object] = {}
    for path in sorted(
        (output_root / "evaluation").glob("keyframe_phase5_*_report.json")
    ):
        report = _read_json(path)
        if not report:
            continue
        aggregate = report.get("aggregate")
        if not isinstance(aggregate, Mapping):
            aggregate = {}
        split = str(report.get("split") or path.stem)
        phase5[split] = {
            "status": report.get("status"),
            "video_count": aggregate.get("video_count"),
            "coverage_violation_count": aggregate.get("coverage_violation_count"),
            "effective_shot_recall": aggregate.get("effective_shot_recall"),
            "detected_protected_event_recall": aggregate.get(
                "detected_protected_event_recall"
            ),
            "manual_end_to_end_event_recall": aggregate.get(
                "manual_end_to_end_event_recall"
            ),
            "false_protection_rate": aggregate.get("false_protection_rate"),
            "retrieval": aggregate.get("retrieval"),
        }

    return {
        "dataset": {
            "video_count": len(corpus),
            "query_count": len(questions),
            "tasks": tasks,
        },
        "workspace": workspace_metrics,
        "canonical": canonical,
        "submission": submission,
        "index": index,
        "phase5": phase5,
    }


def _encode_record(record: Mapping[str, object]) -> str:
    raw = json.dumps(
        dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _load_records(content: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for encoded in ENCODED_RECORD_PATTERN.findall(content):
        try:
            value = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
        except (ValueError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            records.append(value)
    for raw in PLAIN_RECORD_PATTERN.findall(content):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if 0.0 <= result <= 1.0 else None


def _scoreboard_summary(records: Sequence[Mapping[str, object]]) -> str:
    eligible = [
        record
        for record in records
        if record.get("status") in {"completed", "recorded"}
    ]
    public = [
        (_score(record.get("public_score")), str(record.get("experiment_id") or ""))
        for record in eligible
    ]
    private = [
        (_score(record.get("private_score")), str(record.get("experiment_id") or ""))
        for record in eligible
    ]
    public = [(score, experiment) for score, experiment in public if score is not None]
    private = [(score, experiment) for score, experiment in private if score is not None]
    best_public = max(public, default=(None, ""), key=lambda item: item[0])
    best_private = max(private, default=(None, ""), key=lambda item: item[0])

    def value(item: tuple[float | None, str]) -> str:
        return f"{item[0]:.6f}" if item[0] is not None else "N/A"

    return "\n".join(
        [
            SUMMARY_START,
            "| Scoreboard metric | Best score | Experiment |",
            "|---|---:|---|",
            f"| Public | {value(best_public)} | {best_public[1] or 'N/A'} |",
            f"| Private | {value(best_private)} | {best_private[1] or 'N/A'} |",
            SUMMARY_END,
        ]
    )


def _format_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={item}" for key, item in value.items()) or "N/A"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "N/A"
    return str(value)


def _metric_rows(record: Mapping[str, object]) -> list[tuple[str, object]]:
    local = record.get("local_metrics")
    if not isinstance(local, Mapping):
        return []
    dataset = local.get("dataset") if isinstance(local.get("dataset"), Mapping) else {}
    workspace = (
        local.get("workspace") if isinstance(local.get("workspace"), Mapping) else {}
    )
    canonical = (
        local.get("canonical") if isinstance(local.get("canonical"), Mapping) else {}
    )
    submission = (
        local.get("submission") if isinstance(local.get("submission"), Mapping) else {}
    )
    index = local.get("index") if isinstance(local.get("index"), Mapping) else {}
    return [
        ("Dataset videos", dataset.get("video_count")),
        ("Dataset queries", dataset.get("query_count")),
        ("Query tasks", dataset.get("tasks")),
        ("Candidate-pool videos", workspace.get("candidate_pool_video_count")),
        ("Dense candidates", workspace.get("candidate_count")),
        ("SigLIP2 videos", workspace.get("siglip2_video_count")),
        ("Caption videos", workspace.get("caption_video_count")),
        ("OCR videos", workspace.get("ocr_video_count")),
        ("Object videos", workspace.get("object_video_count")),
        ("ASR videos", workspace.get("asr_video_count")),
        ("Complete feature manifests", workspace.get("feature_manifest_count")),
        ("Published videos", canonical.get("published_video_count")),
        ("Selected keyframes", canonical.get("selected_keyframe_count")),
        ("Selection ratio", canonical.get("selection_ratio")),
        ("Hard-constraint pass rate", canonical.get("constraint_pass_rate")),
        ("Detected-event coverage pass rate", canonical.get("detected_event_coverage_pass_rate")),
        ("Observed max temporal gap (s)", canonical.get("observed_max_gap_seconds")),
        ("Submission exists", submission.get("exists")),
        ("Submission SHA256", submission.get("sha256")),
        ("Submission query count", submission.get("query_count")),
        ("Answers per query", submission.get("answers_per_query")),
        ("FAISS vectors", index.get("vector_count")),
        ("FAISS index size (MiB)", index.get("index_file_size_mb")),
    ]


def render_experiment(record: Mapping[str, object]) -> str:
    experiment_id = str(record.get("experiment_id") or "experiment")
    status = str(record.get("status") or "unknown")
    lines = [
        f"## {experiment_id}",
        "",
        f"- Recorded at: `{record.get('recorded_at', 'N/A')}`",
        f"- Source: `{record.get('source', 'runner')}`",
        f"- Status: `{status}`",
        f"- Public score: `{_format_value(record.get('public_score'))}`",
        f"- Private score: `{_format_value(record.get('private_score'))}`",
    ]
    if record.get("note"):
        lines.append(f"- Note: {record['note']}")
    if record.get("error"):
        lines.append(f"- Error: `{record['error']}`")

    run = record.get("run")
    if isinstance(run, Mapping):
        lines.extend(
            [
                "",
                "### Run",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Duration (s) | {_format_value(run.get('duration_seconds'))} |",
                f"| Stages | {_format_value(run.get('stages'))} |",
                f"| Stage durations (s) | {_format_value(run.get('stage_durations_seconds'))} |",
                f"| Device | {_format_value(run.get('device'))} |",
                f"| Candidate interval (s) | {_format_value(run.get('candidate_interval_sec'))} |",
                f"| Max gap (s) | {_format_value(run.get('max_gap_seconds'))} |",
                f"| Batch size | {_format_value(run.get('batch_size'))} |",
            ]
        )

    rows = _metric_rows(record)
    if rows:
        lines.extend(["", "### Artifact-backed metrics", "", "| Metric | Value |", "|---|---:|"])
        lines.extend(f"| {name} | {_format_value(value)} |" for name, value in rows)

    phase5 = None
    local = record.get("local_metrics")
    if isinstance(local, Mapping):
        phase5 = local.get("phase5")
    if isinstance(phase5, Mapping) and phase5:
        lines.extend(["", "### Phase 5", "", "```json", json.dumps(phase5, ensure_ascii=False, indent=2), "```"])

    git = record.get("git")
    if isinstance(git, Mapping):
        lines.extend(
            [
                "",
                "### Reproducibility",
                "",
                f"- Git branch: `{git.get('branch') or 'N/A'}`",
                f"- Git commit: `{git.get('commit') or 'N/A'}`",
                f"- Dirty worktree: `{_format_value(git.get('dirty'))}`",
                f"- Changed paths: `{_format_value(git.get('changed_path_count'))}`",
            ]
        )
    lines.extend(
        [
            "",
            f"<!-- experiment-json-base64:{_encode_record(record)} -->",
            "",
        ]
    )
    return "\n".join(lines)


def _initial_content() -> str:
    return "\n".join(
        [
            "# Competition Experiments",
            "",
            "Lịch sử benchmark append-only cho pipeline TKIS/VKIS. Scoreboard chỉ",
            "được ghi khi có số liệu được cung cấp rõ ràng; metric local được thu từ",
            "artifact và không được xem là ground-truth retrieval quality.",
            "",
            "## Best scoreboard",
            "",
            _scoreboard_summary([]),
            "",
            "## Experiment log",
            "",
            LOG_MARKER,
            "",
        ]
    )


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_experiment(path: Path, record: Mapping[str, object]) -> None:
    content = path.read_text(encoding="utf-8") if path.is_file() else _initial_content()
    if LOG_MARKER not in content:
        raise ValueError(f"experiment log marker is missing from {path}")
    updated = content.rstrip() + "\n\n" + render_experiment(record)
    records = _load_records(updated)
    summary = _scoreboard_summary(records)
    pattern = re.compile(
        re.escape(SUMMARY_START) + r".*?" + re.escape(SUMMARY_END),
        flags=re.DOTALL,
    )
    if not pattern.search(updated):
        raise ValueError(f"scoreboard summary markers are missing from {path}")
    updated = pattern.sub(summary, updated, count=1)
    _write_atomic(path, updated)


def build_runner_record(
    *,
    status: str,
    public_root: Path,
    output_root: Path,
    submission_path: Path,
    stages: Sequence[str],
    stage_durations_seconds: Mapping[str, float],
    duration_seconds: float,
    device: str,
    batch_size: str,
    candidate_interval_sec: float,
    max_gap_seconds: float,
    public_score: float | None,
    private_score: float | None,
    note: str | None,
    error: str | None,
) -> dict[str, object]:
    stamp = datetime.now(timezone.utc)
    return {
        "experiment_id": f"EXP-{stamp.strftime('%Y%m%dT%H%M%S%fZ')}",
        "recorded_at": stamp.isoformat(),
        "source": "end_to_end_runner",
        "status": status,
        "public_score": public_score,
        "private_score": private_score,
        "note": note,
        "error": error,
        "run": {
            "duration_seconds": round(duration_seconds, 3),
            "stages": list(stages),
            "stage_durations_seconds": {
                key: round(value, 3) for key, value in stage_durations_seconds.items()
            },
            "device": device,
            "batch_size": batch_size,
            "candidate_interval_sec": candidate_interval_sec,
            "max_gap_seconds": max_gap_seconds,
        },
        "git": _git_snapshot(),
        "local_metrics": collect_local_metrics(
            public_root=public_root,
            output_root=output_root,
            submission_path=submission_path,
        ),
    }


def _bounded_score(value: str) -> float:
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise argparse.ArgumentTypeError("score must be within [0, 1]")
    return score


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--public-root", type=Path, default=REPO_ROOT / "data" / "public")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "competition")
    parser.add_argument("--submission-path", type=Path, default=None)
    parser.add_argument("--public-score", type=_bounded_score, default=None)
    parser.add_argument("--private-score", type=_bounded_score, default=None)
    parser.add_argument("--note", default="Manual scoreboard update")
    args = parser.parse_args(argv)
    submission_path = args.submission_path or args.output_root / "results" / "submission.csv"
    stamp = datetime.now(timezone.utc)
    record = {
        "experiment_id": f"EXP-{stamp.strftime('%Y%m%dT%H%M%S%fZ')}",
        "recorded_at": stamp.isoformat(),
        "source": "manual_scoreboard_update",
        "status": "recorded",
        "public_score": args.public_score,
        "private_score": args.private_score,
        "note": args.note,
        "git": _git_snapshot(),
        "local_metrics": collect_local_metrics(
            public_root=args.public_root.resolve(),
            output_root=args.output_root.resolve(),
            submission_path=submission_path.resolve(),
        ),
    }
    append_experiment(args.report_path.resolve(), record)
    print(args.report_path.resolve())


if __name__ == "__main__":
    main()
