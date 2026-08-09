"""Versioned, atomic experiment manifests for terminal-only competition runs."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


RUN_MANIFEST_VERSION = 1
RUN_MANIFEST_NAME = "run_manifest.json"
ACTIVE_RUN_NAME = "active_run.json"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def read_run_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root.resolve() / RUN_MANIFEST_NAME
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or int(value.get("version", -1)) != RUN_MANIFEST_VERSION:
        raise ValueError(f"Invalid run manifest: {path}")
    return value


def _git_output(repo_root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return result.stdout


def git_fingerprint(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    sha = str(_git_output(repo_root, "rev-parse", "HEAD")).strip()
    branch = str(_git_output(repo_root, "branch", "--show-current")).strip()
    status = str(_git_output(repo_root, "status", "--porcelain=v1", "--untracked-files=all"))
    diff = _git_output(repo_root, "diff", "--binary", "--no-ext-diff", "HEAD", binary=True)
    assert isinstance(diff, bytes)
    untracked_digest = hashlib.sha256()
    for line in sorted(status.splitlines()):
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        path = repo_root / relative
        untracked_digest.update(relative.encode("utf-8", errors="surrogateescape"))
        if path.is_file():
            untracked_digest.update(sha256_file(path).encode("ascii"))
    combined = hashlib.sha256(diff + untracked_digest.digest()).hexdigest()
    return {
        "sha": sha,
        "branch": branch,
        "dirty": bool(status.strip()),
        "dirty_diff_sha256": combined,
        "status": status.splitlines(),
    }


def dataset_fingerprint(public_root: Path) -> dict[str, Any]:
    public_root = public_root.resolve()
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in sorted(value for value in public_root.rglob("*") if value.is_file()):
        relative = path.relative_to(public_root).as_posix()
        stat = path.stat()
        entry: dict[str, Any] = {"path": relative, "size": stat.st_size}
        # Content hashing is intentionally used for videos/images too.  A run
        # must resume after the public dataset is copied to another machine or
        # extracted again in Colab; filesystem mtimes are not portable lineage.
        entry["sha256"] = sha256_file(path)
        encoded = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(encoded)
        entries.append(entry)
    return {
        "root_name": public_root.name,
        "sha256": digest.hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
    }


def initialize_run_manifest(
    *,
    run_root: Path,
    repo_root: Path,
    public_root: Path,
    run_id: str | None = None,
    baseline: Mapping[str, Any] | None = None,
    offline_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    path = run_root / RUN_MANIFEST_NAME
    if path.exists():
        return read_run_manifest(run_root)
    payload: dict[str, Any] = {
        "version": RUN_MANIFEST_VERSION,
        "run_id": run_id or run_root.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_fingerprint(repo_root),
        "dataset": dataset_fingerprint(public_root),
        "offline": dict(offline_config or {}),
        "retrieval": {},
        "artifacts": {},
        "stages": {},
        "submission": None,
        "leaderboard": [],
        "baseline": dict(baseline or {}),
        "status": "initialized",
    }
    _atomic_json(path, payload)
    return payload


def update_run_manifest(run_root: Path, **changes: Any) -> dict[str, Any]:
    run_root = run_root.resolve()
    payload = read_run_manifest(run_root)
    for key, value in changes.items():
        if isinstance(value, Mapping) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **dict(value)}
        else:
            payload[key] = value
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(run_root / RUN_MANIFEST_NAME, payload)
    return payload


def record_submission(
    run_root: Path,
    submission_path: Path,
    *,
    query_count: int,
    answers_per_query: int,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    submission_path = submission_path.resolve()
    relative = Path(os.path.relpath(submission_path, run_root)).as_posix()
    return update_run_manifest(
        run_root,
        submission={
            "path": relative,
            "sha256": sha256_file(submission_path),
            "query_count": int(query_count),
            "answers_per_query": int(answers_per_query),
        },
    )


def record_leaderboard_score(
    run_root: Path,
    *,
    score: float,
    split: str,
    source: str = "user_reported",
) -> dict[str, Any]:
    payload = read_run_manifest(run_root)
    submission = payload.get("submission")
    if not isinstance(submission, Mapping) or not submission.get("sha256"):
        raise ValueError("Record a validated submission before a leaderboard score")
    leaderboard = list(payload.get("leaderboard") or [])
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "split": str(split),
        "score": float(score),
        "source": str(source),
        "submission_sha256": str(submission["sha256"]),
    }
    identity = {key: value for key, value in record.items() if key != "recorded_at"}
    if not any(
        {key: value for key, value in item.items() if key != "recorded_at"} == identity
        for item in leaderboard
        if isinstance(item, Mapping)
    ):
        leaderboard.append(record)
    return update_run_manifest(run_root, leaderboard=leaderboard)


def set_active_baseline(*, run_root: Path, runs_root: Path) -> dict[str, Any]:
    """Initialize active_run.json once; later changes must use the score gate."""
    run_root = run_root.resolve()
    runs_root = runs_root.resolve()
    active_path = runs_root / ACTIVE_RUN_NAME
    if active_path.exists():
        with active_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    payload = read_run_manifest(run_root)
    submission = payload.get("submission")
    scores = payload.get("leaderboard") or []
    if not isinstance(submission, Mapping) or not scores:
        raise ValueError("Baseline activation requires submission and recorded score")
    best_score = max(float(item["score"]) for item in scores)
    active = {
        "version": 1,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "run_id": payload["run_id"],
        "run_root": Path(os.path.relpath(run_root, runs_root)).as_posix(),
        "submission_sha256": submission["sha256"],
        "leaderboard_score": best_score,
        "role": "immutable_baseline",
    }
    _atomic_json(active_path, active)
    update_run_manifest(run_root, status="baseline_active")
    return active


def promote_active_run(
    *,
    run_root: Path,
    runs_root: Path,
    minimum_score: float = 0.818,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    runs_root = runs_root.resolve()
    payload = read_run_manifest(run_root)
    submission = payload.get("submission")
    if not isinstance(submission, Mapping):
        raise ValueError("Cannot promote a run without a validated submission")
    scores = [
        float(item["score"])
        for item in payload.get("leaderboard", [])
        if isinstance(item, Mapping)
        and str(item.get("submission_sha256")) == str(submission.get("sha256"))
    ]
    best_score = max(scores, default=float("-inf"))
    if best_score <= float(minimum_score):
        raise ValueError(
            f"Promotion requires score > {minimum_score:.6f}; best recorded={best_score:.6f}"
        )
    relative = Path(os.path.relpath(run_root, runs_root)).as_posix()
    active = {
        "version": 1,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "run_id": payload["run_id"],
        "run_root": relative,
        "submission_sha256": submission["sha256"],
        "leaderboard_score": best_score,
    }
    _atomic_json(runs_root / ACTIVE_RUN_NAME, active)
    update_run_manifest(run_root, status="promoted")
    return active
