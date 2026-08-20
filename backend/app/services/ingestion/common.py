from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from PIL import Image


SCHEMA_VERSION = "1.0"
IDENTITY_FIELDS = (
    "candidate_id",
    "candidate_index",
    "frame_id",
    "video_id",
    "segment_id",
    "shot_id",
    "timestamp",
    "frame_index",
    "shot_start",
    "shot_end",
    "candidate_reasons",
    "keyframe_path",
    "source_video_path",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def choose_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("CUDA requires PyTorch, but torch is not installed.")
        return "cpu"
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return "cuda" if available else "cpu"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path} at line {line_number}.")
            records.append(value)
    return records


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def append_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()


def discover_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    files = sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {path}")
    return files


def video_id_from_records(records: Sequence[dict[str, Any]], fallback: str) -> str:
    ids = {str(record.get("video_id")) for record in records if record.get("video_id")}
    if len(ids) > 1:
        raise ValueError(f"One metadata file must contain one video_id, got: {sorted(ids)}")
    return next(iter(ids), fallback.removeprefix("keyframes_"))


def identity(record: dict[str, Any]) -> dict[str, Any]:
    # Keys are copied without coercion so IDs and timestamps retain their source values.
    return {field: record.get(field) for field in IDENTITY_FIELDS}


def processing_fields(
    *,
    pipeline: str,
    model_name: str,
    model_version: str,
    model_revision: str | None = None,
    requested_model_revision: str | None = None,
    status: str,
    run_at: str,
    error: str | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": pipeline,
        "model_name": model_name,
        "model_version": model_version,
        "run_at": run_at,
        "status": status,
    }
    if model_revision:
        value["model_revision"] = model_revision
    if requested_model_revision:
        value["requested_model_revision"] = requested_model_revision
    if error:
        value["error"] = error
    if skip_reason:
        value["skip_reason"] = skip_reason
    return value


def existing_ids(output_path: Path, key: str) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    for record in read_jsonl(output_path):
        value = record.get(key)
        if value is not None:
            ids.add(str(value))
    return ids


def resumable_ids(
    output_path: Path,
    key: str,
    *,
    model_name: str,
    model_revision: str | None = None,
    requested_model_revision: str | None = None,
) -> tuple[set[str], bool]:
    """Return compatible IDs and whether an existing artifact must be replaced.

    A model or explicitly requested revision change invalidates the complete
    modality artifact. Failed/incomplete records also invalidate the artifact
    so resume never mistakes an error checkpoint for completed inference.
    Unrelated historical files are never touched.
    """
    if not output_path.exists():
        return set(), False
    records = read_jsonl(output_path)
    incompatible = False
    ids: set[str] = set()
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        compatible = record.get("model_name") == model_name
        if requested_model_revision is not None:
            compatible = compatible and (
                record.get("requested_model_revision") == requested_model_revision
            )
        elif model_revision is not None:
            compatible = compatible and record.get("model_revision") == model_revision
        if compatible and record.get("status") == "success":
            ids.add(str(value))
        else:
            incompatible = True
    return (set(), True) if incompatible else (ids, False)


def resolve_image_path(record: dict[str, Any], metadata_path: Path) -> Path:
    raw = record.get("keyframe_path") or record.get("frame_path")
    if not raw:
        raise ValueError("missing keyframe_path")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    # This also supports metadata stored beside a portable keyframe directory.
    metadata_relative = metadata_path.parent / path
    return metadata_relative


def verify_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as image:
        image.verify()


def iter_progress(values: Iterable[Any], *, total: int, description: str) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ImportError:
        return values
    return tqdm(values, total=total, desc=description, unit="batch")


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def safe_infer(
    items: Sequence[Any],
    infer: Callable[[Sequence[Any]], Sequence[Any]],
) -> list[tuple[Any | None, Exception | None]]:
    """Infer a batch, then isolate failures per item if the batch fails."""
    try:
        results = list(infer(items))
        if len(results) != len(items):
            raise RuntimeError(f"Backend returned {len(results)} results for {len(items)} inputs.")
        return [(result, None) for result in results]
    except Exception:
        isolated: list[tuple[Any | None, Exception | None]] = []
        for item in items:
            try:
                result = list(infer([item]))
                if len(result) != 1:
                    raise RuntimeError("Backend must return exactly one result for one input.")
                isolated.append((result[0], None))
            except Exception as exc:  # per-record failure is part of the output contract
                isolated.append((None, exc))
        return isolated


def report(
    *,
    pipeline: str,
    input_path: Path,
    output_path: Path,
    model_name: str,
    model_version: str,
    model_revision: str | None = None,
    device: str,
    started_at: str,
    elapsed: float,
    input_count: int,
    success_count: int,
    skipped_count: int,
    error_count: int,
) -> dict[str, Any]:
    processed = success_count + error_count
    value = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": pipeline,
        "started_at": started_at,
        "finished_at": utc_now(),
        "runtime_sec": round(elapsed, 6),
        "throughput_records_per_sec": round(processed / elapsed, 6) if elapsed else 0.0,
        "input_record_count": input_count,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "model_name": model_name,
        "model_version": model_version,
        "device": device,
        "input_path": str(input_path),
        "output_path": str(output_path),
    }
    if model_revision:
        value["model_revision"] = model_revision
    return value


def json_log(service: str, event: str, *, latency: float = 0.0, **extra: Any) -> None:
    payload = {
        "service": service,
        "event": event,
        "timestamp": utc_now(),
        "latency": round(latency, 6),
        **extra,
    }
    logging.getLogger(service).info(json.dumps(payload, ensure_ascii=False))


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )


class Timer:
    def __init__(self) -> None:
        self.started_at = utc_now()
        self._start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start
