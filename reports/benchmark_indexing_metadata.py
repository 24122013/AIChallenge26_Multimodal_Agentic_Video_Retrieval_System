from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

from src.indexing.build_neighbor_index import build_neighbor_index
from src.indexing.build_segment_metadata import build_segment_metadata


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write compact deterministic JSONL used by the synthetic benchmark."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read the generated benchmark artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def synthetic_records(
    *,
    video_count: int,
    frames_per_video: int,
    fps: float,
    keyframe_interval_seconds: float,
    frames_per_shot: int,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic keyframe and multimodal metadata."""
    keyframes: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    ocr: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    for video_offset in range(video_count):
        video_id = f"SYNTH_V{video_offset:03d}"
        for frame_offset in range(frames_per_video):
            timestamp = frame_offset * keyframe_interval_seconds
            frame_index = round(timestamp * fps)
            shot_index = frame_offset // frames_per_shot
            shot_start = shot_index * frames_per_shot * keyframe_interval_seconds
            shot_end = (shot_index + 1) * frames_per_shot * keyframe_interval_seconds
            frame_id = f"FRAME_{video_id}_{frame_offset:06d}"
            shot_id = f"SHOT_{video_id}_{shot_index:06d}"
            keyframes.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "segment_id": shot_id,
                    "shot_id": shot_id,
                    "shot_start": shot_start,
                    "shot_end": shot_end,
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "keyframe_path": f"data/keyframes/{video_id}/{frame_id}.jpg",
                    "thumbnail_path": f"data/keyframes/{video_id}/{frame_id}.jpg",
                    "source_video_path": f"data/raw/video/{video_id}.mp4",
                }
            )
            captions.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "segment_id": shot_id,
                    "timestamp": timestamp,
                    "status": "success",
                    "caption": (
                        f"person walking beside vehicle in scene {shot_index % 7}"
                    ),
                }
            )
            if frame_offset % 3 == 0:
                ocr.append(
                    {
                        "frame_id": frame_id,
                        "video_id": video_id,
                        "timestamp": timestamp,
                        "status": "success",
                        "text_regions": [
                            {
                                "text": f"Street {shot_index % 11}",
                                "confidence": 0.8 + (frame_offset % 10) / 100,
                            }
                        ],
                    }
                )
            objects.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "status": "success",
                    "objects": [
                        {"class_name": "person", "confidence": 0.85},
                        {"class_name": "car", "confidence": 0.75},
                    ],
                }
            )
    return {
        "keyframes": keyframes,
        "captions": captions,
        "ocr": ocr,
        "objects": objects,
    }


def measured_call(call: Callable[[], Any]) -> dict[str, float]:
    """Measure wall time and Python allocation peak for one call."""
    tracemalloc.start()
    started = time.perf_counter()
    call()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "runtime_ms": elapsed * 1000,
        "python_peak_memory_mb": peak / (1024 * 1024),
    }


def summarize_runs(values: list[dict[str, float]]) -> dict[str, float]:
    """Summarize repeated timings with median and observed range."""
    runtimes = [value["runtime_ms"] for value in values]
    peaks = [value["python_peak_memory_mb"] for value in values]
    return {
        "runs": len(values),
        "runtime_median_ms": statistics.median(runtimes),
        "runtime_min_ms": min(runtimes),
        "runtime_max_ms": max(runtimes),
        "python_peak_memory_median_mb": statistics.median(peaks),
    }


def expanded_neighbor_artifact(
    compact_path: Path,
    keyframes: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Create a counterfactual artifact that copies full keyframe metadata."""
    by_identity = {
        (str(record["video_id"]), str(record["frame_id"])): record
        for record in keyframes
    }
    expanded = []
    for record in read_jsonl(compact_path):
        video_id = str(record["video_id"])
        value = dict(record)
        for field in ("neighbors_before", "neighbors_after"):
            value[field] = [
                {
                    **by_identity[(video_id, str(neighbor["frame_id"]))],
                    "delta_seconds": neighbor["delta_seconds"],
                }
                for neighbor in record[field]
            ]
        expanded.append(value)
    write_jsonl(output_path, expanded)


def lookup_benchmark(
    records: list[dict[str, Any]],
    *,
    query_count: int,
    runs: int,
) -> dict[str, Any]:
    """Compare repeated linear metadata lookup with an ID-reference map."""
    queries = [
        (
            str(records[(offset * 7919) % len(records)]["video_id"]),
            str(records[(offset * 7919) % len(records)]["frame_id"]),
        )
        for offset in range(query_count)
    ]
    lookup = {
        (str(record["video_id"]), str(record["frame_id"])): record
        for record in records
    }

    def linear() -> None:
        for query in queries:
            next(
                record
                for record in records
                if (str(record["video_id"]), str(record["frame_id"])) == query
            )

    def indexed() -> None:
        for query in queries:
            lookup[query]

    linear_times = []
    indexed_times = []
    for _ in range(runs):
        started = time.perf_counter()
        linear()
        linear_times.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        indexed()
        indexed_times.append((time.perf_counter() - started) * 1000)
    return {
        "query_count_per_run": query_count,
        "runs": runs,
        "linear_scan_median_ms": statistics.median(linear_times),
        "id_map_median_ms": statistics.median(indexed_times),
        "speedup": statistics.median(linear_times) / statistics.median(indexed_times),
    }


def run_benchmark(
    *,
    output_path: Path,
    video_count: int,
    frames_per_video: int,
    fps: float,
    keyframe_interval_seconds: float,
    frames_per_shot: int,
    window_seconds: float,
    runs: int,
    lookup_queries: int,
) -> dict[str, Any]:
    """Run the reproducible synthetic indexing micro-benchmark."""
    if video_count < 1 or frames_per_video < 1 or runs < 1 or lookup_queries < 1:
        raise ValueError("counts and runs must be >= 1")
    records = synthetic_records(
        video_count=video_count,
        frames_per_video=frames_per_video,
        fps=fps,
        keyframe_interval_seconds=keyframe_interval_seconds,
        frames_per_shot=frames_per_shot,
    )
    with tempfile.TemporaryDirectory(prefix="metadata-benchmark-") as temporary_dir:
        root = Path(temporary_dir)
        paths = {
            name: root / f"{name}.jsonl"
            for name in ("keyframes", "captions", "ocr", "objects")
        }
        for name, path in paths.items():
            write_jsonl(path, records[name])
        neighbor_path = root / "neighbors.jsonl"
        segment_path = root / "segments.jsonl"

        neighbor_runs = [
            measured_call(
                lambda: build_neighbor_index(
                    paths["keyframes"],
                    neighbor_path,
                    window_seconds=window_seconds,
                )
            )
            for _ in range(runs)
        ]
        segment_runs = [
            measured_call(
                lambda: build_segment_metadata(
                    paths["keyframes"],
                    segment_path,
                    captions_path=paths["captions"],
                    ocr_path=paths["ocr"],
                    objects_path=paths["objects"],
                )
            )
            for _ in range(runs)
        ]
        expanded_neighbor_path = root / "neighbors_expanded.jsonl"
        expanded_neighbor_artifact(
            neighbor_path,
            records["keyframes"],
            expanded_neighbor_path,
        )
        frame_map = {
            str(offset): record
            for offset, record in enumerate(records["keyframes"])
        }
        pretty_frame_map_path = root / "frame_map_pretty.json"
        compact_frame_map_path = root / "frame_map_compact.json"
        pretty_frame_map_path.write_text(
            json.dumps(frame_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        compact_frame_map_path.write_text(
            json.dumps(frame_map, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

        baseline_files = list(paths.values())
        baseline_bytes = sum(path.stat().st_size for path in baseline_files)
        optimized_added_bytes = neighbor_path.stat().st_size + segment_path.stat().st_size
        expanded_added_bytes = (
            expanded_neighbor_path.stat().st_size + segment_path.stat().st_size
        )
        neighbor_records = read_jsonl(neighbor_path)
        result = {
            "benchmark_kind": "synthetic_microbenchmark",
            "dataset": {
                "video_count": video_count,
                "frames_per_video": frames_per_video,
                "keyframe_count": len(records["keyframes"]),
                "caption_count": len(records["captions"]),
                "ocr_record_count": len(records["ocr"]),
                "object_record_count": len(records["objects"]),
                "fps": fps,
                "keyframe_interval_seconds": keyframe_interval_seconds,
                "frames_per_shot": frames_per_shot,
                "window_seconds": window_seconds,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or "not reported",
            },
            "build": {
                "neighbor_index": summarize_runs(neighbor_runs),
                "segment_metadata": summarize_runs(segment_runs),
            },
            "size_bytes": {
                "existing_frame_level_metadata": baseline_bytes,
                "compact_neighbor_index": neighbor_path.stat().st_size,
                "expanded_neighbor_counterfactual": expanded_neighbor_path.stat().st_size,
                "segment_metadata": segment_path.stat().st_size,
                "frame_map_pretty_serialization": pretty_frame_map_path.stat().st_size,
                "frame_map_compact_serialization": compact_frame_map_path.stat().st_size,
                "total_existing_plus_compact_artifacts": (
                    baseline_bytes + optimized_added_bytes
                ),
                "total_existing_plus_expanded_artifacts": (
                    baseline_bytes + expanded_added_bytes
                ),
            },
            "lookup": lookup_benchmark(
                neighbor_records,
                query_count=lookup_queries,
                runs=runs,
            ),
            "measurement": {
                "runs": runs,
                "runtime_statistic": "median with min/max",
                "memory_statistic": "median tracemalloc Python allocation peak",
                "vector_index_included": False,
                "quality_benchmark_included": False,
            },
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark indexing metadata changes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/index_size_latency_benchmark.json"),
    )
    parser.add_argument("--videos", type=int, default=4)
    parser.add_argument("--frames-per-video", type=int, default=250)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--keyframe-interval-seconds", type=float, default=1.0)
    parser.add_argument("--frames-per-shot", type=int, default=5)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--lookup-queries", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_benchmark(
        output_path=args.output,
        video_count=args.videos,
        frames_per_video=args.frames_per_video,
        fps=args.fps,
        keyframe_interval_seconds=args.keyframe_interval_seconds,
        frames_per_shot=args.frames_per_shot,
        window_seconds=args.window_seconds,
        runs=args.runs,
        lookup_queries=args.lookup_queries,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
