"""extract_segments — sinh segment record từ keyframe metadata (Team P2).

Segment gom các keyframe cùng `shot_id` thành một đoạn thời gian liên tục
[start, end], dùng cho ASR/temporal. Nếu keyframe không có shot info thì gom
theo cửa sổ thời gian cố định.

Output JSONL khớp Segment schema:
    {segment_id, video_id, start_time, end_time, duration}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"metadata not found: {p}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def segments_from_keyframes(records: list[dict], *, window_sec: float = 5.0) -> list[dict]:
    """Gom keyframe -> segment. Ưu tiên shot_id; fallback theo window thời gian."""
    if not records:
        return []
    video_id = records[0].get("video_id", "")

    # Nếu có shot info thì gom theo shot_id
    has_shot = any(r.get("shot_id") for r in records)
    segments: list[dict] = []

    if has_shot:
        by_shot: dict[str, list[dict]] = {}
        for r in records:
            by_shot.setdefault(r.get("shot_id", ""), []).append(r)
        # sắp shot theo thời điểm bắt đầu
        ordered = sorted(by_shot.items(), key=lambda kv: _seg_bounds(kv[1])[0])
        for i, (_shot_id, frames) in enumerate(ordered, start=1):
            start, end = _seg_bounds(frames)
            segments.append(_make_segment(video_id, i, start, end))
    else:
        # fallback: cắt theo cửa sổ thời gian đều
        ts_sorted = sorted(records, key=lambda r: float(r.get("timestamp") or 0.0))
        idx = 1
        bucket_start = float(ts_sorted[0].get("timestamp") or 0.0)
        bucket_end = bucket_start
        for r in ts_sorted:
            t = float(r.get("timestamp") or 0.0)
            if t - bucket_start > window_sec:
                segments.append(_make_segment(video_id, idx, bucket_start, bucket_end))
                idx += 1
                bucket_start = t
            bucket_end = t
        segments.append(_make_segment(video_id, idx, bucket_start, bucket_end))

    return segments


def _seg_bounds(frames: list[dict]) -> tuple[float, float]:
    starts = []
    ends = []
    for f in frames:
        if f.get("shot_start") is not None:
            starts.append(float(f["shot_start"]))
        else:
            starts.append(float(f.get("timestamp") or 0.0))
        if f.get("shot_end") is not None:
            ends.append(float(f["shot_end"]))
        else:
            ends.append(float(f.get("timestamp") or 0.0))
    return min(starts), max(ends)


def _make_segment(video_id: str, n: int, start: float, end: float) -> dict:
    start = round(float(start), 3)
    end = round(float(max(end, start)), 3)
    return {
        "segment_id": f"SEG_{video_id}_{n:06d}",
        "video_id": video_id,
        "start_time": start,
        "end_time": end,
        "duration": round(end - start, 3),
    }


def extract_segments(
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    window_sec: float = 5.0,
) -> dict:
    records = _load_jsonl(metadata_path)
    segments = segments_from_keyframes(records, window_sec=window_sec)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    report = {
        "metadata_path": str(metadata_path),
        "output_path": str(output_path),
        "total_keyframes": len(records),
        "total_segments": len(segments),
    }
    logger.info("extract_segments: %d frames -> %d segments", len(records), len(segments))
    return report


def main() -> None:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Extract segments from keyframe metadata")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    report = extract_segments(
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        window_sec=args.window_sec,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["extract_segments", "segments_from_keyframes"]
