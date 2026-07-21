"""neighbor_index — tiền tính frame lân cận cho mỗi keyframe (Team P2).

Retrieval hiển thị neighbor cùng shot quanh mỗi hit. Thay vì tính on-the-fly mỗi
query, ta tiền tính từ `frame_map.json` ra một index:
    { "<faiss_index>": [neighbor_faiss_index, ...], ... }

Chiến lược:
- "same_shot": neighbor là frame cùng shot_id, sắp theo |Δtimestamp|.
- "window": nếu không có shot_id, lấy frame cùng video trong cửa sổ thời gian.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_frame_map(path: str | Path) -> dict[int, dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"frame_map.json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[int, dict] = {}
    for key, entry in raw.items():
        try:
            out[int(key)] = entry
        except (ValueError, TypeError):
            logger.warning("Bỏ key không hợp lệ: %r", key)
    return out


def build_neighbor_index(
    frame_map_path: str | Path,
    output_path: str | Path,
    *,
    max_neighbors: int = 4,
    strategy: str = "same_shot",
    window_sec: float = 4.0,
) -> dict:
    """Build neighbor index từ frame_map, ghi ra JSON. Trả report."""
    frame_map = _load_frame_map(frame_map_path)

    # gom theo video để không so cross-video
    by_video: dict[str, list[int]] = {}
    for idx, entry in frame_map.items():
        by_video.setdefault(entry.get("video_id", ""), []).append(idx)

    def ts(i: int) -> float:
        return float(frame_map[i].get("timestamp") or 0.0)

    neighbor_index: dict[str, list[int]] = {}
    total_neighbors = 0
    isolated = 0

    for _video_id, indices in by_video.items():
        indices.sort(key=ts)
        for i in indices:
            entry = frame_map[i]
            shot_id = entry.get("shot_id", "")
            shot_start = entry.get("shot_start")
            shot_end = entry.get("shot_end")

            if strategy == "same_shot" and shot_id:
                candidates = [
                    j for j in indices if j != i and frame_map[j].get("shot_id") == shot_id
                ]
            elif shot_start is not None and shot_end is not None:
                candidates = [
                    j for j in indices
                    if j != i and float(shot_start) <= ts(j) <= float(shot_end)
                ]
            else:  # window fallback
                candidates = [j for j in indices if j != i and abs(ts(j) - ts(i)) <= window_sec]

            candidates.sort(key=lambda j: abs(ts(j) - ts(i)))
            chosen = candidates[: max(0, max_neighbors)]
            chosen.sort(key=ts)  # trả về theo thứ tự thời gian
            neighbor_index[str(i)] = chosen
            total_neighbors += len(chosen)
            if not chosen:
                isolated += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(neighbor_index, f, ensure_ascii=False)

    report = {
        "frame_map_path": str(frame_map_path),
        "output_path": str(output_path),
        "strategy": strategy,
        "max_neighbors": max_neighbors,
        "total_frames": len(frame_map),
        "total_neighbor_links": total_neighbors,
        "isolated_frames": isolated,
        "avg_neighbors": round(total_neighbors / len(frame_map), 3) if frame_map else 0.0,
    }
    logger.info(
        "neighbor_index: %d frames, avg %.2f neighbors, %d isolated",
        report["total_frames"],
        report["avg_neighbors"],
        isolated,
    )
    return report


class NeighborIndex:
    """Load neighbor index đã build để retrieval tra cứu O(1)."""

    def __init__(self, mapping: dict[int, list[int]]) -> None:
        self._map = mapping

    @classmethod
    def load(cls, path: str | Path) -> "NeighborIndex":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls({int(k): list(v) for k, v in raw.items()})

    def neighbors(self, faiss_index: int) -> list[int]:
        return self._map.get(faiss_index, [])


def main() -> None:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Build neighbor index from frame_map.json")
    parser.add_argument("--frame-map-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--strategy", default="same_shot", choices=("same_shot", "window"))
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    report = build_neighbor_index(
        frame_map_path=args.frame_map_path,
        output_path=args.output_path,
        max_neighbors=args.max_neighbors,
        strategy=args.strategy,
        window_sec=args.window_sec,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["build_neighbor_index", "NeighborIndex"]
