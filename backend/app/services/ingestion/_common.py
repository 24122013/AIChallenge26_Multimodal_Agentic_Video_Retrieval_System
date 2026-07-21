"""Tiện ích dùng chung cho các ingestion pipeline (Team P3).

Gói gọn: đọc/ghi JSONL, resolve đường dẫn ảnh keyframe, load ảnh an toàn,
và cấu trúc kết quả pipeline. Không chứa search/rerank logic.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    """Đọc file JSONL thành list dict. Bỏ qua dòng trống / lỗi parse."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")

    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Bỏ dòng %d (%s): parse lỗi %s", line_no, p.name, exc)
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream JSONL để không nạp toàn bộ vào RAM (dùng cho file lớn)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Bỏ dòng %d (%s): parse lỗi %s", line_no, p.name, exc)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Ghi list dict ra JSONL (tạo thư mục cha nếu cần). Trả về số dòng đã ghi."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    logger.info("Đã ghi %d dòng -> %s", count, p)
    return count


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Resolve / load ảnh keyframe
# ---------------------------------------------------------------------------

def resolve_keyframe_path(record: dict, base_dir: str | Path | None = None) -> Path | None:
    """Tìm đường dẫn ảnh keyframe hợp lệ từ một record.

    Ưu tiên keyframe_path -> frame_path -> thumbnail_path. Nếu đường dẫn tương
    đối và base_dir được cung cấp thì join với base_dir.
    """
    raw = record.get("keyframe_path") or record.get("frame_path") or record.get("thumbnail_path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute() and base_dir is not None:
        candidate = Path(base_dir) / path
        if candidate.exists():
            return candidate
    return path


def load_image(path: str | Path):
    """Load ảnh về numpy array (BGR, như OpenCV). None nếu không đọc được.

    Thử OpenCV trước, fallback sang PIL. Import lazy để pipeline nào không cần
    ảnh (vd stub) vẫn chạy khi thiếu thư viện.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(p))
        if img is not None:
            return img
    except Exception:  # pragma: no cover - fallback path
        pass
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()  # RGB->BGR
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Kết quả pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    """Báo cáo tóm tắt sau khi chạy một ingestion pipeline."""

    pipeline: str
    model: str
    input_path: str
    output_path: str
    total_input: int = 0
    total_written: int = 0
    total_skipped: int = 0
    total_empty: int = 0
    errors: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "model": self.model,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "total_input": self.total_input,
            "total_written": self.total_written,
            "total_skipped": self.total_skipped,
            "total_empty": self.total_empty,
            "errors": self.errors[:50],
            "extra": self.extra,
        }
