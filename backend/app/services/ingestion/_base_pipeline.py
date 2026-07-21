"""Khung chung cho các frame-based ingestion pipeline (caption/ocr/objects).

Mỗi pipeline chỉ cần cung cấp một Backend (nhận ảnh + record -> dict metadata).
Khung này lo phần đọc keyframe metadata, load ảnh, gọi backend, validate và ghi
JSONL. Nhờ vậy caption/ocr/objects dùng lại cùng luồng, giảm trùng lặp.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Protocol

from ._common import (
    PipelineReport,
    load_image,
    read_jsonl,
    resolve_keyframe_path,
    write_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)


class FrameBackend(Protocol):
    """Backend xử lý 1 keyframe.

    `name` để ghi vào metadata (vd caption_model). `process` nhận ảnh (numpy
    array BGR hoặc None nếu không đọc được) và record keyframe gốc, trả về dict
    các field metadata muốn merge (không gồm frame_id — khung tự thêm).
    """

    name: str

    def process(self, image, record: dict) -> dict:
        ...


def run_frame_pipeline(
    *,
    pipeline_name: str,
    metadata_path: str | Path,
    output_path: str | Path,
    backend: FrameBackend,
    validate_fn: Callable[[dict], object] | None = None,
    image_base_dir: str | Path | None = None,
    require_image: bool = True,
    report_path: str | Path | None = None,
    limit: int | None = None,
) -> PipelineReport:
    """Chạy pipeline frame-based end-to-end.

    Args:
        pipeline_name: tên pipeline ("caption"/"ocr"/"objects") — để log/report.
        metadata_path: JSONL keyframe metadata (nguồn frame_id + keyframe_path).
        output_path: nơi ghi sidecar JSONL.
        backend: đối tượng có .name và .process(image, record) -> dict.
        validate_fn: hàm validate mỗi record output (trả object có .valid/.errors).
        image_base_dir: base dir nếu keyframe_path là tương đối.
        require_image: nếu True và ảnh không load được thì skip record.
        report_path: nếu set, ghi report JSON ra đây.
        limit: chỉ xử lý N record đầu (debug).

    Returns:
        PipelineReport
    """
    rows = read_jsonl(metadata_path)
    if limit is not None:
        rows = rows[:limit]

    report = PipelineReport(
        pipeline=pipeline_name,
        model=getattr(backend, "name", "unknown"),
        input_path=str(metadata_path),
        output_path=str(output_path),
        total_input=len(rows),
    )

    outputs: list[dict] = []
    for record in rows:
        frame_id = record.get("frame_id")
        if not frame_id:
            report.total_skipped += 1
            report.errors.append("record thiếu frame_id")
            continue

        image = None
        img_path = resolve_keyframe_path(record, image_base_dir)
        if img_path is not None:
            image = load_image(img_path)

        if require_image and image is None:
            report.total_skipped += 1
            report.errors.append(f"{frame_id}: không load được ảnh ({img_path})")
            continue

        try:
            fields = backend.process(image, record) or {}
        except Exception as exc:  # pragma: no cover - phụ thuộc backend
            report.total_skipped += 1
            report.errors.append(f"{frame_id}: backend lỗi {exc}")
            continue

        out = {"frame_id": frame_id, **fields}

        if validate_fn is not None:
            result = validate_fn(out)
            if not getattr(result, "valid", True):
                report.total_skipped += 1
                report.errors.append(f"{frame_id}: validate fail {getattr(result,'errors',[])}")
                continue

        # đếm record "rỗng" (không có nội dung hữu ích) để theo dõi coverage
        if _is_empty_output(pipeline_name, out):
            report.total_empty += 1

        outputs.append(out)

    report.total_written = write_jsonl(output_path, outputs)

    if report_path is not None:
        write_json(report_path, report.to_dict())

    logger.info(
        "[%s] input=%d written=%d skipped=%d empty=%d model=%s",
        pipeline_name,
        report.total_input,
        report.total_written,
        report.total_skipped,
        report.total_empty,
        report.model,
    )
    return report


def _is_empty_output(pipeline_name: str, out: dict) -> bool:
    if pipeline_name == "caption":
        return not str(out.get("caption", "")).strip()
    if pipeline_name == "ocr":
        return not str(out.get("ocr_text", "")).strip()
    if pipeline_name == "objects":
        return not out.get("objects")
    return False
