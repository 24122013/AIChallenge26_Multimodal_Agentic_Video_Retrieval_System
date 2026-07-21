"""metadata_builder — gộp keyframe + caption + ocr + objects + asr thành metadata thống nhất.

Đây là bước cuối của Team P3: nhận các sidecar rời rạc do từng pipeline sinh ra,
join theo `frame_id` (và `segment_id` cho ASR), rồi xuất ra:
  1. unified metadata JSONL (mỗi dòng là 1 UnifiedMetadataRecord)
  2. (tuỳ chọn) enrich frame_map.json với caption/ocr/objects/transcript

Module này KHÔNG search / rerank — chỉ chuẩn bị dữ liệu. Retrieval sẽ đọc output.

ASR gắn với segment thời gian; ta map transcript về frame bằng cách so timestamp
frame nằm trong [start_time, end_time] của segment (ưu tiên khớp segment_id trực
tiếp nếu keyframe đã có segment_id trùng).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.models.metadata import (
    ASR,
    Caption,
    OCR,
    ObjectAnnotation,
    UnifiedMetadataRecord,
)
from backend.app.services.ingestion._common import (
    read_jsonl,
    write_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundle: các sidecar đã index sẵn để join nhanh
# ---------------------------------------------------------------------------

@dataclass
class MetadataBundle:
    """Tập sidecar metadata đã index theo key để join O(1)."""

    caption_by_frame: dict[str, Caption] = field(default_factory=dict)
    ocr_by_frame: dict[str, OCR] = field(default_factory=dict)
    objects_by_frame: dict[str, ObjectAnnotation] = field(default_factory=dict)
    asr_segments: list[ASR] = field(default_factory=list)

    # index phụ cho ASR
    _asr_by_video: dict[str, list[ASR]] = field(default_factory=dict, repr=False)
    _asr_by_segment: dict[str, ASR] = field(default_factory=dict, repr=False)

    def index_asr(self) -> None:
        self._asr_by_video.clear()
        self._asr_by_segment.clear()
        for seg in self.asr_segments:
            if seg.segment_id:
                self._asr_by_segment[seg.segment_id] = seg
            if seg.video_id:
                self._asr_by_video.setdefault(seg.video_id, []).append(seg)
        for segs in self._asr_by_video.values():
            segs.sort(key=lambda s: (s.start_time if s.start_time is not None else 0.0))

    def transcript_for_frame(self, video_id: str, timestamp: float, segment_id: str = "") -> str:
        """Tìm transcript phủ frame: ưu tiên segment_id trùng, sau đó theo timestamp."""
        if segment_id and segment_id in self._asr_by_segment:
            return self._asr_by_segment[segment_id].transcript
        for seg in self._asr_by_video.get(video_id, []):
            start = seg.start_time if seg.start_time is not None else 0.0
            end = seg.end_time if seg.end_time is not None else start
            if start <= timestamp <= end:
                return seg.transcript
        return ""


# ---------------------------------------------------------------------------
# Load sidecar
# ---------------------------------------------------------------------------

def _load_optional(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        logger.warning("Sidecar không tồn tại, bỏ qua: %s", p)
        return []
    return read_jsonl(p)


def load_bundle(
    *,
    caption_path: str | Path | None = None,
    ocr_path: str | Path | None = None,
    objects_path: str | Path | None = None,
    asr_path: str | Path | None = None,
) -> MetadataBundle:
    """Nạp tất cả sidecar hiện có thành một MetadataBundle đã index."""
    bundle = MetadataBundle()

    for row in _load_optional(caption_path):
        cap = Caption.from_dict(row)
        if cap.frame_id:
            bundle.caption_by_frame[cap.frame_id] = cap

    for row in _load_optional(ocr_path):
        ocr = OCR.from_dict(row)
        if ocr.frame_id:
            bundle.ocr_by_frame[ocr.frame_id] = ocr

    for row in _load_optional(objects_path):
        ann = ObjectAnnotation.from_dict(row)
        if ann.frame_id:
            bundle.objects_by_frame[ann.frame_id] = ann

    for row in _load_optional(asr_path):
        bundle.asr_segments.append(ASR.from_dict(row))

    bundle.index_asr()
    logger.info(
        "Bundle loaded: %d caption, %d ocr, %d objects, %d asr-seg",
        len(bundle.caption_by_frame),
        len(bundle.ocr_by_frame),
        len(bundle.objects_by_frame),
        len(bundle.asr_segments),
    )
    return bundle


# ---------------------------------------------------------------------------
# Build unified records
# ---------------------------------------------------------------------------

def build_unified_record(keyframe: dict, bundle: MetadataBundle) -> UnifiedMetadataRecord:
    """Join một keyframe record với các sidecar trong bundle."""
    frame_id = keyframe.get("frame_id", "")
    video_id = keyframe.get("video_id", "")
    timestamp = float(keyframe.get("timestamp") or 0.0)
    segment_id = keyframe.get("segment_id", "") or ""

    cap = bundle.caption_by_frame.get(frame_id)
    ocr = bundle.ocr_by_frame.get(frame_id)
    ann = bundle.objects_by_frame.get(frame_id)

    return UnifiedMetadataRecord(
        video_id=video_id,
        frame_id=frame_id,
        timestamp=timestamp,
        segment_id=segment_id,
        shot_id=keyframe.get("shot_id", "") or "",
        timestamp_source=keyframe.get("timestamp_source", "unknown") or "unknown",
        timestamp_confidence=float(
            keyframe.get("timestamp_confidence")
            if keyframe.get("timestamp_confidence") is not None
            else 1.0
        ),
        frame_index=keyframe.get("frame_index"),
        faiss_index=keyframe.get("faiss_index"),
        keyframe_path=keyframe.get("keyframe_path", "") or keyframe.get("frame_path", "") or "",
        thumbnail_path=keyframe.get("thumbnail_path", "") or keyframe.get("keyframe_path", "") or "",
        caption=cap.caption if cap else "",
        ocr_text=ocr.ocr_text if ocr else "",
        objects=ann.labels if ann else [],
        transcript=bundle.transcript_for_frame(video_id, timestamp, segment_id),
    )


def build_unified_metadata(
    keyframe_metadata_path: str | Path,
    output_path: str | Path,
    *,
    caption_path: str | Path | None = None,
    ocr_path: str | Path | None = None,
    objects_path: str | Path | None = None,
    asr_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict:
    """Đọc keyframe metadata + sidecar -> ghi unified metadata JSONL.

    Returns:
        report dict (coverage của từng sidecar).
    """
    keyframes = read_jsonl(keyframe_metadata_path)
    bundle = load_bundle(
        caption_path=caption_path,
        ocr_path=ocr_path,
        objects_path=objects_path,
        asr_path=asr_path,
    )

    records = [build_unified_record(kf, bundle) for kf in keyframes]
    written = write_jsonl(output_path, (r.to_dict() for r in records))

    total = len(records)
    covered_caption = sum(1 for r in records if r.caption.strip())
    covered_ocr = sum(1 for r in records if r.ocr_text.strip())
    covered_objects = sum(1 for r in records if r.objects)
    covered_transcript = sum(1 for r in records if r.transcript.strip())

    report = {
        "keyframe_metadata_path": str(keyframe_metadata_path),
        "output_path": str(output_path),
        "total_frames": total,
        "written": written,
        "coverage": {
            "caption": covered_caption,
            "ocr": covered_ocr,
            "objects": covered_objects,
            "transcript": covered_transcript,
        },
        "coverage_ratio": {
            "caption": round(covered_caption / total, 4) if total else 0.0,
            "ocr": round(covered_ocr / total, 4) if total else 0.0,
            "objects": round(covered_objects / total, 4) if total else 0.0,
            "transcript": round(covered_transcript / total, 4) if total else 0.0,
        },
    }
    if report_path is not None:
        write_json(report_path, report)
    logger.info("Unified metadata: %d frames -> %s", written, output_path)
    return report


# ---------------------------------------------------------------------------
# Enrich frame_map.json (giữ đúng format key=faiss_index)
# ---------------------------------------------------------------------------

def enrich_frame_map(
    frame_map_path: str | Path,
    output_path: str | Path,
    bundle: MetadataBundle,
) -> dict:
    """Thêm caption/ocr/objects/transcript vào từng entry của frame_map.json.

    frame_map.json giữ nguyên cấu trúc {"<faiss_index>": {...}}; ta chỉ bổ sung
    field, không đổi key hay xoá field cũ.
    """
    p = Path(frame_map_path)
    with p.open("r", encoding="utf-8") as f:
        frame_map = json.load(f)

    if not isinstance(frame_map, dict):
        raise ValueError("frame_map.json phải là dict")

    enriched = 0
    for _idx, entry in frame_map.items():
        if not isinstance(entry, dict):
            continue
        frame_id = entry.get("frame_id", "")
        video_id = entry.get("video_id", "")
        timestamp = float(entry.get("timestamp") or 0.0)
        segment_id = entry.get("segment_id", "") or ""

        cap = bundle.caption_by_frame.get(frame_id)
        ocr = bundle.ocr_by_frame.get(frame_id)
        ann = bundle.objects_by_frame.get(frame_id)

        if cap:
            entry["caption"] = cap.caption
        if ocr:
            entry["ocr_text"] = ocr.ocr_text
        if ann:
            entry["objects"] = ann.labels
        transcript = bundle.transcript_for_frame(video_id, timestamp, segment_id)
        if transcript:
            entry["transcript"] = transcript
        enriched += 1

    write_json(output_path, frame_map)
    logger.info("Enriched %d frame_map entries -> %s", enriched, output_path)
    return {"enriched_entries": enriched, "output_path": str(output_path)}


__all__ = [
    "MetadataBundle",
    "load_bundle",
    "build_unified_record",
    "build_unified_metadata",
    "enrich_frame_map",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Build unified metadata from sidecars")
    parser.add_argument("--keyframe-metadata-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--caption-path", default=None)
    parser.add_argument("--ocr-path", default=None)
    parser.add_argument("--objects-path", default=None)
    parser.add_argument("--asr-path", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--enrich-frame-map", default=None,
                        help="Đường dẫn frame_map.json để enrich (tuỳ chọn)")
    parser.add_argument("--enriched-frame-map-out", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    report = build_unified_metadata(
        keyframe_metadata_path=args.keyframe_metadata_path,
        output_path=args.output_path,
        caption_path=args.caption_path,
        ocr_path=args.ocr_path,
        objects_path=args.objects_path,
        asr_path=args.asr_path,
        report_path=args.report_path,
    )

    if args.enrich_frame_map:
        bundle = load_bundle(
            caption_path=args.caption_path,
            ocr_path=args.ocr_path,
            objects_path=args.objects_path,
            asr_path=args.asr_path,
        )
        out = args.enriched_frame_map_out or args.enrich_frame_map
        enrich_frame_map(args.enrich_frame_map, out, bundle)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
