"""Tests cho metadata pipeline của Team P3 (models + validator + pipelines + builder)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from backend.app.models.metadata import (
    ASR,
    OCR,
    Caption,
    Keyframe,
    ObjectAnnotation,
    UnifiedMetadataRecord,
    make_frame_id,
    make_segment_id,
)
from backend.app.services.ingestion.caption_pipeline import run_caption_pipeline
from backend.app.services.ingestion.metadata_builder import (
    build_unified_metadata,
    enrich_frame_map,
    load_bundle,
)
from backend.app.services.ingestion.object_pipeline import run_object_pipeline
from backend.app.services.ingestion.ocr_pipeline import run_ocr_pipeline
from backend.app.services.ingestion.scheme_validator import (
    validate_caption,
    validate_keyframe,
    validate_objects,
    validate_records,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_keyframes(tmp: Path, n: int = 3, video_id: str = "L01_V001") -> Path:
    kf_dir = tmp / "keyframes" / video_id
    kf_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n + 1):
        frame_id = make_frame_id(video_id, i)
        img_path = kf_dir / f"{frame_id}.jpg"
        if cv2 is not None:
            cv2.imwrite(str(img_path), (np.random.rand(32, 48, 3) * 255).astype("uint8"))
        rows.append(
            {
                "frame_id": frame_id,
                "video_id": video_id,
                "shot_id": f"SHOT_{video_id}_{i:06d}",
                "segment_id": make_segment_id(video_id, 1),
                "timestamp": round(i * 2.0, 3),
                "timestamp_source": "video_fps",
                "timestamp_confidence": 1.0,
                "frame_index": i * 30,
                "keyframe_path": img_path.as_posix(),
            }
        )
    meta = tmp / "metadata" / f"keyframes_{video_id}.jsonl"
    _write_jsonl(meta, rows)
    return meta


class ModelRoundTripTest(unittest.TestCase):
    def test_keyframe_fills_path_aliases(self) -> None:
        kf = Keyframe.from_dict(
            {
                "frame_id": "FRAME_L01_V001_000001",
                "video_id": "L01_V001",
                "shot_id": "SHOT_L01_V001_000001",
                "segment_id": "SEG_L01_V001_000001",
                "timestamp": 1.5,
                "keyframe_path": "a.jpg",
            }
        )
        self.assertEqual(kf.frame_path, "a.jpg")
        self.assertEqual(kf.thumbnail_path, "a.jpg")

    def test_object_annotation_dedup_labels(self) -> None:
        ann = ObjectAnnotation.from_dict(
            {
                "frame_id": "F1",
                "objects": [
                    {"label": "person", "confidence": 0.9},
                    {"label": "person", "confidence": 0.7},
                    {"label": "bus", "confidence": 0.8},
                ],
            }
        )
        self.assertEqual(ann.labels, ["person", "bus"])

    def test_unified_record_roundtrip(self) -> None:
        rec = UnifiedMetadataRecord(
            video_id="L01_V001", frame_id="F1", timestamp=2.0, objects=["car"]
        )
        again = UnifiedMetadataRecord.from_dict(rec.to_dict())
        self.assertEqual(again.objects, ["car"])
        self.assertEqual(again.frame_id, "F1")


class ValidatorTest(unittest.TestCase):
    def test_valid_keyframe(self) -> None:
        row = {
            "frame_id": "FRAME_L01_V001_000001",
            "video_id": "L01_V001",
            "shot_id": "SHOT_L01_V001_000001",
            "segment_id": "SEG_L01_V001_000001",
            "timestamp": 1.0,
            "keyframe_path": "a.jpg",
        }
        self.assertTrue(validate_keyframe(row).valid)

    def test_missing_required_fields(self) -> None:
        result = validate_keyframe({"video_id": "x"})
        self.assertFalse(result.valid)
        self.assertGreaterEqual(len(result.errors), 3)

    def test_caption_empty_is_warning_not_error(self) -> None:
        result = validate_caption({"frame_id": "F1", "caption": ""})
        self.assertTrue(result.valid)
        self.assertTrue(result.warnings)

    def test_objects_requires_label(self) -> None:
        result = validate_objects({"frame_id": "F1", "objects": [{"confidence": 0.9}]})
        self.assertFalse(result.valid)

    def test_batch_summary(self) -> None:
        summary = validate_records(
            "caption",
            [{"frame_id": "F1", "caption": "hi"}, {"caption": "no frame"}],
        )
        self.assertEqual(summary["valid_count"], 1)
        self.assertEqual(summary["invalid_count"], 1)


class StubPipelineTest(unittest.TestCase):
    def test_caption_ocr_object_stub_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            meta = _make_keyframes(tmp)

            cap_out = tmp / "metadata" / "captions.jsonl"
            ocr_out = tmp / "metadata" / "ocr.jsonl"
            obj_out = tmp / "metadata" / "objects.jsonl"

            cap_report = run_caption_pipeline(meta, cap_out, backend="stub")
            ocr_report = run_ocr_pipeline(meta, ocr_out, backend="stub")
            obj_report = run_object_pipeline(meta, obj_out, backend="stub")

            self.assertEqual(cap_report.total_written, 3)
            self.assertEqual(ocr_report.total_written, 3)
            self.assertEqual(obj_report.total_written, 3)
            # caption stub luôn có nội dung
            self.assertEqual(cap_report.total_empty, 0)

            for row in [json.loads(l) for l in cap_out.read_text().splitlines()]:
                self.assertTrue(validate_caption(row).valid)


class BuilderJoinTest(unittest.TestCase):
    def test_join_by_frame_and_segment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            meta = _make_keyframes(tmp, n=3)
            md = tmp / "metadata"

            _write_jsonl(md / "cap.jsonl", [
                {"frame_id": "FRAME_L01_V001_000001", "caption": "a red bus"},
            ])
            _write_jsonl(md / "ocr.jsonl", [
                {"frame_id": "FRAME_L01_V001_000001", "ocr_text": "BUS", "ocr_confidence": 0.9},
            ])
            _write_jsonl(md / "obj.jsonl", [
                {"frame_id": "FRAME_L01_V001_000001",
                 "objects": [{"label": "bus", "confidence": 0.95}]},
            ])
            _write_jsonl(md / "asr.jsonl", [
                {"segment_id": "SEG_L01_V001_000001", "video_id": "L01_V001",
                 "transcript": "welcome", "start_time": 0.0, "end_time": 3.0},
            ])

            out = md / "unified.jsonl"
            report = build_unified_metadata(
                meta, out,
                caption_path=md / "cap.jsonl",
                ocr_path=md / "ocr.jsonl",
                objects_path=md / "obj.jsonl",
                asr_path=md / "asr.jsonl",
            )
            self.assertEqual(report["total_frames"], 3)
            self.assertEqual(report["coverage"]["caption"], 1)

            rows = [json.loads(l) for l in out.read_text().splitlines()]
            first = rows[0]
            self.assertEqual(first["caption"], "a red bus")
            self.assertEqual(first["ocr_text"], "BUS")
            self.assertEqual(first["objects"], ["bus"])
            # cả 3 frame cùng segment_id -> đều nhận transcript qua segment match
            self.assertTrue(all(r["transcript"] == "welcome" for r in rows))

    def test_enrich_frame_map_preserves_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            meta = _make_keyframes(tmp, n=2)
            md = tmp / "metadata"
            rows = [json.loads(l) for l in meta.read_text().splitlines()]
            frame_map = {str(i): {**r, "faiss_index": i} for i, r in enumerate(rows)}
            fm_path = md / "frame_map.json"
            fm_path.write_text(json.dumps(frame_map))

            _write_jsonl(md / "cap.jsonl", [
                {"frame_id": "FRAME_L01_V001_000001", "caption": "hello"},
            ])
            bundle = load_bundle(caption_path=md / "cap.jsonl")
            out = md / "frame_map_enriched.json"
            enrich_frame_map(fm_path, out, bundle)

            enriched = json.loads(out.read_text())
            self.assertIn("0", enriched)
            self.assertEqual(enriched["0"]["caption"], "hello")
            # field gốc vẫn còn
            self.assertEqual(enriched["0"]["frame_id"], "FRAME_L01_V001_000001")


if __name__ == "__main__":
    unittest.main()
