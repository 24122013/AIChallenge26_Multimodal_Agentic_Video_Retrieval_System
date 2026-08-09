from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from backend.app.services.indexing.keyframe_feature_adapter import (
    FeatureAdapterConfig,
    adapt_feature_records,
)
from backend.app.services.ingestion.common import read_jsonl
from backend.app.services.ingestion.ocr_pipeline import run_ocr_file


class _SubtitleOcrBackend:
    model_name = "fake-ocr"
    model_version = "1"

    def infer(self, paths: Sequence[Path]) -> Sequence[Sequence[Any]]:
        return [
            [
                (
                    [[10, 40], [20, 40], [20, 48], [10, 48]],
                    "Score 2026",
                    0.99,
                )
            ]
            for _ in paths
        ]


class _FailingOcrBackend:
    model_name = "failing-ocr"
    model_version = "1"

    def infer(self, paths: Sequence[Path]) -> Sequence[Sequence[Any]]:
        raise RuntimeError("synthetic inference failure")


class OcrGeometryContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_path = self.root / "candidate.png"
        Image.new("RGB", (100, 50), color="white").save(self.image_path)
        self.metadata_path = self.root / "keyframe_candidates_VIDEO.jsonl"
        self.candidate = {
            "candidate_id": "CANDIDATE_VIDEO_000000000",
            "candidate_index": 0,
            "frame_id": "FRAME_VIDEO_000000000",
            "video_id": "VIDEO",
            "timestamp": 0.0,
            "frame_index": 0,
            "shot_index": 0,
            "candidate_reasons": ["dense_interval"],
            "shot_start": 0.0,
            "shot_end": 1.0,
            "keyframe_path": str(self.image_path),
            "source_video_path": "video.mp4",
        }
        self.metadata_path.write_text(
            json.dumps(self.candidate) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_ocr_output_supplies_geometry_to_adapter_without_objects(self) -> None:
        output_path = self.root / "ocr_candidates_VIDEO.jsonl"
        run_ocr_file(
            self.metadata_path,
            output_path=output_path,
            report_path=self.root / "ocr_report.json",
            device="cpu",
            overwrite=True,
            backend=_SubtitleOcrBackend(),
        )

        records = read_jsonl(output_path)
        self.assertEqual(records[0]["image_size"], [100, 50])
        result = adapt_feature_records(
            [self.candidate],
            ocr_records=records,
            object_records=(),
            config=FeatureAdapterConfig(ocr_persistence_candidates=1),
        )

        modality_counts = dict(result.report.modality_available_counts)
        self.assertEqual(modality_counts["ocr"], 1)
        self.assertEqual(modality_counts["objects"], 0)
        self.assertEqual(result.report.suppressed_ocr_subtitle_observations, 1)
        self.assertEqual(result.report.ocr_event_count, 0)
        self.assertEqual(result.protected_events, ())

    def test_inference_error_preserves_verified_image_size(self) -> None:
        output_path = self.root / "ocr_error_VIDEO.jsonl"
        report = run_ocr_file(
            self.metadata_path,
            output_path=output_path,
            report_path=self.root / "ocr_error_report.json",
            device="cpu",
            overwrite=True,
            backend=_FailingOcrBackend(),
        )

        records = read_jsonl(output_path)
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[0]["image_size"], [100, 50])


if __name__ == "__main__":
    unittest.main()
