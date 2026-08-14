from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from backend.app.services.ingestion.caption_pipeline import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    parse_caption_output,
    run_caption_file,
)
from backend.app.services.ingestion.run_caption import build_parser as build_caption_parser
from backend.app.services.ingestion.common import read_jsonl
from backend.app.services.ingestion.object_pipeline import (
    deterministic_class_id,
    normalize_detections,
    run_object_file,
)
from backend.app.services.ingestion.ocr_pipeline import (
    normalize_regions,
    normalize_text,
    run_ocr_file,
    unaccent_text,
)


def structured_caption(name: str) -> str:
    return json.dumps(
        {
            "scene": "studio",
            "people": [{"type": "person", "attributes": ["red shirt"]}],
            "objects": ["table"],
            "actions": ["standing"],
            "relationships": ["person beside table"],
            "colors": ["red"],
            "visible_text": [],
            "caption": f"a visible frame named {name}",
        }
    )


class FakeCaptionBackend:
    model_name = "fake-caption"
    model_version = "1"
    model_revision = "test-revision"

    def infer(self, paths: Sequence[Path]) -> Sequence[str]:
        return [structured_caption(path.stem) for path in paths]


class FailingCaptionBackend(FakeCaptionBackend):
    def infer(self, paths: Sequence[Path]) -> Sequence[str]:
        raise RuntimeError("synthetic model error")


class RevisedCaptionBackend(FakeCaptionBackend):
    model_revision = "new-revision"


class AlternateCaptionModelBackend(FakeCaptionBackend):
    model_name = "alternate-caption"


class FakeOcrBackend:
    model_name = "fake-ocr"
    model_version = "1"
    model_revision = "test-revision"

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, paths: Sequence[Path]) -> Sequence[Sequence[Any]]:
        self.calls += 1
        return [
            [
                (
                    [[0, 0], [20, 0], [20, 10], [0, 10]],
                    "  Xin   chào  ",
                    0.9,
                ),
                (
                    [[0, 12], [20, 12], [20, 20], [0, 20]],
                    "discard",
                    0.1,
                ),
            ]
            if path.stem == "one"
            else []
            for path in paths
        ]


class FakeObjectBackend:
    model_name = "fake-yoloe"
    model_version = "1"
    model_revision = "test-revision"
    prompt_mode = "text"
    vocabulary = ("person", "table")

    def infer(self, paths: Sequence[Path]) -> Sequence[Any]:
        return [
            {
                "image_size": [64, 32],
                "objects": [
                    {
                        "class_name": "person",
                        "confidence": 0.8,
                        "bbox_xyxy": [1, 2, 30, 31],
                    }
                ],
            }
            if path.stem == "one"
            else {"image_size": [64, 32], "objects": []}
            for path in paths
        ]


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class MultimodalIngestionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.one = self.root / "one.png"
        self.two = self.root / "two.png"
        self.broken = self.root / "broken.png"
        Image.new("RGB", (64, 32), color="red").save(self.one)
        Image.new("RGB", (64, 32), color="blue").save(self.two)
        self.broken.write_bytes(b"not an image")
        self.metadata = self.root / "keyframes_TEST.jsonl"
        self.records = [
            self._record("FRAME_1", self.one, 1.0, "SEG_1"),
            self._record("FRAME_2", self.two, 2.0, "SEG_1"),
            self._record("FRAME_3", self.broken, 3.0, "SEG_2"),
        ]
        write_jsonl(self.metadata, self.records)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _record(frame_id: str, path: Path, timestamp: float, segment_id: str) -> dict[str, Any]:
        return {
            "candidate_id": frame_id.replace("FRAME", "CANDIDATE"),
            "candidate_index": int(timestamp),
            "frame_id": frame_id,
            "video_id": "TEST",
            "segment_id": segment_id,
            "shot_id": segment_id.replace("SEG", "SHOT"),
            "timestamp": timestamp,
            "frame_index": int(timestamp * 10),
            "candidate_reasons": ["dense_interval"],
            "shot_start": timestamp - 0.5,
            "shot_end": timestamp + 0.5,
            "keyframe_path": str(path),
            "source_video_path": "source.mp4",
        }

    def test_caption_parser_valid_fenced_and_malformed(self) -> None:
        valid = parse_caption_output(structured_caption("valid"))
        self.assertEqual(valid["caption_parse_status"], "success")
        self.assertEqual(valid["structured_caption"]["colors"], ["red"])

        fenced = parse_caption_output(f"```json\n{structured_caption('fenced')}\n```")
        self.assertEqual(fenced["caption"], "a visible frame named fenced")
        self.assertEqual(fenced["caption_parse_status"], "success")

        malformed = parse_caption_output('{"caption": "useful fallback", broken}')
        self.assertEqual(malformed["caption"], "useful fallback")
        self.assertEqual(malformed["caption_parse_status"], "fallback")
        self.assertTrue(malformed["caption_parse_error"])

        plain = parse_caption_output("person standing beside a red car")
        self.assertEqual(plain["caption"], "person standing beside a red car")
        self.assertEqual(plain["structured_caption"], None)

    def test_caption_cli_uses_pinned_4b_defaults(self) -> None:
        args = build_caption_parser().parse_args(
            ["--metadata-path", str(self.metadata)]
        )
        self.assertEqual(DEFAULT_MODEL_NAME, "Qwen/Qwen3.5-4B")
        self.assertEqual(
            DEFAULT_MODEL_REVISION,
            "c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97",
        )
        self.assertEqual(args.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(args.model_revision, DEFAULT_MODEL_REVISION)

    def test_caption_batch_order_model_error_and_segment_determinism(self) -> None:
        output = self.root / "captions.jsonl"
        report = run_caption_file(
            self.metadata,
            output_path=output,
            report_path=self.root / "caption_report.json",
            device="cpu",
            batch_size=2,
            include_segment_caption=True,
            backend=FakeCaptionBackend(),
        )
        records = read_jsonl(output)
        self.assertEqual(report["success_count"], 2)
        self.assertEqual([item["frame_id"] for item in records], ["FRAME_1", "FRAME_2", "FRAME_3"])
        self.assertEqual(records[0]["caption"], "a visible frame named one")
        self.assertEqual(
            records[0]["segment_caption"],
            "a visible frame named one. a visible frame named two.",
        )
        self.assertEqual(records[2]["status"], "error")

        error_output = self.root / "caption_errors.jsonl"
        run_caption_file(
            self.metadata,
            output_path=error_output,
            report_path=self.root / "caption_error_report.json",
            device="cpu",
            backend=FailingCaptionBackend(),
        )
        failed = read_jsonl(error_output)
        self.assertTrue(all(item["status"] == "error" for item in failed))
        self.assertIn("synthetic model error", failed[0]["error"])

    def test_caption_revision_change_replaces_only_its_artifact(self) -> None:
        output = self.root / "captions.jsonl"
        report_path = self.root / "caption_report.json"
        run_caption_file(
            self.metadata,
            output_path=output,
            report_path=report_path,
            device="cpu",
            backend=FakeCaptionBackend(),
        )
        run_caption_file(
            self.metadata,
            output_path=output,
            report_path=report_path,
            device="cpu",
            backend=RevisedCaptionBackend(),
        )
        values = read_jsonl(output)
        self.assertEqual(len(values), 3)
        self.assertEqual({item["model_revision"] for item in values}, {"new-revision"})

    def test_caption_model_change_replaces_only_its_artifact(self) -> None:
        output = self.root / "captions.jsonl"
        report_path = self.root / "caption_report.json"
        run_caption_file(
            self.metadata,
            output_path=output,
            report_path=report_path,
            device="cpu",
            backend=FakeCaptionBackend(),
        )
        run_caption_file(
            self.metadata,
            output_path=output,
            report_path=report_path,
            device="cpu",
            backend=AlternateCaptionModelBackend(),
        )
        values = read_jsonl(output)
        self.assertEqual(len(values), 3)
        self.assertEqual(
            {item["model_name"] for item in values},
            {"alternate-caption"},
        )

    def test_ppocr_normalization_geometry_unicode_threshold_and_empty(self) -> None:
        regions = normalize_regions(
            [
                ([[0, 0], [10, 0], [10, 5], [0, 5]], "  Cà   phê Việt  ", 0.95),
                ([[0, 6], [10, 6], [10, 9], [0, 9]], "weak", 0.1),
            ],
            0.3,
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["polygon"], [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
        self.assertEqual(regions[0]["text"], "Cà phê Việt")
        self.assertEqual(regions[0]["unaccented_text"], "Ca phe Viet")
        self.assertEqual(regions[0]["language"], "vi")
        self.assertEqual(normalize_regions([], 0.3), [])
        self.assertEqual(normalize_text("a\x00  b"), "a b")
        self.assertEqual(unaccent_text("Đường phố"), "Duong pho")

    def test_ocr_pipeline_image_size_and_resume_provenance(self) -> None:
        output = self.root / "ocr.jsonl"
        backend = FakeOcrBackend()
        first = run_ocr_file(
            self.metadata,
            output_path=output,
            report_path=self.root / "ocr_report.json",
            device="cpu",
            batch_size=2,
            backend=backend,
        )
        values = read_jsonl(output)
        self.assertEqual(first["success_count"], 2)
        self.assertEqual(values[0]["ocr_text"], "Xin chào")
        self.assertEqual(values[0]["ocr_text_unaccented"], "Xin chao")
        self.assertEqual(values[0]["image_size"], [64, 32])
        calls = backend.calls
        resumed = run_ocr_file(
            self.metadata,
            output_path=output,
            report_path=self.root / "ocr_report.json",
            device="cpu",
            backend=backend,
        )
        self.assertEqual(backend.calls, calls)
        self.assertEqual(resumed["skipped_count"], 3)

    def test_yoloe_normalization_bbox_confidence_and_identity(self) -> None:
        first, size = normalize_detections(
            {
                "image_size": [100, 50],
                "objects": [
                    {
                        "class_name": "traffic light",
                        "confidence": 0.87,
                        "bbox_xyxy": [1, 2, 30, 40],
                    }
                ],
            }
        )
        second, _ = normalize_detections(
            {
                "objects": [
                    {
                        "class_name": "traffic light",
                        "confidence": 0.5,
                        "bbox_xyxy": [0, 0, 1, 1],
                    }
                ]
            }
        )
        self.assertEqual(size, [100, 50])
        self.assertEqual(first[0]["bbox_xyxy"], [1.0, 2.0, 30.0, 40.0])
        self.assertEqual(first[0]["class_id"], second[0]["class_id"])
        self.assertEqual(first[0]["class_id"], deterministic_class_id("traffic light"))

        output = self.root / "objects.jsonl"
        run_object_file(
            self.metadata,
            output_path=output,
            report_path=self.root / "objects_report.json",
            device="cpu",
            backend=FakeObjectBackend(),
        )
        values = read_jsonl(output)
        self.assertEqual(values[0]["object_counts"], {"person": 1})
        self.assertTrue(values[0]["evidence_only"])
        self.assertEqual(values[1]["objects"], [])

    def test_imports_are_lazy_and_cli_help_has_no_audio_options(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        code = (
            "import sys; "
            "import backend.app.services.ingestion.caption_pipeline; "
            "import backend.app.services.ingestion.ocr_pipeline; "
            "import backend.app.services.ingestion.object_pipeline; "
            "assert 'transformers' not in sys.modules; "
            "assert 'paddleocr' not in sys.modules; "
            "assert 'ultralytics' not in sys.modules"
        )
        imported = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        for script in ("run_caption.py", "run_ocr.py", "run_object_detection.py"):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(repo_root / "backend/app/services/ingestion" / script),
                    "--help",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
