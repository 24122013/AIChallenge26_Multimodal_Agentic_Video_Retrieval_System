from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from backend.app.services.ingestion.asr_pipeline import (
    map_transcript_to_segments,
    run_asr_file,
)
from backend.app.services.ingestion.caption_pipeline import (
    BlipCaptionBackend,
    run_caption_file,
)
from backend.app.services.ingestion.common import read_jsonl
from backend.app.services.ingestion.object_pipeline import run_object_file
from backend.app.services.ingestion.ocr_pipeline import run_ocr_file


class FakeCaptionBackend:
    model_name = "fake-caption"
    model_version = "1"

    def infer(self, paths: Sequence[Path]) -> Sequence[str]:
        return [f"a visible frame named {path.stem}" for path in paths]


class FakeBlipProcessor:
    def __init__(self) -> None:
        self.inputs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        import torch

        self.inputs = kwargs
        return {
            "pixel_values": torch.zeros(
                (len(kwargs["images"]), 3, 2, 2),
                dtype=torch.float32,
            )
        }

    def batch_decode(
        self,
        tokens: Any,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        self.skip_special_tokens = skip_special_tokens
        return ["a red object", "a blue object"]


class FakeBlipModel:
    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.inputs = kwargs
        return [[1], [2]]


class FakeOcrBackend:
    model_name = "fake-ocr"
    model_version = "1"

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, paths: Sequence[Path]) -> Sequence[Sequence[Any]]:
        self.calls += 1
        values: list[Sequence[Any]] = []
        for path in paths:
            if path.stem == "one":
                values.append(
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
                )
            else:
                values.append([])
        return values


class FakeObjectBackend:
    model_name = "fake-yolo"
    model_version = "1"

    def infer(self, paths: Sequence[Path]) -> Sequence[Any]:
        values: list[Any] = []
        for path in paths:
            if path.stem == "one":
                values.append(
                    {
                        "image_size": [64, 32],
                        "objects": [
                            {
                                "class_id": 0,
                                "class_name": "person",
                                "confidence": 0.8,
                                "bbox_xyxy": [1, 2, 30, 31],
                            }
                        ],
                    }
                )
            else:
                values.append({"image_size": [64, 32], "objects": []})
        return values


class FakeAsrBackend:
    model_name = "fake-whisper"
    model_version = "1"

    def transcribe(self, video_path: Path) -> tuple[list[dict[str, Any]], str]:
        return (
            [
                {
                    "start": 0.5,
                    "end": 1.5,
                    "text": "hello",
                    "language": "en",
                    "confidence": 0.9,
                },
                {
                    "start": 2.1,
                    "end": 2.8,
                    "text": "world",
                    "language": "en",
                    "confidence": 0.8,
                },
            ],
            "en",
        )


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
        self.three = self.root / "three.png"
        self.four = self.root / "four.png"
        self.five = self.root / "five.png"
        self.broken = self.root / "broken.png"
        Image.new("RGB", (64, 32), color="red").save(self.one)
        Image.new("RGB", (64, 32), color="blue").save(self.two)
        Image.new("RGB", (64, 32), color="green").save(self.three)
        Image.new("RGB", (64, 32), color="yellow").save(self.four)
        Image.new("RGB", (64, 32), color="purple").save(self.five)
        self.broken.write_bytes(b"not an image")
        self.metadata = self.root / "keyframes_TEST.jsonl"
        self.records = [
            self._record("FRAME_1", self.one, 1.0, 0.0, 2.0, "SEG_1"),
            self._record("FRAME_2", self.two, 2.5, 2.0, 3.0, "SEG_2"),
            self._record("FRAME_3", self.three, 4.0, 3.0, 5.0, "SEG_3"),
            self._record("FRAME_4", self.four, 5.5, 5.0, 6.0, "SEG_4"),
            self._record("FRAME_5", self.five, 6.5, 6.0, 7.0, "SEG_5"),
            self._record("FRAME_6", self.broken, 8.0, 7.0, 9.0, "SEG_6"),
        ]
        write_jsonl(self.metadata, self.records)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _record(
        frame_id: str,
        path: Path,
        timestamp: float,
        start: float,
        end: float,
        segment_id: str,
    ) -> dict[str, Any]:
        return {
            "frame_id": frame_id,
            "video_id": "TEST",
            "segment_id": segment_id,
            "shot_id": segment_id.replace("SEG", "SHOT"),
            "timestamp": timestamp,
            "shot_start": start,
            "shot_end": end,
            "keyframe_path": str(path),
            "source_video_path": "source.mp4",
        }

    def test_metadata_reader_preserves_ids_timestamps_and_rejects_bad_json(self) -> None:
        loaded = read_jsonl(self.metadata)
        self.assertEqual(loaded[0]["frame_id"], "FRAME_1")
        self.assertEqual(loaded[0]["timestamp"], 1.0)
        invalid = self.root / "invalid.jsonl"
        invalid.write_text('{"ok": 1}\n{broken}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            read_jsonl(invalid)

    def test_caption_ocr_objects_write_valid_jsonl_and_isolate_bad_image(self) -> None:
        caption_path = self.root / "captions.jsonl"
        run_caption_file(
            self.metadata,
            output_path=caption_path,
            report_path=self.root / "caption_report.json",
            device="cpu",
            batch_size=2,
            include_segment_caption=True,
            backend=FakeCaptionBackend(),
        )
        captions = read_jsonl(caption_path)
        self.assertEqual(
            [value["status"] for value in captions],
            ["success", "success", "success", "success", "success", "error"],
        )
        self.assertEqual(captions[0]["frame_id"], self.records[0]["frame_id"])
        self.assertEqual(captions[0]["timestamp"], self.records[0]["timestamp"])
        self.assertTrue(captions[0]["segment_caption"])

        ocr_path = self.root / "ocr.jsonl"
        fake_ocr = FakeOcrBackend()
        first_report = run_ocr_file(
            self.metadata,
            output_path=ocr_path,
            report_path=self.root / "ocr_report.json",
            device="cpu",
            batch_size=2,
            conf_threshold=0.3,
            backend=fake_ocr,
        )
        ocr = read_jsonl(ocr_path)
        self.assertEqual(first_report["success_count"], 5)
        self.assertEqual(ocr[0]["ocr_text"], "Xin chào")
        self.assertEqual(len(ocr[0]["text_regions"]), 1)
        self.assertEqual(ocr[1]["ocr_text"], "")
        self.assertEqual(ocr[1]["status"], "success")
        self.assertEqual(ocr[5]["status"], "error")

        calls_before_resume = fake_ocr.calls
        resumed = run_ocr_file(
            self.metadata,
            output_path=ocr_path,
            report_path=self.root / "ocr_report.json",
            device="cpu",
            backend=fake_ocr,
        )
        self.assertEqual(fake_ocr.calls, calls_before_resume)
        self.assertEqual(resumed["skipped_count"], 6)
        self.assertEqual(len(read_jsonl(ocr_path)), 6)

        object_path = self.root / "objects.jsonl"
        run_object_file(
            self.metadata,
            output_path=object_path,
            report_path=self.root / "objects_report.json",
            device="cpu",
            batch_size=2,
            backend=FakeObjectBackend(),
        )
        objects = read_jsonl(object_path)
        self.assertEqual(objects[0]["class_counts"], {"person": 1})
        self.assertEqual(objects[1]["objects"], [])
        self.assertEqual(objects[1]["status"], "success")
        self.assertEqual(objects[5]["status"], "error")

    def test_blip_caption_backend_does_not_send_instruction_prompt(self) -> None:
        processor = FakeBlipProcessor()
        model = FakeBlipModel()
        backend = BlipCaptionBackend(device="cpu")
        backend._processor = processor
        backend._model = model

        values = backend.infer([self.one, self.two])

        self.assertEqual(values, ["a red object", "a blue object"])
        self.assertNotIn("text", processor.inputs)
        self.assertTrue(processor.skip_special_tokens)

    def test_asr_mapping_uses_only_overlapping_time_ranges(self) -> None:
        transcript = FakeAsrBackend().transcribe(Path("unused"))[0]
        mapped = map_transcript_to_segments(self.records[:2], transcript)
        self.assertEqual(mapped[0]["transcript_text"], "hello")
        self.assertEqual(mapped[1]["transcript_text"], "world")
        self.assertNotIn("world", mapped[0]["transcript_text"])
        self.assertEqual(mapped[0]["frame_ids"], ["FRAME_1"])

    def test_video_without_audio_is_skipped_not_failed(self) -> None:
        video = self.root / "silent.mp4"
        video.write_bytes(b"fake")
        output = self.root / "asr_silent.jsonl"
        result = run_asr_file(
            video,
            output_path=output,
            segments_output_path=self.root / "asr_segments_silent.jsonl",
            report_path=self.root / "asr_silent_report.json",
            device="cpu",
            backend=FakeAsrBackend(),
            audio_probe=lambda _: False,
        )
        value = read_jsonl(output)[0]
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(value["status"], "skipped")
        self.assertEqual(value["skip_reason"], "no_audio_stream")

    def test_asr_writes_timeline_and_segment_mapping(self) -> None:
        video = self.root / "TEST.mp4"
        video.write_bytes(b"fake")
        timeline = self.root / "asr_TEST.jsonl"
        mapped_path = self.root / "asr_segments_TEST.jsonl"
        result = run_asr_file(
            video,
            metadata_path=self.metadata,
            output_path=timeline,
            segments_output_path=mapped_path,
            report_path=self.root / "asr_TEST_report.json",
            device="cpu",
            backend=FakeAsrBackend(),
            audio_probe=lambda _: True,
        )
        self.assertEqual(result["success_count"], 2)
        values = read_jsonl(timeline)
        self.assertEqual(values[0]["language"], "en")
        mapped = read_jsonl(mapped_path)
        self.assertEqual(mapped[0]["transcript_text"], "hello")
        self.assertEqual(mapped[1]["transcript_text"], "world")

    def test_imports_do_not_load_model_packages_and_cli_help_works(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        code = (
            "import sys; "
            "import backend.app.services.ingestion.caption_pipeline; "
            "import backend.app.services.ingestion.ocr_pipeline; "
            "import backend.app.services.ingestion.object_pipeline; "
            "import backend.app.services.ingestion.asr_pipeline; "
            "assert 'easyocr' not in sys.modules; "
            "assert 'ultralytics' not in sys.modules; "
            "assert 'faster_whisper' not in sys.modules; "
            "assert 'whisper' not in sys.modules"
        )
        imported = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        for script in ("run_caption.py", "run_ocr.py", "run_object_detection.py", "run_asr.py"):
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
