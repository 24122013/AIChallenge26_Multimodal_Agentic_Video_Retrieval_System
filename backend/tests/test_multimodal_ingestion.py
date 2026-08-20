from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence
from unittest import mock

import numpy as np
from PIL import Image

from backend.app.services.ingestion.caption_pipeline import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TASK_PROMPT,
    FlorenceCaptionBackend,
    FlorenceCaptionOutput,
    parse_caption_output,
    run_caption_file,
)
from backend.app.services.ingestion.run_caption import build_parser as build_caption_parser
from backend.app.services.ingestion.common import read_jsonl, resumable_ids
from backend.app.services.ingestion.object_pipeline import (
    YoloEBackend,
    deterministic_class_id,
    normalize_detections,
    run_object_file,
)
from backend.app.services.ingestion.ocr_pipeline import (
    PaddleOcrBackend,
    normalize_regions,
    normalize_text,
    run_ocr_file,
    unaccent_text,
)


def florence_caption(name: str) -> FlorenceCaptionOutput:
    return FlorenceCaptionOutput(
        caption=f"a visible frame named {name}",
        raw_output=f"<s>a visible frame named {name}</s>",
    )


class FakeCaptionBackend:
    model_name = "fake-caption"
    model_version = "1"
    model_revision = "test-revision"

    def infer(self, paths: Sequence[Path]) -> Sequence[FlorenceCaptionOutput]:
        return [florence_caption(path.stem) for path in paths]


class FailingCaptionBackend(FakeCaptionBackend):
    def infer(self, paths: Sequence[Path]) -> Sequence[str]:
        raise RuntimeError("synthetic model error")


class RevisedCaptionBackend(FakeCaptionBackend):
    model_revision = "new-revision"


class AlternateCaptionModelBackend(FakeCaptionBackend):
    model_name = "alternate-caption"


class CountingCaptionBackend(FakeCaptionBackend):
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, paths: Sequence[Path]) -> Sequence[FlorenceCaptionOutput]:
        self.calls += 1
        return super().infer(paths)


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
        Image.new("RGB", (80, 48), color="blue").save(self.two)
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

    def test_florence_caption_adapter_is_nonempty_and_schema_compatible(self) -> None:
        valid = parse_caption_output(florence_caption("valid"))
        self.assertEqual(valid["caption_parse_status"], "success")
        self.assertEqual(valid["structured_caption"], None)
        self.assertEqual(valid["raw_caption_output"], "<s>a visible frame named valid</s>")

        plain = parse_caption_output("person standing beside a red car")
        self.assertEqual(plain["caption"], "person standing beside a red car")
        self.assertEqual(plain["structured_caption"], None)
        self.assertRaises(ValueError, parse_caption_output, "   ")

    def test_caption_cli_uses_pinned_florence_2_defaults(self) -> None:
        args = build_caption_parser().parse_args(
            ["--metadata-path", str(self.metadata)]
        )
        self.assertEqual(DEFAULT_MODEL_NAME, "florence-community/Florence-2-base-ft")
        self.assertEqual(
            DEFAULT_MODEL_REVISION,
            "0b03b6f15a4a211370fb204aee4e7dd48887ea37",
        )
        self.assertEqual(args.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(args.model_revision, DEFAULT_MODEL_REVISION)
        self.assertEqual(args.task_prompt, "<MORE_DETAILED_CAPTION>")
        legacy = build_caption_parser().parse_args(
            ["--metadata-path", str(self.metadata), "--prompt", "<CAPTION>"]
        )
        self.assertEqual(legacy.task_prompt, "<CAPTION>")

    def test_florence_loader_and_batch_postprocessing_without_download(self) -> None:
        import torch

        transformers = ModuleType("transformers")
        model_calls: list[tuple[str, dict[str, Any]]] = []
        processor_calls: list[tuple[str, dict[str, Any]]] = []
        processor_inputs: list[dict[str, Any]] = []
        post_process_calls: list[tuple[str, str, tuple[int, int]]] = []
        generate_calls: list[dict[str, Any]] = []

        class FakeConfig:
            _commit_hash = DEFAULT_MODEL_REVISION

        class FakeModel:
            config = FakeConfig()

            def to(self, device: str) -> "FakeModel":
                self.device = device
                return self

            def eval(self) -> None:
                return None

            def generate(self, **kwargs: Any) -> Any:
                generate_calls.append(kwargs)
                return torch.tensor([[1, 2], [3, 4]])

        class FakeModelFactory:
            @classmethod
            def from_pretrained(cls, name: str, **kwargs: Any) -> FakeModel:
                model_calls.append((name, kwargs))
                return FakeModel()

        class FakeProcessorFactory:
            @classmethod
            def from_pretrained(cls, name: str, **kwargs: Any) -> "FakeProcessorFactory":
                processor_calls.append((name, kwargs))
                return cls()

            def __call__(self, **kwargs: Any) -> dict[str, Any]:
                processor_inputs.append(kwargs)
                return {
                    "input_ids": torch.tensor([[1], [1]]),
                    "pixel_values": torch.zeros((2, 3, 2, 2)),
                }

            def batch_decode(self, tokens: Any, **kwargs: Any) -> list[str]:
                self.decode_kwargs = kwargs
                return ["decoded first", "decoded second"]

            def post_process_generation(
                self,
                text: str,
                *,
                task: str,
                image_size: tuple[int, int],
            ) -> dict[str, str]:
                post_process_calls.append((text, task, image_size))
                return {task: f"caption for {image_size[0]}x{image_size[1]}"}

        transformers.AutoModelForImageTextToText = FakeModelFactory  # type: ignore[attr-defined]
        transformers.AutoProcessor = FakeProcessorFactory  # type: ignore[attr-defined]
        backend = FlorenceCaptionBackend(device="cpu", cache_dir=self.root / "caption-cache")
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            outputs = backend.infer([self.one, self.two])

        self.assertEqual(model_calls[0][0], DEFAULT_MODEL_NAME)
        self.assertEqual(processor_calls[0][0], DEFAULT_MODEL_NAME)
        self.assertEqual(model_calls[0][1]["revision"], DEFAULT_MODEL_REVISION)
        self.assertIs(model_calls[0][1]["trust_remote_code"], False)
        self.assertIs(processor_calls[0][1]["trust_remote_code"], False)
        self.assertEqual(processor_inputs[0]["text"], [DEFAULT_TASK_PROMPT] * 2)
        self.assertEqual(processor_inputs[0]["images"][0].size, (64, 32))
        self.assertIs(processor_inputs[0]["padding"], True)
        self.assertEqual(
            post_process_calls,
            [
                ("decoded first", DEFAULT_TASK_PROMPT, (64, 32)),
                ("decoded second", DEFAULT_TASK_PROMPT, (80, 48)),
            ],
        )
        self.assertEqual(
            [item.caption for item in outputs],
            ["caption for 64x32", "caption for 80x48"],
        )
        self.assertIs(generate_calls[0]["do_sample"], False)
        self.assertEqual(generate_calls[0]["num_beams"], 3)
        with self.assertRaises(ValueError):
            processor_inputs[0]["images"][0].getpixel((0, 0))

    def test_florence_rejects_untested_quantization(self) -> None:
        with self.assertRaisesRegex(ValueError, "not supported or tested"):
            FlorenceCaptionBackend(device="cuda", quantization="4bit")

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

    def test_caption_resume_skips_compatible_success_records(self) -> None:
        metadata = self.root / "keyframes_SUCCESS.jsonl"
        write_jsonl(metadata, self.records[:2])
        output = self.root / "captions_success.jsonl"
        report_path = self.root / "caption_success_report.json"
        backend = CountingCaptionBackend()
        first = run_caption_file(
            metadata,
            output_path=output,
            report_path=report_path,
            device="cpu",
            backend=backend,
        )
        calls = backend.calls
        resumed = run_caption_file(
            metadata,
            output_path=output,
            report_path=report_path,
            device="cpu",
            backend=backend,
        )

        self.assertEqual(first["success_count"], 2)
        self.assertEqual(resumed["skipped_count"], 2)
        self.assertEqual(backend.calls, calls)
        self.assertEqual(len(read_jsonl(output)), 2)

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

    def test_ppocr_normalization_accepts_numpy_polygon(self) -> None:
        regions = normalize_regions(
            {
                "res": {
                    "rec_polys": [
                        np.asarray(
                            [[0, 0], [10, 0], [10, 5], [0, 5]],
                            dtype=np.float32,
                        )
                    ],
                    "rec_texts": ["screen text"],
                    "rec_scores": np.asarray([0.95], dtype=np.float32),
                }
            },
            0.3,
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(
            regions[0]["polygon"],
            [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
        )
        self.assertEqual(regions[0]["text"], "screen text")

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
        self.assertGreater(backend.calls, calls)
        self.assertEqual(resumed["skipped_count"], 0)
        self.assertEqual(resumed["error_count"], 1)

    def test_resume_invalidates_compatible_error_records(self) -> None:
        output = self.root / "failed.jsonl"
        output.write_text(
            json.dumps(
                {
                    "frame_id": "FRAME_1",
                    "model_name": "test-model",
                    "model_revision": "revision-1",
                    "status": "error",
                    "error": "synthetic inference failure",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        processed, stale = resumable_ids(
            output,
            "frame_id",
            model_name="test-model",
            model_revision="revision-1",
        )

        self.assertEqual(processed, set())
        self.assertTrue(stale)

    def test_paddle_backend_explicitly_disables_mkldnn(self) -> None:
        captured: dict[str, Any] = {}
        paddleocr = ModuleType("paddleocr")

        def fake_paddle_ocr(**kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

        paddleocr.PaddleOCR = fake_paddle_ocr  # type: ignore[attr-defined]
        backend = PaddleOcrBackend(device="cuda", cache_dir=self.root / "ocr-cache")
        with (
            mock.patch.dict(sys.modules, {"paddleocr": paddleocr}),
            mock.patch.dict(
                os.environ,
                {
                    "PADDLE_PDX_CACHE_HOME": "stale-cache",
                    "PADDLE_HOME": "stale-cache",
                },
            ),
        ):
            backend._load()
            expected_cache = str((self.root / "ocr-cache").resolve())
            self.assertEqual(os.environ["PADDLE_PDX_CACHE_HOME"], expected_cache)
            self.assertEqual(os.environ["PADDLE_HOME"], expected_cache)

        self.assertEqual(captured["device"], "gpu:0")
        self.assertIs(captured["enable_mkldnn"], False)

    def test_yoloe_backend_uses_explicit_cache_path_on_first_load(self) -> None:
        model_refs: list[str] = []
        configured_classes: list[list[str]] = []
        ultralytics = ModuleType("ultralytics")

        class FakeYoloE:
            def __init__(self, model_ref: str) -> None:
                model_refs.append(model_ref)

            def set_classes(self, classes: list[str]) -> None:
                configured_classes.append(classes)

        ultralytics.YOLOE = FakeYoloE  # type: ignore[attr-defined]
        ultralytics.settings = {}  # type: ignore[attr-defined]
        cache_dir = self.root / "objects-cache"
        backend = YoloEBackend(cache_dir=cache_dir, vocabulary=("person", "bus"))
        with mock.patch.dict(sys.modules, {"ultralytics": ultralytics}):
            backend._load()

        self.assertEqual(model_refs, [str((cache_dir / backend.model_name).resolve())])
        self.assertEqual(configured_classes, [["person", "bus"]])

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

        numpy_bbox, _ = normalize_detections(
            {
                "objects": [
                    {
                        "class_name": "screen",
                        "confidence": 0.75,
                        "bbox_xyxy": np.asarray([2, 3, 20, 30], dtype=np.float32),
                    }
                ]
            }
        )
        self.assertEqual(numpy_bbox[0]["bbox_xyxy"], [2.0, 3.0, 20.0, 30.0])

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
