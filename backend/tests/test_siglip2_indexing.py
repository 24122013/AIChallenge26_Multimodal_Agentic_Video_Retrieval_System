from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from backend.app.services.indexing.build_siglip2_index import (
    DEFAULT_MODEL_NAME,
    encode_keyframes,
    load_siglip2_model_processor,
    tune_batch_size,
)


class FakeConfig:
    _commit_hash = "fake-commit"


class FakeProcessor:
    def __init__(self) -> None:
        self.image_processor = type("ImageProcessor", (), {"size": {"height": 384}})()
        self.seen_modes: list[list[str]] = []

    def __call__(self, *, images, return_tensors: str):
        if return_tensors != "pt":
            raise AssertionError(return_tensors)
        self.seen_modes.append([image.mode for image in images])
        values = [
            float(np.asarray(image, dtype=np.float32).mean()) / 255.0
            for image in images
        ]
        return {"pixel_values": torch.tensor(values, dtype=torch.float32).reshape(-1, 1)}


class FakeSiglip2Model:
    def __init__(self) -> None:
        self.config = FakeConfig()
        self.get_image_features_calls = 0
        self.eval_called = False

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.get_image_features_calls += 1
        value = pixel_values.reshape(-1, 1)
        return torch.cat([value + 1.0, value + 2.0, value + 3.0], dim=1)


class FakeInput:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size


class OomProcessor:
    def __call__(self, *, images, return_tensors: str):
        return {"pixel_values": FakeInput(len(images))}


class OomModel(FakeSiglip2Model):
    def get_image_features(self, pixel_values: FakeInput) -> torch.Tensor:
        if pixel_values.batch_size >= 4:
            raise RuntimeError("CUDA out of memory (simulated)")
        return torch.ones((pixel_values.batch_size, 3), dtype=torch.float32)


def make_record(path: Path, number: int) -> dict:
    return {
        "frame_id": f"FRAME_L01_V001_{number:06d}",
        "video_id": "L01_V001",
        "shot_id": f"SHOT_L01_V001_{number:06d}",
        "segment_id": f"SEG_L01_V001_{number:06d}",
        "timestamp": float(number),
        "frame_index": number,
        "keyframe_path": path.as_posix(),
        "candidate_index": number,
        "candidate_id": f"CANDIDATE_L01_V001_{number:09d}",
        "candidate_reasons": ["dense_interval"],
        "keyframe_strategy": "dense_coverage",
        "selection_phase": "coverage_fill",
        "selection_rank": number,
        "selection_reasons": ["temporal_coverage"],
        "covered_event_ids": [],
        "selection_score": None,
        "protected": False,
        "coverage_added": True,
    }


class Siglip2IndexingTest(unittest.TestCase):
    def test_cuda_loader_materializes_model_directly_in_inference_dtype(self) -> None:
        calls: dict[str, dict] = {}

        class FakeLoadedModel:
            def to(self, device: str):
                calls["to"] = {"device": device}
                return self

            def eval(self):
                calls["eval"] = {}
                return self

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(_name: str, **kwargs):
                calls["model"] = kwargs
                return FakeLoadedModel()

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(_name: str, **kwargs):
                calls["processor"] = kwargs
                return object()

        fake_transformers = SimpleNamespace(
            AutoModel=FakeAutoModel,
            AutoProcessor=FakeAutoProcessor,
        )
        with (
            patch.dict(sys.modules, {"transformers": fake_transformers}),
            patch(
                "backend.app.services.indexing.build_siglip2_index.compute_dtype_for",
                return_value=torch.bfloat16,
            ),
        ):
            load_siglip2_model_processor(
                model_name=DEFAULT_MODEL_NAME,
                model_revision=None,
                device="cuda",
                model_cache_dir=None,
                use_autocast=True,
            )

        self.assertIs(calls["model"]["dtype"], torch.bfloat16)
        self.assertNotIn("dtype", calls["processor"])
        self.assertEqual(calls["to"]["device"], "cuda")
        self.assertIn("eval", calls)

    def test_fake_encoder_normalizes_and_skips_bad_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            broken = root / "broken.png"
            third = root / "third.png"
            Image.new("L", (16, 16), color=64).save(first)
            broken.write_bytes(b"not an image")
            Image.new("RGBA", (16, 16), color=(128, 0, 0, 255)).save(third)

            model = FakeSiglip2Model()
            processor = FakeProcessor()
            embeddings, metadata, skipped, benchmark = encode_keyframes(
                records=[
                    make_record(first, 1),
                    make_record(broken, 2),
                    make_record(third, 3),
                ],
                batch_size=3,
                num_workers=0,
                device="cpu",
                model=model,
                processor=processor,
            )

            self.assertTrue(model.eval_called)
            self.assertGreater(model.get_image_features_calls, 0)
            self.assertTrue(all(mode == "RGB" for batch in processor.seen_modes for mode in batch))
            self.assertEqual(embeddings.shape, (2, 3))
            self.assertEqual(embeddings.dtype, np.float32)
            np.testing.assert_allclose(
                np.linalg.norm(embeddings, axis=1),
                np.ones(2, dtype=np.float32),
                atol=1e-6,
            )
            self.assertEqual(len(metadata), 2)
            self.assertEqual([record["embedding_index"] for record in metadata], [0, 1])
            self.assertEqual({record["vector_dim"] for record in metadata}, {3})
            self.assertEqual(
                {record["model_name"] for record in metadata}, {DEFAULT_MODEL_NAME}
            )
            self.assertTrue(all(record["normalized"] is True for record in metadata))
            self.assertEqual(
                [record["candidate_id"] for record in metadata],
                ["CANDIDATE_L01_V001_000000001", "CANDIDATE_L01_V001_000000003"],
            )
            self.assertTrue(
                all(
                    record["keyframe_strategy"] == "dense_coverage"
                    for record in metadata
                )
            )
            self.assertTrue(
                all(record["coverage_added"] is True for record in metadata)
            )
            self.assertEqual(
                [record["candidate_index"] for record in metadata],
                [1, 3],
            )
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["skip_reason"], "image_load_error")
            self.assertEqual(benchmark["embedding_shape"], [2, 3])
            self.assertEqual(benchmark["output_dtype"], "float32")

    def test_auto_batch_falls_back_after_simulated_cuda_oom(self) -> None:
        sample = [Image.new("RGB", (8, 8), color="white")]
        with (
            patch(
                "backend.app.services.indexing.build_siglip2_index.synchronize_cuda"
            ),
            patch(
                "backend.app.services.indexing.build_siglip2_index.clear_cuda_cache"
            ) as clear_cache,
        ):
            selected, results = tune_batch_size(
                model=OomModel(),
                processor=OomProcessor(),
                sample_images=sample,
                input_record_count=16,
                device="cuda",
                use_autocast=False,
                compute_dtype=torch.float32,
            )

        self.assertIn(selected, {1, 2})
        self.assertEqual(results[-1]["batch_size"], 4)
        self.assertEqual(results[-1]["status"], "cuda_oom")
        clear_cache.assert_called_once_with("cuda")


if __name__ == "__main__":
    unittest.main()
