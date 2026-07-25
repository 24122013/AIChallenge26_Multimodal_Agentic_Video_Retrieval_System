from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.app.services.indexing.build_siglip2_index import (
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    encode_keyframes,
)


@unittest.skipUnless(
    os.getenv("RUN_SIGLIP2_SMOKE") == "1",
    "Set RUN_SIGLIP2_SMOKE=1 to run the cached real-checkpoint smoke test",
)
class Siglip2CachedSmokeTest(unittest.TestCase):
    def test_two_images_with_cached_checkpoint(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            self.skipTest("transformers is not installed")
        if not DEFAULT_MODEL_CACHE_DIR.exists():
            self.skipTest(f"model cache does not exist: {DEFAULT_MODEL_CACHE_DIR}")

        try:
            model = AutoModel.from_pretrained(
                DEFAULT_MODEL_NAME,
                cache_dir=DEFAULT_MODEL_CACHE_DIR.as_posix(),
                local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(
                DEFAULT_MODEL_NAME,
                cache_dir=DEFAULT_MODEL_CACHE_DIR.as_posix(),
                local_files_only=True,
            )
        except (OSError, ValueError) as exc:
            self.skipTest(f"SigLIP2 checkpoint is not fully cached: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = []
            for index, color in enumerate(("red", "blue"), start=1):
                image_path = root / f"{index}.png"
                Image.new("RGB", (64, 64), color=color).save(image_path)
                records.append(
                    {
                        "frame_id": f"FRAME_SMOKE_{index:06d}",
                        "video_id": "SMOKE",
                        "shot_id": f"SHOT_SMOKE_{index:06d}",
                        "segment_id": f"SEG_SMOKE_{index:06d}",
                        "timestamp": float(index),
                        "keyframe_path": image_path.as_posix(),
                    }
                )

            embeddings, metadata, skipped, _ = encode_keyframes(
                records,
                batch_size=2,
                device="cpu",
                model=model,
                processor=processor,
            )
            self.assertEqual(embeddings.shape[0], 2)
            self.assertEqual(embeddings.shape[1], metadata[0]["vector_dim"])
            self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
