from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from backend.app.api import video as video_api


@unittest.skipIf(video_api.video_router is None, "FastAPI is not installed")
class VideoApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.selected_root = self.root / "keyframes"
        self.dense_root = self.root / "dense_keyframes"
        self.video_id = "L27_V001"
        self.config = SimpleNamespace(
            RETRIEVAL_KEYFRAME_ROOT=str(self.selected_root),
            RETRIEVAL_DENSE_KEYFRAME_ROOT=str(self.dense_root),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_frame(self, root: Path, frame_id: str, content: bytes) -> Path:
        frame_path = root / self.video_id / f"{frame_id}.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(content)
        return frame_path

    def test_frame_image_prefers_selected_keyframe(self) -> None:
        frame_id = "FRAME_L27_V001_000013572"
        selected_path = self._write_frame(self.selected_root, frame_id, b"selected")
        self._write_frame(self.dense_root, frame_id, b"dense")

        with mock.patch.object(video_api, "STREAM_CONFIG", self.config):
            response = video_api.get_frame_image(self.video_id, frame_id)

        self.assertEqual(Path(response.path), selected_path)
        self.assertEqual(response.media_type, "image/jpeg")

    def test_frame_image_falls_back_to_dense_candidate(self) -> None:
        frame_id = "FRAME_L27_V001_000009739"
        dense_path = self._write_frame(self.dense_root, frame_id, b"dense")

        with mock.patch.object(video_api, "STREAM_CONFIG", self.config):
            response = video_api.get_frame_image(self.video_id, frame_id)

        self.assertEqual(Path(response.path), dense_path)
        self.assertEqual(response.media_type, "image/jpeg")

    def test_frame_image_returns_404_when_both_files_are_missing(self) -> None:
        with mock.patch.object(video_api, "STREAM_CONFIG", self.config):
            with self.assertRaises(HTTPException) as raised:
                video_api.get_frame_image(
                    self.video_id,
                    "FRAME_L27_V001_999999999",
                )

        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
