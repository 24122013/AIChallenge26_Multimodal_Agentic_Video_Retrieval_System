from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.app.services.metadata.metadata_store import MetadataStore
import backend.app.services.indexing.extract_keyframes as keyframe_extractor
from backend.app.services.indexing.extract_keyframes import (
    Shot,
    extract_keyframes_for_video,
    select_frame_indices,
)
from backend.app.services.retrieval.search_visual import (
    VisualSearchConfig,
    VisualSearchEngine,
    normalize_query_vector,
)


class FakeEncoder:
    def encode(self, query: str) -> np.ndarray:
        if query != "a man cooking in a kitchen":
            raise AssertionError(f"unexpected query: {query}")
        return np.array([3.0, 4.0], dtype="float32")


class FakeSearcher:
    def __init__(self) -> None:
        self.seen_vector = None
        self.seen_top_k = None

    def search(self, vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        self.seen_vector = vector
        self.seen_top_k = top_k
        return (
            np.array([[0.91, 0.75, -1.0]], dtype="float32"),
            np.array([[1, 0, -1]], dtype="int64"),
        )


class VisualSearchEngineTest(unittest.TestCase):
    def test_normalize_query_vector_returns_single_unit_vector(self) -> None:
        vector = normalize_query_vector(np.array([3.0, 4.0], dtype="float32"))

        self.assertEqual(vector.shape, (1, 2))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_search_maps_faiss_indices_to_retrieval_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_map_path = Path(tmp_dir) / "frame_map.json"
            frame_map = {
                "0": {
                    "frame_id": "FRAME_L01_V001_000001",
                    "video_id": "L01_V001",
                    "shot_id": "SHOT_L01_V001_000001",
                    "segment_id": "SEG_L01_V001_000001",
                    "timestamp": 0.0,
                    "timestamp_source": "interval",
                    "timestamp_confidence": 0.5,
                    "keyframe_path": "data/keyframes/L01_V001/000001.jpg",
                    "thumbnail_path": "data/keyframes/L01_V001/000001.jpg",
                },
                "1": {
                    "frame_id": "FRAME_L01_V001_000002",
                    "video_id": "L01_V001",
                    "shot_id": "SHOT_L01_V001_000002",
                    "segment_id": "SEG_L01_V001_000002",
                    "timestamp": 2.0,
                    "timestamp_source": "interval",
                    "timestamp_confidence": 0.5,
                    "keyframe_path": "data/keyframes/L01_V001/000002.jpg",
                    "thumbnail_path": "data/keyframes/L01_V001/000002.jpg",
                },
            }
            frame_map_path.write_text(json.dumps(frame_map), encoding="utf-8")

            searcher = FakeSearcher()
            engine = VisualSearchEngine(
                config=VisualSearchConfig(default_top_k=3),
                encoder=FakeEncoder(),
                searcher=searcher,
                metadata_store=MetadataStore.from_frame_map(frame_map_path),
            )

            response = engine.search("a man cooking in a kitchen")

            self.assertEqual(searcher.seen_top_k, 3)
            self.assertAlmostEqual(float(np.linalg.norm(searcher.seen_vector)), 1.0, places=6)
            self.assertEqual(len(response.results), 2)
            self.assertEqual(response.results[0].faiss_index, 1)
            self.assertEqual(response.results[0].video_id, "L01_V001")
            self.assertEqual(response.results[0].timestamp, 2.0)
            self.assertEqual(response.results[0].score, 0.91)
            self.assertEqual(response.results[1].frame_id, "FRAME_L01_V001_000001")

    def test_search_returns_neighbors_from_same_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_map_path = Path(tmp_dir) / "frame_map.json"
            frame_map = {
                "0": {
                    "frame_id": "FRAME_L01_V001_000001",
                    "video_id": "L01_V001",
                    "shot_id": "SHOT_L01_V001_000001",
                    "segment_id": "SHOT_L01_V001_000001",
                    "timestamp": 1.0,
                    "keyframe_path": "data/keyframes/L01_V001/000001.jpg",
                },
                "1": {
                    "frame_id": "FRAME_L01_V001_000002",
                    "video_id": "L01_V001",
                    "shot_id": "SHOT_L01_V001_000001",
                    "segment_id": "SHOT_L01_V001_000001",
                    "timestamp": 3.0,
                    "keyframe_path": "data/keyframes/L01_V001/000002.jpg",
                },
            }
            frame_map_path.write_text(json.dumps(frame_map), encoding="utf-8")

            engine = VisualSearchEngine(
                config=VisualSearchConfig(default_top_k=1),
                encoder=FakeEncoder(),
                searcher=FakeSearcher(),
                metadata_store=MetadataStore.from_frame_map(frame_map_path),
            )

            response = engine.search("a man cooking in a kitchen", top_k=1)

            self.assertEqual(response.results[0].frame_id, "FRAME_L01_V001_000002")
            self.assertEqual(len(response.results[0].neighbors), 1)
            self.assertEqual(response.results[0].neighbors[0].frame_id, "FRAME_L01_V001_000001")

    def test_select_frame_indices_uses_competition_sampling_rules(self) -> None:
        short_shot = Shot(shot_index=1, start_frame=0, end_frame=89, fps=30.0)
        medium_shot = Shot(shot_index=2, start_frame=0, end_frame=179, fps=30.0)
        long_shot = Shot(shot_index=3, start_frame=0, end_frame=299, fps=30.0)

        self.assertEqual(len(select_frame_indices(short_shot)), 1)
        self.assertEqual(len(select_frame_indices(medium_shot)), 2)
        self.assertGreaterEqual(len(select_frame_indices(long_shot)), 2)

    def test_extract_keyframes_records_successful_ffmpeg_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / "T01.mp4"
            output_dir = tmp_path / "keyframes"
            metadata_path = tmp_path / "metadata" / "keyframes_T01.jsonl"
            report_path = tmp_path / "metadata" / "keyframes_T01_report.json"

            video_path.write_bytes(b"placeholder")

            original_read_video_info = keyframe_extractor.read_video_info
            original_detect = keyframe_extractor.detect_shots_transnetv2
            original_extract = keyframe_extractor.extract_frame_ffmpeg
            try:
                keyframe_extractor.read_video_info = lambda path: keyframe_extractor.VideoInfo(
                    video_id="T01",
                    fps=10.0,
                    frame_count=30,
                )
                keyframe_extractor.detect_shots_transnetv2 = lambda path, info, threshold, device: (
                    [Shot(shot_index=1, start_frame=0, end_frame=29, fps=10.0)],
                    "transnetv2_test",
                )

                def fake_extract_frame(
                    video_path: Path,
                    timestamp: float,
                    output_path: Path,
                    jpeg_quality: int,
                ) -> None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    image = np.full((32, 48, 3), 128, dtype=np.uint8)
                    ok = cv2.imwrite(str(output_path), image)
                    if not ok:
                        raise AssertionError(f"failed to write {output_path}")

                keyframe_extractor.extract_frame_ffmpeg = fake_extract_frame

                report = extract_keyframes_for_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    metadata_path=metadata_path,
                    report_path=report_path,
                )
            finally:
                keyframe_extractor.read_video_info = original_read_video_info
                keyframe_extractor.detect_shots_transnetv2 = original_detect
                keyframe_extractor.extract_frame_ffmpeg = original_extract

            self.assertEqual(report["keyframe_count"], 1)
            self.assertEqual(report["skipped_count"], 0)
            records = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source_video_path"], video_path.as_posix())
            self.assertTrue(Path(records[0]["keyframe_path"]).exists())


if __name__ == "__main__":
    unittest.main()
