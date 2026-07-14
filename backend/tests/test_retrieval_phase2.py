from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.services.indexing.build_text_index import load_records, write_text_index
from backend.app.services.retrieval.retrieval_config import load_retrieval_runtime_config
from backend.app.services.retrieval.text_index import TextIndexSearcher


class Phase2RetrievalTest(unittest.TestCase):
    def test_build_text_index_and_search_modalities(self) -> None:
        records = [
            {
                "faiss_index": 0,
                "frame_id": "F001",
                "video_id": "V001",
                "timestamp": 3.0,
                "keyframe_path": "data/keyframes/V001/000001.jpg",
                "caption": "a person holding a phone",
                "ocr": "OPEN 24H",
                "objects": ["person", "phone"],
            },
            {
                "faiss_index": 1,
                "frame_id": "F002",
                "video_id": "V002",
                "timestamp": 9.0,
                "keyframe_path": "data/keyframes/V002/000001.jpg",
                "caption": "a car on the street",
                "ocr": "BUS STOP",
                "objects": ["car"],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "text_index.json"
            summary = write_text_index(records, index_path)

            self.assertEqual(summary["modalities"]["caption"]["doc_count"], 2)
            searcher = TextIndexSearcher(index_path)

            caption_results = searcher.search("holding phone", "caption", top_k=1).results
            ocr_results = searcher.search("open", "ocr", top_k=1).results
            object_results = searcher.search("phone", "objects", top_k=1).results

            self.assertEqual(caption_results[0].frame_id, "F001")
            self.assertEqual(ocr_results[0].frame_id, "F001")
            self.assertEqual(object_results[0].frame_id, "F001")

    def test_load_records_accepts_frame_map_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_map_path = Path(tmp_dir) / "frame_map.json"
            frame_map_path.write_text(
                json.dumps({"3": {"frame_id": "F003", "video_id": "V003"}}),
                encoding="utf-8",
            )

            records = load_records(frame_map_path)

            self.assertEqual(records[0]["faiss_index"], 3)
            self.assertEqual(records[0]["frame_id"], "F003")

    def test_retrieval_config_loads_yaml_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "retrieval.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "hybrid:",
                        "  stage1_top_k: 40",
                        "  rerank_pool_size: 8",
                        "  default_top_k: 5",
                        "  max_gap_seconds: 30.0",
                        "weights:",
                        "  visual: 0.5",
                        "  caption: 0.3",
                        "  ocr: 0.1",
                        "  objects: 0.05",
                        "  temporal: 0.05",
                        "text_index:",
                        "  path: data/indexes/test_text.json",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_retrieval_runtime_config(config_path)

            self.assertEqual(config.hybrid.stage1_top_k, 40)
            self.assertEqual(config.hybrid.rerank_pool_size, 8)
            self.assertAlmostEqual(config.rerank.weights.caption, 0.3)
            self.assertEqual(config.text_index.path.as_posix(), "data/indexes/test_text.json")


if __name__ == "__main__":
    unittest.main()
