from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.services.indexing.build_text_index import (
    load_records,
    write_text_index,
)
from backend.app.services.retrieval.retrieval_config import (
    load_retrieval_runtime_config,
)
from backend.app.services.retrieval.text_index import (
    TextIndexSearcher,
    build_text_index,
    text_for_modality,
    tokenize,
)


class Phase2RetrievalTest(unittest.TestCase):
    def test_unicode_tokenizer_preserves_vietnamese_words(self) -> None:
        self.assertEqual(
            tokenize("Người đàn ông nấu ăn ở Thành phố Hồ Chí Minh"),
            ["người", "đàn", "ông", "nấu", "ăn", "ở", "thành", "phố", "hồ", "chí", "minh"],
        )

    def test_current_segment_schema_is_indexed_for_all_modalities(self) -> None:
        record = {
            "video_id": "L27_V001",
            "segment_id": "SHOT_L27_V001_000001",
            "start_keyframe": "FRAME_L27_V001_000001",
            "start_time": 1.2,
            "captions_aggregated": "a person cooking in a kitchen",
            "ocr": [{"text": "BẾP VIỆT"}],
            "asr": [{"text": "hôm nay chúng ta nấu phở"}],
            "objects": [{"label": "person"}, {"class_name": "bowl"}],
        }

        self.assertIn("cooking", text_for_modality(record, "caption"))
        self.assertEqual(text_for_modality(record, "ocr"), "BẾP VIỆT")
        self.assertIn("nấu phở", text_for_modality(record, "asr"))
        self.assertEqual(text_for_modality(record, "objects"), "person bowl")

        payload = build_text_index([record])
        for modality in ("caption", "ocr", "asr", "objects"):
            self.assertEqual(
                payload["modalities"][modality]["stats"]["doc_count"],
                1,
            )

    def test_text_search_returns_modality_scores_and_metadata(self) -> None:
        records = [
            {
                "video_id": "L27_V001",
                "frame_id": "F001",
                "segment_id": "S001",
                "timestamp": 2.0,
                "caption": "a person cooking noodles in a kitchen",
                "ocr_text": "BẾP VIỆT",
                "asr_text": "hôm nay chúng ta nấu phở",
                "objects": [{"class_name": "person"}, {"class_name": "bowl"}],
            },
            {
                "video_id": "L27_V001",
                "frame_id": "F002",
                "segment_id": "S002",
                "timestamp": 8.0,
                "caption": "orange clouds at sunset",
                "ocr_text": "HTV",
                "asr_text": "dự báo thời tiết",
                "objects": ["cloud"],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "retrieval_text_index.json"
            write_text_index(records, index_path)
            searcher = TextIndexSearcher(index_path)

            caption = searcher.search_results("person cooking", "caption", 5)
            ocr = searcher.search_results("BẾP VIỆT", "ocr", 5)
            asr = searcher.search_results("nấu phở", "asr", 5)
            objects = searcher.search_results("bowl", "objects", 5)

            self.assertEqual(caption[0].frame_id, "F001")
            self.assertIn("caption", caption[0].modality_scores)
            self.assertEqual(ocr[0].frame_id, "F001")
            self.assertEqual(asr[0].asr_text, "hôm nay chúng ta nấu phở")
            self.assertEqual(objects[0].objects, ["person", "bowl"])

    def test_directory_loader_prefers_segments_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "segments_all.jsonl").write_text(
                json.dumps({"video_id": "V1", "segment_id": "S1"}) + "\n",
                encoding="utf-8",
            )
            (root / "captions_V1.jsonl").write_text(
                json.dumps({"video_id": "V1", "frame_id": "F1"}) + "\n",
                encoding="utf-8",
            )

            records = load_records(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["segment_id"], "S1")

    def test_runtime_config_loads_asr_weight_and_text_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "retrieval.yaml"
            path.write_text(
                "\n".join(
                    [
                        "weights:",
                        "  visual: 0.4",
                        "  asr: 0.3",
                        "text_index:",
                        "  path: custom/index.json",
                        "  default_top_k: 7",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_retrieval_runtime_config(path)

            self.assertEqual(config.rerank.weights.visual, 0.4)
            self.assertEqual(config.rerank.weights.asr, 0.3)
            self.assertEqual(config.text_index.path, Path("custom/index.json"))
            self.assertEqual(config.text_index.default_top_k, 7)


if __name__ == "__main__":
    unittest.main()
