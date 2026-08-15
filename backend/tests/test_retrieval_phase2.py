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
    RetrievalConfigError,
    load_retrieval_runtime_config,
)
from backend.app.services.retrieval.query_terms import (
    content_tokens,
    weighted_query_coverage,
)
from backend.app.services.retrieval.text_index import (
    TextIndexSearcher,
    build_text_index,
    text_for_modality,
    tokenize,
)


class Phase2RetrievalTest(unittest.TestCase):
    def assert_invalid_config(
        self,
        content: str,
        *,
        line_number: int,
        message: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "retrieval.yaml"
            path.write_text(content, encoding="utf-8")

            with self.assertRaises(RetrievalConfigError) as raised:
                load_retrieval_runtime_config(path)

            self.assertIn(f"{path}:{line_number}", str(raised.exception))
            self.assertIn(message, str(raised.exception))

    def test_unicode_tokenizer_preserves_vietnamese_words(self) -> None:
        self.assertEqual(
            tokenize("Người đàn ông nấu ăn ở Thành phố Hồ Chí Minh"),
            ["người", "đàn", "ông", "nấu", "ăn", "ở", "thành", "phố", "hồ", "chí", "minh"],
        )

    def test_query_terms_normalize_actions_and_downweight_generic_subjects(
        self,
    ) -> None:
        self.assertEqual(
            content_tokens("a person sits down", fallback_to_all=True),
            ["person", "sit", "down"],
        )
        self.assertGreater(
            weighted_query_coverage(
                "a person sits down",
                "a person is sitting down",
            ),
            weighted_query_coverage(
                "a person sits down",
                "a person is walking down the street",
            ),
        )

    def test_current_segment_schema_is_indexed_for_all_modalities(self) -> None:
        record = {
            "video_id": "L27_V001",
            "segment_id": "SHOT_L27_V001_000001",
            "start_keyframe": "FRAME_L27_V001_000001",
            "start_time": 1.2,
            "captions_aggregated": "a person cooking in a kitchen",
            "ocr": [{"text": "BẾP VIỆT"}],
            "objects": [{"label": "person"}, {"class_name": "bowl"}],
        }

        self.assertIn("cooking", text_for_modality(record, "caption"))
        self.assertEqual(text_for_modality(record, "ocr"), "BẾP VIỆT")
        self.assertEqual(text_for_modality(record, "objects"), "person bowl")

        payload = build_text_index([record])
        for modality in ("caption", "ocr", "objects"):
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
                "objects": [{"class_name": "person"}, {"class_name": "bowl"}],
            },
            {
                "video_id": "L27_V001",
                "frame_id": "F002",
                "segment_id": "S002",
                "timestamp": 8.0,
                "caption": "orange clouds at sunset",
                "ocr_text": "HTV",
                "objects": ["cloud"],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "retrieval_text_index.json"
            write_text_index(records, index_path)
            searcher = TextIndexSearcher(index_path)

            caption = searcher.search_results("person cooking", "caption", 5)
            ocr = searcher.search_results("BẾP VIỆT", "ocr", 5)
            objects = searcher.search_results("bowl", "objects", 5)

            self.assertEqual(caption[0].frame_id, "F001")
            self.assertIn("caption", caption[0].modality_scores)
            self.assertEqual(ocr[0].frame_id, "F001")
            self.assertEqual(objects[0].objects, ["person", "bowl"])

    def test_text_search_prefers_complete_action_over_partial_overlap(
        self,
    ) -> None:
        records = [
            {
                "video_id": "V1",
                "frame_id": "walking",
                "timestamp": 1.0,
                "caption": "a person is walking down the street",
            },
            {
                "video_id": "V1",
                "frame_id": "sitting",
                "timestamp": 2.0,
                "caption": "a person is sitting down at a table",
            },
            {
                "video_id": "V1",
                "frame_id": "entering",
                "timestamp": 3.0,
                "caption": "a person is entering a room",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "retrieval_text_index.json"
            write_text_index(records, index_path)
            searcher = TextIndexSearcher(index_path)

            sitting = searcher.search_results("a person sits down", "caption", 3)
            entering = searcher.search_results("a person enters", "caption", 3)

            self.assertEqual(sitting[0].frame_id, "sitting")
            self.assertEqual(entering[0].frame_id, "entering")
            self.assertGreater(sitting[0].score, sitting[1].score)

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

    def test_runtime_config_loads_caption_weight_and_text_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "retrieval.yaml"
            path.write_text(
                "\n".join(
                    [
                        "weights:",
                        "  visual: 0.4",
                        "  caption: 0.3",
                        "text_index:",
                        "  path: custom/index.json",
                        "  default_top_k: 7",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_retrieval_runtime_config(path)

            self.assertEqual(config.rerank.weights.visual, 0.4)
            self.assertEqual(config.rerank.weights.caption, 0.3)
            self.assertEqual(config.text_index.path, Path("custom/index.json"))
            self.assertEqual(config.text_index.default_top_k, 7)

    def test_runtime_config_reports_yaml_and_schema_errors_at_exact_line(self) -> None:
        cases = (
            (
                "- hybrid",
                1,
                "document root must be a mapping",
            ),
            (
                "hybrid:\n  nested:",
                2,
                "unknown setting hybrid.nested",
            ),
            (
                "hybrid:\n  stage1_top_k: [20]",
                2,
                "hybrid.stage1_top_k must be a positive integer",
            ),
            (
                "hybrid:\n\tstage1_top_k: 20",
                2,
                "cannot start any token",
            ),
            (
                'text_index:\n  path: "unterminated',
                2,
                "unexpected end of stream",
            ),
        )
        for content, line_number, message in cases:
            with self.subTest(content=content):
                self.assert_invalid_config(
                    content,
                    line_number=line_number,
                    message=message,
                )

    def test_runtime_config_rejects_duplicates_and_unknown_settings(self) -> None:
        cases = (
            (
                "hybrid:\n  stage1_top_k: 20\nhybrid:",
                3,
                "duplicate section 'hybrid'",
            ),
            (
                "hybrid:\n  stage1_top_k: 20\n  stage1_top_k: 30",
                3,
                "duplicate setting hybrid.stage1_top_k",
            ),
            (
                "unknown:\n  value: 1",
                1,
                "unknown section 'unknown'",
            ),
            (
                "hybrid:\n  stage_one_top_k: 20",
                2,
                "unknown setting hybrid.stage_one_top_k",
            ),
        )
        for content, line_number, message in cases:
            with self.subTest(content=content):
                self.assert_invalid_config(
                    content,
                    line_number=line_number,
                    message=message,
                )

    def test_runtime_config_rejects_wrong_types_and_invalid_values(self) -> None:
        cases = (
            (
                "hybrid:\n  stage1_top_k: many",
                "hybrid.stage1_top_k must be a positive integer",
            ),
            (
                "dedupe:\n  same_shot: maybe",
                "dedupe.same_shot must be a boolean",
            ),
            (
                "weights:\n  visual: -0.1",
                "weights.visual must be a non-negative number",
            ),
            (
                'text_index:\n  path: ""',
                "text_index.path must be a non-empty string",
            ),
            (
                "hybrid:\n  max_gap_seconds: 0",
                "hybrid.max_gap_seconds must be a positive number",
            ),
        )
        for content, message in cases:
            with self.subTest(content=content):
                self.assert_invalid_config(
                    content,
                    line_number=2,
                    message=message,
                )

    def test_runtime_config_accepts_standard_yaml_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "retrieval.yaml"
            path.write_text(
                "\n".join(
                    [
                        "hybrid: {stage1_top_k: 33, default_top_k: 7}",
                        "weights:",
                        "    visual: &shared_weight 0.40",
                        "    caption: *shared_weight",
                        "text_index:",
                        "    path: data/indexes/custom.json",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_retrieval_runtime_config(path)

            self.assertEqual(config.hybrid.stage1_top_k, 33)
            self.assertEqual(config.hybrid.default_top_k, 7)
            self.assertEqual(config.rerank.weights.visual, 0.40)
            self.assertEqual(config.rerank.weights.caption, 0.40)
            self.assertEqual(
                config.text_index.path,
                Path("data/indexes/custom.json"),
            )

    def test_runtime_config_preserves_hashes_inside_scalar_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "retrieval.yaml"
            path.write_text(
                'text_index:\n  path: "data/index # current.json" # selected index',
                encoding="utf-8",
            )

            config = load_retrieval_runtime_config(path)

            self.assertEqual(
                config.text_index.path,
                Path("data/index # current.json"),
            )


if __name__ == "__main__":
    unittest.main()
