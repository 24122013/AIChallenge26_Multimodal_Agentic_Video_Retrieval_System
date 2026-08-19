from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.agent.query_expansion import QueryExpansionConfig
from backend.app.services.retrieval import advanced_search as advanced_search_module
from backend.app.services.retrieval.advanced_rerank import AdvancedRerankWeights
from backend.app.services.retrieval.advanced_search import (
    AdvancedSearchConfig,
    advanced_vector_search,
)


class _DenseIndex:
    def __init__(self) -> None:
        self.records = [
            {
                "candidate_id": "A0",
                "frame_id": "FRAME_A0",
                "video_id": "V1",
                "shot_id": "SHOT_A",
                "segment_id": "SHOT_A",
                "timestamp": 1.0,
                "frame_index": 25,
                "keyframe_path": "data/dense/V1/FRAME_A0.jpg",
                "caption": "a person closes a door",
                "ocr_text": "",
                "objects": ["person", "door"],
                "protected_event_ids": ["SHOT_A"],
            },
            {
                "candidate_id": "B0",
                "frame_id": "FRAME_B0",
                "video_id": "V2",
                "shot_id": "SHOT_B",
                "segment_id": "SHOT_B",
                "timestamp": 2.0,
                "frame_index": 50,
                "keyframe_path": "data/dense/V2/FRAME_B0.jpg",
                "caption": "a person opens a refrigerator and takes a bottle",
                "ocr_text": "MILK",
                "objects": ["person", "refrigerator", "bottle"],
                "protected_event_ids": ["OCR_MILK"],
            },
        ]
        self.vectors = np.asarray(
            [
                [0.8, 0.6],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
        # Row zero is deliberately repeated under the rescued clip.  The public
        # search response must still contain each exact dense candidate once.
        self.rows_by_clip = {
            ("V1", "SHOT_A"): [0],
            ("V2", "SHOT_B"): [0, 1],
        }
        self.calls: list[tuple[np.ndarray, int]] = []

    def search(self, query_vector: np.ndarray, top_k: int):
        self.calls.append((np.asarray(query_vector), top_k))
        # -1 is FAISS's short-index sentinel and must never index records[-1].
        return [(1, 1.0), (0, 0.8), (-1, float("-inf"))]


class AdvancedSearchRuntimeTest(unittest.TestCase):
    def test_dense_rescue_timing_dedupe_and_existing_result_mapping(self) -> None:
        dense = _DenseIndex()
        coarse = [
            RetrievalResult(
                video_id="V1",
                frame_id="SELECTED_A",
                timestamp=1.0,
                score=0.9,
                shot_id="SHOT_A",
                segment_id="SHOT_A",
            )
        ]
        weights = AdvancedRerankWeights(
            coarse_rrf=0.10,
            dense_visual=0.50,
            caption=0.10,
            ocr=0.10,
            objects=0.05,
            cses_gain=0.05,
            temporal_consistency=0.05,
            modality_alignment=0.05,
        )
        config = AdvancedSearchConfig(
            coarse_top_n=1,
            dense_global_top_k=10,
            dense_rescue_clips=1,
            max_total_clips=2,
            dense_frames_per_clip=2,
            rerank_weights=weights,
            query_expansion=QueryExpansionConfig(enabled=False),
        )

        with mock.patch.object(
            advanced_search_module,
            "rerank_dense_candidates",
            wraps=advanced_search_module.rerank_dense_candidates,
        ) as reranker:
            response = advanced_vector_search(
                np.asarray([1.0, 0.0], dtype=np.float32),
                coarse_results=coarse,
                dense_index=dense,
                config=config,
            )

        self.assertEqual(response.dense_rescue_clip_count, 1)
        self.assertEqual(response.exact_duplicate_count, 1)
        self.assertEqual(response.selected_row_count, 2)
        self.assertEqual({item.dense_row for item in response.results}, {0, 1})
        self.assertIs(reranker.call_args.kwargs["weights"], weights)
        self.assertEqual(dense.calls[0][1], 10)

        expected_timing = {
            "selected_visual_ms",
            "text_retrieval_ms",
            "fusion_ms",
            "dense_global_ms",
            "dense_rescue_ms",
            "cses_ms",
            "deterministic_rerank_ms",
            "total_ms",
        }
        self.assertTrue(expected_timing <= set(response.stage_latency_ms))
        self.assertGreaterEqual(response.stage_latency_ms["total_ms"], 0.0)

        payload = response.to_dict(top_k=1)
        self.assertEqual(payload["query"], "visual image instance")
        self.assertEqual(payload["top_k"], 1)
        self.assertEqual(len(payload["results"]), 1)
        result = payload["results"][0]
        self.assertIn(result["candidate_id"], {"A0", "B0"})
        self.assertTrue(str(result["frame_id"]).startswith("FRAME_"))
        self.assertTrue(str(result["keyframe_path"]).endswith(".jpg"))
        self.assertEqual(result["score"], result["rerank_score"])
        self.assertIn("dense_visual", result["score_breakdown"])
        self.assertIn("selection_gain", result["cses"])
        self.assertEqual(result["cses_selection"], result["cses"])
        self.assertEqual(
            result["modality_scores"]["visual"],
            result["score_breakdown"]["dense_visual"],
        )
        self.assertEqual(payload["trace"]["latency"], response.stage_latency_ms)
        self.assertNotIn("stage_latency_ms", payload["trace"])
        self.assertNotIn("results", payload["trace"])

    def test_config_and_weight_bounds_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least coarse_top_n"):
            AdvancedSearchConfig(
                coarse_top_n=50,
                max_total_clips=49,
            )
        with self.assertRaisesRegex(ValueError, "similarity_threshold"):
            AdvancedSearchConfig(similarity_threshold=1.1)
        with self.assertRaisesRegex(ValueError, "finite"):
            AdvancedRerankWeights(dense_visual=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            AdvancedRerankWeights(dense_visual=True)

    def test_to_dict_rejects_non_positive_top_k(self) -> None:
        dense = _DenseIndex()
        response = advanced_vector_search(
            np.asarray([1.0, 0.0], dtype=np.float32),
            coarse_results=[],
            dense_index=dense,
            config=AdvancedSearchConfig(
                coarse_top_n=1,
                dense_rescue_clips=1,
                max_total_clips=2,
                dense_frames_per_clip=1,
            ),
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            response.to_dict(top_k=0)

    def test_negative_visual_coarse_scores_preserve_rank_order(self) -> None:
        dense = _DenseIndex()
        dense.rows_by_clip = {
            ("V1", "SHOT_A"): [0],
            ("V2", "SHOT_B"): [1],
        }
        dense.vectors = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        coarse = [
            RetrievalResult(
                video_id="V1",
                frame_id="SELECTED_A",
                timestamp=1.0,
                score=-0.1,
                shot_id="SHOT_A",
                segment_id="SHOT_A",
            ),
            RetrievalResult(
                video_id="V2",
                frame_id="SELECTED_B",
                timestamp=2.0,
                score=-0.9,
                shot_id="SHOT_B",
                segment_id="SHOT_B",
            ),
        ]

        response = advanced_vector_search(
            np.asarray([1.0, 0.0], dtype=np.float32),
            coarse_results=coarse,
            dense_index=dense,
            config=AdvancedSearchConfig(
                coarse_top_n=2,
                dense_rescue_clips=0,
                max_total_clips=2,
                dense_frames_per_clip=1,
                dense_rescue_enabled=False,
            ),
        )

        coarse_by_id = {
            str(item.record["candidate_id"]): item.breakdown["coarse_rrf"]
            for item in response.results
        }
        self.assertGreater(coarse_by_id["A0"], coarse_by_id["B0"])


if __name__ == "__main__":
    unittest.main()
