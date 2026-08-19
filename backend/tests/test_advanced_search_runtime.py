from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.agent.query_expansion import QueryExpansionConfig
from backend.app.services.retrieval import advanced_search as advanced_search_module
from backend.app.services.retrieval.advanced_rerank import (
    AdvancedRerankWeights,
    ContextRerankConfig,
    rerank_dense_candidates,
)
from backend.app.services.retrieval.advanced_search import (
    AdvancedSearchConfig,
    advanced_vector_search,
)
from backend.app.services.retrieval.cses import CSESSelection
from backend.app.services.retrieval.online_context import OnlineContextIndex
from backend.app.services.retrieval.query_plan import build_query_plan


class _DenseIndex:
    def __init__(self) -> None:
        self.records = [
            {
                "artifact_role": "dense_candidate",
                "layer": "dense_visual",
                "candidate_id": "A0",
                "frame_id": "FRAME_A0",
                "video_id": "V1",
                "shot_id": "SHOT_A",
                "segment_id": "SHOT_A",
                "timestamp": 1.0,
                "frame_index": 25,
                "keyframe_path": "data/dense/V1/FRAME_A0.jpg",
            },
            {
                "artifact_role": "dense_candidate",
                "layer": "dense_visual",
                "candidate_id": "B0",
                "frame_id": "FRAME_B0",
                "video_id": "V2",
                "shot_id": "SHOT_B",
                "segment_id": "SHOT_B",
                "timestamp": 2.0,
                "frame_index": 50,
                "keyframe_path": "data/dense/V2/FRAME_B0.jpg",
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
        self.row_by_frame = {
            (str(record["video_id"]), str(record["frame_id"])): row
            for row, record in enumerate(self.records)
        }
        self.calls: list[tuple[np.ndarray, int]] = []

    def search(self, query_vector: np.ndarray, top_k: int):
        self.calls.append((np.asarray(query_vector), top_k))
        # -1 is FAISS's short-index sentinel and must never index records[-1].
        return [(1, 1.0), (0, 0.8), (-1, float("-inf"))]


class AdvancedSearchRuntimeTest(unittest.TestCase):
    @staticmethod
    def _unit_vector(cosine: float) -> list[float]:
        return [cosine, float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))]

    @staticmethod
    def _selection(row: int, relevance: float) -> CSESSelection:
        return CSESSelection(
            row=row,
            selection_rank=row + 1,
            selection_gain=0.5,
            relevance=relevance,
            visual_coverage_gain=0.0,
            temporal_coverage_gain=0.0,
            preserved_event_ids=(),
        )

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
            context_config=ContextRerankConfig(neighbor_enabled=True),
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
        self.assertTrue(payload["trace"]["context_scoring"]["neighbor"]["requested"])
        self.assertEqual(
            payload["trace"]["context_scoring"]["neighbor"]["fallback_reason"],
            "context_index_unavailable",
        )

    def test_config_and_weight_bounds_fail_closed(self) -> None:
        self.assertAlmostEqual(
            sum(AdvancedRerankWeights().__dict__.values()),
            1.0,
        )
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
        with self.assertRaisesRegex(ValueError, "non-negative"):
            AdvancedRerankWeights(neighbor_support=-0.01)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            ContextRerankConfig(
                segment_candidate_limit=2,
                segment_top_k=3,
            )
        with self.assertRaisesRegex(ValueError, "within"):
            ContextRerankConfig(max_bonus=1.01)

    def test_visual_only_rows_renormalize_without_fake_semantic_zeros(self) -> None:
        records = [
            {
                "candidate_id": candidate_id,
                "frame_id": frame_id,
                "video_id": "V1",
                "segment_id": segment_id,
                "timestamp": timestamp,
            }
            for candidate_id, frame_id, segment_id, timestamp in (
                ("A", "F_A", "S_A", 1.0),
                ("B", "F_B", "S_B", 2.0),
            )
        ]
        weights = AdvancedRerankWeights(
            coarse_rrf=0.20,
            dense_visual=0.30,
            caption=0.25,
            ocr=0.10,
            objects=0.05,
            cses_gain=0.05,
            temporal_consistency=0.03,
            modality_alignment=0.02,
            neighbor_support=0.0,
            segment_support=0.0,
        )

        ranked = rerank_dense_candidates(
            plan=build_query_plan("target action", profile="kis"),
            selections=[self._selection(0, 0.8), self._selection(1, 0.8)],
            records=records,
            vectors=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
            coarse_scores={("V1", "S_A"): 1.0, ("V1", "S_B"): 1.0},
            weights=weights,
        )

        self.assertEqual(ranked[0].score, ranked[1].score)
        for item in ranked:
            self.assertEqual(
                item.context_trace["active_terms"],
                (
                    "coarse_rrf",
                    "dense_visual",
                    "cses_gain",
                    "temporal_consistency",
                ),
            )
            for name in ("caption", "ocr", "objects", "modality_alignment"):
                self.assertNotIn(name, item.breakdown)
                self.assertNotIn(name, item.contributions)
                self.assertNotIn(name, item.to_result_mapping()["modality_scores"])
            self.assertAlmostEqual(item.score, 1.0)

    def test_semantic_evidence_is_activated_per_record(self) -> None:
        records = [
            {
                "candidate_id": "VISUAL_ONLY",
                "frame_id": "F0",
                "video_id": "V1",
                "segment_id": "S0",
                "timestamp": 1.0,
            },
            {
                "candidate_id": "WITH_CAPTION",
                "frame_id": "F1",
                "video_id": "V1",
                "segment_id": "S1",
                "timestamp": 2.0,
                "caption": "unrelated words",
            },
        ]
        weights = AdvancedRerankWeights(
            coarse_rrf=0.20,
            dense_visual=0.30,
            caption=0.50,
            ocr=0.0,
            objects=0.0,
            cses_gain=0.0,
            temporal_consistency=0.0,
            modality_alignment=0.0,
            neighbor_support=0.0,
            segment_support=0.0,
        )

        ranked = rerank_dense_candidates(
            plan=build_query_plan("target action", profile="kis"),
            selections=[self._selection(0, 0.8), self._selection(1, 0.8)],
            records=records,
            vectors=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
            coarse_scores={("V1", "S0"): 1.0, ("V1", "S1"): 1.0},
            weights=weights,
        )

        by_id = {str(item.record["candidate_id"]): item for item in ranked}
        self.assertNotIn("caption", by_id["VISUAL_ONLY"].contributions)
        self.assertIn("caption", by_id["WITH_CAPTION"].contributions)
        self.assertAlmostEqual(by_id["VISUAL_ONLY"].score, 1.0)
        self.assertAlmostEqual(by_id["WITH_CAPTION"].score, 0.5)

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

    def test_neighbor_support_can_break_close_tie_but_is_capped_for_weak_frame(
        self,
    ) -> None:
        records = [
            {
                "candidate_id": candidate_id,
                "frame_id": frame_id,
                "video_id": "V1",
                "segment_id": segment_id,
                "timestamp": timestamp,
            }
            for candidate_id, frame_id, segment_id, timestamp in (
                ("A", "A", "SA", 10.0),
                ("B", "B", "SB", 20.0),
                ("C", "C", "SC", 30.0),
                ("AN", "AN", "SA", 9.8),
                ("BN", "BN", "SB", 19.8),
                ("CN", "CN", "SC", 29.8),
            )
        ]
        vectors = np.asarray(
            [
                self._unit_vector(0.40),  # direct score 0.70
                self._unit_vector(0.42),  # direct score 0.71
                self._unit_vector(-0.80),  # very weak direct score 0.10
                self._unit_vector(1.0),
                self._unit_vector(-1.0),
                self._unit_vector(1.0),
            ],
            dtype=np.float32,
        )
        context = OnlineContextIndex(
            neighbor_records=[
                {
                    "video_id": "V1",
                    "frame_id": center,
                    "neighbors_before": [
                        {"frame_id": neighbor, "delta_seconds": -0.2}
                    ],
                    "neighbors_after": [],
                }
                for center, neighbor in (("A", "AN"), ("B", "BN"), ("C", "CN"))
            ]
        )
        ranked = rerank_dense_candidates(
            plan=build_query_plan("target action", profile="kis"),
            selections=[
                self._selection(0, 0.70),
                self._selection(1, 0.71),
                self._selection(2, 0.10),
                # AN is both A's bounded context frame and a direct CSES
                # candidate. Context lookup must never materialize a second
                # output candidate for the same canonical frame identity.
                self._selection(3, 1.00),
            ],
            records=records,
            vectors=vectors,
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
            row_by_frame={
                ("V1", str(record["frame_id"])): row
                for row, record in enumerate(records)
            },
            context_index=context,
            context_config=ContextRerankConfig(
                neighbor_enabled=True,
                max_bonus=0.04,
            ),
        )

        ordered = [str(item.record["candidate_id"]) for item in ranked]
        self.assertLess(ordered.index("A"), ordered.index("B"))
        self.assertNotEqual(ordered[0], "C")
        by_id = {str(item.record["candidate_id"]): item for item in ranked}
        self.assertTrue(by_id["A"].context_trace["neighbor_used_for_scoring"])
        self.assertTrue(
            by_id["C"].context_trace["cap_applied"],
            by_id["C"].context_trace,
        )
        self.assertLessEqual(
            by_id["C"].contributions["context_bonus_after_cap"],
            0.04,
        )
        self.assertEqual(len(ordered), len(set(ordered)))
        self.assertEqual(ordered.count("AN"), 1)

    def test_segment_support_uses_bounded_top_mean_not_raw_frame_count(self) -> None:
        centers = [
            ("A", "SA", 10.0, 0.40),
            ("B", "SB", 20.0, 0.42),
            ("D", "SD", 30.0, 0.40),
        ]
        children = [
            ("A1", "SA", 9.8, 1.0),
            ("A2", "SA", 10.2, 1.0),
            ("D1", "SD", 29.8, 1.0),
            *[
                (f"D{index}", "SD", 30.0 + index / 10.0, -1.0)
                for index in range(2, 10)
            ],
        ]
        values = centers + children
        records = [
            {
                "candidate_id": frame_id,
                "frame_id": frame_id,
                "video_id": "V1",
                "segment_id": segment_id,
                "timestamp": timestamp,
            }
            for frame_id, segment_id, timestamp, _cosine in values
        ]
        vectors = np.asarray(
            [self._unit_vector(cosine) for *_prefix, cosine in values],
            dtype=np.float32,
        )
        context = OnlineContextIndex(
            segment_records=[
                {
                    "video_id": "V1",
                    "segment_id": "SA",
                    "start_time": 9.0,
                    "end_time": 11.0,
                    "keyframe_ids": ["A1", "A", "A2"],
                },
                {
                    "video_id": "V1",
                    "segment_id": "SB",
                    "start_time": 19.0,
                    "end_time": 21.0,
                    "keyframe_ids": ["B"],
                },
                {
                    "video_id": "V1",
                    "segment_id": "SD",
                    "start_time": 29.0,
                    "end_time": 32.0,
                    "keyframe_ids": ["D1", "D", *[f"D{i}" for i in range(2, 10)]],
                },
            ]
        )
        ranked = rerank_dense_candidates(
            plan=build_query_plan("target action", profile="kis"),
            selections=[
                self._selection(0, 0.70),
                self._selection(1, 0.71),
                self._selection(2, 0.70),
            ],
            records=records,
            vectors=vectors,
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
            row_by_frame={
                ("V1", str(record["frame_id"])): row
                for row, record in enumerate(records)
            },
            context_index=context,
            context_config=ContextRerankConfig(
                segment_enabled=True,
                segment_candidate_limit=12,
                segment_top_k=3,
            ),
        )

        by_id = {str(item.record["candidate_id"]): item for item in ranked}
        ordered = [str(item.record["candidate_id"]) for item in ranked]
        self.assertLess(ordered.index("A"), ordered.index("B"))
        self.assertNotIn("segment_support", by_id["B"].breakdown)
        self.assertFalse(by_id["B"].context_trace["segment_used_for_scoring"])
        self.assertGreater(
            by_id["A"].breakdown["segment_support"],
            by_id["B"].breakdown.get("segment_support", 0.0),
        )
        self.assertGreater(
            by_id["A"].breakdown["segment_support"],
            by_id["D"].breakdown["segment_support"],
        )
        self.assertLessEqual(by_id["D"].context_trace["segment_evidence_count"], 12)


if __name__ == "__main__":
    unittest.main()
