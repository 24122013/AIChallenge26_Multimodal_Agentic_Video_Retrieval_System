from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.app.api.search import search
from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.qa_evidence import (
    QaEvidenceSearchEngine,
    QaRoutingConfig,
    plan_qa_question,
)


def _result(
    frame_id: str,
    *,
    score: float,
    timestamp: float,
    shot_id: str,
    caption: str,
) -> RetrievalResult:
    return RetrievalResult(
        video_id="V001",
        frame_id=frame_id,
        timestamp=timestamp,
        score=score,
        shot_id=shot_id,
        segment_id=shot_id,
        keyframe_path=f"data/keyframes/V001/{frame_id}.jpg",
        thumbnail_path=f"data/keyframes/V001/{frame_id}.jpg",
        caption=caption,
        modality_scores={"caption": score},
    )


class FakeHybridEngine:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> VisualSearchResponse:
        self.queries.append(query)
        woman = _result(
            "woman_red",
            score=0.72 if query.startswith("a woman") else 0.68,
            timestamp=12.0,
            shot_id="S001",
            caption="a woman in a red shirt sitting at a table",
        )
        unrelated = _result(
            "unrelated",
            score=0.40,
            timestamp=30.0,
            shot_id="S002",
            caption="a car driving down a road",
        )
        return VisualSearchResponse(
            query=query,
            top_k=top_k or 2,
            latency_ms=1.0,
            results=[woman, unrelated][:top_k],
        )


class QueryResultHybrid:
    def __init__(
        self,
        results_by_query: dict[str, list[RetrievalResult]],
        *,
        default: list[RetrievalResult] | None = None,
    ) -> None:
        self.results_by_query = results_by_query
        self.default = list(default or [])
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> VisualSearchResponse:
        self.queries.append(query)
        rows = self.results_by_query.get(query, self.default)
        return VisualSearchResponse(
            query=query,
            top_k=top_k or len(rows),
            latency_ms=1.0,
            results=rows[:top_k],
        )


class RecordingReranker:
    def __init__(self) -> None:
        self.top_k: int | None = None
        self.candidate_count = 0
        self.query = ""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.query = query
        self.top_k = top_k
        self.candidate_count = len(candidates)
        return list(candidates[:top_k])


class QaEvidenceTest(unittest.TestCase):
    def test_vietnamese_held_object_question_builds_visual_queries(self) -> None:
        plan = plan_qa_question(
            "Người phụ nữ mặc áo đỏ đang ngồi trên bàn cầm cái gì?"
        )

        self.assertEqual(plan.answer_target, "held_object")
        self.assertIn("cầm một vật", plan.retrieval_queries[0])
        self.assertEqual(plan.answer_type, "object")
        self.assertIn("người phụ nữ", plan.constraints["subject"])
        self.assertIn("áo đỏ", plan.constraints["attributes"])
        self.assertEqual(plan.expansions, ())

    def test_qa_evidence_fuses_queries_and_returns_image_paths(self) -> None:
        hybrid = FakeHybridEngine()
        engine = QaEvidenceSearchEngine(hybrid)

        response = engine.search(
            "Người phụ nữ mặc áo đỏ đang ngồi trên bàn cầm cái gì?",
            top_k=2,
            expanded_queries=["a woman in a red shirt holding an object"],
        )

        self.assertEqual(response["answer_mode"], "manual_visual_inspection")
        self.assertEqual(response["answer_target"], "held_object")
        self.assertEqual(len(hybrid.queries), 2)
        self.assertEqual(
            response["routing_trace"]["queries"][1]["source"],
            "external_expansion",
        )
        self.assertEqual(response["results"][0]["frame_id"], "woman_red")
        self.assertTrue(response["results"][0]["keyframe_path"].endswith(".jpg"))
        self.assertGreater(
            response["results"][0]["score"],
            0.72,
        )

    def test_unknown_question_type_still_returns_best_effort_evidence(
        self,
    ) -> None:
        engine = QaEvidenceSearchEngine(FakeHybridEngine())

        response = engine.search("Biển hiệu trong cảnh ghi nội dung gì?", top_k=1)

        self.assertEqual(response["answer_target"], "ocr")
        self.assertEqual(response["evidence_count"], 1)
        self.assertEqual(len(response["results"]), 1)

    def test_constraint_rerank_uses_context_and_exposes_scores(self) -> None:
        bad = _result(
            "bad",
            score=0.9,
            timestamp=1.0,
            shot_id="S001",
            caption="a car on an empty road",
        )
        good = _result(
            "good",
            score=0.8,
            timestamp=2.0,
            shot_id="S002",
            caption="the woman is holding a bag",
        )
        response = QaEvidenceSearchEngine(
            QueryResultHybrid({}, default=[bad, good])
        ).search("What is the woman holding?", top_k=2)

        self.assertTrue(response["routing_trace"]["constraint_rerank"]["applied"])
        self.assertEqual(response["results"][0]["frame_id"], "good")
        evidence = response["evidence"][0]
        self.assertGreater(evidence["constraint_score"], 0.0)
        self.assertIn("subject:woman", evidence["matched_constraints"])
        self.assertNotEqual(
            evidence["base_retrieval_score"],
            evidence["retrieval_score"],
        )

    def test_constraint_rerank_below_signal_preserves_exact_order(self) -> None:
        first = _result(
            "first",
            score=0.9,
            timestamp=1.0,
            shot_id="S001",
            caption="",
        )
        second = _result(
            "second",
            score=0.8,
            timestamp=2.0,
            shot_id="S002",
            caption="",
        )
        response = QaEvidenceSearchEngine(
            QueryResultHybrid({}, default=[first, second])
        ).search("What is the woman holding?", top_k=2)

        self.assertEqual(
            response["routing_trace"]["constraint_rerank"]["status"],
            "below_min_signal",
        )
        self.assertEqual(
            [row["frame_id"] for row in response["results"]],
            ["first", "second"],
        )
        self.assertEqual(
            response["evidence"][0]["base_retrieval_score"],
            response["evidence"][0]["retrieval_score"],
        )

    def test_constraint_scorer_error_fails_open_with_exact_order(self) -> None:
        first = _result(
            "first",
            score=0.9,
            timestamp=1.0,
            shot_id="S001",
            caption="an unrelated scene",
        )
        second = _result(
            "second",
            score=0.8,
            timestamp=2.0,
            shot_id="S002",
            caption="another unrelated scene",
        )
        engine = QaEvidenceSearchEngine(
            QueryResultHybrid({}, default=[first, second])
        )
        with patch(
            "backend.app.services.retrieval.qa_evidence.weighted_query_coverage",
            side_effect=RuntimeError("scorer unavailable"),
        ):
            response = engine.search("What is the woman holding?", top_k=2)

        self.assertEqual(
            response["routing_trace"]["constraint_rerank"]["status"],
            "scorer_error",
        )
        self.assertEqual(
            [row["frame_id"] for row in response["results"]],
            ["first", "second"],
        )
        self.assertIn(
            "constraint_rerank:RuntimeError",
            response["routing_trace"]["fallback_reasons"],
        )
        self.assertEqual(
            response["routing_trace"]["constraint_rerank"]["error_code"],
            "constraint_rerank_failed",
        )
        self.assertNotIn(
            "scorer unavailable",
            str(response["routing_trace"]),
        )

    def test_no_constraints_preserves_order_without_reranking(self) -> None:
        rows = [
            _result(
                "first",
                score=0.9,
                timestamp=1.0,
                shot_id="S001",
                caption="first scene",
            ),
            _result(
                "second",
                score=0.8,
                timestamp=2.0,
                shot_id="S002",
                caption="second scene",
            ),
        ]
        response = QaEvidenceSearchEngine(
            QueryResultHybrid({}, default=rows)
        ).search("Khung cảnh này thế nào?", top_k=2)

        self.assertEqual(
            response["routing_trace"]["constraint_rerank"]["status"],
            "no_context_constraints",
        )
        self.assertEqual(
            [row["frame_id"] for row in response["results"]],
            ["first", "second"],
        )

    def test_bge_reranker_receives_pool_of_one_hundred(self) -> None:
        rows = [
            _result(
                f"F{index:03d}",
                score=1.0 - index / 1000.0,
                timestamp=float(index),
                shot_id=f"S{index:03d}",
                caption="generic scene",
            )
            for index in range(120)
        ]
        reranker = RecordingReranker()
        engine = QaEvidenceSearchEngine(
            QueryResultHybrid({}, default=rows),
            candidate_reranker=reranker,
        )

        engine.search("Khung cảnh này thế nào?", top_k=5)

        self.assertEqual(reranker.top_k, 100)
        self.assertEqual(reranker.candidate_count, 100)

    def test_yes_no_uses_neutral_and_proposition_without_hypothesis_boost(
        self,
    ) -> None:
        traffic_light = _result(
            "light",
            score=0.8,
            timestamp=1.0,
            shot_id="S001",
            caption="a green traffic light at an intersection",
        )
        hybrid = QueryResultHybrid({}, default=[traffic_light])

        reranker = RecordingReranker()
        response = QaEvidenceSearchEngine(
            hybrid,
            candidate_reranker=reranker,
        ).search(
            "Is the traffic light green?",
            top_k=1,
        )

        self.assertEqual(
            [item["source"] for item in response["routing_trace"]["queries"]],
            ["neutral_context", "full_proposition"],
        )
        context = response["routing_trace"]["constraint_rerank"][
            "context_constraints"
        ]
        self.assertEqual(context, {"objects": ["traffic light"]})
        self.assertNotIn(
            "attributes:green",
            response["evidence"][0]["matched_constraints"],
        )
        self.assertEqual(reranker.query, "traffic light")

    def test_non_temporal_empty_evidence_is_not_answer_eligible(self) -> None:
        response = QaEvidenceSearchEngine(QueryResultHybrid({})).search(
            "What is happening?",
            top_k=1,
        )

        self.assertFalse(response["answer_eligible"])
        self.assertEqual(response["preflight_block_reason"], "no_evidence")

    def test_temporal_route_retrieves_each_event_and_flattens_strict_chain(
        self,
    ) -> None:
        events = {
            "a man enters the room": [
                _result(
                    "enter",
                    score=0.9,
                    timestamp=10.0,
                    shot_id="S001",
                    caption="a man enters the room",
                )
            ],
            "the man sits down": [
                _result(
                    "sit",
                    score=0.8,
                    timestamp=20.0,
                    shot_id="S002",
                    caption="the man sits down",
                )
            ],
        }
        hybrid = QueryResultHybrid(events)
        response = QaEvidenceSearchEngine(
            hybrid,
            config=QaRoutingConfig(evidence_limit=1),
        ).search(
            "a man enters the room then the man sits down",
            top_k=1,
            expanded_queries=["ignored whole-query expansion"],
        )

        self.assertEqual(hybrid.queries, list(events))
        self.assertEqual(response["routing_trace"]["temporal_route"]["match_mode"], "strict")
        self.assertTrue(response["answer_eligible"])
        self.assertIsNone(response["preflight_block_reason"])
        self.assertEqual(response["evidence_count"], 2)
        self.assertEqual(
            [item["temporal_event_index"] for item in response["evidence"]],
            [0, 1],
        )
        chain_ids = {
            item["temporal_chain_id"] for item in response["evidence"]
        }
        self.assertEqual(len(chain_ids), 1)
        self.assertEqual(
            [item["temporal_event_query"] for item in response["evidence"]],
            list(events),
        )
        self.assertEqual(
            [item["temporal_event_role"] for item in response["evidence"]],
            ["context", "answer_target"],
        )
        self.assertEqual(
            response["evidence"][0]["temporal_chain_score"],
            response["temporal_matches"][0]["score"],
        )
        self.assertIn(
            "external_expansions_ignored_for_temporal",
            response["routing_trace"]["fallback_reasons"],
        )

    def test_temporal_answer_target_can_be_the_event_before_then(self) -> None:
        hybrid = QueryResultHybrid(
            {
                "the man": [
                    _result(
                        "wave",
                        score=0.9,
                        timestamp=10.0,
                        shot_id="S001",
                        caption="the man waves",
                    )
                ],
                "he left": [
                    _result(
                        "leave",
                        score=0.8,
                        timestamp=20.0,
                        shot_id="S002",
                        caption="he left",
                    )
                ],
            }
        )

        response = QaEvidenceSearchEngine(hybrid).search(
            "What did the man do, then he left?",
            top_k=1,
        )

        self.assertEqual(response["query_plan"]["answer_event_index"], 0)
        self.assertEqual(
            [item["temporal_event_role"] for item in response["evidence"]],
            ["answer_target", "context"],
        )
        self.assertTrue(response["answer_eligible"])

    def test_relaxed_temporal_chain_is_manual_only(self) -> None:
        hybrid = QueryResultHybrid(
            {
                "a man enters the room": [
                    _result(
                        "enter",
                        score=0.9,
                        timestamp=10.0,
                        shot_id="S001",
                        caption="a man enters the room",
                    )
                ],
                "the man sits down": [
                    _result(
                        "sit",
                        score=0.8,
                        timestamp=500.0,
                        shot_id="S002",
                        caption="the man sits down",
                    )
                ],
            }
        )
        response = QaEvidenceSearchEngine(hybrid).search(
            "a man enters the room then the man sits down",
            top_k=1,
        )

        route = response["routing_trace"]["temporal_route"]
        self.assertEqual(route["match_mode"], "relaxed_gap")
        self.assertFalse(route["answer_eligible"])
        self.assertEqual(
            response["preflight_block_reason"],
            "temporal_match_not_strict:relaxed_gap",
        )
        self.assertTrue(response["evidence"])

    def test_sparse_temporal_chain_is_manual_only(self) -> None:
        same = _result(
            "same",
            score=0.9,
            timestamp=10.0,
            shot_id="S001",
            caption="a man enters the room and sits down",
        )
        hybrid = QueryResultHybrid(
            {
                "a man enters the room": [same],
                "the man sits down": [same],
            }
        )
        response = QaEvidenceSearchEngine(hybrid).search(
            "a man enters the room then the man sits down",
            top_k=1,
        )

        route = response["routing_trace"]["temporal_route"]
        self.assertEqual(route["match_mode"], "sparse_compat")
        self.assertFalse(response["answer_eligible"])
        self.assertEqual(response["evidence_count"], 2)
        self.assertIn(
            "temporal_sparse_compatibility",
            response["evidence"][0]["warnings"],
        )

    def test_missing_temporal_event_returns_no_chain(self) -> None:
        hybrid = QueryResultHybrid(
            {
                "a man enters the room": [
                    _result(
                        "enter",
                        score=0.9,
                        timestamp=10.0,
                        shot_id="S001",
                        caption="a man enters the room",
                    )
                ],
                "the man sits down": [],
            }
        )
        response = QaEvidenceSearchEngine(hybrid).search(
            "a man enters the room then the man sits down",
            top_k=1,
        )

        self.assertFalse(response["answer_eligible"])
        self.assertEqual(response["preflight_block_reason"], "temporal_no_chain")
        self.assertEqual(response["evidence"], [])
        self.assertEqual(response["temporal_matches"], [])

    def test_disabled_temporal_router_keeps_manual_evidence_but_blocks_answer(
        self,
    ) -> None:
        fallback = _result(
            "fallback",
            score=0.8,
            timestamp=10.0,
            shot_id="S001",
            caption="a man enters a room and sits down",
        )
        response = QaEvidenceSearchEngine(
            QueryResultHybrid({}, default=[fallback]),
            config=QaRoutingConfig(temporal_routing_enabled=False),
        ).search(
            "a man enters the room then the man sits down",
            top_k=1,
        )

        self.assertEqual(response["evidence_count"], 1)
        self.assertFalse(response["answer_eligible"])
        self.assertEqual(
            response["preflight_block_reason"],
            "temporal_routing_disabled",
        )
        self.assertFalse(
            response["routing_trace"]["temporal_route"]["executed"]
        )

    def test_unified_search_accepts_qa_alias(self) -> None:
        expected = {"answer_mode": "manual_visual_inspection", "results": []}
        with patch(
            "backend.app.api.search.search_online",
            return_value=expected,
        ) as mocked:
            response = search("người phụ nữ cầm gì", 5, "qa")

        self.assertEqual(response, expected)
        mocked.assert_called_once_with(
            query="người phụ nữ cầm gì",
            task="qa",
            top_k=5,
            expanded_queries=[],
        )


if __name__ == "__main__":
    unittest.main()
