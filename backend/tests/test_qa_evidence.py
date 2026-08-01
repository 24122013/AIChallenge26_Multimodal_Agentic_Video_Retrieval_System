from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.app.api.search import search
from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.qa_evidence import (
    QaEvidenceSearchEngine,
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


class QaEvidenceTest(unittest.TestCase):
    def test_vietnamese_held_object_question_builds_visual_queries(self) -> None:
        plan = plan_qa_question(
            "Người phụ nữ mặc áo đỏ đang ngồi trên bàn cầm cái gì?"
        )

        self.assertEqual(plan.answer_target, "held_object")
        self.assertIn("cầm một vật", plan.retrieval_queries[0])
        self.assertTrue(
            any(
                "a woman" in query
                and "red shirt" in query
                and "holding an object" in query
                for query in plan.retrieval_queries
            )
        )
        self.assertTrue(
            any("on a table" in query for query in plan.retrieval_queries)
        )

    def test_qa_evidence_fuses_queries_and_returns_image_paths(self) -> None:
        hybrid = FakeHybridEngine()
        engine = QaEvidenceSearchEngine(hybrid)

        response = engine.search(
            "Người phụ nữ mặc áo đỏ đang ngồi trên bàn cầm cái gì?",
            top_k=2,
        )

        self.assertEqual(response["answer_mode"], "manual_visual_inspection")
        self.assertEqual(response["answer_target"], "held_object")
        self.assertGreaterEqual(len(hybrid.queries), 2)
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

        self.assertEqual(response["answer_target"], "unknown")
        self.assertEqual(response["evidence_count"], 1)
        self.assertEqual(len(response["results"]), 1)

    def test_unified_search_accepts_qa_alias(self) -> None:
        expected = {"answer_mode": "manual_visual_inspection", "results": []}
        with patch(
            "backend.app.api.search.search_qa_evidence",
            return_value=expected,
        ) as mocked:
            response = search("người phụ nữ cầm gì", 5, "qa")

        self.assertEqual(response, expected)
        mocked.assert_called_once_with(
            question="người phụ nữ cầm gì",
            top_k=5,
        )


if __name__ == "__main__":
    unittest.main()
