from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.api import retrieval as retrieval_api
from backend.app.api.search import search
from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.qa_evidence import (
    QaEvidenceSearchEngine,
    QaRoutingConfig,
)
from backend.app.services.retrieval.qa_pipeline import (
    QaPipelineConfig,
    QaSearchPipeline,
    RequiredQaPipelineError,
)
from backend.app.services.retrieval.query_plan import build_query_plan


def _result(
    frame_id: str,
    *,
    shot_id: str,
    score: float,
    modality: str,
    caption: str = "",
    ocr_text: str = "",
) -> RetrievalResult:
    return RetrievalResult(
        video_id="V001",
        frame_id=frame_id,
        timestamp=float(frame_id[-1]) if frame_id[-1].isdigit() else 0.0,
        score=score,
        shot_id=shot_id,
        segment_id=shot_id,
        keyframe_path=f"data/keyframes/V001/{frame_id}.jpg",
        caption=caption,
        ocr_text=ocr_text,
        objects=["person", "phone"] if "phone" in caption else [],
        modality_scores={modality: score},
    )


class FakeVisual:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        self.queries.append(query)
        rows = [
            _result(
                "F1",
                shot_id="S1",
                score=0.8,
                modality="visual",
                caption="a woman in red holding a phone",
            ),
            _result("F2", shot_id="S1", score=0.7, modality="visual"),
            _result("F3", shot_id="S2", score=0.6, modality="visual"),
        ]
        return VisualSearchResponse(query, top_k or 3, 1.0, rows[:top_k])


class FakeText:
    def __init__(self, modality: str) -> None:
        self.modality = modality
        self.queries: list[str] = []

    def search_results(self, query: str, top_k: int | None = None):
        self.queries.append(query)
        text = "a woman in red holding a phone" if self.modality == "caption" else ""
        ocr = "OPEN" if self.modality == "ocr" else ""
        return [
            _result(
                "F1",
                shot_id="S1",
                score=0.9,
                modality=self.modality,
                caption=text,
                ocr_text=ocr,
            )
        ][:top_k]


class FakeHybrid:
    def __init__(self) -> None:
        self.visual_engine = FakeVisual()
        self.text_engines = {
            "caption": FakeText("caption"),
            "ocr": FakeText("ocr"),
            "objects": FakeText("objects"),
        }

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        return self.visual_engine.search(query, top_k)


class QueryParserV2Test(unittest.TestCase):
    def test_typed_vietnamese_object_contract(self) -> None:
        plan = build_query_plan(
            "Người phụ nữ áo đỏ cầm vật gì?",
            task_mode="qa",
        )
        self.assertEqual(plan.task_mode, "qa")
        self.assertEqual(plan.answer_type, "object")
        self.assertEqual(plan.constraints["subject"], ["người phụ nữ"])
        self.assertIn("đỏ", plan.constraints["attributes"][0])
        self.assertIn("cầm", plan.constraints["actions"])
        self.assertEqual(plan.expansions, ())
        self.assertFalse(plan.needs_temporal)

    def test_explicit_mode_wins_and_temporal_only_hands_off(self) -> None:
        plan = build_query_plan(
            "người đàn ông vào bếp rồi ngồi xuống",
            task_mode="qa",
        )
        self.assertEqual(plan.profile, "qa")
        self.assertEqual(plan.profile_source, "explicit")
        self.assertTrue(plan.needs_temporal)
        self.assertEqual(plan.temporal_relation, "then")

    def test_unknown_keeps_original_without_generated_expansion(self) -> None:
        plan = build_query_plan("Khung cảnh này thế nào?", task_mode="qa")
        self.assertEqual(plan.answer_type, "unknown")
        self.assertEqual(plan.retrieval_statement, "Khung cảnh này thế nào")
        self.assertEqual(plan.expansions, ())


class QaRouterEvidenceTest(unittest.TestCase):
    def test_external_expansion_is_passthrough_and_traceable(self) -> None:
        hybrid = FakeHybrid()
        response = QaEvidenceSearchEngine(hybrid).search(
            "Người phụ nữ áo đỏ cầm gì?",
            top_k=5,
            expanded_queries=["woman red shirt holding object"],
        )
        self.assertEqual(
            response["routing_trace"]["queries"][1],
            {
                "query": "woman red shirt holding object",
                "source": "external_expansion",
            },
        )
        self.assertEqual(response["evidence"][0]["evidence_id"], "E001")
        self.assertEqual(response["evidence_count"], 2)
        self.assertEqual(
            len({row["shot_id"] for row in response["evidence"]}),
            response["evidence_count"],
        )
        self.assertTrue(response["evidence"][0]["image_path"].endswith(".jpg"))
        self.assertIn("caption", response["evidence"][0]["source_modalities"])

    def test_ocr_route_records_hint_and_boost(self) -> None:
        response = QaEvidenceSearchEngine(FakeHybrid()).search(
            "Biển hiệu ghi nội dung gì?",
            top_k=1,
        )
        self.assertEqual(response["query_plan"]["answer_type"], "ocr")
        self.assertIn("ocr", response["routing_trace"]["modality_hints"])
        self.assertEqual(response["routing_trace"]["hint_boost"], 1.5)
        self.assertEqual(response["routing_trace"]["rrf_k"], 60)

    def test_router_flag_rolls_back_to_existing_hybrid_engine(self) -> None:
        response = QaEvidenceSearchEngine(
            FakeHybrid(),
            config=QaRoutingConfig(router_enabled=False),
        ).search("What is the woman holding?", top_k=1)
        self.assertTrue(response["routing_trace"]["fallback_used"])
        self.assertFalse(
            response["routing_trace"]["feature_flags"]["qa_router"]
        )
        self.assertEqual(response["evidence_count"], 1)


class QaPipelineApiTest(unittest.TestCase):
    def test_pipeline_answers_from_top_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = QaSearchPipeline(
                QaEvidenceSearchEngine(FakeHybrid()),
                config=QaPipelineConfig(
                    answer_mode="optional",
                    answer_cache_root=Path(temporary),
                ),
                answer_runner=lambda *_: {
                    "status": "answered",
                    "answer": "một chiếc điện thoại",
                    "answer_type": "object",
                    "confidence": 0.86,
                    "evidence_ids": ["E001"],
                },
            )
            response = pipeline.search(
                "Người phụ nữ áo đỏ cầm gì?",
                top_k=5,
            )
        self.assertEqual(response["answer"]["status"], "answered")
        self.assertEqual(response["answer"]["evidence_ids"], ["E001"])
        self.assertTrue(response["evidence"])
        self.assertGreaterEqual(response["latency_ms"], 0)

    def test_bundle_rollback_keeps_manual_evidence_and_skips_answerer(self) -> None:
        called = False

        def runner(*_: object) -> dict[str, object]:
            nonlocal called
            called = True
            raise AssertionError("answerer must be disabled without evidence bundle")

        pipeline = QaSearchPipeline(
            QaEvidenceSearchEngine(
                FakeHybrid(),
                config=QaRoutingConfig(evidence_bundle_enabled=False),
            ),
            config=QaPipelineConfig(answer_mode="optional"),
            answer_runner=runner,
        )
        response = pipeline.search("What is the woman holding?", top_k=1)
        self.assertFalse(called)
        self.assertEqual(response["answer"]["status"], "disabled")
        self.assertEqual(
            response["answer"]["reason"],
            "qa_evidence_bundle_disabled",
        )
        self.assertEqual(len(response["evidence"]), 1)
        self.assertEqual(response["evidence"][0]["evidence_id"], "E001")
        self.assertIn(
            "evidence_bundle_disabled",
            response["evidence"][0]["warnings"],
        )
        self.assertNotIn("asr_text", response["evidence"][0])

    def test_required_answer_failure_raises_with_evidence_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = QaSearchPipeline(
                QaEvidenceSearchEngine(FakeHybrid()),
                config=QaPipelineConfig(
                    answer_mode="required",
                    answer_cache_root=Path(temporary),
                ),
                answer_runner=lambda *_: (_ for _ in ()).throw(
                    RuntimeError("model failed")
                ),
            )
            with self.assertRaises(RequiredQaPipelineError) as caught:
                pipeline.search("What is the woman holding?", top_k=1)
        payload = caught.exception.response
        self.assertEqual(payload["answer"]["status"], "error")
        self.assertTrue(payload["evidence"])
        self.assertIn("model failed", payload["required_answer_error"])

    def test_required_failure_api_response_is_503_and_keeps_evidence(self) -> None:
        if retrieval_api.router is None:
            self.skipTest("FastAPI is not installed")
        failure = RequiredQaPipelineError(
            "required answer failed",
            {"answer": {"status": "error"}, "evidence": [{"evidence_id": "E001"}]},
        )

        def fail() -> dict[str, object]:
            raise failure

        response = retrieval_api._response(fail)
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["data"]["evidence"][0]["evidence_id"], "E001")

    def test_qa_answer_alias_passes_external_queries(self) -> None:
        expected = {
            "query_plan": {},
            "routing_trace": {},
            "answer": {},
            "evidence": [],
            "latency_ms": 0,
        }
        with patch(
            "backend.app.api.search.search_qa",
            return_value=expected,
        ) as mocked:
            response = search(
                "What is she holding?",
                5,
                "qa_answer",
                task_mode="qa",
                expanded_queries=["woman holding an object"],
            )
        self.assertEqual(response, expected)
        mocked.assert_called_once_with(
            query="What is she holding?",
            top_k=5,
            task_mode="qa",
            expanded_queries=["woman holding an object"],
        )

    def test_legacy_qa_alias_is_unchanged(self) -> None:
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
