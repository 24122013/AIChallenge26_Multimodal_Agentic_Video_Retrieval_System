from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from backend.app.services.retrieval.bge_dense import BGE_M3_SCHEMA_VERSION
from backend.app.services.retrieval.qa_answerer import DEFAULT_QA_MODEL_REVISION
from backend.app.services.retrieval.run_task_smoke import (
    SmokeValidationError,
    _validate_task_payload,
    build_parser,
    run,
)


def _runtime_lineage() -> dict:
    digest = "a" * 64
    return {
        "answer_model": {
            "enabled": True,
            "mode": "required",
            "name": "Qwen/Qwen3.5-9B",
            "revision": DEFAULT_QA_MODEL_REVISION,
            "prompt_revision": "grounded-qa-v2",
        },
        "dense_text": {
            "enabled": True,
            "model_name": "BAAI/bge-m3",
            "model_revision": "pinned-bge",
            "index_schema_version": BGE_M3_SCHEMA_VERSION,
            "vector_count": 1,
            "source_contract": {
                "canonical_only": True,
                "source_kind": "canonical_segments",
            },
            "artifact_checksums": {
                "index": {"sha256": digest},
                "frame_map": {"sha256": digest},
            },
        },
        "reranker": {
            "enabled": True,
            "model_name": "BAAI/bge-reranker-v2-m3",
            "model_revision": "pinned-reranker",
            "last_report": {
                "status": "passed",
                "model_name": "BAAI/bge-reranker-v2-m3",
                "model_revision": "pinned-reranker",
                "candidate_count": 1,
                "scored_count": 1,
                "output_count": 1,
            },
        },
    }


def _response(task: str, image_path: Path) -> dict:
    return {
        "query_plan": {"task_mode": task, "needs_temporal": False},
        "routing_trace": {
            "reranker": "applied",
            "fallback_used": False,
            "fallback_reasons": [],
            "modality_queries": [
                {"modality": "dense_text", "candidate_count": 1}
            ],
        },
        "answer": {
            "status": "answered",
            "answer": "phone",
            "evidence_ids": ["E001"],
        },
        "answer_report": {
            "status": "answered",
            "mode": "required",
            "cache_hit": False,
            "model_invoked": True,
            "model_name": "Qwen/Qwen3.5-9B",
            "model_revision": DEFAULT_QA_MODEL_REVISION,
            "prompt_revision": "grounded-qa-v2",
            "evidence_count": 1,
            "manual_evidence_available": True,
        },
        "answer_eligible": True,
        "preflight_block_reason": None,
        "evidence": [
            {
                "evidence_id": "E001",
                "video_id": "L01_V001",
                "frame_id": "000001",
                "shot_id": "S001",
                "timestamp": 1.0,
                "image_path": str(image_path),
                "caption": "a woman holding a phone",
                "ocr_text": "",
                "objects": ["phone"],
                "source_modalities": ["visual", "dense_text"],
                "retrieval_score": 0.9,
                "warnings": [],
                "ignored_large_field": "not serialized",
            }
        ],
        "latency_ms": 12.0,
    }


def _temporal_response(image_paths: list[Path]) -> dict:
    event_queries = [f"event {index}" for index in range(1, 6)]
    chain_id = "TC-five-events"
    score = 0.75
    evidence = []
    match_events = []
    for index, (event_query, image_path) in enumerate(
        zip(event_queries, image_paths, strict=True)
    ):
        event = {
            "video_id": "L01_V001",
            "frame_id": f"{index + 1:06d}",
            "shot_id": f"S{index + 1:03d}",
            "timestamp": float(index + 1),
        }
        match_events.append(event)
        evidence.append(
            {
                "evidence_id": f"E{index + 1:03d}",
                **event,
                "image_path": str(image_path),
                "caption": event_query,
                "ocr_text": "",
                "objects": ["person"],
                "source_modalities": ["visual", "dense_text"],
                "retrieval_score": 0.9,
                "base_retrieval_score": 0.9,
                "constraint_score": 0.5,
                "matched_constraints": ["person"],
                "temporal_event_index": index,
                "temporal_match_rank": 1,
                "temporal_match_mode": "strict",
                "temporal_chain_id": chain_id,
                "temporal_event_query": event_query,
                "temporal_event_role": (
                    "answer_target" if index == 4 else "context"
                ),
                "temporal_chain_score": score,
                "warnings": [],
            }
        )
    return {
        "query_plan": {
            "task_mode": "qa",
            "needs_temporal": True,
            "answer_type": "action",
            "answer_event_index": 4,
            "temporal_events": event_queries,
        },
        "routing_trace": {
            "reranker": "applied",
            "fallback_used": False,
            "fallback_reasons": [],
            "modality_queries": [
                {
                    "modality": "dense_text",
                    "candidate_count": 1,
                    "query": query,
                }
                for query in event_queries
            ],
            "temporal_route": {
                "executed": True,
                "match_mode": "strict",
                "event_count": 5,
                "event_queries": event_queries,
            },
        },
        "answer": {
            "status": "answered",
            "answer": "event 5",
            "evidence_ids": ["E005"],
        },
        "answer_report": {
            "status": "answered",
            "mode": "required",
            "cache_hit": False,
            "model_invoked": True,
            "model_name": "Qwen/Qwen3.5-9B",
            "model_revision": DEFAULT_QA_MODEL_REVISION,
            "prompt_revision": "grounded-qa-v2",
            "evidence_count": 5,
            "manual_evidence_available": True,
        },
        "answer_eligible": True,
        "preflight_block_reason": None,
        "temporal_matches": [
            {
                "match_mode": "strict",
                "chain_id": chain_id,
                "score": score,
                "events": match_events,
            }
        ],
        "evidence": evidence,
        "latency_ms": 20.0,
    }


class RunTaskSmokeTest(unittest.TestCase):
    @mock.patch("backend.app.services.retrieval.run_task_smoke.search_qa")
    @mock.patch("backend.app.services.retrieval.run_task_smoke.get_qa_evidence_search_engine")
    @mock.patch("backend.app.services.retrieval.run_task_smoke.get_qa_runtime_lineage")
    @mock.patch("backend.app.services.retrieval.run_task_smoke.clear_retrieval_caches")
    def test_all_tasks_keep_answerer_qa_only(
        self,
        clear_mock: mock.Mock,
        lineage_mock: mock.Mock,
        evidence_factory: mock.Mock,
        qa_mock: mock.Mock,
    ) -> None:
        engine = mock.Mock()
        evidence_factory.return_value = engine
        lineage_mock.return_value = _runtime_lineage()
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "frame.jpg"
            Image.new("RGB", (2, 2), color="red").save(image_path)
            engine.search.side_effect = (
                lambda query, top_k, task_mode: _response(task_mode, image_path)
            )
            qa_mock.return_value = _response("qa", image_path)
            output = Path(temporary) / "task_smoke.json"
            args = build_parser().parse_args(["--task", "all", "--output", str(output)])
            payload = run(args)
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload, saved)
        self.assertEqual([item["task"] for item in payload["results"]], ["kis", "avs", "qa"])
        self.assertIsNone(payload["results"][0]["answer"])
        self.assertEqual(payload["results"][2]["answer"]["answer"], "phone")
        self.assertNotIn("ignored_large_field", payload["results"][0]["evidence"][0])
        self.assertEqual(engine.search.call_count, 2)
        qa_mock.assert_called_once()
        self.assertEqual(clear_mock.call_count, 2)

    @mock.patch("backend.app.services.retrieval.run_task_smoke.search_qa")
    @mock.patch("backend.app.services.retrieval.run_task_smoke.get_qa_runtime_lineage")
    @mock.patch("backend.app.services.retrieval.run_task_smoke.clear_retrieval_caches")
    def test_fail_loud_writes_report_when_qa_reuses_cache(
        self,
        _: mock.Mock,
        lineage_mock: mock.Mock,
        qa_mock: mock.Mock,
    ) -> None:
        lineage_mock.return_value = _runtime_lineage()
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "frame.jpg"
            Image.new("RGB", (2, 2), color="blue").save(image_path)
            response = _response("qa", image_path)
            response["answer_report"]["cache_hit"] = True
            response["answer_report"]["model_invoked"] = False
            qa_mock.return_value = response
            output = Path(temporary) / "task_smoke.json"
            args = build_parser().parse_args(
                ["--task", "qa", "--output", str(output)]
            )
            with self.assertRaises(SmokeValidationError):
                run(args)
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertIn(
            "qa_answer_cache_not_miss",
            saved["validation_errors"][0]["issues"],
        )

    @mock.patch("backend.app.services.retrieval.run_task_smoke.search_qa")
    @mock.patch("backend.app.services.retrieval.run_task_smoke.get_qa_runtime_lineage")
    @mock.patch("backend.app.services.retrieval.run_task_smoke.clear_retrieval_caches")
    def test_temporal_smoke_retains_complete_chain_when_top_k_is_one(
        self,
        _: mock.Mock,
        lineage_mock: mock.Mock,
        qa_mock: mock.Mock,
    ) -> None:
        lineage_mock.return_value = _runtime_lineage()
        with tempfile.TemporaryDirectory() as temporary:
            image_paths = []
            for index in range(5):
                image_path = Path(temporary) / f"frame-{index}.jpg"
                Image.new("RGB", (2, 2), color="purple").save(image_path)
                image_paths.append(image_path)
            qa_mock.return_value = _temporal_response(image_paths)
            args = build_parser().parse_args(["--task", "qa", "--top-k", "1"])

            payload = run(args)

        self.assertEqual(len(payload["results"][0]["evidence"]), 5)
        self.assertEqual(
            [
                item["temporal_event_index"]
                for item in payload["results"][0]["evidence"]
            ],
            [0, 1, 2, 3, 4],
        )

    def test_rejects_unbounded_top_k(self) -> None:
        args = build_parser().parse_args(["--top-k", "21"])
        with self.assertRaisesRegex(ValueError, "within"):
            run(args)

    def test_validator_rejects_noop_models_and_inconsistent_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "frame.jpg"
            Image.new("RGB", (2, 2), color="green").save(image_path)
            payload = {
                "task": "qa",
                **_response("qa", image_path),
                "runtime_lineage": _runtime_lineage(),
            }
            payload["routing_trace"]["modality_queries"][0]["candidate_count"] = 0
            payload["runtime_lineage"]["reranker"]["last_report"][
                "scored_count"
            ] = 0
            payload["answer_report"]["status"] = "failed"
            payload["answer_report"]["model_revision"] = "wrong"
            payload["answer_eligible"] = False
            payload["preflight_block_reason"] = "blocked"
            payload["answer"]["answer"] = ""

            issues = _validate_task_payload(payload)

        self.assertIn("routing_bge_dense_not_applied", issues)
        self.assertIn("bge_reranker_scored_count_invalid", issues)
        self.assertIn("qa_answer_report_model_revision_mismatch", issues)
        self.assertIn("qa_answer_report_status_mismatch", issues)
        self.assertIn("qa_answer_not_eligible", issues)
        self.assertIn("qa_preflight_blocked", issues)
        self.assertIn("qa_answer_text_missing", issues)

    def test_validator_requires_an_answer_not_only_an_abstention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "frame.jpg"
            Image.new("RGB", (2, 2), color="yellow").save(image_path)
            payload = {
                "task": "qa",
                **_response("qa", image_path),
                "runtime_lineage": _runtime_lineage(),
            }
            payload["answer"] = {
                "status": "insufficient_evidence",
                "answer": None,
                "evidence_ids": [],
            }
            payload["answer_report"]["status"] = "insufficient_evidence"

            issues = _validate_task_payload(payload)

        self.assertIn("qa_answer_not_answered", issues)


if __name__ == "__main__":
    unittest.main()
