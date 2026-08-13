from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.app.services.retrieval.run_task_smoke import build_parser, run


def _response(task: str) -> dict:
    return {
        "query_plan": {"task_mode": task},
        "routing_trace": {"reranker": "bge-reranker-v2-m3"},
        "answer": {"status": "answered", "answer": "phone"},
        "answer_report": {"status": "passed"},
        "evidence": [
            {
                "evidence_id": "E001",
                "video_id": "L01_V001",
                "frame_id": "000001",
                "shot_id": "S001",
                "timestamp": 1.0,
                "image_path": "/tmp/frame.jpg",
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


class RunTaskSmokeTest(unittest.TestCase):
    @mock.patch("backend.app.services.retrieval.run_task_smoke.search_qa")
    @mock.patch("backend.app.services.retrieval.run_task_smoke.get_qa_evidence_search_engine")
    @mock.patch("backend.app.services.retrieval.run_task_smoke.clear_retrieval_caches")
    def test_all_tasks_keep_answerer_qa_only(
        self,
        clear_mock: mock.Mock,
        evidence_factory: mock.Mock,
        qa_mock: mock.Mock,
    ) -> None:
        engine = mock.Mock()
        engine.search.side_effect = lambda query, top_k, task_mode: _response(task_mode)
        evidence_factory.return_value = engine
        qa_mock.return_value = _response("qa")
        with tempfile.TemporaryDirectory() as temporary:
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
        clear_mock.assert_called_once()

    def test_rejects_unbounded_top_k(self) -> None:
        args = build_parser().parse_args(["--top-k", "21"])
        with self.assertRaisesRegex(ValueError, "within"):
            run(args)


if __name__ == "__main__":
    unittest.main()
