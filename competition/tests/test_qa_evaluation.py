from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from competition.evaluation.qa_evaluation import (
    QA_ANSWER_TYPES,
    LockedTestReuseError,
    evaluate_locked_test_once,
    evaluate_qa_predictions,
    load_qa_dataset,
    load_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "qa_example_dataset.jsonl"
MANIFEST_PATH = ROOT / "evaluation" / "qa_split_manifest.example.json"


def _perfect_prediction(label: dict[str, object]) -> dict[str, object]:
    gold_answers = label["gold_answer"]
    if isinstance(gold_answers, list):
        answer = gold_answers[0]
    else:
        answer = gold_answers
    evidence = [dict(item) for item in label["gold_evidence"]]  # type: ignore[union-attr]
    return {
        "query_id": label["query_id"],
        "query_plan": {
            "task_mode": label["task_mode"],
            "answer_type": label["answer_type"],
            "known_constraints": label["known_constraints"],
        },
        "evidence": evidence,
        "answer": {
            "status": "answered" if label["answerable"] else "insufficient_evidence",
            "answer": answer,
        },
        "latency_ms": 10.0,
        "peak_vram_mb": 256.0,
    }


class QaFixtureTest(unittest.TestCase):
    def test_fixture_is_bilingual_complete_and_explicitly_non_quality(self) -> None:
        labels = load_qa_dataset(DATASET_PATH)
        manifest = load_split_manifest(MANIFEST_PATH)
        self.assertEqual(len(labels), 18)
        self.assertEqual({row["language"] for row in labels.values()}, {"vi", "en"})
        self.assertEqual({row["answer_type"] for row in labels.values()}, QA_ANSWER_TYPES)
        self.assertTrue(all(row["example_only"] is True for row in labels.values()))
        self.assertIs(manifest["quality_claim_allowed"], False)
        dev = set(manifest["splits"]["dev"]["query_ids"])
        locked = set(manifest["splits"]["locked_test"]["query_ids"])
        self.assertFalse(dev & locked)
        self.assertEqual(dev | locked, set(labels))

    def test_loader_rejects_invalid_unanswerable_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_id": "bad",
                        "question": "Unknown?",
                        "task_mode": "qa",
                        "answer_type": "object",
                        "known_constraints": {},
                        "gold_evidence": [],
                        "gold_answer": None,
                        "answerable": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unanswerable type"):
                load_qa_dataset(path)


class QaMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.labels = load_qa_dataset(DATASET_PATH)
        cls.predictions = {
            query_id: _perfect_prediction(label)
            for query_id, label in cls.labels.items()
        }

    def test_perfect_predictions_score_all_metric_families(self) -> None:
        report = evaluate_qa_predictions(
            labels=self.labels,
            predictions=self.predictions,
        )
        self.assertEqual(report["query_count"], 18)
        self.assertEqual(report["answerable_count"], 16)
        for value in report["evidence"].values():
            self.assertAlmostEqual(value, 1.0)
        self.assertAlmostEqual(report["answer"]["answer_EM"], 1.0)
        self.assertAlmostEqual(report["answer"]["answer_F1"], 1.0)
        self.assertAlmostEqual(report["abstention"]["accuracy"], 1.0)
        self.assertAlmostEqual(report["parser"]["task_mode_accuracy"], 1.0)
        self.assertAlmostEqual(report["parser"]["answer_type_macro_F1"], 1.0)
        self.assertAlmostEqual(report["parser"]["constraint_F1"], 1.0)
        self.assertEqual(report["latency_ms"]["count"], 18)
        self.assertEqual(report["latency_ms"]["p95"], 10.0)
        self.assertEqual(report["peak_vram_mb"]["max"], 256.0)

    def test_rank_metrics_and_parser_errors_are_visible(self) -> None:
        query_id = "qa-dev-vi-object"
        prediction = _perfect_prediction(self.labels[query_id])
        prediction["evidence"] = [
            {"evidence_id": "wrong"},
            {"video_id": "example-v01", "frame_id": "120"},
        ]
        prediction["query_plan"] = {
            "task_mode": "kis",
            "answer_type": "color",
            "known_constraints": {"subject": ["người phụ nữ"]},
        }
        report = evaluate_qa_predictions(
            labels=self.labels,
            predictions={query_id: prediction},
            query_ids=[query_id],
        )
        self.assertEqual(report["evidence"]["Evidence Hit@1"], 0.0)
        self.assertEqual(report["evidence"]["Evidence Hit@5"], 1.0)
        self.assertAlmostEqual(report["evidence"]["MRR"], 0.5)
        self.assertAlmostEqual(report["evidence"]["nDCG@10"], 1.0 / 1.584962500721156)
        self.assertEqual(report["parser"]["task_mode_accuracy"], 0.0)
        self.assertEqual(report["parser"]["answer_type_macro_F1"], 0.0)
        self.assertGreater(report["parser"]["constraint_F1"], 0.0)
        self.assertLess(report["parser"]["constraint_F1"], 1.0)

    def test_missing_predictions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing labelled queries"):
            evaluate_qa_predictions(labels=self.labels, predictions={})

    def test_duplicate_evidence_does_not_inflate_ndcg(self) -> None:
        query_id = "qa-dev-vi-object"
        prediction = _perfect_prediction(self.labels[query_id])
        prediction["evidence"] = [
            {"video_id": "example-v01", "frame_id": "120"},
            {"video_id": "example-v01", "frame_id": "120"},
        ]
        report = evaluate_qa_predictions(
            labels=self.labels,
            predictions={query_id: prediction},
            query_ids=[query_id],
        )
        self.assertEqual(report["evidence"]["nDCG@10"], 1.0)
        self.assertEqual(report["evidence"]["Evidence Hit@1"], 1.0)

    def test_locked_test_receipt_prevents_silent_reuse(self) -> None:
        manifest = load_split_manifest(MANIFEST_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            receipt_path = Path(temp_dir) / "locked-test-receipt.json"
            report = evaluate_locked_test_once(
                labels=self.labels,
                predictions=self.predictions,
                manifest=manifest,
                receipt_path=receipt_path,
            )
            self.assertEqual(report["query_count"], 9)
            self.assertTrue(receipt_path.is_file())
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["reuse_policy"], "single_use")
            self.assertEqual(len(receipt["prediction_sha256"]), 64)
            with self.assertRaises(LockedTestReuseError):
                evaluate_locked_test_once(
                    labels=self.labels,
                    predictions=self.predictions,
                    manifest=manifest,
                    receipt_path=receipt_path,
                )


if __name__ == "__main__":
    unittest.main()
