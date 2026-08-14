from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.services.retrieval.query_plan import build_query_plan
from competition.evaluation.qa_evaluation import (
    QA_ANSWER_TYPES,
    LockedTestReuseError,
    evaluate_locked_test_once,
    evaluate_answerer_only_predictions,
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
    cited_ids = [
        str(item["evidence_id"])
        for item in evidence
        if item.get("evidence_id")
    ]
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
            "evidence_ids": cited_ids,
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

    def test_loader_rejects_unanswerable_as_an_answer_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_id": "bad",
                        "question": "Unknown?",
                        "task_mode": "qa",
                        "answer_type": "unanswerable",
                        "known_constraints": {},
                        "gold_evidence": [],
                        "gold_answer": None,
                        "answerable": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid answer_type"):
                load_qa_dataset(path)

    def test_unanswerable_outcome_keeps_its_semantic_answer_type(self) -> None:
        labels = load_qa_dataset(DATASET_PATH)
        self.assertEqual(labels["qa-dev-vi-unanswerable"]["answer_type"], "identity")
        self.assertEqual(labels["qa-locked-en-unanswerable"]["answer_type"], "location")
        self.assertFalse(labels["qa-dev-vi-unanswerable"]["answerable"])

    def test_gold_evidence_rejects_local_only_citation_ids(self) -> None:
        row = dict(load_qa_dataset(DATASET_PATH)["qa-dev-vi-object"])
        row["gold_evidence"] = [{"evidence_id": "E001"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "local-only.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stable video lineage"):
                load_qa_dataset(path)

    def test_checked_in_parser_matches_the_workflow_fixture(self) -> None:
        labels = load_qa_dataset(DATASET_PATH)
        predictions: dict[str, dict[str, object]] = {}
        for query_id, label in labels.items():
            plan = build_query_plan(str(label["question"]), task_mode="qa")
            predictions[query_id] = {
                "query_plan": plan.to_dict(),
                "evidence": [],
                "answer": {
                    "status": "insufficient_evidence",
                    "answer": None,
                    "evidence_ids": [],
                },
            }
        report = evaluate_qa_predictions(labels=labels, predictions=predictions)
        self.assertEqual(report["parser"]["task_mode_accuracy"], 1.0)
        self.assertEqual(report["parser"]["answer_type_macro_F1"], 1.0)
        self.assertEqual(report["parser"]["constraint_F1"], 1.0)


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
        self.assertEqual(report["validity"]["invalid_response_count"], 0)
        self.assertEqual(report["validity"]["invalid_citation_count"], 0)
        self.assertEqual(report["integrity"]["unsupported_answer_rate"], 0.0)
        self.assertEqual(report["integrity"]["invalid_response_rate"], 0.0)
        self.assertEqual(report["temporal"]["query_count"], 0)
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

    def test_shot_lineage_matches_prediction_that_also_has_a_frame(self) -> None:
        query_id = "qa-dev-vi-object"
        label = dict(self.labels[query_id])
        label["gold_evidence"] = [{"video_id": "V001", "shot_id": "S001"}]
        prediction = _perfect_prediction(self.labels[query_id])
        prediction["evidence"] = [
            {
                "evidence_id": "E001",
                "video_id": "V001",
                "shot_id": "S001",
                "frame_id": "F010",
            }
        ]
        prediction["answer"]["evidence_ids"] = ["E001"]  # type: ignore[index]
        report = evaluate_qa_predictions(
            labels={query_id: label},
            predictions={query_id: prediction},
        )
        self.assertEqual(report["evidence"]["Evidence Hit@1"], 1.0)

    def test_point_timestamp_matches_gold_window_but_outside_point_does_not(self) -> None:
        query_id = "qa-dev-vi-object"
        label = dict(self.labels[query_id])
        label["gold_evidence"] = [
            {"video_id": "V001", "start_time": 10.0, "end_time": 20.0}
        ]
        prediction = _perfect_prediction(self.labels[query_id])
        prediction["evidence"] = [
            {"evidence_id": "E001", "video_id": "V001", "timestamp": 15.0}
        ]
        prediction["answer"]["evidence_ids"] = ["E001"]  # type: ignore[index]
        report = evaluate_qa_predictions(
            labels={query_id: label}, predictions={query_id: prediction}
        )
        self.assertEqual(report["evidence"]["Evidence Hit@1"], 1.0)
        self.assertFalse(report["queries"][0]["unsupported_answer"])

        prediction["evidence"][0]["timestamp"] = 20.01  # type: ignore[index]
        report = evaluate_qa_predictions(
            labels={query_id: label}, predictions={query_id: prediction}
        )
        self.assertEqual(report["evidence"]["Evidence Hit@1"], 0.0)
        self.assertTrue(report["queries"][0]["unsupported_answer"])

    def test_gold_frame_set_matches_any_member(self) -> None:
        query_id = "qa-dev-vi-object"
        label = dict(self.labels[query_id])
        label["gold_evidence"] = [
            {"video_id": "V001", "frame_ids": ["F010", "F011"]}
        ]
        prediction = _perfect_prediction(self.labels[query_id])
        prediction["evidence"] = [
            {"evidence_id": "E001", "video_id": "V001", "frame_id": "F011"}
        ]
        prediction["answer"]["evidence_ids"] = ["E001"]  # type: ignore[index]
        report = evaluate_qa_predictions(
            labels={query_id: label}, predictions={query_id: prediction}
        )
        self.assertEqual(report["evidence"]["Evidence Hit@1"], 1.0)

    def test_temporal_chain_metric_requires_ordered_strict_match(self) -> None:
        query_id = "qa-dev-vi-action"
        label = dict(self.labels[query_id])
        chain = [
            {"video_id": "V001", "shot_id": "S001"},
            {"video_id": "V001", "shot_id": "S002"},
        ]
        label["gold_temporal_chains"] = [chain]
        prediction = _perfect_prediction(self.labels[query_id])
        prediction["temporal_matches"] = [
            {"match_mode": "relaxed_gap", "events": chain},
            {"match_mode": "strict", "events": chain},
        ]
        report = evaluate_qa_predictions(
            labels={query_id: label}, predictions={query_id: prediction}
        )
        self.assertEqual(report["temporal"]["query_count"], 1)
        self.assertEqual(report["temporal"]["Temporal Chain Hit@5"], 1.0)

        prediction["temporal_matches"] = [
            {"match_mode": "relaxed_gap", "events": chain}
        ]
        report = evaluate_qa_predictions(
            labels={query_id: label}, predictions={query_id: prediction}
        )
        self.assertEqual(report["temporal"]["Temporal Chain Hit@5"], 0.0)

    def test_only_explicit_null_insufficient_evidence_is_valid_abstention(self) -> None:
        query_id = "qa-dev-vi-unanswerable"
        cases = (
            ({"status": "error", "answer": None}, "error"),
            ({"status": "disabled", "answer": None}, "disabled"),
            ({"answer": None}, "missing"),
            ({"status": "insufficient_evidence", "answer": "guess"}, "contradictory"),
            ({"status": "insufficient_evidence"}, "missing_answer"),
        )
        for payload, label in cases:
            with self.subTest(label=label):
                prediction = _perfect_prediction(self.labels[query_id])
                prediction["answer"] = payload
                report = evaluate_qa_predictions(
                    labels=self.labels,
                    predictions={query_id: prediction},
                    query_ids=[query_id],
                )
                self.assertEqual(report["abstention"]["accuracy"], 0.0)
                self.assertEqual(report["validity"]["invalid_response_count"], 1)

    def test_answered_response_citations_must_reference_local_evidence_ids(self) -> None:
        query_id = "qa-dev-vi-object"
        prediction = _perfect_prediction(self.labels[query_id])
        prediction["answer"]["evidence_ids"] = ["E999"]  # type: ignore[index]
        report = evaluate_qa_predictions(
            labels=self.labels,
            predictions={query_id: prediction},
            query_ids=[query_id],
        )
        self.assertEqual(report["validity"]["invalid_citation_count"], 1)
        self.assertEqual(report["integrity"]["unsupported_answer_rate"], 1.0)

    def test_duplicate_query_ids_and_non_finite_resources_are_rejected(self) -> None:
        query_id = "qa-dev-vi-object"
        with self.assertRaisesRegex(ValueError, "must be unique"):
            evaluate_qa_predictions(
                labels=self.labels,
                predictions=self.predictions,
                query_ids=[query_id, query_id],
            )
        prediction = _perfect_prediction(self.labels[query_id])
        prediction["latency_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "latency_ms"):
            evaluate_qa_predictions(
                labels={query_id: self.labels[query_id]},
                predictions={query_id: prediction},
            )

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

    def test_answerer_only_report_is_bound_to_exact_gold_lineage(self) -> None:
        report = evaluate_answerer_only_predictions(
            labels=self.labels,
            predictions=self.predictions,
        )
        self.assertEqual(report["evaluation_mode"], "answerer_only_gold_evidence")
        self.assertEqual(report["evidence_source"], "gold")
        self.assertEqual(
            report["gold_evidence_sha256"],
            report["evaluated_evidence_lineage_sha256"],
        )

        query_id = "qa-dev-vi-object"
        prediction = _perfect_prediction(self.labels[query_id])
        prediction["evidence"] = [
            {"evidence_id": "E001", "video_id": "other", "frame_id": "F1"}
        ]
        with self.assertRaisesRegex(ValueError, "identical to gold lineage"):
            evaluate_answerer_only_predictions(
                labels={query_id: self.labels[query_id]},
                predictions={query_id: prediction},
            )


if __name__ == "__main__":
    unittest.main()
