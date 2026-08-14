from __future__ import annotations

import copy
import hashlib
import json
import unittest

from competition.evaluation.qa_quality_gates import (
    evaluate_answerer_only_gates,
    evaluate_real_dev_gates,
    validate_locked_split_manifest,
)


def _manifest() -> dict:
    def summary(prefix: str) -> dict:
        query_ids = [f"{prefix}-{index:03d}" for index in range(80)]
        query_hash = hashlib.sha256(
            json.dumps(query_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "query_count": 80,
            "query_ids": query_ids,
            "languages": {"vi": 40, "en": 40},
            "unanswerable_count": 16,
            "temporal_count": 16,
            "constraint_heavy_count": 24,
            "query_ids_sha256": query_hash,
        }
    dev = summary("dev")
    locked = summary("locked")
    return {
        "example_only": False,
        "labels_adjudicated": True,
        "dataset_sha256": "a" * 64,
        "splits": {"dev": copy.deepcopy(dev), "locked_test": locked},
        "locked_test_policy": {"reuse": "single_use", "labels_hidden": True},
    }


def _report() -> dict:
    manifest = _manifest()
    return {
        "query_count": 80,
        "query_ids_sha256": manifest["splits"]["dev"]["query_ids_sha256"],
        "dataset_sha256": "a" * 64,
        "evaluation_mode": "answerer_only_gold_evidence",
        "evidence_source": "gold",
        "gold_evidence_sha256": "d" * 64,
        "evaluated_evidence_lineage_sha256": "d" * 64,
        "parser": {"answer_type_macro_F1": 0.91, "constraint_F1": 0.81},
        "evidence": {"Evidence Hit@5": 0.76, "nDCG@10": 0.66},
        "temporal": {"Temporal Chain Hit@5": 0.71},
        "answer": {"answer_F1": 0.81},
        "abstention": {
            "unanswerable_recall": 0.91,
            "answerable_response_rate": 0.91,
        },
        "integrity": {
            "unsupported_answer_rate": 0.05,
            "pipeline_error_rate": 0.0,
            "invalid_response_rate": 0.0,
            "invalid_citation_rate": 0.0,
        },
    }


class QaQualityGateTests(unittest.TestCase):
    def test_real_dev_passes_only_with_adjudicated_balanced_manifest(self) -> None:
        result = evaluate_real_dev_gates(_report(), _manifest())
        self.assertTrue(result["passed"])
        validate_locked_split_manifest(_manifest())

    def test_synthetic_fixture_is_rejected(self) -> None:
        manifest = _manifest()
        manifest["example_only"] = True
        with self.assertRaisesRegex(ValueError, "example_only=false"):
            evaluate_real_dev_gates(_report(), manifest)

    def test_missing_integrity_metric_fails_closed(self) -> None:
        report = _report()
        del report["integrity"]["invalid_citation_rate"]
        with self.assertRaisesRegex(ValueError, "invalid_citation_rate"):
            evaluate_real_dev_gates(report, _manifest())

    def test_failed_metric_is_reported_without_relaxing_threshold(self) -> None:
        report = _report()
        report["evidence"]["Evidence Hit@5"] = 0.74
        result = evaluate_real_dev_gates(report, _manifest())
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["evidence_Hit@5"])

    def test_answerer_only_uses_stricter_gold_evidence_gate(self) -> None:
        result = evaluate_answerer_only_gates(_report())
        self.assertTrue(result["passed"])
        report = _report()
        report["answer"]["answer_F1"] = 0.79
        self.assertFalse(evaluate_answerer_only_gates(report)["passed"])

    def test_answerer_only_rejects_unproven_retrieval_evidence(self) -> None:
        report = _report()
        report["evaluation_mode"] = "retrieval_e2e"
        with self.assertRaisesRegex(ValueError, "answerer-only report"):
            evaluate_answerer_only_gates(report)
        report = _report()
        report["evaluated_evidence_lineage_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "does not match gold lineage"):
            evaluate_answerer_only_gates(report)

    def test_nonfinite_and_out_of_range_rates_fail_closed(self) -> None:
        for value in (float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(value=value):
                report = _report()
                report["answer"]["answer_F1"] = value
                with self.assertRaisesRegex(ValueError, "finite|within"):
                    evaluate_real_dev_gates(report, _manifest())

    def test_fractional_or_negative_counts_are_never_truncated(self) -> None:
        for value in (80.5, -80):
            with self.subTest(value=value):
                manifest = _manifest()
                manifest["splits"]["dev"]["query_count"] = value
                with self.assertRaisesRegex(ValueError, "exact nonnegative"):
                    evaluate_real_dev_gates(_report(), manifest)

    def test_manifest_rejects_placeholder_hashes(self) -> None:
        for field, value in (
            ("dataset_sha256", "REPLACE_AFTER_ADJUDICATION"),
            ("query_ids_sha256", "REPLACE"),
        ):
            with self.subTest(field=field):
                manifest = _manifest()
                if field == "dataset_sha256":
                    manifest[field] = value
                else:
                    manifest["splits"]["dev"][field] = value
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    evaluate_real_dev_gates(_report(), manifest)

    def test_report_is_bound_to_manifest_count_and_query_hash(self) -> None:
        report = _report()
        report["query_count"] = 79
        with self.assertRaisesRegex(ValueError, "query_count does not match"):
            evaluate_real_dev_gates(report, _manifest())
        report = _report()
        report["query_ids_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "query_ids_sha256 does not match"):
            evaluate_real_dev_gates(report, _manifest())
        report = _report()
        report["dataset_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "dataset_sha256 does not match"):
            evaluate_real_dev_gates(report, _manifest())


if __name__ == "__main__":
    unittest.main()
