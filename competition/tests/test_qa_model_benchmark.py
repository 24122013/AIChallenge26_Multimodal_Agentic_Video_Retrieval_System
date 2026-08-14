from __future__ import annotations

import hashlib
import json
import unittest

from competition.evaluation.qa_model_benchmark import (
    BASELINE_MODEL,
    BASELINE_REVISION,
    CANDIDATE_MODEL,
    CANDIDATE_REVISION,
    evaluate_4b_eligibility,
)


QUERY_IDS_SHA256 = hashlib.sha256(
    json.dumps(
        sorted(f"q{index}" for index in range(80)),
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _manifest(model_name: str, model_revision: str) -> dict:
    return {
        "model_name": model_name,
        "model_revision": model_revision,
        "dataset_sha256": "a" * 64,
        "evidence_sha256": "b" * 64,
        "prompt_revision": "grounded-qa-v2",
        "decoding": {"do_sample": False, "max_new_tokens": 128},
        "hardware": {"gpu": "NVIDIA L4", "gpu_memory_gb": 24},
        "software": {"transformers": "pinned"},
        "quantization": "4bit",
        "cache_policy": "miss_only",
        "split": "real_dev",
        "labels_adjudicated": True,
        "languages": {"vi": 40, "en": 40},
        "unanswerable_count": 16,
        "temporal_count": 16,
        "constraint_heavy_count": 24,
        "query_count": 80,
        "query_ids_sha256": QUERY_IDS_SHA256,
    }


def _report(*, latency: float, vram: float, candidate: bool = False) -> dict:
    rows = []
    for index in range(80):
        answerable = index < 64
        rows.append(
            {
                "query_id": f"q{index}",
                "answerable": answerable,
                "answer_F1": (0.79 if candidate else 0.80) if answerable else 0.0,
                "abstention_correct": 1.0,
                "unsupported_answer": False,
                "response_valid": True,
                "citation_valid": True,
                "model_invoked": True,
                "cache_hit": False,
            }
        )
    return {
        "query_count": 80,
        "query_ids_sha256": QUERY_IDS_SHA256,
        "dataset_sha256": "a" * 64,
        "evidence_sha256": "b" * 64,
        "queries": rows,
        "latency_ms": {"p95": latency},
        "peak_vram_mb": {"max": vram},
    }


class QaModelBenchmarkTests(unittest.TestCase):
    def test_4b_is_only_eligible_when_quality_integrity_and_efficiency_pass(self) -> None:
        result = evaluate_4b_eligibility(
            baseline_report=_report(latency=100.0, vram=8_000.0),
            candidate_report=_report(latency=75.0, vram=5_500.0, candidate=True),
            baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
            candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
            bootstrap_samples=200,
        )
        self.assertTrue(result["eligible"])
        self.assertFalse(result["default_model_changed"])

    def test_missing_integrity_field_fails_closed(self) -> None:
        candidate = _report(latency=75.0, vram=5_500.0, candidate=True)
        del candidate["queries"][0]["citation_valid"]
        with self.assertRaisesRegex(ValueError, "citation_valid"):
            evaluate_4b_eligibility(
                baseline_report=_report(latency=100.0, vram=8_000.0),
                candidate_report=candidate,
                baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
                candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                bootstrap_samples=100,
            )

    def test_unpaired_manifest_is_rejected(self) -> None:
        candidate_manifest = _manifest(CANDIDATE_MODEL, CANDIDATE_REVISION)
        candidate_manifest["evidence_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "not paired"):
            evaluate_4b_eligibility(
                baseline_report=_report(latency=100.0, vram=8_000.0),
                candidate_report=_report(latency=75.0, vram=5_500.0, candidate=True),
                baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
                candidate_manifest=candidate_manifest,
                bootstrap_samples=100,
            )

    def test_unsupported_answer_regression_blocks_promotion(self) -> None:
        candidate = _report(latency=75.0, vram=5_500.0, candidate=True)
        candidate["queries"][0]["unsupported_answer"] = True
        result = evaluate_4b_eligibility(
            baseline_report=_report(latency=100.0, vram=8_000.0),
            candidate_report=candidate,
            baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
            candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
            bootstrap_samples=100,
        )
        self.assertFalse(result["eligible"])
        self.assertFalse(result["gates"]["unsupported_answer_delta"])

    def test_default_model_pin_cannot_be_silently_changed(self) -> None:
        baseline_manifest = _manifest(BASELINE_MODEL, BASELINE_REVISION)
        baseline_manifest["model_revision"] = "main"
        with self.assertRaisesRegex(ValueError, "pinned"):
            evaluate_4b_eligibility(
                baseline_report=_report(latency=100.0, vram=8_000.0),
                candidate_report=_report(latency=75.0, vram=5_500.0, candidate=True),
                baseline_manifest=baseline_manifest,
                candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                bootstrap_samples=100,
            )

    def test_point_drop_boundary_gates_quality_while_ci_stays_advisory(self) -> None:
        candidate = _report(latency=75.0, vram=5_500.0, candidate=True)
        for row in candidate["queries"]:
            if row["answerable"]:
                row["answer_F1"] = 0.78
        result = evaluate_4b_eligibility(
            baseline_report=_report(latency=100.0, vram=8_000.0),
            candidate_report=candidate,
            baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
            candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
            bootstrap_samples=100,
        )
        self.assertAlmostEqual(result["answer_F1_delta"], -0.02)
        self.assertTrue(result["gates"]["quality_non_inferiority"])
        for row in candidate["queries"]:
            if row["answerable"]:
                row["answer_F1"] = 0.779
        result = evaluate_4b_eligibility(
            baseline_report=_report(latency=100.0, vram=8_000.0),
            candidate_report=candidate,
            baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
            candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
            bootstrap_samples=100,
        )
        self.assertFalse(result["gates"]["quality_non_inferiority"])

    def test_nonfinite_out_of_range_and_wrong_row_types_fail_closed(self) -> None:
        mutations = (
            ("answer_F1", float("nan")),
            ("answer_F1", 1.01),
            ("abstention_correct", -0.01),
            ("answerable", "true"),
            ("response_valid", 1),
            ("cache_hit", True),
            ("model_invoked", False),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                candidate = _report(latency=75.0, vram=5_500.0, candidate=True)
                candidate["queries"][0][field] = value
                with self.assertRaisesRegex(ValueError, "finite|within|boolean|prove"):
                    evaluate_4b_eligibility(
                        baseline_report=_report(latency=100.0, vram=8_000.0),
                        candidate_report=candidate,
                        baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
                        candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                        bootstrap_samples=100,
                    )

    def test_resource_values_must_be_finite_numeric_values(self) -> None:
        for value in (float("nan"), float("inf"), True, "75"):
            with self.subTest(value=value):
                candidate = _report(latency=75.0, vram=5_500.0, candidate=True)
                candidate["latency_ms"]["p95"] = value
                with self.assertRaisesRegex(ValueError, "numeric|finite"):
                    evaluate_4b_eligibility(
                        baseline_report=_report(latency=100.0, vram=8_000.0),
                        candidate_report=candidate,
                        baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
                        candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                        bootstrap_samples=100,
                    )

    def test_query_id_must_be_a_string(self) -> None:
        candidate = _report(latency=75.0, vram=5_500.0, candidate=True)
        candidate["queries"][0]["query_id"] = 0
        with self.assertRaisesRegex(ValueError, "query_id must be a string"):
            evaluate_4b_eligibility(
                baseline_report=_report(latency=100.0, vram=8_000.0),
                candidate_report=candidate,
                baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
                candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                bootstrap_samples=100,
            )

    def test_synthetic_or_undersized_hardware_benchmark_is_rejected(self) -> None:
        baseline = _manifest(BASELINE_MODEL, BASELINE_REVISION)
        baseline["query_count"] = 8
        with self.assertRaisesRegex(ValueError, "exactly 80"):
            evaluate_4b_eligibility(
                baseline_report=_report(latency=100.0, vram=8_000.0),
                candidate_report=_report(latency=75.0, vram=5_500.0, candidate=True),
                baseline_manifest=baseline,
                candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                bootstrap_samples=100,
            )
        baseline = _manifest(BASELINE_MODEL, BASELINE_REVISION)
        candidate = _manifest(CANDIDATE_MODEL, CANDIDATE_REVISION)
        baseline["hardware"]["gpu_memory_gb"] = 6
        candidate["hardware"]["gpu_memory_gb"] = 6
        with self.assertRaisesRegex(ValueError, "at least 16 GB"):
            evaluate_4b_eligibility(
                baseline_report=_report(latency=100.0, vram=8_000.0),
                candidate_report=_report(latency=75.0, vram=5_500.0, candidate=True),
                baseline_manifest=baseline,
                candidate_manifest=candidate,
                bootstrap_samples=100,
            )

    def test_manifest_and_report_hashes_counts_are_bound(self) -> None:
        for field, value in (
            ("dataset_sha256", "REPLACE"),
            ("evidence_sha256", "not-a-hash"),
        ):
            with self.subTest(field=field):
                manifest = _manifest(BASELINE_MODEL, BASELINE_REVISION)
                manifest[field] = value
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    evaluate_4b_eligibility(
                        baseline_report=_report(latency=100.0, vram=8_000.0),
                        candidate_report=_report(latency=75.0, vram=5_500.0, candidate=True),
                        baseline_manifest=manifest,
                        candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                        bootstrap_samples=100,
                    )

        report = _report(latency=100.0, vram=8_000.0)
        report["query_count"] = 7.5
        with self.assertRaisesRegex(ValueError, "exact nonnegative"):
            evaluate_4b_eligibility(
                baseline_report=report,
                candidate_report=_report(latency=75.0, vram=5_500.0, candidate=True),
                baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
                candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                bootstrap_samples=100,
            )

        report = _report(latency=100.0, vram=8_000.0)
        report["query_ids_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_4b_eligibility(
                baseline_report=report,
                candidate_report=_report(latency=75.0, vram=5_500.0, candidate=True),
                baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
                candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                bootstrap_samples=100,
            )

        report = _report(latency=100.0, vram=8_000.0)
        report["evidence_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "evidence_sha256 does not match"):
            evaluate_4b_eligibility(
                baseline_report=report,
                candidate_report=_report(latency=75.0, vram=5_500.0, candidate=True),
                baseline_manifest=_manifest(BASELINE_MODEL, BASELINE_REVISION),
                candidate_manifest=_manifest(CANDIDATE_MODEL, CANDIDATE_REVISION),
                bootstrap_samples=100,
            )


if __name__ == "__main__":
    unittest.main()
