from __future__ import annotations

import random
import unittest

from backend.app.services.evaluation.evaluator import (
    aggregate_keyframe_reports,
    evaluate_keyframe_video,
    evaluate_retrieval_evidence,
)
from backend.app.services.evaluation.metrics import temporal_coverage_metrics


def _candidate(candidate_id: str, timestamp: float, shot_index: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "video_id": "video_001",
        "timestamp": timestamp,
        "shot_index": shot_index,
    }


class TemporalCoverageMetricsTest(unittest.TestCase):
    def test_head_middle_tail_and_exact_threshold_are_counted(self) -> None:
        result = temporal_coverage_metrics(
            [4.0, 2.0],
            video_duration=6.0,
            max_gap_seconds=2.0,
        )

        self.assertEqual(result.gaps_seconds, (2.0, 2.0, 2.0))
        self.assertEqual(result.coverage_violation_count, 0)
        self.assertEqual(result.max_gap_seconds, 2.0)
        self.assertEqual(result.p95_gap_seconds, 2.0)

        violation = temporal_coverage_metrics(
            [2.000001, 4.0],
            video_duration=6.0,
            max_gap_seconds=2.0,
        )
        self.assertEqual(violation.coverage_violation_count, 1)

        tolerated = temporal_coverage_metrics(
            [2.000001, 4.0],
            video_duration=6.0,
            max_gap_seconds=2.0,
            tolerance_seconds=0.000001,
        )
        self.assertEqual(tolerated.coverage_violation_count, 0)

    def test_seeded_fuzz_is_order_invariant_and_conserves_duration(self) -> None:
        generator = random.Random(20260809)
        for _ in range(100):
            duration = generator.uniform(1.0, 300.0)
            timestamps = [
                generator.uniform(0.0, duration)
                for _ in range(generator.randint(0, 50))
            ]
            forward = temporal_coverage_metrics(
                timestamps,
                video_duration=duration,
                max_gap_seconds=5.0,
            )
            reverse = temporal_coverage_metrics(
                reversed(timestamps),
                video_duration=duration,
                max_gap_seconds=5.0,
            )
            self.assertEqual(forward, reverse)
            self.assertAlmostEqual(sum(forward.gaps_seconds), duration, places=4)


class KeyframeEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            _candidate("C0", 1.0, 0),
            _candidate("C1", 4.0, 1),
        ]
        self.selected = [
            {**self.candidates[0], "protected": True},
            {**self.candidates[1], "protected": True},
        ]
        self.events = [
            {
                "event_id": "OCR_TITLE",
                "source": "feature_adapter",
                "event_type": "ocr_new",
                "candidate_ids": ["C0"],
            },
            {
                "event_id": "OBJECT_BIKE",
                "source": "feature_adapter",
                "event_type": "object_new",
                "candidate_ids": ["C1"],
            },
        ]

    def test_recomputes_hard_manual_budget_and_resource_metrics(self) -> None:
        report = evaluate_keyframe_video(
            video_id="video_001",
            final_records=self.selected,
            candidate_records=self.candidates,
            event_records=self.events,
            video_duration=5.0,
            max_gap_seconds=3.0,
            target_keyframes=1,
            manual_events=[
                {"event_id": "M0", "start_time": 0.9, "end_time": 1.1},
                {"event_id": "M1", "start_time": 3.9, "end_time": 4.1},
            ],
            protection_reviews=[
                {"detected_event_id": "OCR_TITLE", "is_true_event": True},
                {"detected_event_id": "OBJECT_BIKE", "is_true_event": False},
            ],
            resource_usage={"runtime_sec": 2.5, "peak_ram_mb": 128.0},
            disk_bytes=4096,
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["coverage_violation_count"], 0)
        self.assertEqual(report["effective_shot_recall"], 1.0)
        self.assertEqual(report["detected_protected_event_recall"], 1.0)
        self.assertEqual(report["manual_end_to_end_event_recall"], 1.0)
        self.assertEqual(report["manual_detector_event_recall"], 1.0)
        self.assertEqual(report["false_protection_rate"], 0.5)
        self.assertEqual(report["soft_budget_overrun_count"], 1)
        self.assertEqual(report["soft_budget_overrun_ratio"], 1.0)
        self.assertEqual(report["keyframes_per_minute"], 24.0)
        self.assertEqual(report["runtime_sec"], 2.5)
        self.assertEqual(report["peak_ram_mb"], 128.0)
        self.assertEqual(report["disk_bytes"], 4096)

    def test_missing_shot_event_and_temporal_coverage_fail(self) -> None:
        report = evaluate_keyframe_video(
            video_id="video_001",
            final_records=[self.selected[0]],
            candidate_records=self.candidates,
            event_records=self.events,
            video_duration=10.0,
            max_gap_seconds=3.0,
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["coverage_violation_count"], 1)
        self.assertEqual(report["effective_shot_recall"], 0.5)
        self.assertEqual(report["detected_protected_event_recall"], 0.5)
        self.assertIsNone(report["manual_end_to_end_event_recall"])
        self.assertIsNone(report["false_protection_rate"])

    def test_selected_ledger_mismatch_is_rejected(self) -> None:
        selected = [{**self.selected[0], "timestamp": 1.5}]
        with self.assertRaisesRegex(ValueError, "does not match candidate ledger"):
            evaluate_keyframe_video(
                video_id="video_001",
                final_records=selected,
                candidate_records=self.candidates,
                event_records=[],
                video_duration=5.0,
                max_gap_seconds=5.0,
            )

    def test_retrieval_hit_at_k_uses_relevant_frame_intervals(self) -> None:
        retrieval = evaluate_retrieval_evidence(
            [
                {
                    "query_id": "Q0",
                    "relevant": [
                        {"video_id": "video_001", "start_frame": 10, "end_frame": 20}
                    ],
                    "ranked_results": [
                        {"video_id": "video_001", "frame_index": 15}
                    ],
                },
                {
                    "query_id": "Q1",
                    "relevant": [
                        {"video_id": "video_001", "start_frame": 30, "end_frame": 40}
                    ],
                    "ranked_results": [
                        {"video_id": "video_001", "frame_index": 15},
                        {"video_id": "video_001", "frame_index": 35},
                    ],
                },
            ],
            split_video_ids={"video_001"},
        )

        self.assertEqual(retrieval["retrieval_evidence_query_count"], 2)
        self.assertEqual(retrieval["hit_at_1"], 0.5)
        self.assertEqual(retrieval["hit_at_5"], 1.0)

    def test_aggregate_keeps_missing_human_and_resource_evidence_null(self) -> None:
        report = evaluate_keyframe_video(
            video_id="video_001",
            final_records=self.selected,
            candidate_records=self.candidates,
            event_records=self.events,
            video_duration=5.0,
            max_gap_seconds=3.0,
        )

        aggregate = aggregate_keyframe_reports([report])
        self.assertIsNone(aggregate["manual_end_to_end_event_recall"])
        self.assertIsNone(aggregate["false_protection_rate"])
        self.assertIsNone(aggregate["runtime_sec_sum"])
        self.assertEqual(aggregate["runtime_measurement_coverage"], 0.0)
        self.assertIsNone(aggregate["retrieval"]["hit_at_1"])


if __name__ == "__main__":
    unittest.main()
