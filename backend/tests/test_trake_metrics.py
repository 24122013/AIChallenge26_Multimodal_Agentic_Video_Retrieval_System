from __future__ import annotations

import unittest

from backend.app.services.evaluation.trake_metrics import (
    best_r_at_k,
    final_trake_score,
    trake_final_score,
    trake_metrics_report,
    trake_r_score,
    validate_trake_predictions,
)


GROUND_TRUTH = {
    "video_id": "L10_V010",
    "intervals": [[95, 105], [145, 155], [195, 205], [245, 255]],
}


class TrakeMetricsTests(unittest.TestCase):
    def test_official_example_scores_three_of_four_events(self) -> None:
        prediction = {
            "video_id": "L10_V010",
            "frame_ids": [101, 156, 203, 251],
        }
        self.assertEqual(trake_r_score(prediction, GROUND_TRUTH), 0.75)

    def test_wrong_video_is_zero_even_when_all_frames_hit(self) -> None:
        prediction = {
            "video_id": "L10_V999",
            "frame_ids": [101, 150, 203, 251],
        }
        self.assertEqual(trake_r_score(prediction, GROUND_TRUTH), 0.0)

    def test_interval_endpoints_are_inclusive(self) -> None:
        prediction = {
            "video_id": "L10_V010",
            "frame_ids": [95, 155, 195, 255],
        }
        self.assertEqual(trake_r_score(prediction, GROUND_TRUTH), 1.0)

    def test_best_r_at_k_and_final_score_use_ranked_maxima(self) -> None:
        predictions = [
            {"video_id": "wrong", "frame_ids": [1, 2, 3, 4]},
            {"video_id": "L10_V010", "frame_ids": [101, 999, 999, 999]},
            {"video_id": "wrong-2", "frame_ids": [5, 6, 7, 8]},
            {"video_id": "wrong-3", "frame_ids": [9, 10, 11, 12]},
            {"video_id": "L10_V010", "frame_ids": [101, 150, 999, 999]},
            {"video_id": "L10_V010", "frame_ids": [101, 150, 203, 251]},
        ]
        self.assertEqual(best_r_at_k(predictions, GROUND_TRUTH, 1), 0.0)
        self.assertEqual(best_r_at_k(predictions, GROUND_TRUTH, 5), 0.5)
        self.assertEqual(best_r_at_k(predictions, GROUND_TRUTH, 20), 1.0)
        expected = (0.0 + 0.5 + 1.0 + 1.0 + 1.0) / 5.0
        self.assertEqual(trake_final_score(predictions, GROUND_TRUTH), expected)
        self.assertEqual(final_trake_score(predictions, GROUND_TRUTH), expected)

    def test_empty_ranked_list_scores_zero(self) -> None:
        self.assertEqual(best_r_at_k([], GROUND_TRUTH, 100), 0.0)
        self.assertEqual(trake_final_score([], GROUND_TRUTH), 0.0)

    def test_validation_rejects_event_count_interval_and_negative_frames(self) -> None:
        invalid_cases = [
            (
                {"video_id": "L10_V010", "frame_ids": [101]},
                GROUND_TRUTH,
                "ground truth has 4",
            ),
            (
                {"video_id": "L10_V010", "frame_ids": [101, 150, 203, 251]},
                {"video_id": "L10_V010", "intervals": [[5, 4]] * 4},
                "reversed",
            ),
            (
                {"video_id": "L10_V010", "frame_ids": [101, 150, 203, 251]},
                {"video_id": "L10_V010", "intervals": [[-1, 4]] * 4},
                "non-negative",
            ),
            (
                {"video_id": "L10_V010", "frame_ids": [-1, 150, 203, 251]},
                GROUND_TRUTH,
                "non-negative",
            ),
        ]
        for prediction, ground_truth, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                (TypeError, ValueError),
                message,
            ):
                trake_r_score(prediction, ground_truth)

    def test_duplicate_whole_hypothesis_is_rejected(self) -> None:
        duplicate = {"video_id": "L10_V010", "frame_ids": [101, 150, 203, 251]}
        with self.assertRaisesRegex(ValueError, "duplicate TRAKE hypothesis"):
            validate_trake_predictions([duplicate, dict(duplicate)], GROUND_TRUTH)

        # Reusing individual frames is valid when the complete sequence differs.
        ranked, _ = validate_trake_predictions(
            [
                duplicate,
                {"video_id": "L10_V010", "frame_ids": [101, 151, 203, 251]},
                {"video_id": "another", "frame_ids": [101, 150, 203, 251]},
            ],
            GROUND_TRUTH,
        )
        self.assertEqual(len(ranked), 3)

    def test_validation_rejects_more_than_one_hundred_hypotheses(self) -> None:
        predictions = [
            {"video_id": f"V{index}", "frame_ids": [1, 2, 3, 4]}
            for index in range(101)
        ]
        with self.assertRaisesRegex(ValueError, "maximum is 100"):
            best_r_at_k(predictions, GROUND_TRUTH, 100)

    def test_report_adds_video_and_alignment_diagnostics(self) -> None:
        predictions = {
            "hypotheses": [
                {"video_id": "wrong", "frame_ids": [1, 2, 3, 4]},
                {"video_id": "L10_V010", "frame_ids": [101, 150, 999, 251]},
            ]
        }
        report = trake_metrics_report(predictions, GROUND_TRUTH)
        self.assertEqual(report["video_at_1"], 0.0)
        self.assertEqual(report["video_at_5"], 1.0)
        self.assertEqual(report["video_at_20"], 1.0)
        self.assertEqual(report["per_event_hit_rate"], [1.0, 1.0, 0.0, 1.0])
        self.assertEqual(report["matched_event_ratio"], 0.75)
        self.assertEqual(report["r_at_1"], 0.0)
        self.assertEqual(report["r_at_5"], 0.75)
        self.assertEqual(report["final_score"], 0.6)


if __name__ == "__main__":
    unittest.main()
