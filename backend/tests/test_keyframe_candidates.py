from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from backend.app.services.indexing.keyframe_candidates import (
    REASON_DENSE_INTERVAL,
    REASON_SHOT_BOUNDARY_END,
    REASON_SHOT_BOUNDARY_START,
    REASON_TINY_SHOT_MIDPOINT,
    generate_keyframe_candidates,
)


@dataclass(frozen=True)
class FakeShot:
    shot_index: int
    start_frame: int
    end_frame: int


class KeyframeCandidateGenerationTest(unittest.TestCase):
    def test_generates_half_second_dense_samples_and_guarded_anchors(self) -> None:
        candidates = generate_keyframe_candidates(
            "L01_V001",
            [FakeShot(shot_index=3, start_frame=0, end_frame=59)],
            fps=10.0,
        )

        self.assertEqual(
            [candidate.frame_index for candidate in candidates],
            [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 57],
        )
        self.assertEqual(candidates[0].reasons, (REASON_DENSE_INTERVAL,))
        self.assertEqual(candidates[1].reasons, (REASON_SHOT_BOUNDARY_START,))
        self.assertEqual(candidates[-1].reasons, (REASON_SHOT_BOUNDARY_END,))
        self.assertEqual(candidates[0].shot_start_sec, 0.0)
        self.assertEqual(candidates[0].shot_end_sec, 6.0)

    def test_tiny_shot_uses_one_midpoint_candidate(self) -> None:
        candidates = generate_keyframe_candidates(
            "L01_V001",
            [FakeShot(shot_index=1, start_frame=100, end_frame=102)],
            fps=10.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].frame_index, 101)
        self.assertEqual(candidates[0].timestamp_sec, 10.1)
        self.assertEqual(candidates[0].reasons, (REASON_TINY_SHOT_MIDPOINT,))

    def test_one_frame_shot_uses_midpoint_when_tiny_threshold_is_disabled(self) -> None:
        candidates = generate_keyframe_candidates(
            "ONE_FRAME",
            [FakeShot(shot_index=0, start_frame=7, end_frame=7)],
            fps=29.97,
            tiny_shot_max_sec=0.0,
            boundary_guard_sec=0.0,
        )

        self.assertEqual([candidate.frame_index for candidate in candidates], [7])
        self.assertEqual(candidates[0].reasons, (REASON_TINY_SHOT_MIDPOINT,))
        self.assertAlmostEqual(candidates[0].timestamp_sec, 7 / 29.97)

    def test_rounding_collisions_merge_reasons_in_stable_order(self) -> None:
        candidates = generate_keyframe_candidates(
            "MERGE",
            [FakeShot(shot_index=4, start_frame=0, end_frame=3)],
            fps=2.0,
            interval_sec=0.5,
            boundary_guard_sec=0.5,
            tiny_shot_max_sec=0.0,
        )

        by_frame = {candidate.frame_index: candidate for candidate in candidates}
        self.assertEqual(sorted(by_frame), [0, 1, 2, 3])
        self.assertEqual(
            by_frame[1].reasons,
            (REASON_DENSE_INTERVAL, REASON_SHOT_BOUNDARY_START),
        )
        self.assertEqual(
            by_frame[2].reasons,
            (REASON_DENSE_INTERVAL, REASON_SHOT_BOUNDARY_END),
        )

    def test_output_and_ids_do_not_depend_on_input_shot_order(self) -> None:
        early = FakeShot(shot_index=8, start_frame=0, end_frame=19)
        late = FakeShot(shot_index=2, start_frame=30, end_frame=49)

        forward = generate_keyframe_candidates("VIDEO_7", [early, late], fps=10.0)
        reverse = generate_keyframe_candidates("VIDEO_7", [late, early], fps=10.0)

        self.assertEqual(forward, reverse)
        self.assertTrue(
            all(candidate.candidate_id.startswith("CANDIDATE_VIDEO_7_") for candidate in forward)
        )
        self.assertEqual(
            len({candidate.candidate_id for candidate in forward}),
            len(forward),
        )

    def test_interval_below_frame_period_returns_each_frame_once(self) -> None:
        candidates = generate_keyframe_candidates(
            "LOW_FPS",
            [FakeShot(shot_index=1, start_frame=0, end_frame=5)],
            fps=2.0,
            interval_sec=0.1,
            boundary_guard_sec=0.0,
            tiny_shot_max_sec=0.0,
        )

        self.assertEqual([candidate.frame_index for candidate in candidates], list(range(6)))
        self.assertEqual(
            candidates[0].reasons,
            (REASON_DENSE_INTERVAL, REASON_SHOT_BOUNDARY_START),
        )
        self.assertEqual(
            candidates[-1].reasons,
            (REASON_DENSE_INTERVAL, REASON_SHOT_BOUNDARY_END),
        )

    def test_empty_shot_sequence_is_valid(self) -> None:
        self.assertEqual(generate_keyframe_candidates("EMPTY", [], fps=25.0), [])

    def test_rejects_invalid_numeric_configuration(self) -> None:
        shot = [FakeShot(shot_index=1, start_frame=0, end_frame=10)]
        invalid_calls = [
            lambda: generate_keyframe_candidates("V", shot, fps=0.0),
            lambda: generate_keyframe_candidates("V", shot, fps=math.inf),
            lambda: generate_keyframe_candidates("V", shot, fps=True),
            lambda: generate_keyframe_candidates("V", shot, fps=10.0, interval_sec=0.0),
            lambda: generate_keyframe_candidates(
                "V", shot, fps=10.0, boundary_guard_sec=-0.1
            ),
            lambda: generate_keyframe_candidates(
                "V", shot, fps=10.0, tiny_shot_max_sec=-0.1
            ),
        ]

        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()

    def test_rejects_invalid_shot_bounds_and_video_frame_bounds(self) -> None:
        invalid_shot_sets = [
            [FakeShot(shot_index=1, start_frame=-1, end_frame=4)],
            [FakeShot(shot_index=1, start_frame=4, end_frame=3)],
            [FakeShot(shot_index=-1, start_frame=0, end_frame=3)],
            [
                FakeShot(shot_index=1, start_frame=0, end_frame=5),
                FakeShot(shot_index=2, start_frame=5, end_frame=9),
            ],
            [
                FakeShot(shot_index=1, start_frame=0, end_frame=4),
                FakeShot(shot_index=1, start_frame=5, end_frame=9),
            ],
        ]

        for shots in invalid_shot_sets:
            with self.subTest(shots=shots):
                with self.assertRaises(ValueError):
                    generate_keyframe_candidates("V", shots, fps=10.0)

        with self.assertRaisesRegex(ValueError, "smaller than frame_count"):
            generate_keyframe_candidates(
                "V",
                [FakeShot(shot_index=1, start_frame=0, end_frame=10)],
                fps=10.0,
                frame_count=10,
            )

    def test_rejects_missing_shot_fields_and_empty_video_id(self) -> None:
        with self.assertRaisesRegex(TypeError, "must expose"):
            generate_keyframe_candidates("V", [object()], fps=10.0)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            generate_keyframe_candidates(" ", [], fps=10.0)


if __name__ == "__main__":
    unittest.main()
