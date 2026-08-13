from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import backend.app.services.indexing.extract_keyframes as extractor
from backend.app.services.indexing.extract_keyframes import (
    KEYFRAME_STRATEGY_DENSE_COVERAGE,
    Shot,
    VideoInfo,
    extract_keyframes_for_video,
    scenes_to_shots,
)


class ExtractKeyframeStrategyTest(unittest.TestCase):
    def test_scene_normalization_drops_phantom_tail_and_trims_overlap(self) -> None:
        info = VideoInfo(video_id="video8573", fps=25.0, frame_count=555)

        shots = scenes_to_shots(
            [(0, 245), (246, 554), (555, 555)],
            info,
        )
        self.assertEqual(
            [(shot.shot_index, shot.start_frame, shot.end_frame) for shot in shots],
            [(1, 0, 245), (2, 246, 554)],
        )

        shared_boundary = scenes_to_shots([(0, 10), (10, 20)], info)
        self.assertEqual(
            [(shot.start_frame, shot.end_frame) for shot in shared_boundary],
            [(0, 10), (11, 20)],
        )

    def _run_extraction(
        self,
        root: Path,
        *,
        video_id: str,
        frame_count: int,
        shots: list[Shot],
        fail_first: bool = False,
        fail_frame_index: int | None = None,
        **kwargs: object,
    ) -> tuple[dict, list[dict], Path]:
        video_path = root / f"{video_id}.mp4"
        video_path.write_bytes(b"placeholder")
        output_dir = root / "keyframes"
        metadata_path = root / "metadata" / f"keyframes_{video_id}.jsonl"
        report_path = root / "metadata" / f"keyframes_{video_id}_report.json"
        attempts = 0

        def fake_extract_frame(
            video_path: Path,
            timestamp: float,
            output_path: Path,
            jpeg_quality: int,
        ) -> None:
            nonlocal attempts
            attempts += 1
            current_frame_index = int(round(timestamp * 10.0))
            if fail_first and attempts == 1:
                raise RuntimeError("synthetic extraction failure")
            if fail_frame_index is not None and current_frame_index == fail_frame_index:
                raise RuntimeError("synthetic extraction failure")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image = np.full((32, 48, 3), 128, dtype=np.uint8)
            if not cv2.imwrite(str(output_path), image):
                raise AssertionError(f"failed to write {output_path}")

        with (
            patch.object(
                extractor,
                "read_video_info",
                return_value=VideoInfo(
                    video_id=video_id,
                    fps=10.0,
                    frame_count=frame_count,
                ),
            ),
            patch.object(
                extractor,
                "detect_shots_transnetv2",
                return_value=(shots, "transnetv2_test"),
            ),
            patch.object(extractor, "extract_frame_ffmpeg", side_effect=fake_extract_frame),
        ):
            report = extract_keyframes_for_video(
                video_path=video_path,
                output_dir=output_dir,
                metadata_path=metadata_path,
                report_path=report_path,
                **kwargs,
            )

        records = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8")),
            report,
        )
        return report, records, output_dir / video_id

    def test_default_legacy_strategy_keeps_existing_ids_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report, records, _ = self._run_extraction(
                Path(temp_dir),
                video_id="LEGACY",
                frame_count=30,
                shots=[Shot(shot_index=1, start_frame=0, end_frame=29, fps=10.0)],
            )

        self.assertEqual(report["keyframe_count"], 1)
        self.assertEqual(report["keyframe_strategy"], "legacy")
        self.assertEqual(records[0]["frame_id"], "FRAME_LEGACY_000001")
        self.assertEqual(records[0]["selection_reason"], "midpoint_lt_4s")
        self.assertNotIn("candidate_id", records[0])

    def test_dense_strategy_satisfies_boundary_and_temporal_guarantees(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report, records, _ = self._run_extraction(
                Path(temp_dir),
                video_id="DENSE",
                frame_count=120,
                shots=[Shot(shot_index=1, start_frame=0, end_frame=119, fps=10.0)],
                strategy=KEYFRAME_STRATEGY_DENSE_COVERAGE,
                max_gap_seconds=3.0,
            )

        self.assertEqual(report["status"], "satisfied")
        self.assertTrue(report["constraints_satisfied"])
        self.assertTrue(report["coverage_satisfied"])
        self.assertLessEqual(report["selection"]["max_gap_after"], 3.0)
        self.assertEqual(report["keyframe_count"], len(records))
        self.assertGreater(len(records), 1)
        self.assertTrue(any(record["protected"] for record in records))
        self.assertTrue(any(record["coverage_added"] for record in records))
        for record in records:
            self.assertEqual(
                record["frame_id"],
                record["candidate_id"].replace("CANDIDATE_", "FRAME_", 1),
            )
            self.assertEqual(record["keyframe_strategy"], "dense_coverage")
            self.assertIn("selection_phase", record)
            self.assertIn("selection_reasons", record)
            self.assertIn("covered_event_ids", record)

    def test_dense_strategy_reselects_after_extraction_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report, records, keyframe_dir = self._run_extraction(
                Path(temp_dir),
                video_id="RESELECT",
                frame_count=20,
                shots=[Shot(shot_index=1, start_frame=0, end_frame=19, fps=10.0)],
                fail_first=True,
                strategy=KEYFRAME_STRATEGY_DENSE_COVERAGE,
                max_gap_seconds=10.0,
            )
            output_names = sorted(path.name for path in keyframe_dir.glob("*.jpg"))
            failed_path_exists = Path(report["skipped"][0]["output_path"]).exists()

        self.assertEqual(report["status"], "satisfied")
        self.assertEqual(report["skipped_count"], 1)
        self.assertEqual(len(records), 1)
        self.assertNotEqual(records[0]["candidate_id"], report["skipped"][0]["candidate_id"])
        self.assertEqual(output_names, [f"{records[0]['frame_id']}.jpg"])
        self.assertFalse(failed_path_exists)

    def test_dense_strategy_retains_protected_phash_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report, records, _ = self._run_extraction(
                Path(temp_dir),
                video_id="DUPLICATE",
                frame_count=40,
                shots=[
                    Shot(shot_index=1, start_frame=0, end_frame=19, fps=10.0),
                    Shot(shot_index=2, start_frame=20, end_frame=39, fps=10.0),
                ],
                strategy=KEYFRAME_STRATEGY_DENSE_COVERAGE,
                max_gap_seconds=10.0,
            )

        self.assertEqual(report["status"], "satisfied")
        self.assertEqual(report["keyframe_count"], 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(report["skipped_count"], 0)
        self.assertGreaterEqual(report["duplicate_retained_count"], 1)
        self.assertTrue(all(record["protected"] for record in records))
        self.assertTrue(any("phash_duplicate_of" in record for record in records))

    def test_dense_strategy_retains_protected_clip_duplicates(self) -> None:
        class FakeClipDeduper:
            def __init__(self, **kwargs: object) -> None:
                self.kept: list[str] = []

            def encode(self, frame: np.ndarray) -> np.ndarray:
                return np.array([1.0, 0.0], dtype="float32")

            def find_duplicate(
                self,
                embedding: np.ndarray,
                timestamp: float,
            ) -> tuple[str, float] | None:
                return (self.kept[0], 0.999) if self.kept else None

            def add(self, embedding: np.ndarray, frame_id: str, timestamp: float) -> None:
                self.kept.append(frame_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(extractor, "ClipDeduper", FakeClipDeduper):
                report, records, _ = self._run_extraction(
                    Path(temp_dir),
                    video_id="CLIP_DUPLICATE",
                    frame_count=40,
                    shots=[
                        Shot(shot_index=1, start_frame=0, end_frame=19, fps=10.0),
                        Shot(shot_index=2, start_frame=20, end_frame=39, fps=10.0),
                    ],
                    strategy=KEYFRAME_STRATEGY_DENSE_COVERAGE,
                    max_gap_seconds=10.0,
                    enable_clip_dedup=True,
                )

        self.assertEqual(report["keyframe_count"], 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(report["skipped_count"], 0)
        self.assertTrue(any("clip_duplicate_of" in record for record in records))
        self.assertTrue(
            any(
                item["reason"] == "clip_duplicate_retained"
                for item in report["duplicate_retained"]
            )
        )

    def test_dense_strategy_cleans_successful_frames_deselected_after_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report, records, keyframe_dir = self._run_extraction(
                Path(temp_dir),
                video_id="CLEANUP",
                frame_count=40,
                shots=[
                    Shot(shot_index=1, start_frame=0, end_frame=19, fps=10.0),
                    Shot(shot_index=2, start_frame=20, end_frame=39, fps=10.0),
                ],
                fail_frame_index=22,
                strategy=KEYFRAME_STRATEGY_DENSE_COVERAGE,
                max_gap_seconds=1.0,
            )
            output_names = sorted(path.name for path in keyframe_dir.glob("*.jpg"))

        self.assertEqual(report["status"], "satisfied")
        self.assertEqual(report["skipped_count"], 1)
        self.assertGreaterEqual(report["deselected_cached_count"], 1)
        self.assertEqual(
            output_names,
            sorted(f"{record['frame_id']}.jpg" for record in records),
        )

    def test_dense_strategy_writes_partial_report_when_hard_cap_is_infeasible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report, records, _ = self._run_extraction(
                Path(temp_dir),
                video_id="PARTIAL",
                frame_count=40,
                shots=[
                    Shot(shot_index=1, start_frame=0, end_frame=19, fps=10.0),
                    Shot(shot_index=2, start_frame=20, end_frame=39, fps=10.0),
                ],
                strategy=KEYFRAME_STRATEGY_DENSE_COVERAGE,
                max_gap_seconds=10.0,
                hard_max_keyframes=1,
            )

        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["constraints_satisfied"])
        self.assertEqual(len(records), 1)
        self.assertTrue(report["selection"]["unsatisfied_event_ids"])
        self.assertEqual(
            report["selection"]["stop_reason"],
            "hard_constraints_infeasible_within_cap",
        )

    def test_cli_accepts_dense_selection_configuration(self) -> None:
        argv = [
            "extract_keyframes.py",
            "--video-path",
            "video.mp4",
            "--strategy",
            "dense_coverage",
            "--candidate-interval-sec",
            "0.25",
            "--boundary-guard-sec",
            "0.1",
            "--tiny-shot-max-sec",
            "0.4",
            "--max-gap-seconds",
            "4.0",
            "--gap-tolerance-seconds",
            "0.02",
            "--target-keyframes",
            "20",
            "--hard-max-keyframes",
            "30",
        ]
        with patch.object(sys, "argv", argv):
            args = extractor.parse_args()

        self.assertEqual(args.strategy, "dense_coverage")
        self.assertEqual(args.candidate_interval_sec, 0.25)
        self.assertEqual(args.boundary_guard_sec, 0.1)
        self.assertEqual(args.tiny_shot_max_sec, 0.4)
        self.assertEqual(args.max_gap_seconds, 4.0)
        self.assertEqual(args.gap_tolerance_seconds, 0.02)
        self.assertEqual(args.target_keyframes, 20)
        self.assertEqual(args.hard_max_keyframes, 30)


if __name__ == "__main__":
    unittest.main()
