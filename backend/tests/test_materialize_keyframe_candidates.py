from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import backend.app.services.indexing.materialize_keyframe_candidates as materializer
from backend.app.services.indexing.extract_keyframes import Shot, VideoInfo
from backend.app.services.indexing.keyframe_candidates import (
    KeyframeCandidate,
    generate_keyframe_candidates,
)
from backend.app.services.indexing.materialize_keyframe_candidates import (
    FRAME_EXTRACTOR,
    MATERIALIZATION_MODE,
    CandidateFrameDecode,
    decode_candidate_frames_sequential,
    materialize_keyframe_candidates_for_video,
)


class MaterializeKeyframeCandidatesTest(unittest.TestCase):
    def _run(
        self,
        root: Path,
        *,
        fail_frame_index: int | None = None,
    ) -> tuple[dict, list[dict], list[tuple[Path, tuple]], list]:
        video_id = "DENSE_POOL"
        video_path = root / f"{video_id}.mp4"
        video_path.write_bytes(b"placeholder")
        output_dir = root / "candidate_keyframes"
        metadata_path = root / "work" / "candidates.jsonl"
        report_path = root / "work" / "candidate_report.json"
        info = VideoInfo(video_id=video_id, fps=10.0, frame_count=40)
        shots = [
            Shot(shot_index=1, start_frame=0, end_frame=19, fps=10.0),
            Shot(shot_index=2, start_frame=20, end_frame=39, fps=10.0),
        ]
        expected = generate_keyframe_candidates(
            video_id,
            shots,
            info.fps,
            interval_sec=0.5,
            boundary_guard_sec=0.2,
            tiny_shot_max_sec=0.5,
            frame_count=info.frame_count,
        )
        decode_passes: list[tuple[Path, tuple]] = []

        def fake_sequential_decode(
            video_path: Path,
            candidates: tuple,
        ):
            ordered = tuple(candidates)
            decode_passes.append((video_path, ordered))
            for candidate in ordered:
                if candidate.frame_index == fail_frame_index:
                    yield CandidateFrameDecode(
                        candidate=candidate,
                        frame=None,
                        error="synthetic sequential decode failure",
                    )
                else:
                    # Identical images exercise annotate-only pHash handling.
                    yield CandidateFrameDecode(
                        candidate=candidate,
                        frame=np.full((32, 48, 3), 127, dtype=np.uint8),
                    )

        with (
            patch.object(materializer, "read_video_info", return_value=info),
            patch.object(
                materializer,
                "detect_shots_transnetv2",
                return_value=(shots, "transnetv2_test"),
            ),
            patch.object(
                materializer,
                "decode_candidate_frames_sequential",
                side_effect=fake_sequential_decode,
            ),
        ):
            report = materialize_keyframe_candidates_for_video(
                video_path=video_path,
                output_dir=output_dir,
                metadata_path=metadata_path,
                report_path=report_path,
                phash_threshold=6,
                phash_window_sec=12.0,
                jpeg_quality=93,
                shot_threshold=0.42,
                shot_device="cpu",
                candidate_interval_sec=0.5,
                boundary_guard_sec=0.2,
                tiny_shot_max_sec=0.5,
            )

        records = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)
        return report, records, decode_passes, expected

    def test_materializes_every_generated_candidate_once_without_dedup_drop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report, records, decode_passes, expected = self._run(Path(temp_dir))
            image_paths = [Path(record["keyframe_path"]) for record in records]
            images_exist = all(
                path.is_file() and path.stat().st_size > 0 for path in image_paths
            )

        self.assertEqual(len(decode_passes), 1)
        self.assertEqual(decode_passes[0][1], tuple(expected))
        self.assertEqual(report["candidate_count"], len(expected))
        self.assertEqual(report["keyframe_count"], len(expected))
        self.assertEqual(report["skipped_count"], 0)
        self.assertEqual(report["status"], "satisfied")
        self.assertTrue(report["constraints_satisfied"])
        self.assertTrue(report["coverage_satisfied"])
        self.assertTrue(images_exist)
        self.assertEqual(report["frame_extractor"], FRAME_EXTRACTOR)
        self.assertEqual(report["sequential_decode_passes"], 1)
        self.assertEqual(report["jpeg_quality"], 93)
        self.assertEqual(report["materialization_mode"], MATERIALIZATION_MODE)
        self.assertFalse(report["selection_applied"])
        self.assertEqual(report["deduplication_mode"], "annotate_only")
        self.assertEqual(report["duplicate_retained_count"], len(expected) - 1)
        self.assertEqual(
            report["candidate_config"],
            {
                "candidate_interval_sec": 0.5,
                "boundary_guard_sec": 0.2,
                "tiny_shot_max_sec": 0.5,
            },
        )
        self.assertEqual(
            [
                (record["candidate_id"], record["frame_index"])
                for record in records
            ],
            [
                (candidate.candidate_id, candidate.frame_index)
                for candidate in expected
            ],
        )
        for record in records:
            self.assertEqual(
                record["frame_id"],
                record["candidate_id"].replace("CANDIDATE_", "FRAME_", 1),
            )
            self.assertEqual(record["frame_path"], record["keyframe_path"])
            self.assertEqual(record["thumbnail_path"], record["keyframe_path"])
            self.assertIn("shot_id", record)
            self.assertIn("segment_id", record)
            self.assertIn("shot_start", record)
            self.assertIn("shot_end", record)
            self.assertIn("candidate_reasons", record)
            self.assertIn("phash", record)

    def test_one_failure_keeps_processing_and_persists_partial_pool(self) -> None:
        probe_shots = [
            Shot(shot_index=1, start_frame=0, end_frame=19, fps=10.0),
            Shot(shot_index=2, start_frame=20, end_frame=39, fps=10.0),
        ]
        probe_candidates = generate_keyframe_candidates(
            "DENSE_POOL",
            probe_shots,
            10.0,
            interval_sec=0.5,
            boundary_guard_sec=0.2,
            tiny_shot_max_sec=0.5,
            frame_count=40,
        )
        failed = probe_candidates[2]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report, records, decode_passes, expected = self._run(
                root,
                fail_frame_index=failed.frame_index,
            )
            failed_output = Path(report["skipped"][0]["output_path"])
            report_exists = (root / "work" / "candidate_report.json").is_file()
            metadata_exists = (root / "work" / "candidates.jsonl").is_file()

        self.assertEqual(len(decode_passes), 1)
        self.assertEqual(decode_passes[0][1], tuple(expected))
        self.assertEqual(report["candidate_count"], len(expected))
        self.assertEqual(report["keyframe_count"], len(expected) - 1)
        self.assertEqual(report["skipped_count"], 1)
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["constraints_satisfied"])
        self.assertFalse(report["coverage_satisfied"])
        self.assertEqual(report["skipped"][0]["candidate_id"], failed.candidate_id)
        self.assertFalse(failed_output.exists())
        self.assertTrue(report_exists)
        self.assertTrue(metadata_exists)
        self.assertNotIn(failed.candidate_id, {record["candidate_id"] for record in records})
        candidate_row = next(
            row for row in report["candidates"] if row["candidate_id"] == failed.candidate_id
        )
        self.assertFalse(candidate_row["materialized"])
        self.assertEqual(candidate_row["extraction_status"], "failed")

    def test_transnet_failure_preserves_cuda_root_cause_and_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "video.mp4"
            video_path.write_bytes(b"placeholder")
            info = VideoInfo(video_id="video", fps=25.0, frame_count=100)
            with (
                patch.object(materializer, "read_video_info", return_value=info),
                patch.object(
                    materializer,
                    "detect_shots_transnetv2",
                    side_effect=AssertionError("Torch not compiled with CUDA enabled"),
                ),
                self.assertRaises(RuntimeError) as raised,
            ):
                materialize_keyframe_candidates_for_video(
                    video_path=video_path,
                    output_dir=root / "keyframes",
                    metadata_path=root / "metadata.jsonl",
                    report_path=root / "report.json",
                    shot_device="cuda",
                )

        message = str(raised.exception)
        self.assertIn("Torch not compiled with CUDA enabled", message)
        self.assertIn("CUDA-enabled PyTorch build", message)

    def test_decoder_uses_one_forward_pass_and_retrieves_exact_target_indices(self) -> None:
        candidates = tuple(
            KeyframeCandidate(
                candidate_id=f"CANDIDATE_EXACT_{frame_index:09d}",
                video_id="EXACT",
                shot_index=1,
                frame_index=frame_index,
                timestamp_sec=frame_index / 10.0,
                shot_start_sec=0.0,
                shot_end_sec=0.8,
                reasons=("dense_interval",),
            )
            for frame_index in (0, 2, 5)
        )

        class FakeCapture:
            def __init__(self, path: str) -> None:
                self.path = path
                self.position = -1
                self.grab_count = 0
                self.retrieve_indices: list[int] = []
                self.released = False

            def isOpened(self) -> bool:
                return True

            def grab(self) -> bool:
                self.position += 1
                self.grab_count += 1
                return self.position < 8

            def retrieve(self):
                self.retrieve_indices.append(self.position)
                if self.position == 2:
                    return False, None
                return (
                    True,
                    np.full((4, 6, 3), self.position, dtype=np.uint8),
                )

            def release(self) -> None:
                self.released = True

        fake_capture = FakeCapture("unused")
        with patch.object(
            materializer.cv2,
            "VideoCapture",
            return_value=fake_capture,
        ) as video_capture:
            decoded = list(
                decode_candidate_frames_sequential(Path("exact.mp4"), candidates)
            )

        video_capture.assert_called_once_with("exact.mp4")
        self.assertEqual(fake_capture.grab_count, 6)
        self.assertEqual(fake_capture.retrieve_indices, [0, 2, 5])
        self.assertTrue(fake_capture.released)
        self.assertIsNone(decoded[0].error)
        self.assertEqual(int(decoded[0].frame[0, 0, 0]), 0)
        self.assertIn("target frame 2", decoded[1].error)
        self.assertIsNone(decoded[2].error)
        self.assertEqual(int(decoded[2].frame[0, 0, 0]), 5)


if __name__ == "__main__":
    unittest.main()
