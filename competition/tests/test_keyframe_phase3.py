from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from competition.keyframe_phase3 import (
    CandidatePoolConfig,
    atomic_save_npy,
    atomic_write_json,
    atomic_write_jsonl,
    candidate_run_contract,
    materialize_candidate_pool,
    read_json,
    read_jsonl,
    workspace_paths,
)


class _FakeDenseExtractor:
    def __init__(self, *, candidate_count: int = 3, skipped_count: int = 0) -> None:
        self.candidate_count = candidate_count
        self.skipped_count = skipped_count
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict:
        self.calls.append(dict(kwargs))
        video_path = Path(kwargs["video_path"])
        output_dir = Path(kwargs["output_dir"])
        metadata_path = Path(kwargs["metadata_path"])
        report_path = Path(kwargs["report_path"])
        video_id = video_path.stem
        materialized_count = self.candidate_count - self.skipped_count
        image_dir = output_dir / video_id
        image_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, object]] = []
        for index in range(materialized_count):
            image_path = image_dir / f"candidate_{index:06d}.jpg"
            image_path.write_bytes(f"candidate-image-{index}".encode("ascii"))
            records.append(
                {
                    "candidate_id": f"CAND_{video_id}_{index:06d}",
                    "frame_id": f"FRAME_{video_id}_{index:06d}",
                    "video_id": video_id,
                    "shot_id": f"SHOT_{video_id}_000000",
                    "shot_index": 0,
                    "frame_index": index * 5,
                    "timestamp": index * 0.5,
                    "keyframe_path": image_path.as_posix(),
                }
            )
        metadata_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        report = {
            "video_id": video_id,
            "keyframe_strategy": "dense_coverage",
            "candidate_count": self.candidate_count,
            "keyframe_count": materialized_count,
            "skipped_count": self.skipped_count,
            "constraints_satisfied": True,
            "coverage_satisfied": True,
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report


class KeyframePhase3WorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video_path = self.root / "video_001.mp4"
        self.video_path.write_bytes(b"source-video-v1")
        self.output_root = self.root / "artifacts"
        self.config = CandidatePoolConfig()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _materialize(
        self,
        extractor: _FakeDenseExtractor,
        *,
        resume: bool = False,
        config: CandidatePoolConfig | None = None,
    ):
        return materialize_candidate_pool(
            video_path=self.video_path,
            video_id="video_001",
            frame_count=40,
            output_root=self.output_root,
            config=config or self.config,
            resume=resume,
            extractor=extractor,
        )

    def test_materializes_every_planned_dense_candidate(self) -> None:
        extractor = _FakeDenseExtractor(candidate_count=4)

        paths, report, records = self._materialize(extractor)

        self.assertEqual(len(extractor.calls), 1)
        call = extractor.calls[0]
        for selector_only_argument in (
            "strategy",
            "max_gap_seconds",
            "gap_tolerance_seconds",
            "target_keyframes",
            "hard_max_keyframes",
        ):
            self.assertNotIn(selector_only_argument, call)
        self.assertEqual(call["phash_threshold"], self.config.phash_threshold)
        self.assertEqual(call["phash_window_sec"], self.config.phash_window_sec)
        self.assertEqual(call["jpeg_quality"], self.config.jpeg_quality)
        self.assertEqual(call["shot_threshold"], self.config.shot_threshold)
        self.assertEqual(call["shot_device"], self.config.shot_device)
        self.assertEqual(
            call["candidate_interval_sec"],
            self.config.candidate_interval_sec,
        )
        self.assertEqual(call["boundary_guard_sec"], self.config.boundary_guard_sec)
        self.assertEqual(call["tiny_shot_max_sec"], self.config.tiny_shot_max_sec)
        self.assertEqual(report["phase3_status"], "passed")
        self.assertEqual(report["planned_candidate_count"], 4)
        self.assertEqual(report["materialized_candidate_count"], 4)
        self.assertEqual(len(records), 4)
        self.assertTrue(paths.candidate_validation.parent.is_dir())
        self.assertTrue(all(row["artifact_role"] == "dense_candidate" for row in records))
        self.assertTrue(
            all(row["candidate_pool_run_id"] == paths.run_id for row in records)
        )

    def test_workspace_contract_and_run_id_are_stable(self) -> None:
        first_contract = candidate_run_contract(
            video_path=self.video_path,
            video_id="video_001",
            frame_count=40,
            config=self.config,
        )
        second_contract = candidate_run_contract(
            video_path=self.video_path,
            video_id="video_001",
            frame_count=40,
            config=self.config,
        )

        first_paths = workspace_paths(self.output_root, "video_001", first_contract)
        second_paths = workspace_paths(self.output_root, "video_001", second_contract)

        self.assertEqual(first_contract, second_contract)
        self.assertEqual(first_paths.run_id, second_paths.run_id)
        self.assertEqual(first_paths.root, second_paths.root)
        self.assertIn(first_paths.run_id, first_paths.candidate_images_dir.parts)

    def test_resume_with_exact_lineage_skips_extractor(self) -> None:
        extractor = _FakeDenseExtractor()
        first_paths, first_report, first_records = self._materialize(extractor)

        second_paths, second_report, second_records = self._materialize(
            extractor,
            resume=True,
        )

        self.assertEqual(len(extractor.calls), 1)
        self.assertEqual(second_paths, first_paths)
        self.assertEqual(second_report, first_report)
        self.assertEqual(second_records, first_records)

    def test_source_or_config_change_uses_a_new_run(self) -> None:
        extractor = _FakeDenseExtractor()
        first_paths, _, _ = self._materialize(extractor)

        self.video_path.write_bytes(b"source-video-v2")
        source_changed_paths, _, _ = self._materialize(extractor, resume=True)
        config_changed_paths, _, _ = self._materialize(
            extractor,
            resume=True,
            config=replace(self.config, candidate_interval_sec=0.25),
        )

        self.assertEqual(len(extractor.calls), 3)
        self.assertNotEqual(source_changed_paths.run_id, first_paths.run_id)
        self.assertNotEqual(config_changed_paths.run_id, source_changed_paths.run_id)
        self.assertEqual(len({first_paths.root, source_changed_paths.root, config_changed_paths.root}), 3)

    def test_corrupt_candidate_image_invalidates_resume_and_reruns(self) -> None:
        extractor = _FakeDenseExtractor()
        paths, _, records = self._materialize(extractor)
        corrupt_path = Path(records[0]["keyframe_path"])
        corrupt_path.write_bytes(b"corrupt")

        resumed_paths, report, resumed_records = self._materialize(
            extractor,
            resume=True,
        )

        self.assertEqual(len(extractor.calls), 2)
        self.assertEqual(resumed_paths.run_id, paths.run_id)
        self.assertEqual(report["phase3_status"], "passed")
        self.assertEqual(corrupt_path.read_bytes(), b"candidate-image-0")
        self.assertEqual(len(resumed_records), 3)

    def test_incomplete_or_skipped_pool_raises_and_records_partial_status(self) -> None:
        extractor = _FakeDenseExtractor(candidate_count=3, skipped_count=1)

        with self.assertRaisesRegex(RuntimeError, "materialization was incomplete"):
            self._materialize(extractor)

        contract = candidate_run_contract(
            video_path=self.video_path,
            video_id="video_001",
            frame_count=40,
            config=self.config,
        )
        paths = workspace_paths(self.output_root, "video_001", contract)
        report = read_json(paths.candidate_report)
        self.assertEqual(report["phase3_status"], "partial")
        self.assertEqual(report["planned_candidate_count"], 3)
        self.assertEqual(report["materialized_candidate_count"], 2)
        self.assertEqual(report["skipped_count"], 1)

    def test_atomic_npy_and_json_helpers_write_readable_artifacts(self) -> None:
        json_path = self.root / "atomic" / "manifest.json"
        jsonl_path = self.root / "atomic" / "records.jsonl"
        npy_path = self.root / "atomic" / "embeddings.npy"
        matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        atomic_write_json(json_path, {"status": "passed", "count": 2})
        atomic_write_jsonl(jsonl_path, [{"id": "a"}, {"id": "b"}])
        atomic_save_npy(npy_path, matrix)

        self.assertEqual(read_json(json_path), {"status": "passed", "count": 2})
        self.assertEqual(read_jsonl(jsonl_path), [{"id": "a"}, {"id": "b"}])
        np.testing.assert_array_equal(np.load(npy_path, allow_pickle=False), matrix)
        self.assertEqual(list(json_path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
