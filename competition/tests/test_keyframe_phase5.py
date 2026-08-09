from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from competition.keyframe_phase3 import sha256_file
from competition.keyframe_phase5 import (
    create_config_lock,
    create_split_manifest,
    evaluate_split_artifacts,
    load_split_manifest,
    validate_config_lock,
    validate_split_manifest,
    write_config_lock,
    write_split_manifest,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _artifact_entry(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class KeyframePhase5Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output_root = self.root / "competition"
        self.metadata_dir = self.output_root / "metadata"
        self.video_ids = [f"video_{index:03d}" for index in range(16)]
        self.split_path = self.root / "phase5_split.json"
        self.split_manifest = write_split_manifest(
            self.split_path,
            self.video_ids,
            seed=42,
        )
        self.common_config = {
            "dense_interval_seconds": 0.5,
            "max_gap_seconds": 3.0,
            "feature_backend": "fake-ci",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish_phase3(
        self,
        video_id: str,
        *,
        extract_config: dict[str, object] | None = None,
    ) -> None:
        candidate_ids = [f"CAND_{video_id}_0", f"CAND_{video_id}_1"]
        image_dir = self.output_root / "keyframes" / video_id
        image_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [image_dir / "000001.jpg", image_dir / "000004.jpg"]
        for index, image_path in enumerate(image_paths):
            image_path.write_bytes(f"synthetic-frame-{video_id}-{index}".encode("ascii"))

        candidates = [
            {
                "candidate_id": candidate_ids[0],
                "video_id": video_id,
                "timestamp": 1.0,
                "shot_index": 0,
            },
            {
                "candidate_id": candidate_ids[1],
                "video_id": video_id,
                "timestamp": 4.0,
                "shot_index": 1,
            },
        ]
        selected = [
            {
                **candidates[0],
                "keyframe_path": image_paths[0].as_posix(),
                "protected": True,
            },
            {
                **candidates[1],
                "keyframe_path": image_paths[1].as_posix(),
                "protected": False,
            },
        ]
        events = [
            {
                "event_id": f"OCR_{video_id}",
                "source": "feature_adapter",
                "event_type": "ocr_new",
                "candidate_ids": [candidate_ids[0]],
            }
        ]
        metadata_path = self.metadata_dir / f"keyframes_{video_id}.jsonl"
        candidate_path = self.metadata_dir / f"candidate_scores_{video_id}.jsonl"
        event_path = self.metadata_dir / f"protected_events_{video_id}.jsonl"
        _write_jsonl(metadata_path, selected)
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(event_path, events)

        run_id = f"RUN_{video_id}"
        phase3_manifest_path = self.metadata_dir / f"phase3_{video_id}.json"
        phase3_manifest = {
            "status": "passed",
            "video_id": video_id,
            "selection_run_id": run_id,
            "degraded": False,
            "allow_partial_features": False,
            "feature_config": {
                "siglip2": {"model": "fake-ci", "revision": "fixed"},
                "ocr": {"backend": "fake-ci"},
                "objects": {"backend": "fake-ci"},
            },
            "adapter_config": {
                "ocr_min_confidence": 0.8,
                "object_min_confidence": 0.8,
            },
            "selection_config": {
                "max_gap_seconds": 3.0,
                "gap_tolerance_seconds": 0.0,
                "target_keyframes": 1,
            },
            "canonical_artifacts": {
                "keyframe_metadata": _artifact_entry(metadata_path),
                "candidate_scores": _artifact_entry(candidate_path),
                "protected_events": _artifact_entry(event_path),
            },
        }
        _write_json(phase3_manifest_path, phase3_manifest)
        extract_report = {
            "video_id": video_id,
            "keyframe_strategy": "multimodal_coverage",
            "status": "satisfied",
            "duration": 5.0,
            "phase3_manifest_path": phase3_manifest_path.as_posix(),
            "phase3_manifest_sha256": sha256_file(phase3_manifest_path),
            "phase3_selection_run_id": run_id,
            "selection_config": {
                "max_gap_seconds": 3.0,
                "gap_tolerance_seconds": 0.0,
                "target_keyframes": 1,
            },
            "competition_extract_config": dict(
                extract_config if extract_config is not None else self.common_config
            ),
            "guarantees": {
                "coverage_satisfied": True,
                "shot_coverage_satisfied": True,
                "protected_events_satisfied": True,
            },
        }
        _write_json(
            self.metadata_dir / f"keyframes_{video_id}_extract_report.json",
            extract_report,
        )

    def _publish_all(self, *, validation_config: dict | None = None) -> None:
        validation_ids = set(self.split_manifest["splits"]["validation"])
        for video_id in self.video_ids:
            config = validation_config if video_id in validation_ids else None
            self._publish_phase3(video_id, extract_config=config)

    def _canonical_sources(self, split: str) -> list[dict[str, str]]:
        return [
            {"video_id": video_id}
            for video_id in self.split_manifest["splits"][split]
        ]

    def test_split_is_deterministic_exact_and_tamper_evident(self) -> None:
        first = create_split_manifest(list(reversed(self.video_ids)), seed=42)
        second = create_split_manifest(self.video_ids, seed=42)

        self.assertEqual(first, second)
        self.assertEqual({name: len(ids) for name, ids in first["splits"].items()}, {
            "dev": 4,
            "validation": 4,
            "test": 8,
        })
        self.assertEqual(load_split_manifest(self.split_path), self.split_manifest)

        tampered = dict(first)
        tampered["seed"] = 43
        with self.assertRaisesRegex(ValueError, "modified or is inconsistent"):
            validate_split_manifest(tampered)

    def test_synthetic_validation_report_uses_lock_and_optional_evidence(self) -> None:
        self._publish_all()
        lock_path = self.root / "phase5_config_lock.json"
        lock = write_config_lock(
            lock_path,
            output_root=self.output_root,
            split_manifest_path=self.split_path,
        )
        validation_id = self.split_manifest["splits"]["validation"][0]
        event_id = f"OCR_{validation_id}"
        manual_path = self.root / "manual_events.jsonl"
        review_path = self.root / "protection_reviews.jsonl"
        resource_path = self.root / "resource_usage.jsonl"
        retrieval_path = self.root / "retrieval.jsonl"
        _write_jsonl(
            manual_path,
            [{
                "video_id": validation_id,
                "event_id": "MANUAL_TITLE",
                "start_time": 0.9,
                "end_time": 1.1,
            }],
        )
        _write_jsonl(
            review_path,
            [{
                "video_id": validation_id,
                "detected_event_id": event_id,
                "is_true_event": True,
            }],
        )
        _write_jsonl(
            resource_path,
            [{
                "video_id": validation_id,
                "runtime_sec": 2.0,
                "peak_ram_mb": 64.0,
            }],
        )
        _write_jsonl(
            retrieval_path,
            [{
                "query_id": "QUERY_TITLE",
                "relevant": [{
                    "video_id": validation_id,
                    "start_frame": 10,
                    "end_frame": 20,
                }],
                "ranked_results": [{
                    "video_id": validation_id,
                    "frame_index": 15,
                }],
            }],
        )

        report = evaluate_split_artifacts(
            output_root=self.output_root,
            split_manifest_path=self.split_path,
            split="validation",
            canonical_sources=self._canonical_sources("validation"),
            config_lock_path=lock_path,
            manual_events_path=manual_path,
            protection_reviews_path=review_path,
            resource_usage_path=resource_path,
            retrieval_evidence_path=retrieval_path,
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["config_sha256"], lock["config_sha256"])
        self.assertIn("feature_config", lock["phase3_config"])
        self.assertEqual(report["aggregate"]["video_count"], 4)
        self.assertEqual(report["aggregate"]["coverage_violation_count"], 0)
        self.assertEqual(report["aggregate"]["effective_shot_recall"], 1.0)
        self.assertEqual(
            report["aggregate"]["detected_protected_event_recall"],
            1.0,
        )
        self.assertEqual(report["aggregate"]["manual_end_to_end_event_recall"], 1.0)
        self.assertEqual(report["aggregate"]["false_protection_rate"], 0.0)
        self.assertEqual(report["aggregate"]["soft_budget_overrun_count"], 4)
        self.assertEqual(report["aggregate"]["soft_budget_overrun_ratio"], 1.0)
        self.assertEqual(report["aggregate"]["runtime_measurement_coverage"], 0.25)
        self.assertEqual(report["aggregate"]["retrieval"]["hit_at_1"], 1.0)
        self.assertEqual(len(report["artifact_lineage"]), 4)

    def test_validation_rejects_config_drift_from_dev_lock(self) -> None:
        drifted_config = {**self.common_config, "dense_interval_seconds": 1.0}
        self._publish_all(validation_config=drifted_config)
        lock_path = self.root / "phase5_config_lock.json"
        _write_json(
            lock_path,
            create_config_lock(
                output_root=self.output_root,
                split_manifest_path=self.split_path,
            ),
        )

        with self.assertRaisesRegex(ValueError, "config lock is missing, stale"):
            evaluate_split_artifacts(
                output_root=self.output_root,
                split_manifest_path=self.split_path,
                split="validation",
                canonical_sources=self._canonical_sources("validation"),
                config_lock_path=lock_path,
            )

    def test_config_lock_rejects_tampered_source_lineage(self) -> None:
        self._publish_all()
        lock = create_config_lock(
            output_root=self.output_root,
            split_manifest_path=self.split_path,
        )
        lock_path = self.root / "phase5_config_lock.json"
        lock["source_selection_runs"][self.split_manifest["splits"]["dev"][0]] = (
            "TAMPERED"
        )
        _write_json(lock_path, lock)

        with self.assertRaisesRegex(ValueError, "config lock is missing, stale"):
            validate_config_lock(
                lock_path,
                split_manifest_path=self.split_path,
                expected_config_sha256=lock["config_sha256"],
            )

    def test_locked_test_requires_dev_config_lock(self) -> None:
        self._publish_all()

        with self.assertRaisesRegex(ValueError, "requires a dev-derived config lock"):
            evaluate_split_artifacts(
                output_root=self.output_root,
                split_manifest_path=self.split_path,
                split="test",
                canonical_sources=self._canonical_sources("test"),
            )


if __name__ == "__main__":
    unittest.main()
