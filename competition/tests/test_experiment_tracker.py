from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from competition.experiment_tracker import (
    _load_records,
    append_experiment,
    collect_local_metrics,
)


class ExperimentTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.public_root = self.root / "public"
        self.output_root = self.root / "competition"
        self.report_path = self.root / "reports" / "Experiment.md"
        self.public_root.mkdir(parents=True)
        (self.public_root / "corpus.csv").write_text(
            "video,path,duration_seconds,fps,frame_count,width,height\n"
            "video001.mp4,videos/video001.mp4,10,25,250,640,360\n",
            encoding="utf-8",
        )
        (self.public_root / "questions.csv").write_text(
            "query_id,task,text,query_image\n"
            "Q1,TKIS,test,\nQ2,VKIS,,queries/q.jpg\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_collects_artifact_backed_metrics(self) -> None:
        work = self.output_root / "work" / "keyframe_v3" / "video001" / "run1"
        work.mkdir(parents=True)
        (work / "candidates.jsonl").write_text("{}\n{}\n", encoding="utf-8")
        for name in (
            "siglip2.npy",
            "captions.jsonl",
            "ocr.jsonl",
            "objects.jsonl",
            "asr.jsonl",
            "feature_manifest.json",
        ):
            (work / name).write_bytes(b"artifact")

        metadata = self.output_root / "metadata"
        metadata.mkdir(parents=True)
        (metadata / "keyframes_video001_extract_report.json").write_text(
            json.dumps(
                {
                    "candidate_count": 2,
                    "keyframe_count": 1,
                    "guarantees": {
                        "constraints_satisfied": True,
                        "event_recall_satisfied": True,
                        "observed_max_gap_seconds": 4.5,
                    },
                }
            ),
            encoding="utf-8",
        )
        (metadata / "siglip2_so400m_patch16_384_faiss_manifest.json").write_text(
            json.dumps(
                {
                    "index_type": "IndexFlatIP",
                    "metric": "ip",
                    "vector_count": 1,
                    "index_file_size_mb": 0.01,
                }
            ),
            encoding="utf-8",
        )
        submission = self.output_root / "results" / "submission.csv"
        submission.parent.mkdir(parents=True)
        submission.write_text("query_id,a1\nQ1,video001.mp4,0\n", encoding="utf-8")

        metrics = collect_local_metrics(
            public_root=self.public_root,
            output_root=self.output_root,
            submission_path=submission,
        )

        self.assertEqual(metrics["dataset"]["video_count"], 1)
        self.assertEqual(metrics["dataset"]["tasks"], {"TKIS": 1, "VKIS": 1})
        self.assertEqual(metrics["workspace"]["candidate_count"], 2)
        self.assertEqual(metrics["workspace"]["feature_manifest_count"], 1)
        self.assertEqual(metrics["canonical"]["selection_ratio"], 0.5)
        self.assertEqual(metrics["canonical"]["constraint_pass_rate"], 1.0)
        self.assertEqual(metrics["index"]["vector_count"], 1)
        self.assertTrue(metrics["submission"]["exists"])

    def test_append_preserves_history_and_updates_best_score(self) -> None:
        first = {
            "experiment_id": "EXP-1",
            "recorded_at": "2026-08-09T00:00:00+00:00",
            "source": "test",
            "status": "completed",
            "public_score": 0.8,
            "private_score": 0.7,
        }
        second = {
            "experiment_id": "EXP-2",
            "recorded_at": "2026-08-09T01:00:00+00:00",
            "source": "test",
            "status": "completed",
            "public_score": 0.82,
            "private_score": 0.75,
        }

        append_experiment(self.report_path, first)
        append_experiment(self.report_path, second)
        content = self.report_path.read_text(encoding="utf-8")

        self.assertEqual(len(_load_records(content)), 2)
        self.assertIn("| Public | 0.820000 | EXP-2 |", content)
        self.assertIn("| Private | 0.750000 | EXP-2 |", content)
        self.assertIn("## EXP-1", content)
        self.assertIn("## EXP-2", content)

    def test_failed_run_does_not_replace_best_score(self) -> None:
        append_experiment(
            self.report_path,
            {
                "experiment_id": "GOOD",
                "status": "completed",
                "public_score": 0.81,
                "private_score": 0.80,
            },
        )
        append_experiment(
            self.report_path,
            {
                "experiment_id": "FAILED",
                "status": "failed",
                "public_score": 0.99,
                "private_score": 0.99,
            },
        )
        content = self.report_path.read_text(encoding="utf-8")
        self.assertIn("| Public | 0.810000 | GOOD |", content)
        self.assertIn("| Private | 0.800000 | GOOD |", content)


if __name__ == "__main__":
    unittest.main()
