from __future__ import annotations

import tempfile
import unittest
import shutil
from pathlib import Path
from unittest import mock

from competition.run_retrieval_v2 import (
    STAGES,
    build_parser,
    build_stage_commands,
    run,
    selected_stage_names,
)
from competition.run_manifest import dataset_fingerprint


class RetrievalV2RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.public_root = self.root / "public"
        self.public_root.mkdir()
        self.run_root = self.root / "runs" / "r1"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, *extra: str):
        return build_parser().parse_args(
            [
                "--public-root",
                str(self.public_root),
                "--run-root",
                str(self.run_root),
                *extra,
            ]
        )

    def test_command_chain_builds_one_coherent_advanced_run(self) -> None:
        commands = build_stage_commands(self._args(), python_executable="python-test")

        self.assertEqual(tuple(commands), STAGES)
        self.assertIn("--endpoint-protection", commands["keyframes"])
        self.assertEqual(
            commands["keyframes"][commands["keyframes"].index("--endpoint-protection") + 1],
            "on",
        )
        self.assertIn("--resume", commands["keyframes"])
        self.assertEqual(
            commands["keyframes"][commands["keyframes"].index("--caption-quantization") + 1],
            "none",
        )
        dense = commands["dense-index"]
        self.assertEqual(
            dense[dense.index("--source-output-root") + 1],
            str(self.run_root.resolve()),
        )
        predict = commands["predict"]
        self.assertEqual(predict[predict.index("--retrieval-mode") + 1], "advanced")
        self.assertEqual(
            predict[predict.index("--dense-run-root") + 1],
            str(self.run_root.resolve()),
        )
        self.assertEqual(predict[predict.index("--vlm-mode") + 1], "off")
        self.assertEqual(predict[predict.index("--bge-dense-mode") + 1], "off")
        self.assertEqual(predict[predict.index("--bge-reranker-mode") + 1], "off")
        self.assertIn("--model-revision", commands["bge-text-index"])
        self.assertTrue(predict[predict.index("--retrieval-config") + 1].endswith("retrieval.yaml"))

    def test_stage_range_is_ordered(self) -> None:
        self.assertEqual(
            selected_stage_names("dense-index", "validate-submission"),
            ("dense-index", "predict", "validate-submission"),
        )
        with self.assertRaisesRegex(ValueError, "must not come after"):
            selected_stage_names("predict", "keyframes")

    def test_dataset_fingerprint_survives_copy_and_mtime_change(self) -> None:
        source = self.root / "dataset-a"
        source.mkdir()
        (source / "video.mp4").write_bytes(b"stable-public-bytes")
        moved = self.root / "dataset-b"
        shutil.copytree(source, moved)
        moved_file = moved / "video.mp4"
        moved_file.touch()

        self.assertEqual(
            dataset_fingerprint(source)["sha256"],
            dataset_fingerprint(moved)["sha256"],
        )

    def test_dry_run_does_not_create_run_or_experiment(self) -> None:
        report = self.root / "Experiment.md"
        run(self._args("--dry-run", "--experiment-report", str(report)))

        self.assertFalse(self.run_root.exists())
        self.assertFalse(report.exists())

    @mock.patch("competition.run_retrieval_v2.runtime_preflight")
    @mock.patch("competition.run_retrieval_v2._run_stage_with_native_retries")
    def test_partial_run_does_not_append_experiment(
        self,
        stage_mock: mock.Mock,
        preflight_mock: mock.Mock,
    ) -> None:
        preflight_mock.return_value = {"resolved_device": "cpu"}
        report = self.root / "Experiment.md"
        run(
            self._args(
                "--stop-after",
                "validate-input",
                "--experiment-report",
                str(report),
            )
        )

        stage_mock.assert_called_once()
        self.assertFalse(report.exists())

    @mock.patch("competition.run_retrieval_v2.append_experiment")
    @mock.patch("competition.run_retrieval_v2.build_runner_record")
    @mock.patch("competition.run_retrieval_v2._collect_offline_lineage")
    @mock.patch("competition.run_retrieval_v2.runtime_preflight")
    @mock.patch("competition.run_retrieval_v2._run_stage_with_native_retries")
    def test_full_validated_chain_is_the_only_experiment_commit_point(
        self,
        stage_mock: mock.Mock,
        preflight_mock: mock.Mock,
        _lineage_mock: mock.Mock,
        record_mock: mock.Mock,
        append_mock: mock.Mock,
    ) -> None:
        preflight_mock.return_value = {"resolved_device": "cuda"}
        record_mock.return_value = {"experiment_id": "EXP-test"}

        run(self._args())

        # The disabled BGE index stage is an explicit passed/skipped stage and
        # therefore does not spawn a subprocess.
        self.assertEqual(stage_mock.call_count, len(STAGES) - 1)
        append_mock.assert_called_once()
        manifest = (self.run_root / "run_manifest.json").read_text(encoding="utf-8")
        self.assertIn('"status": "architecture_complete"', manifest)
        self.assertIn('"recorded_after_full_validation": true', manifest)


if __name__ == "__main__":
    unittest.main()
