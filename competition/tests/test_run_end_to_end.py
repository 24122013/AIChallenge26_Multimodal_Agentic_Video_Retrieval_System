from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from competition.run_end_to_end import (
    STAGES,
    build_parser,
    build_stage_commands,
    _run_stage_with_native_retries,
    run,
    runtime_preflight,
    selected_stage_names,
)


class EndToEndRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, *extra: str):
        return build_parser().parse_args(
            [
                "--public-root",
                str(self.root / "public"),
                "--output-root",
                str(self.root / "artifacts"),
                *extra,
            ]
        )

    def test_default_command_chain_uses_phase3_algorithm_and_exact_submission(self) -> None:
        args = self._args()
        commands = build_stage_commands(args, python_executable="python-test")

        self.assertEqual(tuple(commands), STAGES)
        keyframes = commands["keyframes"]
        self.assertEqual(keyframes[:4], [
            "python-test",
            "-m",
            "competition.pipeline",
            "keyframes",
        ])
        self.assertIn("--resume", keyframes)
        self.assertEqual(
            keyframes[keyframes.index("--candidate-interval-sec") + 1],
            "0.5",
        )
        self.assertEqual(
            keyframes[keyframes.index("--max-gap-seconds") + 1],
            "2.0",
        )
        submission = str((self.root / "artifacts" / "results" / "submission.csv").resolve())
        self.assertIn(submission, commands["predict"])
        self.assertIn(submission, commands["validate-submission"])

    def test_fresh_and_explicit_submission_options_are_forwarded(self) -> None:
        submission = self.root / "final.csv"
        args = self._args(
            "--fresh",
            "--submission-path",
            str(submission),
            "--device",
            "cuda",
            "--batch-size",
            "8",
            "--no-autocast",
        )
        commands = build_stage_commands(args)

        self.assertNotIn("--resume", commands["keyframes"])
        self.assertIn("--no-autocast", commands["keyframes"])
        self.assertIn("--no-autocast", commands["predict"])
        self.assertEqual(
            commands["predict"][commands["predict"].index("--device") + 1],
            "cuda",
        )
        self.assertIn(str(submission.resolve()), commands["validate-submission"])

    def test_restart_range_is_ordered_and_rejects_reverse_range(self) -> None:
        self.assertEqual(
            selected_stage_names("segments", "validate-submission"),
            ("segments", "text-index", "predict", "validate-submission"),
        )
        with self.assertRaisesRegex(ValueError, "must not come after"):
            selected_stage_names("predict", "index")

    def test_batch_size_parser_rejects_invalid_values(self) -> None:
        with self.assertRaises(SystemExit):
            self._args("--batch-size", "0")
        with self.assertRaises(SystemExit):
            self._args("--batch-size", "many")

    def test_score_parser_rejects_values_outside_unit_interval(self) -> None:
        with self.assertRaises(SystemExit):
            self._args("--public-score", "1.01")
        with self.assertRaises(SystemExit):
            self._args("--private-score", "-0.01")

    @mock.patch("competition.run_end_to_end._run_stage_with_native_retries")
    def test_completed_run_appends_experiment_report(
        self,
        stage_mock: mock.Mock,
    ) -> None:
        report_path = self.root / "reports" / "Experiment.md"
        args = self._args(
            "--start-at",
            "validate-input",
            "--stop-after",
            "validate-input",
            "--experiment-report",
            str(report_path),
        )

        run(args)

        stage_mock.assert_called_once()
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("end_to_end_runner", content)
        self.assertIn("Status: `completed`", content)

    @mock.patch("competition.run_end_to_end._run_stage_with_native_retries")
    def test_failed_run_is_also_recorded(self, stage_mock: mock.Mock) -> None:
        report_path = self.root / "reports" / "Experiment.md"
        args = self._args(
            "--start-at",
            "validate-input",
            "--stop-after",
            "validate-input",
            "--experiment-report",
            str(report_path),
        )
        stage_mock.side_effect = subprocess.CalledProcessError(2, ["python-test"])

        with self.assertRaises(subprocess.CalledProcessError):
            run(args)

        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Status: `failed`", content)
        self.assertIn("CalledProcessError", content)

    @mock.patch("competition.run_end_to_end.subprocess.run")
    def test_native_access_violation_retries_resumable_keyframes(
        self,
        run_mock: mock.Mock,
    ) -> None:
        run_mock.side_effect = [
            SimpleNamespace(returncode=3221225477),
            SimpleNamespace(returncode=0),
        ]

        _run_stage_with_native_retries(
            ["python-test", "-m", "competition.pipeline", "keyframes"],
            stage="keyframes",
            environment={},
            retries=2,
        )

        self.assertEqual(run_mock.call_count, 2)

    @mock.patch("competition.run_end_to_end.subprocess.run")
    def test_native_access_violation_does_not_retry_other_stages(
        self,
        run_mock: mock.Mock,
    ) -> None:
        run_mock.return_value = SimpleNamespace(returncode=3221225477)

        with self.assertRaises(subprocess.CalledProcessError):
            _run_stage_with_native_retries(
                ["python-test", "-m", "competition.pipeline", "predict"],
                stage="predict",
                environment={},
                retries=2,
            )

        self.assertEqual(run_mock.call_count, 1)

    @mock.patch("competition.run_end_to_end.subprocess.run")
    def test_cuda_oom_exit_retries_keyframes(self, run_mock: mock.Mock) -> None:
        run_mock.side_effect = [
            SimpleNamespace(returncode=75),
            SimpleNamespace(returncode=0),
        ]

        _run_stage_with_native_retries(
            ["python-test", "-m", "competition.pipeline", "keyframes"],
            stage="keyframes",
            environment={},
            retries=1,
        )

        self.assertEqual(run_mock.call_count, 2)

    @mock.patch("competition.run_end_to_end.shutil.which", return_value="available")
    def test_cuda_preflight_rejects_cpu_only_torch(self, _which: mock.Mock) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.13.0+cpu",
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        )

        with self.assertRaisesRegex(ValueError, "CPU-only PyTorch build"):
            runtime_preflight(
                device="cuda",
                require_ffmpeg=True,
                torch_module=fake_torch,
            )

        report = runtime_preflight(
            device="auto",
            require_ffmpeg=True,
            torch_module=fake_torch,
        )
        self.assertEqual(report["resolved_device"], "cpu")

    @mock.patch("competition.run_end_to_end.shutil.which", return_value="available")
    def test_cuda_preflight_reports_selected_gpu(self, _which: mock.Mock) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.13.0+cu130",
            version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _index: "Fake GPU",
            ),
        )

        report = runtime_preflight(
            device="cuda",
            require_ffmpeg=True,
            torch_module=fake_torch,
        )
        self.assertEqual(report["resolved_device"], "cuda")
        self.assertEqual(report["device_name"], "Fake GPU")

    @mock.patch("competition.run_end_to_end.shutil.which", return_value="available")
    def test_cuda_preflight_rejects_cpu_only_paddle(self, _which: mock.Mock) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.13.0+cu130",
            version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _index: "Fake GPU",
            ),
        )
        fake_paddle = SimpleNamespace(
            __version__="3.2.2",
            device=SimpleNamespace(is_compiled_with_cuda=lambda: False),
        )

        with self.assertRaisesRegex(ValueError, "CUDA-enabled PaddlePaddle wheel"):
            runtime_preflight(
                device="cuda",
                require_ffmpeg=True,
                require_paddle=True,
                torch_module=fake_torch,
                paddle_module=fake_paddle,
            )

    @mock.patch("competition.run_end_to_end.shutil.which", return_value="available")
    @mock.patch("competition.run_end_to_end.importlib.import_module")
    def test_cuda_preflight_smokes_full_model_stack(
        self,
        import_mock: mock.Mock,
        _which: mock.Mock,
    ) -> None:
        class FakeTensor:
            def numpy(self):
                return [1.0]

        fake_torch = SimpleNamespace(
            __version__="2.13.0+cu130",
            version=SimpleNamespace(cuda="13.0"),
            ones=lambda *_args, **_kwargs: FakeTensor(),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _index: "Fake GPU",
                synchronize=lambda: None,
            ),
        )
        fake_paddle = SimpleNamespace(
            __version__="3.2.2",
            ones=lambda *_args, **_kwargs: FakeTensor(),
            device=SimpleNamespace(
                is_compiled_with_cuda=lambda: True,
                set_device=lambda _device: None,
                cuda=SimpleNamespace(empty_cache=lambda: None),
            ),
        )
        import_mock.side_effect = [
            SimpleNamespace(PaddleOCR=object()),
            SimpleNamespace(YOLOE=object()),
            SimpleNamespace(AutoModelForMultimodalLM=object()),
            SimpleNamespace(TransNetV2=object()),
        ]

        report = runtime_preflight(
            device="cuda",
            require_ffmpeg=False,
            require_paddle=True,
            torch_module=fake_torch,
            paddle_module=fake_paddle,
        )

        self.assertTrue(report["torch_cuda_smoke_tested"])
        self.assertTrue(report["paddle_cuda_smoke_tested"])
        self.assertEqual(len(report["model_stack_imports"]), 4)


if __name__ == "__main__":
    unittest.main()
