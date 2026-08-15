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
from competition.pipeline import Question, precompute_tkis_query_plans
from backend.app.services.agent.query_expansion import (
    ProviderResponse,
    QueryExpansionConfig,
)


class _Provider:
    provider_name = "test-production"
    model_name = "test/model"
    model_revision = "revision"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False

    def expand(self, query, _protected):
        self.calls.append(query)
        return ProviderResponse(
            {
                "paraphrases": ["a person beside a bus"],
                "objects": ["man", "bus"],
                "attributes": [],
                "actions": [],
                "relations": ["beside"],
                "ocr_literals": [],
                "scene_terms": [],
            }
        )

    def close(self) -> None:
        self.closed = True


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
<<<<<<< HEAD
            commands["keyframes"][commands["keyframes"].index("--caption-quantization") + 1],
            "none",
=======
            commands["keyframes"][
                commands["keyframes"].index("--caption-model-name") + 1
            ],
            "Qwen/Qwen3.5-4B",
        )
        self.assertEqual(
            commands["keyframes"][
                commands["keyframes"].index("--caption-model-revision") + 1
            ],
            "c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97",
>>>>>>> origin/main
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
<<<<<<< HEAD
        self.assertEqual(predict[predict.index("--bge-dense-mode") + 1], "required")
        self.assertEqual(
            predict[predict.index("--bge-reranker-mode") + 1],
            "required",
        )
        self.assertIn("--model-revision", commands["bge-text-index"])
        self.assertIn("--canonical-only", commands["bge-text-index"])
        self.assertEqual(
            commands["bge-text-index"][
                commands["bge-text-index"].index("--metadata") + 1
            ],
            str(self.run_root.resolve() / "metadata"),
        )
=======
        self.assertEqual(predict[predict.index("--tkis-routing") + 1], "hybrid")
        self.assertEqual(predict[predict.index("--retrieval-profile") + 1], "kis")
        self.assertIn("--query-expansion-cache-dir", predict)
        self.assertNotIn("--no-query-expansion", predict)
>>>>>>> origin/main
        self.assertTrue(predict[predict.index("--retrieval-config") + 1].endswith("retrieval.yaml"))

    def test_query_expansion_ablation_is_forwarded(self) -> None:
        commands = build_stage_commands(
            self._args("--no-query-expansion"),
            python_executable="python-test",
        )
        self.assertIn("--no-query-expansion", commands["predict"])

    def test_colab_caption_memory_settings_are_forwarded(self) -> None:
        commands = build_stage_commands(
            self._args(
                "--caption-batch-size",
                "1",
                "--caption-quantization",
                "4bit",
            ),
            python_executable="python-test",
        )
        keyframes = commands["keyframes"]
        self.assertEqual(
            keyframes[keyframes.index("--caption-batch-size") + 1],
            "1",
        )
        self.assertEqual(
            keyframes[keyframes.index("--caption-quantization") + 1],
            "4bit",
        )

    def test_offline_model_cache_applies_to_keyframes_and_predict(self) -> None:
        commands = build_stage_commands(
            self._args("--offline-model-cache"),
            python_executable="python-test",
        )
        self.assertIn("--offline-model-cache", commands["keyframes"])
        self.assertIn("--offline-model-cache", commands["predict"])

    def test_precompute_calls_production_provider_only_for_tkis(self) -> None:
        provider = _Provider()
        factory_calls: list[dict] = []

        def factory(**kwargs):
            factory_calls.append(kwargs)
            return provider

        plans = precompute_tkis_query_plans(
            [
                Question("t", "TKIS", "a man next to a bus", ""),
                Question("v", "VKIS", "", "queries/v.jpg"),
            ],
            profile="kis",
            expansion_config=QueryExpansionConfig(),
            device="cpu",
            cache_dir=self.root / "responses",
            model_cache_dir=self.root / "models",
            local_files_only=True,
            provider_factory=factory,
        )
        self.assertEqual(list(plans), ["t"])
        self.assertEqual(provider.calls, ["a man next to a bus"])
        self.assertTrue(provider.closed)
        self.assertEqual(len(factory_calls), 1)

    def test_precompute_does_not_construct_provider_for_vkis_or_ablation(self) -> None:
        def forbidden_factory(**_kwargs):
            raise AssertionError("provider must not be constructed")

        self.assertEqual(
            precompute_tkis_query_plans(
                [Question("v", "VKIS", "", "queries/v.jpg")],
                profile="kis",
                expansion_config=QueryExpansionConfig(),
                device="cpu",
                cache_dir=self.root / "responses",
                model_cache_dir=self.root / "models",
                local_files_only=True,
                provider_factory=forbidden_factory,
            ),
            {},
        )
        plans = precompute_tkis_query_plans(
            [Question("t", "TKIS", "a bus", "")],
            profile="kis",
            expansion_config=QueryExpansionConfig(enabled=False),
            device="cpu",
            cache_dir=self.root / "responses",
            model_cache_dir=self.root / "models",
            local_files_only=True,
            provider_factory=forbidden_factory,
        )
        self.assertEqual(plans["t"].expansion_plan.status, "disabled")

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

        self.assertEqual(stage_mock.call_count, len(STAGES))
        append_mock.assert_called_once()
        manifest = (self.run_root / "run_manifest.json").read_text(encoding="utf-8")
        self.assertIn('"status": "architecture_complete"', manifest)
        self.assertIn('"recorded_after_full_validation": true', manifest)


if __name__ == "__main__":
    unittest.main()
