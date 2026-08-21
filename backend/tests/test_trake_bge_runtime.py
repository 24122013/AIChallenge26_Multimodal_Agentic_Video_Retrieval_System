from __future__ import annotations

import gc
import os
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.app.services.retrieval import retrieval_manager
from backend.app.services.retrieval.qa_evidence import BgeCandidateReranker
from backend.app.services.retrieval.bge_reranker import (
    clear_shared_bge_reranker_runners,
)
from backend.app.services.retrieval.retrieval_config import (
    RetrievalConfigError,
    TrakeConfig,
    load_retrieval_runtime_config,
)
from backend.app.services.trake import RequiredTrakePipelineError


class TrakeBgeConfigTest(unittest.TestCase):
    def test_defaults_are_safe_and_independent_from_qa_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "retrieval.yaml"
            path.write_text("{}\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "QA_BGE_DENSE_ENABLED": "true",
                    "QA_BGE_RERANKER_ENABLED": "true",
                },
                clear=True,
            ):
                config = load_retrieval_runtime_config(path).trake

        self.assertFalse(config.bge_dense_enabled)
        self.assertEqual(config.bge_dense_top_k, 300)
        self.assertFalse(config.bge_reranker_enabled)
        self.assertEqual(config.bge_reranker_top_k, 150)
        self.assertEqual(config.retrieval_fusion, "rrf")
        self.assertEqual(config.rrf_k, 60)
        self.assertEqual(config.hybrid_rrf_weight, 1.0)
        self.assertEqual(config.bge_rrf_weight, 1.0)
        self.assertFalse(config.bge_required)

    def test_every_bge_setting_has_an_environment_override(self) -> None:
        environment = {
            "RETRIEVAL_TRAKE_BGE_DENSE_ENABLED": "true",
            "RETRIEVAL_TRAKE_BGE_DENSE_TOP_K": "700",
            "RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED": "true",
            "RETRIEVAL_TRAKE_BGE_RERANKER_TOP_K": "175",
            "RETRIEVAL_TRAKE_RETRIEVAL_FUSION": "RRF",
            "RETRIEVAL_TRAKE_RRF_K": "75",
            "RETRIEVAL_TRAKE_HYBRID_RRF_WEIGHT": "0.75",
            "RETRIEVAL_TRAKE_BGE_RRF_WEIGHT": "1.25",
            "RETRIEVAL_TRAKE_BGE_REQUIRED": "true",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "retrieval.yaml"
            path.write_text("{}\n", encoding="utf-8")
            with mock.patch.dict(os.environ, environment, clear=True):
                config = load_retrieval_runtime_config(path).trake

        self.assertTrue(config.bge_dense_enabled)
        self.assertEqual(config.bge_dense_top_k, 700)
        self.assertTrue(config.bge_reranker_enabled)
        self.assertEqual(config.bge_reranker_top_k, 175)
        self.assertEqual(config.retrieval_fusion, "rrf")
        self.assertEqual(config.rrf_k, 75)
        self.assertEqual(config.hybrid_rrf_weight, 0.75)
        self.assertEqual(config.bge_rrf_weight, 1.25)
        self.assertTrue(config.bge_required)

    def test_yaml_boolean_is_not_coerced_from_a_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "retrieval.yaml"
            path.write_text(
                'trake:\n  bge_dense_enabled: "false"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RetrievalConfigError,
                "trake.bge_dense_enabled must be a boolean",
            ):
                load_retrieval_runtime_config(path)

    def test_bge_validation_rejects_unsafe_or_ineffective_configs(self) -> None:
        invalid = (
            ({"bge_dense_top_k": 0}, "positive integer"),
            ({"bge_dense_top_k": 10_001}, "must not exceed 10000"),
            ({"bge_reranker_top_k": 10_001}, "must not exceed 10000"),
            ({"rrf_k": 0}, "positive integer"),
            ({"retrieval_fusion": "score_sum"}, "must be rrf"),
            ({"hybrid_rrf_weight": -0.1}, "non-negative"),
            (
                {"hybrid_rrf_weight": 0.0, "bge_rrf_weight": 0.0},
                "must include a positive value",
            ),
            (
                {"hybrid_rrf_weight": 0.0, "bge_dense_enabled": False},
                "must be positive when BGE dense is disabled",
            ),
            (
                {"bge_dense_enabled": True, "bge_rrf_weight": 0.0},
                "must be positive when BGE dense is enabled",
            ),
            ({"bge_dense_enabled": 1}, "must be a boolean"),
            ({"bge_required": True}, "requires bge_dense_enabled"),
        )
        for kwargs, message in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                TrakeConfig(**kwargs)

    def test_invalid_environment_boolean_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "retrieval.yaml"
            path.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.dict(
                    os.environ,
                    {"RETRIEVAL_TRAKE_BGE_REQUIRED": "truthy"},
                    clear=True,
                ),
                self.assertRaisesRegex(ValueError, "must be a boolean value"),
            ):
                load_retrieval_runtime_config(path)


class TrakeBgeManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        retrieval_manager.clear_retrieval_caches()
        self.corpus_key = retrieval_manager._CorpusCacheKey(
            manifest_path="legacy",
            bundle_generation=None,
            manifest_contract_sha256=None,
        )

    def tearDown(self) -> None:
        retrieval_manager.clear_retrieval_caches()

    def _build_pipeline(
        self,
        config: TrakeConfig,
        *,
        dense: object | None = None,
        reranker: object | None = None,
    ) -> tuple[object, mock.Mock, mock.Mock]:
        visual_encoder = SimpleNamespace(
            encode=mock.Mock(),
            encode_images=mock.Mock(),
        )
        hybrid = SimpleNamespace(
            corpus_generation=None,
            visual_engine=SimpleNamespace(encoder=visual_encoder),
        )
        dense_getter = mock.Mock(return_value=dense)
        reranker_factory = mock.Mock(return_value=reranker)

        class FakeTrakePipeline:
            def __init__(
                self,
                *,
                retrieval_engine,
                dense_event_engine,
                event_reranker,
                bge_contract,
                local_scorer,
                config,
            ) -> None:
                self.retrieval_engine = retrieval_engine
                self.dense_event_engine = dense_event_engine
                self.event_reranker = event_reranker
                self.bge_contract = bge_contract
                self.local_scorer = local_scorer
                self.config = config

        with (
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=SimpleNamespace(trake=config),
            ),
            mock.patch.object(
                retrieval_manager,
                "get_hybrid_search_engine",
                return_value=hybrid,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_trake_bge_dense_search_engine",
                dense_getter,
            ),
            mock.patch.object(
                retrieval_manager,
                "build_trake_bge_candidate_reranker",
                reranker_factory,
            ),
            mock.patch.object(
                retrieval_manager,
                "_validate_expected_corpus",
                return_value=None,
            ),
            mock.patch(
                "backend.app.services.trake.pipeline.TrakePipeline",
                FakeTrakePipeline,
            ),
        ):
            pipeline = retrieval_manager.get_trake_pipeline.__wrapped__(
                self.corpus_key
            )
        return pipeline, dense_getter, reranker_factory

    def test_default_pipeline_does_not_construct_bge_components(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "QA_BGE_DENSE_ENABLED": "true",
                "QA_BGE_RERANKER_ENABLED": "true",
            },
            clear=False,
        ):
            pipeline, dense_getter, reranker_factory = self._build_pipeline(
                TrakeConfig()
            )

        self.assertIsNone(pipeline.dense_event_engine)
        self.assertIsNone(pipeline.event_reranker)
        self.assertIsNotNone(pipeline.local_scorer)
        dense_getter.assert_not_called()
        reranker_factory.assert_not_called()

    def test_dense_is_injected_but_model_reranker_is_removed(self) -> None:
        dense = SimpleNamespace(corpus_generation=None)
        reranker = object()
        pipeline, dense_getter, reranker_factory = self._build_pipeline(
            TrakeConfig(
                bge_dense_enabled=True,
                bge_reranker_enabled=True,
            ),
            dense=dense,
            reranker=reranker,
        )

        self.assertIs(pipeline.dense_event_engine, dense)
        self.assertIsNone(pipeline.event_reranker)
        dense_getter.assert_called_once_with()
        reranker_factory.assert_not_called()

    def test_optional_dense_initialization_failure_is_sanitized_and_fails_open(
        self,
    ) -> None:
        config = TrakeConfig(bge_dense_enabled=True)
        runtime = SimpleNamespace(trake=config)
        with (
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=runtime,
            ),
            mock.patch.object(
                retrieval_manager,
                "_bge_artifact_overrides",
                return_value={"bge_index": Path("secret/index.faiss")},
            ),
            mock.patch.object(
                retrieval_manager,
                "_validate_expected_corpus",
                return_value=None,
            ),
            mock.patch.object(
                retrieval_manager,
                "build_bge_m3_dense_search_engine",
                side_effect=FileNotFoundError("secret/index.faiss"),
            ),
            self.assertLogs(retrieval_manager.__name__, level="WARNING") as logs,
        ):
            engine = retrieval_manager.get_trake_bge_dense_search_engine.__wrapped__(
                self.corpus_key
            )

        self.assertIsNone(engine)
        message = " ".join(logs.output)
        self.assertIn("reason=initialization_failed", message)
        self.assertNotIn("secret/index.faiss", message)

    def test_required_dense_initialization_failure_has_fixed_public_error(self) -> None:
        config = TrakeConfig(bge_dense_enabled=True, bge_required=True)
        runtime = SimpleNamespace(trake=config)
        with (
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=runtime,
            ),
            mock.patch.object(
                retrieval_manager,
                "_bge_artifact_overrides",
                return_value={"bge_index": Path("secret/index.faiss")},
            ),
            mock.patch.object(
                retrieval_manager,
                "_validate_expected_corpus",
                return_value=None,
            ),
            mock.patch.object(
                retrieval_manager,
                "build_bge_m3_dense_search_engine",
                side_effect=FileNotFoundError("secret/index.faiss"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "^Required TRAKE BGE-M3 dense retrieval failed to initialize$",
            ),
        ):
            retrieval_manager.get_trake_bge_dense_search_engine.__wrapped__(
                self.corpus_key
            )

    def test_required_dense_rejects_mutable_manifest_revision(self) -> None:
        config = TrakeConfig(bge_dense_enabled=True, bge_required=True)
        engine = SimpleNamespace(
            artifacts=SimpleNamespace(
                manifest={
                    "model": {
                        "name": "BAAI/bge-m3",
                        "revision": "main",
                    }
                }
            )
        )
        with (
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=SimpleNamespace(trake=config),
            ),
            mock.patch.object(
                retrieval_manager,
                "_bge_artifact_overrides",
                return_value={},
            ),
            mock.patch.object(
                retrieval_manager,
                "_validate_expected_corpus",
                return_value={"bundle_generation": "generation"},
            ),
            mock.patch.object(
                retrieval_manager,
                "_shared_bge_m3_dense_search_engine",
                return_value=engine,
            ),
            self.assertRaises(RequiredTrakePipelineError) as captured,
        ):
            retrieval_manager.get_trake_bge_dense_search_engine.__wrapped__(
                self.corpus_key
            )

        self.assertEqual(
            captured.exception.failure_code,
            "required_bge_dense_revision_unpinned",
        )

    def test_trake_reranker_factory_uses_configured_candidate_limit(self) -> None:
        sentinel = object()
        config = TrakeConfig(
            bge_reranker_enabled=True,
            bge_reranker_top_k=175,
        )
        with (
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=SimpleNamespace(trake=config),
            ),
            mock.patch.object(
                retrieval_manager,
                "build_bge_candidate_reranker",
                return_value=sentinel,
            ) as factory,
        ):
            result = retrieval_manager.build_trake_bge_candidate_reranker()

        self.assertIs(result, sentinel)
        self.assertEqual(factory.call_args.kwargs["candidate_limit"], 175)


class SharedBgeRerankerAdapterTest(unittest.TestCase):
    def test_candidate_limit_is_explicit_and_forwarded(self) -> None:
        reranker = BgeCandidateReranker(candidate_limit=150)
        report = SimpleNamespace(to_dict=lambda: {"status": "passed"})
        with mock.patch(
            "backend.app.services.retrieval.bge_reranker.rerank_with_bge",
            return_value=([], report),
        ) as run:
            self.assertEqual(reranker.rerank("query", [], top_k=150), [])

        self.assertEqual(run.call_args.kwargs["candidate_limit"], 150)
        self.assertEqual(run.call_args.kwargs["output_k"], 150)

    def test_candidate_limit_rejects_boolean_or_non_positive_values(self) -> None:
        for value in (True, 0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "positive integer",
            ):
                BgeCandidateReranker(candidate_limit=value)

    def test_identical_adapters_share_runner_until_cache_clear(self) -> None:
        clear_shared_bge_reranker_runners()
        first = BgeCandidateReranker()
        second = BgeCandidateReranker()
        self.assertIs(first._runner, second._runner)
        prior = weakref.ref(first._runner)

        clear_shared_bge_reranker_runners()
        third = BgeCandidateReranker()
        self.assertIsNot(first._runner, third._runner)
        self.assertIsNotNone(prior())

    def test_call_local_report_is_sanitized_and_updates_snapshot(self) -> None:
        reranker = BgeCandidateReranker()
        report = SimpleNamespace(
            to_dict=lambda: {
                "status": "fallback",
                "candidate_count": 1,
                "scored_count": 0,
                "output_count": 1,
                "retrieval_alpha": 0.5,
                "fallback_reason": "RuntimeError: secret/path/token",
            }
        )
        with mock.patch(
            "backend.app.services.retrieval.bge_reranker.rerank_with_bge",
            return_value=([], report),
        ):
            _results, call_report = reranker.rerank_with_report("query", [])

        self.assertEqual(call_report, reranker.last_report)
        self.assertEqual(call_report["failure_type"], "RuntimeError")
        self.assertNotIn("secret", repr(call_report))


class SharedBgeDenseEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        retrieval_manager.clear_retrieval_caches()

    def tearDown(self) -> None:
        retrieval_manager.clear_retrieval_caches()

    def test_identical_qa_and_trake_contracts_share_one_dense_engine(self) -> None:
        class FakeDense:
            pass

        built: list[FakeDense] = []

        def build(**_kwargs):
            engine = FakeDense()
            built.append(engine)
            return engine

        key = retrieval_manager._CorpusCacheKey("manifest", "generation", "sha")
        kwargs = {
            "corpus_key": key,
            "artifact_root": Path("data/indexes/bge_m3"),
            "model_revision": None,
            "device": "cpu",
            "cache_dir": Path("data/model_cache/bge_m3"),
        }
        with mock.patch.object(
            retrieval_manager,
            "build_bge_m3_dense_search_engine",
            side_effect=build,
        ):
            qa_engine = retrieval_manager._shared_bge_m3_dense_search_engine(**kwargs)
            trake_engine = retrieval_manager._shared_bge_m3_dense_search_engine(**kwargs)

        self.assertIs(qa_engine, trake_engine)
        self.assertEqual(len(built), 1)
        engine_ref = weakref.ref(qa_engine)
        del qa_engine, trake_engine
        built.clear()
        gc.collect()
        self.assertIsNone(engine_ref())
        self.assertEqual(len(retrieval_manager._BGE_DENSE_ENGINES), 0)

    def test_online_factory_keeps_task_specific_pipelines_lazy(self) -> None:
        runtime = SimpleNamespace(
            query_expansion=SimpleNamespace(enabled=False),
            hybrid=SimpleNamespace(max_top_k=200),
            trake=TrakeConfig(),
        )
        trake = SimpleNamespace(
            corpus_generation=None,
            search=mock.Mock(return_value={"task": "trake", "hypotheses": []}),
        )
        key = retrieval_manager._CorpusCacheKey("legacy", None, None)
        with (
            mock.patch.object(retrieval_manager, "get_runtime_config", return_value=runtime),
            mock.patch.object(retrieval_manager, "get_online_context_index", return_value=None),
            mock.patch.object(retrieval_manager, "get_hybrid_search_engine", return_value=object()),
            mock.patch.object(
                retrieval_manager,
                "get_qa_search_pipeline",
                side_effect=RuntimeError("private QA initialization failure"),
            ) as qa_getter,
            mock.patch.object(
                retrieval_manager,
                "get_qa_evidence_search_engine",
                side_effect=RuntimeError("private evidence initialization failure"),
            ) as evidence_getter,
            mock.patch.object(retrieval_manager, "get_trake_pipeline", return_value=trake) as getter,
        ):
            pipeline = retrieval_manager.get_online_pipeline.__wrapped__(key)
            getter.assert_not_called()
            qa_getter.assert_not_called()
            evidence_getter.assert_not_called()
            pipeline.trake_pipeline.search("event", top_k=5)
            qa_getter.assert_not_called()
            evidence_getter.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "QA initialization"):
                pipeline.qa_pipeline.search("question", top_k=5)
            with self.assertRaisesRegex(RuntimeError, "evidence initialization"):
                pipeline.qa_evidence_engine.search("event", top_k=5)

        getter.assert_called_once_with()
        qa_getter.assert_called_once_with()
        evidence_getter.assert_called_once_with()

    def test_bge_bundle_validation_rejects_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing-corpus-manifest.json"
            with (
                mock.patch.dict(
                    os.environ,
                    {"RETRIEVAL_CORPUS_MANIFEST_PATH": str(missing)},
                    clear=False,
                ),
                self.assertRaisesRegex(
                    FileNotFoundError,
                    "committed offline corpus manifest",
                ),
            ):
                retrieval_manager.validate_runtime_corpus_bundle(
                    require_bge=True
                )

    def test_required_trake_rejects_legacy_corpus_before_model_init(self) -> None:
        runtime = SimpleNamespace(
            trake=TrakeConfig(
                bge_reranker_enabled=True,
                bge_required=True,
            )
        )
        with (
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=runtime,
            ),
            mock.patch.object(
                retrieval_manager,
                "build_trake_bge_candidate_reranker",
            ) as reranker_factory,
            self.assertRaises(RequiredTrakePipelineError) as captured,
        ):
            retrieval_manager.get_trake_pipeline.__wrapped__(
                retrieval_manager._CorpusCacheKey("legacy", None, None)
            )

        reranker_factory.assert_not_called()
        self.assertEqual(
            captured.exception.failure_code,
            "required_corpus_manifest_unavailable",
        )

    def test_public_bge_contract_redacts_local_model_identifiers(self) -> None:
        dense = SimpleNamespace(
            artifacts=SimpleNamespace(
                manifest={
                    "model": {
                        "name": r"C:\Users\alice\private\model",
                        "revision": "secret-token-revision",
                    },
                    "artifacts": {},
                }
            )
        )
        reranker = SimpleNamespace(
            model_name="BAAI/bge-reranker-v2-m3",
            model_revision="main",
            candidate_limit=150,
        )
        runtime = SimpleNamespace(
            trake=TrakeConfig(
                bge_dense_enabled=True,
                bge_reranker_enabled=True,
            )
        )
        with mock.patch.object(
            retrieval_manager,
            "get_runtime_config",
            return_value=runtime,
        ):
            contract = retrieval_manager._trake_bge_contract(
                corpus_key=retrieval_manager._CorpusCacheKey(
                    "manifest",
                    "generation",
                    "sha",
                ),
                dense_event_engine=dense,
                event_reranker=reranker,
            )

        rendered = repr(contract)
        self.assertNotIn("alice", rendered)
        self.assertNotIn("secret-token", rendered)
        self.assertEqual(contract["dense"]["model_name"], "local_or_redacted")
        self.assertEqual(contract["dense"]["model_revision"], "redacted")
        self.assertFalse(contract["dense"]["revision_pinned"])
        self.assertEqual(
            contract["reranker"]["model_name"],
            "BAAI/bge-reranker-v2-m3",
        )
        self.assertEqual(contract["reranker"]["model_revision"], "main")
        self.assertFalse(contract["reranker"]["revision_pinned"])

    def test_required_reranker_rejects_mutable_revision_before_construction(
        self,
    ) -> None:
        runtime = SimpleNamespace(
            trake=TrakeConfig(
                bge_reranker_enabled=True,
                bge_required=True,
            )
        )
        with (
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=runtime,
            ),
            mock.patch.dict(
                os.environ,
                {"RETRIEVAL_TRAKE_BGE_RERANKER_REVISION": "main"},
                clear=False,
            ),
            mock.patch.object(
                retrieval_manager,
                "build_bge_candidate_reranker",
            ) as factory,
            self.assertRaises(RequiredTrakePipelineError) as captured,
        ):
            retrieval_manager.build_trake_bge_candidate_reranker()

        factory.assert_not_called()
        self.assertEqual(
            captured.exception.failure_code,
            "required_bge_reranker_revision_unpinned",
        )


if __name__ == "__main__":
    unittest.main()
