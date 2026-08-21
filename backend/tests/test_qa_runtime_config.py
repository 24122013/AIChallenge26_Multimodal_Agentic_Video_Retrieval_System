from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.app.services.retrieval.qa_answerer import (
    DEFAULT_QA_MODEL,
    DEFAULT_QA_MODEL_REVISION,
)
from backend.app.services.retrieval import retrieval_manager


class QaRuntimeConfigTest(unittest.TestCase):
    def tearDown(self) -> None:
        retrieval_manager.clear_retrieval_caches()

    def test_local_defaults_keep_models_off_but_new_routing_on(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = retrieval_manager._qa_routing_config()
            self.assertTrue(config.constraint_rerank_enabled)
            self.assertEqual(config.constraint_weight, 0.15)
            self.assertEqual(config.constraint_min_signal, 0.20)
            self.assertTrue(config.temporal_routing_enabled)
            self.assertEqual(config.evidence_limit, 100)
            self.assertEqual(config.fusion_pool_size, 100)

            retrieval_manager.get_qa_search_pipeline.cache_clear()
            evidence_engine = mock.Mock()
            evidence_engine.corpus_generation = (
                retrieval_manager._current_corpus_cache_key().bundle_generation
            )
            with mock.patch.object(
                retrieval_manager,
                "get_qa_evidence_search_engine",
                return_value=evidence_engine,
            ):
                pipeline = retrieval_manager.get_qa_search_pipeline()

        self.assertEqual(pipeline.config.answer_mode, "off")
        self.assertIsNone(pipeline.answer_runner)
        self.assertEqual(pipeline.config.model_name, DEFAULT_QA_MODEL)
        self.assertEqual(
            pipeline.config.model_revision,
            DEFAULT_QA_MODEL_REVISION,
        )

    def test_routing_environment_values_are_wired(self) -> None:
        environment = {
            "QA_CONSTRAINT_RERANK_ENABLED": "false",
            "QA_CONSTRAINT_WEIGHT": "0.25",
            "QA_CONSTRAINT_MIN_SIGNAL": "0.35",
            "QA_TEMPORAL_ROUTING_ENABLED": "false",
            "QA_TEMPORAL_MAX_EVENTS": "4",
            "QA_TEMPORAL_MAX_GAP_SECONDS": "42",
            "QA_PER_MODALITY_LIMIT": "80",
            "QA_RERANK_POOL_SIZE": "70",
            "QA_FUSION_POOL_SIZE": "60",
            "QA_EVIDENCE_LIMIT": "50",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            config = retrieval_manager._qa_routing_config()
        self.assertFalse(config.constraint_rerank_enabled)
        self.assertEqual(config.constraint_weight, 0.25)
        self.assertEqual(config.constraint_min_signal, 0.35)
        self.assertFalse(config.temporal_routing_enabled)
        self.assertEqual(config.temporal_max_events, 4)
        self.assertEqual(config.temporal_max_gap_seconds, 42.0)
        self.assertEqual(config.per_modality_limit, 80)
        self.assertEqual(config.rerank_pool_size, 70)
        self.assertEqual(config.fusion_pool_size, 60)
        self.assertEqual(config.evidence_limit, 50)

    def test_qwen35_2b_runtime_report_uses_cuda_4bit_and_separate_cache(self) -> None:
        environment = {
            "QA_ANSWER_MODE": "required",
            "QA_ANSWER_MODEL": "Qwen/Qwen3.5-2B",
            "QA_ANSWER_MODEL_REVISION": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "QA_ANSWER_DEVICE": "cuda",
            "QA_ANSWER_QUANTIZATION": "auto",
            "QA_ANSWER_MODEL_CACHE_DIR": "data/model_cache/qa_answer",
            "QA_MODELS_LOCAL_ONLY": "true",
            "QUERY_EXPANSION_MODEL_CACHE_DIR": "data/model_cache/query_expansion",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            settings = retrieval_manager._qa_answer_runtime_settings()

        self.assertEqual(DEFAULT_QA_MODEL, "Qwen/Qwen3.5-2B")
        self.assertEqual(
            DEFAULT_QA_MODEL_REVISION,
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
        )
        self.assertEqual(settings["mode"], "required")
        self.assertEqual(settings["model_name"], DEFAULT_QA_MODEL)
        self.assertEqual(settings["model_revision"], DEFAULT_QA_MODEL_REVISION)
        self.assertEqual(settings["device"], "cuda")
        self.assertEqual(settings["requested_quantization"], "auto")
        self.assertEqual(settings["effective_quantization"], "4bit")
        self.assertEqual(settings["model_cache_dir"], Path("data/model_cache/qa_answer"))
        self.assertTrue(settings["local_files_only"])
        self.assertNotEqual(
            settings["model_cache_dir"],
            Path(environment["QUERY_EXPANSION_MODEL_CACHE_DIR"]),
        )

        expansion = SimpleNamespace(
            enabled=True,
            model_name="Qwen/Qwen3.5-2B",
            model_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
            quantization="4bit",
        )
        evidence_engine = SimpleNamespace(
            dense_text_engine=None,
            candidate_reranker=None,
        )
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(
                retrieval_manager,
                "get_qa_evidence_search_engine",
                return_value=evidence_engine,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=SimpleNamespace(query_expansion=expansion),
            ),
        ):
            lineage = retrieval_manager.get_qa_runtime_lineage()
        self.assertEqual(lineage["answer_model"]["effective_quantization"], "4bit")
        self.assertEqual(
            lineage["answer_model"]["model_cache_dir"],
            "data/model_cache/qa_answer",
        )
        self.assertEqual(
            lineage["query_expansion"]["model_cache_dir"],
            "data/model_cache/query_expansion",
        )
        self.assertFalse(lineage["query_expansion"]["shares_qa_answer_cache"])


if __name__ == "__main__":
    unittest.main()
