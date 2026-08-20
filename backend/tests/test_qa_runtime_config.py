from __future__ import annotations

import os
import unittest
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

            retrieval_manager.get_qa_search_pipeline.cache_clear()
            with mock.patch.object(
                retrieval_manager,
                "get_qa_evidence_search_engine",
                return_value=mock.Mock(),
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
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            config = retrieval_manager._qa_routing_config()
        self.assertFalse(config.constraint_rerank_enabled)
        self.assertEqual(config.constraint_weight, 0.25)
        self.assertEqual(config.constraint_min_signal, 0.35)
        self.assertFalse(config.temporal_routing_enabled)
        self.assertEqual(config.temporal_max_events, 4)
        self.assertEqual(config.temporal_max_gap_seconds, 42.0)


if __name__ == "__main__":
    unittest.main()
