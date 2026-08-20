from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from backend.app.models.retrieval import VisualSearchResponse
from backend.app.services.agent.query_expansion import (
    ProviderResponse,
    QueryExpansionConfig,
)
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine
from backend.app.services.retrieval.planned_hybrid import planned_hybrid_search
from backend.app.services.retrieval.query_plan import build_query_plan
from backend.app.services.retrieval.retrieval_config import RetrievalRuntimeConfig
from backend.app.services.retrieval import retrieval_manager


def _payload(paraphrases: list[str]) -> dict[str, list[str]]:
    return {
        "paraphrases": paraphrases,
        "objects": ["man", "bus"],
        "attributes": [],
        "actions": [],
        "relations": ["next to"],
        "ocr_literals": ["CITY"],
        "scene_terms": [],
    }


class _Provider:
    provider_name = "fake-generative"
    model_name = "fake/model"
    model_revision = "revision-1"

    def __init__(self, payload=None, error: Exception | None = None) -> None:
        self.payload = payload or _payload([])
        self.error = error
        self.calls: list[str] = []
        self.close_calls = 0

    def expand(self, query, _protected):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return ProviderResponse(self.payload)

    def close(self) -> None:
        self.close_calls += 1


class _Visual:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int | None = None):
        self.queries.append(query)
        return VisualSearchResponse(query, top_k or 1, 0.0, [])


class _Text:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_results(self, query: str, top_k: int | None = None):
        self.queries.append(query)
        return []


def _engine():
    visual = _Visual()
    caption = _Text()
    ocr = _Text()
    objects = _Text()
    engine = HybridSearchEngine(
        visual,
        {"caption": caption, "ocr": ocr, "objects": objects},
    )
    return engine, visual, caption, ocr, objects


class PlannedHybridTest(unittest.TestCase):
    def test_parser_expansion_decomposition_retrieval_and_trace(self) -> None:
        query = "a man next to a bus with the text CITY"
        accepted = "a person beside a bus with the text CITY"
        rejected = "a woman beside one blue car with the text CITY"
        provider = _Provider(_payload([accepted, rejected]))
        plan = build_query_plan(
            query,
            profile="kis",
            expansion_provider=provider,
        )
        engine, visual, caption, ocr, objects = _engine()

        response = planned_hybrid_search(engine, plan, top_k=3)

        self.assertEqual(provider.calls, [query])
        self.assertEqual(visual.queries, [query, accepted])
        self.assertEqual(caption.queries, [query, accepted])
        self.assertNotIn(rejected, visual.queries)
        self.assertEqual(ocr.queries, ["CITY"])
        self.assertEqual(objects.queries, ["person bus"])
        trace = response.to_dict()["trace"]
        self.assertEqual(trace["query_plan"]["original_query"], query)
        self.assertEqual(trace["query_plan"]["profile"], "kis")
        self.assertEqual(trace["query_plan"]["expansion_plan"]["status"], "passed")
        self.assertEqual(trace["variant_weights"], {"original": 1.0, "paraphrase_1": 0.6})
        self.assertEqual(
            trace["searched_modalities"],
            ["visual", "caption", "ocr", "objects"],
        )

    def test_provider_failure_keeps_original_and_decomposition(self) -> None:
        query = "a man next to a bus with the text CITY"
        provider = _Provider(error=TimeoutError("deadline"))
        plan = build_query_plan(
            query,
            profile="kis",
            expansion_provider=provider,
        )
        engine, visual, caption, ocr, objects = _engine()

        response = planned_hybrid_search(engine, plan)

        self.assertEqual(visual.queries, [query])
        self.assertEqual(caption.queries, [query])
        self.assertEqual(ocr.queries, ["CITY"])
        self.assertEqual(objects.queries, ["bus person"])
        expansion = response.trace["query_plan"]["expansion_plan"]
        self.assertEqual(expansion["status"], "fallback")
        self.assertIn("TimeoutError", expansion["fallback_reason"])

    def test_disabled_expansion_never_calls_provider(self) -> None:
        provider = _Provider(_payload(["a red vehicle"]))
        plan = build_query_plan(
            "a red bus",
            profile="kis",
            expansion_provider=provider,
            expansion_config=QueryExpansionConfig(enabled=False),
        )
        engine, visual, *_ = _engine()

        planned_hybrid_search(engine, plan)

        self.assertEqual(provider.calls, [])
        self.assertEqual(visual.queries, ["a red bus"])

    def test_empty_query_fails_before_retrieval(self) -> None:
        engine, visual, *_ = _engine()
        with self.assertRaises(ValueError):
            build_query_plan("   ", profile="kis")
        self.assertEqual(visual.queries, [])

    def test_qa_and_temporal_qa_do_not_call_generative_provider(self) -> None:
        provider = _Provider(_payload(["unsafe whole question rewrite"]))
        question = "What does the man hold after he enters, then sits down?"

        plan = build_query_plan(
            question,
            profile="qa",
            expansion_provider=provider,
        )

        self.assertEqual(provider.calls, [])
        self.assertEqual(plan.expansion_plan.status, "disabled")
        self.assertGreater(len(plan.temporal_events), 1)
        self.assertEqual(plan.retrieval_queries, (plan.retrieval_statement,))


class RuntimeProviderTest(unittest.TestCase):
    def tearDown(self) -> None:
        retrieval_manager.clear_retrieval_caches()

    def test_factory_is_lazy_cached_and_clear_resets_provider(self) -> None:
        retrieval_manager.clear_retrieval_caches()
        runtime = RetrievalRuntimeConfig(
            query_expansion=QueryExpansionConfig(enabled=True)
        )
        first = _Provider()
        second = _Provider()
        with (
            mock.patch.object(retrieval_manager, "get_runtime_config", return_value=runtime),
            mock.patch.object(
                retrieval_manager,
                "build_production_query_expansion_provider",
                side_effect=[first, second],
            ) as factory,
        ):
            self.assertIs(retrieval_manager.get_query_expansion_provider(), first)
            self.assertIs(retrieval_manager.get_query_expansion_provider(), first)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(first.calls, [])

            retrieval_manager.clear_retrieval_caches()

            self.assertEqual(first.close_calls, 1)
            self.assertIs(retrieval_manager.get_query_expansion_provider(), second)
            self.assertEqual(factory.call_count, 2)

    def test_disabled_runtime_does_not_construct_provider(self) -> None:
        retrieval_manager.clear_retrieval_caches()
        runtime = RetrievalRuntimeConfig(
            query_expansion=QueryExpansionConfig(enabled=False)
        )
        with (
            mock.patch.object(retrieval_manager, "get_runtime_config", return_value=runtime),
            mock.patch.object(
                retrieval_manager,
                "build_production_query_expansion_provider",
            ) as factory,
        ):
            self.assertIsNone(retrieval_manager.get_query_expansion_provider())
            factory.assert_not_called()

    def test_runtime_factory_receives_device_and_cache_environment(self) -> None:
        retrieval_manager.clear_retrieval_caches()
        runtime = RetrievalRuntimeConfig(
            query_expansion=QueryExpansionConfig(enabled=True)
        )
        provider = _Provider()
        environment = {
            "QUERY_EXPANSION_DEVICE": "cuda:1",
            "QUERY_EXPANSION_CACHE_DIR": "custom/response-cache",
            "QUERY_EXPANSION_MODEL_CACHE_DIR": "custom/model-cache",
            "QUERY_EXPANSION_LOCAL_FILES_ONLY": "true",
        }
        with (
            mock.patch.dict(retrieval_manager.os.environ, environment),
            mock.patch.object(retrieval_manager, "get_runtime_config", return_value=runtime),
            mock.patch.object(
                retrieval_manager,
                "build_production_query_expansion_provider",
                return_value=provider,
            ) as factory,
        ):
            self.assertIs(retrieval_manager.get_query_expansion_provider(), provider)

        kwargs = factory.call_args.kwargs
        self.assertIs(kwargs["config"], runtime.query_expansion)
        self.assertEqual(kwargs["device"], "cuda:1")
        self.assertEqual(kwargs["cache_dir"], Path("custom/response-cache"))
        self.assertEqual(kwargs["model_cache_dir"], Path("custom/model-cache"))
        self.assertTrue(kwargs["local_files_only"])

    def test_public_hybrid_entrypoint_injects_provider_and_returns_trace(self) -> None:
        retrieval_manager.clear_retrieval_caches()
        query = "a man next to a bus with the text CITY"
        provider = _Provider(_payload(["a person beside a bus with the text CITY"]))
        runtime = RetrievalRuntimeConfig(
            query_expansion=QueryExpansionConfig(enabled=True)
        )
        engine, visual, _caption, _ocr, _objects = _engine()
        with (
            mock.patch.object(retrieval_manager, "get_runtime_config", return_value=runtime),
            mock.patch.object(
                retrieval_manager,
                "get_query_expansion_provider",
                return_value=provider,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_hybrid_search_engine",
                return_value=engine,
            ),
        ):
            response = retrieval_manager.search_hybrid(query, top_k=2)

        self.assertEqual(provider.calls, [query])
        self.assertEqual(visual.queries[0], query)
        self.assertEqual(response.trace["query_plan"]["profile"], "kis")
        self.assertEqual(
            response.trace["query_plan"]["expansion_plan"]["provider_name"],
            provider.provider_name,
        )


if __name__ == "__main__":
    unittest.main()
