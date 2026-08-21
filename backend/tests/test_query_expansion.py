from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.agent.query_expansion import (
    DEFAULT_QUERY_EXPANSION_MODEL,
    DEFAULT_QUERY_EXPANSION_MODEL_REVISION,
    ProviderResponse,
    QueryExpansionConfig,
    QwenQueryExpansionProvider,
    build_production_query_expansion_provider,
    build_query_expansion_plan,
    protect_literals,
)
from backend.app.services.retrieval.qa_answerer import DEFAULT_QA_MODEL
from backend.app.services.retrieval.advanced_rerank import rerank_dense_candidates
from backend.app.services.retrieval.advanced_search import (
    AdvancedSearchConfig,
    advanced_text_search,
)
from backend.app.services.retrieval.cses import CSESSelection
from backend.app.services.retrieval.query_plan import build_query_plan
from backend.app.services.retrieval.rank_fusion import fuse_query_variants


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


class _FakeProvider:
    provider_name = "fake-generative"
    model_name = "fake/model"
    model_revision = "deadbeef"

    def __init__(self, payload: dict[str, list[str]]) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def expand(self, query, _protected):
        self.calls.append(query)
        return ProviderResponse(self.payload)

    def close(self) -> None:
        return None


class _FailingProvider(_FakeProvider):
    def expand(self, query, _protected):
        self.calls.append(query)
        raise TimeoutError("provider deadline")


class _ColdStartProvider(QwenQueryExpansionProvider):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loaded = False

    def _load(self) -> None:
        if not self.loaded:
            time.sleep(0.02)
            self.loaded = True

    def _run_local(self, _prompt: str) -> str:
        self._load()
        return json.dumps(_payload([]))


class QueryExpansionPlanTest(unittest.TestCase):
    def test_query_expansion_defaults_to_pinned_2b_4bit_only(self) -> None:
        config = QueryExpansionConfig()
        self.assertEqual(DEFAULT_QUERY_EXPANSION_MODEL, "Qwen/Qwen3.5-2B")
        self.assertEqual(
            DEFAULT_QUERY_EXPANSION_MODEL_REVISION,
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
        )
        self.assertEqual(config.model_name, DEFAULT_QUERY_EXPANSION_MODEL)
        self.assertEqual(config.model_revision, DEFAULT_QUERY_EXPANSION_MODEL_REVISION)
        self.assertEqual(config.quantization, "4bit")
        self.assertEqual(DEFAULT_QA_MODEL, "Qwen/Qwen3.5-2B")

    def test_more_than_two_paraphrases_keeps_first_two_valid(self) -> None:
        original = "a man in a red shirt next to two cars"
        provider = _FakeProvider(
            {
                "paraphrases": [
                    "a person in a red shirt beside two cars",
                    "a woman in a blue shirt beside one car",
                    "two cars are next to a man in a red shirt",
                    "a man in a red shirt is beside two cars",
                ],
                "objects": ["man", "car", "shirt"],
                "attributes": ["red"],
                "actions": [],
                "relations": ["next to"],
                "ocr_literals": [],
                "scene_terms": [],
            }
        )

        plan = build_query_expansion_plan(
            original,
            provider=provider,
            config=QueryExpansionConfig(),
        )

        accepted = [value for value in plan.variants if value.accepted]
        self.assertEqual([value.type for value in accepted], ["original", "paraphrase", "paraphrase"])
        self.assertEqual(accepted[1].text, "a person in a red shirt beside two cars")
        self.assertEqual(accepted[2].text, "two cars are next to a man in a red shirt")
        rejected = {value.text: value.rejection_reason for value in plan.variants if not value.accepted}
        self.assertIn("color_changed", rejected["a woman in a blue shirt beside one car"])
        self.assertEqual(
            rejected["a man in a red shirt is beside two cars"],
            "max_paraphrases_exceeded",
        )
        self.assertEqual(plan.status, "passed")

    def test_literal_protector_and_drift_validator_are_conservative(self) -> None:
        original = 'Find a red bus numbered 152 with the text "BEN THANH" and no people'
        protected = protect_literals(original)
        self.assertIn("BEN THANH", protected.quoted)
        self.assertIn("152", protected.numbers)
        self.assertIn("red", protected.colors)
        self.assertIn("no", protected.negations)
        provider = _FakeProvider(
            {
                "paraphrases": ['Find a blue bus numbered 153 with the text "CITY" and people'],
                "objects": ["bus"],
                "attributes": ["blue"],
                "actions": [],
                "relations": [],
                "ocr_literals": ["CITY"],
                "scene_terms": ["street"],
            }
        )
        plan = build_query_expansion_plan(original, provider=provider)
        rejected = plan.variants[1]
        self.assertFalse(rejected.accepted)
        for reason in ("literal_missing", "number_changed", "color_changed", "negation_changed"):
            self.assertIn(reason, rejected.rejection_reason)
        self.assertEqual(plan.decomposition.scene_terms, ())
        self.assertTrue(any(value.startswith("scene_term_not_grounded") for value in plan.decomposition_rejections))
        self.assertEqual(plan.status, "fallback")

    def test_valid_zero_paraphrase_is_success_not_failure(self) -> None:
        provider = _FakeProvider(_payload([]))
        plan = build_query_expansion_plan(
            "a man next to a bus with the text CITY",
            provider=provider,
        )
        self.assertEqual(plan.status, "passed")
        self.assertEqual(plan.fallback_reason, "")
        self.assertEqual(len(plan.accepted_variants), 1)

    def test_blank_paraphrase_is_not_valid_zero_output(self) -> None:
        provider = _FakeProvider(_payload(["   "]))
        plan = build_query_expansion_plan("a red bus", provider=provider)
        self.assertEqual(plan.status, "fallback")
        self.assertEqual(plan.fallback_reason, "no_valid_paraphrase")
        self.assertEqual(plan.variants[1].rejection_reason, "empty_paraphrase")

    def test_empty_query_never_calls_provider(self) -> None:
        provider = _FakeProvider(_payload(["a red bus"]))
        plan = build_query_expansion_plan("   ", provider=provider)
        self.assertEqual(plan.status, "fallback")
        self.assertEqual(plan.fallback_reason, "empty_query")
        self.assertEqual(provider.calls, [])

    def test_provider_timeout_is_original_only_fallback(self) -> None:
        provider = _FailingProvider(_payload([]))
        plan = build_query_expansion_plan("a red bus", provider=provider)
        self.assertEqual(plan.status, "fallback")
        self.assertIn("TimeoutError", plan.fallback_reason)
        self.assertEqual([value.type for value in plan.accepted_variants], ["original"])

    def test_explicit_ablation_is_original_only_without_provider_call(self) -> None:
        provider = _FakeProvider(_payload(["a red vehicle"]))
        plan = build_query_expansion_plan(
            "a red bus",
            provider=provider,
            config=QueryExpansionConfig(enabled=False),
        )
        self.assertEqual(plan.status, "disabled")
        self.assertEqual(plan.fallback_reason, "explicit_ablation")
        self.assertEqual(provider.calls, [])
        self.assertEqual([value.type for value in plan.accepted_variants], ["original"])

    def test_missing_production_provider_is_explicit_fallback(self) -> None:
        plan = build_query_expansion_plan("a red bus", provider=None)
        self.assertEqual(plan.status, "fallback")
        self.assertEqual(plan.fallback_reason, "production_provider_unavailable")
        self.assertEqual([value.type for value in plan.accepted_variants], ["original"])

    def test_count_words_and_digits_are_equivalent(self) -> None:
        provider = _FakeProvider(
            {
                "paraphrases": ["a man beside 2 cars"],
                "objects": ["man", "car"],
                "attributes": [],
                "actions": [],
                "relations": ["beside"],
                "ocr_literals": [],
                "scene_terms": [],
            }
        )
        plan = build_query_expansion_plan("a man next to two cars", provider=provider)
        self.assertTrue(plan.variants[1].accepted, plan.variants[1].rejection_reason)

    def test_child_cannot_be_relaxed_to_generic_person(self) -> None:
        provider = _FakeProvider(
            {
                "paraphrases": ["a person next to a red bus"],
                "objects": ["person", "bus"],
                "attributes": ["red"],
                "actions": [],
                "relations": ["next to"],
                "ocr_literals": [],
                "scene_terms": [],
            }
        )
        plan = build_query_expansion_plan("a child next to a red bus", provider=provider)
        self.assertFalse(plan.variants[1].accepted)
        self.assertIn("subject_changed", plan.variants[1].rejection_reason)

    def test_provider_cannot_add_known_object_or_action(self) -> None:
        provider = _FakeProvider(
            {
                "paraphrases": ["a red bus is driving"],
                "objects": ["bus"],
                "attributes": ["red"],
                "actions": ["driving"],
                "relations": [],
                "ocr_literals": [],
                "scene_terms": [],
            }
        )
        plan = build_query_expansion_plan("a red scene", provider=provider)
        self.assertFalse(plan.variants[1].accepted)
        self.assertIn("object_changed", plan.variants[1].rejection_reason)
        self.assertIn("action_changed", plan.variants[1].rejection_reason)

    def test_production_provider_adapter_generates_and_caches(self) -> None:
        calls: list[str] = []

        def runner(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps(_payload(["a person beside a bus with the text CITY"]))

        with tempfile.TemporaryDirectory() as raw:
            provider = QwenQueryExpansionProvider(
                cache_dir=Path(raw),
                runner=runner,
            )
            protected = protect_literals("a man next to a bus with the text CITY")
            first = provider.expand("a man next to a bus with the text CITY", protected)
            second = provider.expand("a man next to a bus with the text CITY", protected)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(len(calls), 1)
        self.assertIn("Do not translate", calls[0])
        self.assertIn("untrusted data", calls[0])
        self.assertIn('Original query JSON string:', calls[0])
        summary = provider.runtime_summary()
        self.assertEqual(summary["provider_call_count"], 2)
        self.assertEqual(summary["cache_hit_count"], 1)
        self.assertEqual(summary["model_load_count"], 0)

    def test_cache_hit_does_not_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            provider = QwenQueryExpansionProvider(cache_dir=Path(raw))
            query = "a red bus"
            cache_path = Path(raw) / "responses" / f"{provider._cache_key(query)}.json"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text(json.dumps(_payload([])), encoding="utf-8")
            with mock.patch.object(provider, "_load") as load:
                response = provider.expand(query, protect_literals(query))

        self.assertTrue(response.cache_hit)
        load.assert_not_called()

    def test_cuda_loader_passes_real_4bit_bitsandbytes_config(self) -> None:
        captured: dict[str, object] = {}

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                captured["processor"] = (model_name, kwargs)
                return cls()

        class FakeLoadedModel:
            def eval(self):
                captured["eval"] = True

        class FakeModelFactory:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                captured["model"] = (model_name, kwargs)
                return FakeLoadedModel()

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs):
                captured["bnb"] = kwargs

        fake_torch = SimpleNamespace(
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
            cuda=SimpleNamespace(is_bf16_supported=lambda: True),
        )
        fake_transformers = SimpleNamespace(
            AutoModelForMultimodalLM=FakeModelFactory,
            AutoProcessor=FakeProcessor,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
        )
        provider = QwenQueryExpansionProvider(device="cuda")

        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            provider._load()

        model_name, model_kwargs = captured["model"]
        self.assertEqual(model_name, "Qwen/Qwen3.5-2B")
        self.assertEqual(
            model_kwargs["revision"],
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
        )
        self.assertIsInstance(
            model_kwargs["quantization_config"],
            FakeBitsAndBytesConfig,
        )
        self.assertEqual(
            captured["bnb"],
            {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
        )
        self.assertEqual(model_kwargs["device_map"], "auto")
        self.assertEqual(provider.runtime_summary()["model_load_count"], 1)

    def test_cache_identity_includes_generation_configuration(self) -> None:
        calls: list[str] = []

        def runner(_prompt: str) -> str:
            calls.append("called")
            return json.dumps(_payload([]))

        with tempfile.TemporaryDirectory() as raw:
            protected = protect_literals("a red bus")
            first = QwenQueryExpansionProvider(
                config=QueryExpansionConfig(max_new_tokens=64),
                cache_dir=Path(raw),
                runner=runner,
            )
            second = QwenQueryExpansionProvider(
                config=QueryExpansionConfig(max_new_tokens=128),
                cache_dir=Path(raw),
                runner=runner,
            )
            first.expand("a red bus", protected)
            second.expand("a red bus", protected)
        self.assertEqual(len(calls), 2)

    def test_corrupt_cache_is_regenerated(self) -> None:
        calls: list[str] = []

        def runner(_prompt: str) -> str:
            calls.append("called")
            return json.dumps(_payload([]))

        with tempfile.TemporaryDirectory() as raw:
            provider = QwenQueryExpansionProvider(cache_dir=Path(raw), runner=runner)
            query = "a red bus"
            cache_path = Path(raw) / "responses" / f"{provider._cache_key(query)}.json"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text("not-json", encoding="utf-8")
            response = provider.expand(query, protect_literals(query))
            cached = provider.expand(query, protect_literals(query))
        self.assertFalse(response.cache_hit)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(len(calls), 1)

    def test_cold_model_load_is_outside_inference_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            provider = _ColdStartProvider(
                config=QueryExpansionConfig(timeout_seconds=0.005),
                cache_dir=Path(raw),
            )
            response = provider.expand("a red bus", protect_literals("a red bus"))
        self.assertFalse(response.cache_hit)
        self.assertTrue(provider.loaded)

    def test_injected_provider_runner_still_obeys_deadline(self) -> None:
        def slow_runner(_prompt: str) -> str:
            time.sleep(0.02)
            return json.dumps(_payload([]))

        with tempfile.TemporaryDirectory() as raw:
            provider = QwenQueryExpansionProvider(
                config=QueryExpansionConfig(timeout_seconds=0.005),
                cache_dir=Path(raw),
                runner=slow_runner,
            )
            with self.assertRaises(TimeoutError):
                provider.expand("a red bus", protect_literals("a red bus"))

    def test_local_generation_disables_thinking_and_receives_deadline(self) -> None:
        import torch

        class Processor:
            def __init__(self) -> None:
                self.template_kwargs: dict[str, object] = {}

            def apply_chat_template(self, _messages, **kwargs):
                self.template_kwargs = kwargs
                return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

            def batch_decode(self, _tokens, **_kwargs):
                return [json.dumps(_payload([]))]

        class Model:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self.generation_kwargs: dict[str, object] = {}

            def generate(self, **kwargs):
                self.generation_kwargs = kwargs
                return torch.tensor([[1, 2, 3]], dtype=torch.long)

        processor = Processor()
        model = Model()
        provider = QwenQueryExpansionProvider(
            config=QueryExpansionConfig(timeout_seconds=7.0, max_new_tokens=123),
            cache_dir=Path("unused-test-cache"),
        )
        provider._processor = processor
        provider._model = model
        provider._run_local("prompt")
        self.assertFalse(processor.template_kwargs["enable_thinking"])
        self.assertEqual(model.generation_kwargs["max_time"], 7.0)
        self.assertEqual(model.generation_kwargs["max_new_tokens"], 123)
        self.assertFalse(model.generation_kwargs["do_sample"])

    def test_default_factory_returns_real_lazy_provider(self) -> None:
        provider = build_production_query_expansion_provider(
            config=QueryExpansionConfig(),
            device="cpu",
            cache_dir=Path("unused-test-cache"),
            local_files_only=True,
        )
        self.assertIsInstance(provider, QwenQueryExpansionProvider)
        self.assertIsNone(provider._model)

    def test_next_to_is_not_a_temporal_split(self) -> None:
        plan = build_query_plan(
            "a man standing next to a bus",
            profile="auto",
            expansion_config=QueryExpansionConfig(enabled=False),
        )
        self.assertEqual(plan.temporal_events, ("a man standing next to a bus",))
        self.assertNotEqual(plan.profile, "temporal")

    def test_explicit_kis_profile_does_not_execute_temporal_chain(self) -> None:
        query = "a man enters a bus, then sits down"
        plan = build_query_plan(
            query,
            profile="kis",
            expansion_config=QueryExpansionConfig(enabled=False),
        )
        self.assertEqual(plan.profile, "kis")
        self.assertEqual(plan.temporal_relation, "none")
        self.assertEqual(plan.temporal_events, (query,))

    def test_original_variant_is_not_replaced_by_typo_normalization(self) -> None:
        original = "a person outside a resturant"
        plan = build_query_plan(
            original,
            profile="kis",
            expansion_config=QueryExpansionConfig(enabled=False),
        )
        self.assertEqual(plan.original_query, original)
        self.assertEqual(plan.expansion_plan.accepted_variants[0].text, original)
        self.assertIn("restaurant", plan.normalized_query)


class IntraModalityFusionTest(unittest.TestCase):
    def test_expansion_contribution_uses_defined_rank_one_budget(self) -> None:
        original_only = RetrievalResult("v", "original", 0.0, 999.0)
        rescued = RetrievalResult("v", "rescued", 1.0, -999.0)
        fused = fuse_query_variants(
            {
                "original": [original_only],
                "paraphrase_1": [rescued, original_only],
                "paraphrase_2": [rescued],
            },
            weights={"original": 1.0, "paraphrase_1": 0.6, "paraphrase_2": 0.6},
            k=60,
            max_expansion_contribution=1.0,
        )
        by_frame = {value.result.frame_id: value for value in fused}
        expected_budget = 1.0 / 61.0
        self.assertAlmostEqual(by_frame["rescued"].max_expansion_budget, expected_budget)
        self.assertGreater(by_frame["rescued"].raw_expansion_contribution, expected_budget)
        self.assertAlmostEqual(by_frame["rescued"].expansion_contribution, expected_budget)
        self.assertAlmostEqual(by_frame["original"].original_contribution, expected_budget)
        self.assertGreater(by_frame["original"].intra_score, expected_budget)

    def test_budget_changes_with_k_and_is_deterministic(self) -> None:
        result = RetrievalResult("v", "f", 0.0, 123.0)
        kwargs = dict(
            groups={"original": [], "paraphrase_1": [result]},
            weights={"original": 1.0, "paraphrase_1": 0.6},
            max_expansion_contribution=1.0,
        )
        first = fuse_query_variants(k=10, **kwargs)
        second = fuse_query_variants(k=10, **kwargs)
        larger_k = fuse_query_variants(k=20, **kwargs)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0].max_expansion_budget, 1.0 / 11.0)
        self.assertAlmostEqual(larger_k[0].max_expansion_budget, 1.0 / 21.0)

    def test_budget_scales_by_configured_ratio_and_original_weight(self) -> None:
        result = RetrievalResult("v", "expanded", 0.0, 0.0)
        fused = fuse_query_variants(
            {"original": [], "paraphrase_1": [result]},
            weights={"original": 2.0, "paraphrase_1": 1.0},
            k=9,
            max_expansion_contribution=0.25,
        )
        # 0.25 * w_orig(2.0) / (k(9) + rank_one(1)) = 0.05.
        self.assertAlmostEqual(fused[0].raw_expansion_contribution, 0.1)
        self.assertAlmostEqual(fused[0].max_expansion_budget, 0.05)
        self.assertAlmostEqual(fused[0].expansion_contribution, 0.05)
        self.assertAlmostEqual(fused[0].intra_score, 0.05)


class _EmptyVisualEngine:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int):
        self.queries.append(query)
        return SimpleNamespace(results=[])


class _EmptyTextEngine:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_results(self, query: str, top_k: int):
        self.queries.append(query)
        return []


class _RecordingEncoder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode(self, query: str):
        self.queries.append(query)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _EmptyDenseIndex:
    records: list[dict[str, object]] = []
    vectors = np.empty((0, 2), dtype=np.float32)
    rows_by_clip: dict[tuple[str, str], list[int]] = {}
    row_by_frame: dict[tuple[str, str], int] = {}

    def search(self, _query, _top_k):
        return []


class AdvancedSearchExpansionIntegrationTest(unittest.TestCase):
    def test_variants_route_but_dense_rerank_vector_uses_original(self) -> None:
        original = "a man next to a bus with the text CITY"
        provider = _FakeProvider(
            _payload([
                "a person beside a bus with the text CITY",
                "a bus with the text CITY next to a man",
            ])
        )
        plan = build_query_plan(original, profile="kis", expansion_provider=provider)
        visual = _EmptyVisualEngine()
        caption = _EmptyTextEngine()
        ocr = _EmptyTextEngine()
        objects = _EmptyTextEngine()
        hybrid = SimpleNamespace(
            visual_engine=visual,
            text_engines={"caption": caption, "ocr": ocr, "objects": objects},
        )
        encoder = _RecordingEncoder()
        response = advanced_text_search(
            original,
            hybrid_engine=hybrid,
            text_encoder=encoder,
            dense_index=_EmptyDenseIndex(),
            profile="kis",
            config=AdvancedSearchConfig(coarse_top_n=1),
            plan=plan,
        )
        self.assertEqual(visual.queries, [value.text for value in plan.expansion_plan.accepted_variants])
        self.assertEqual(caption.queries, visual.queries)
        self.assertEqual(ocr.queries, ["CITY"])
        self.assertEqual(objects.queries, ["person bus"])
        self.assertEqual(encoder.queries, [original])
        self.assertEqual(response.trace()["rerank_canonical_query"], original)

    def test_candidate_metadata_remains_rerank_evidence(self) -> None:
        plan = build_query_plan(
            "red bus",
            profile="kis",
            expansion_config=QueryExpansionConfig(enabled=False),
        )
        records = [
            {"candidate_id": "match", "video_id": "v", "timestamp": 0.0, "caption": "red bus"},
            {"candidate_id": "miss", "video_id": "v", "timestamp": 1.0, "caption": "blue car"},
        ]
        selections = [
            CSESSelection(0, 1, 1.0, 1.0, 0.0, 0.0, ()),
            CSESSelection(1, 2, 1.0, 1.0, 0.0, 0.0, ()),
        ]
        ranked = rerank_dense_candidates(
            plan=plan,
            selections=selections,
            records=records,
            vectors=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(ranked[0].record["candidate_id"], "match")
        self.assertGreater(ranked[0].breakdown["caption"], ranked[1].breakdown["caption"])

    def test_metadata_rerank_tokens_come_from_original_query(self) -> None:
        plan = build_query_plan(
            "red resturant",
            profile="kis",
            expansion_config=QueryExpansionConfig(enabled=False),
        )
        self.assertEqual(plan.normalized_query, "red restaurant")
        records = [
            {"candidate_id": "original", "video_id": "v", "timestamp": 0.0, "caption": "red resturant"},
            {"candidate_id": "normalized", "video_id": "v", "timestamp": 1.0, "caption": "red restaurant"},
        ]
        selections = [
            CSESSelection(0, 1, 1.0, 1.0, 0.0, 0.0, ()),
            CSESSelection(1, 2, 1.0, 1.0, 0.0, 0.0, ()),
        ]
        ranked = rerank_dense_candidates(
            plan=plan,
            selections=selections,
            records=records,
            vectors=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(ranked[0].record["candidate_id"], "original")
        self.assertGreater(ranked[0].breakdown["caption"], ranked[1].breakdown["caption"])


if __name__ == "__main__":
    unittest.main()
