from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.pipelines.online_pipeline import OnlinePipeline, OnlinePipelineConfig
from backend.app.services.agent.query_expansion import (
    ProviderResponse,
    QueryExpansionConfig,
)
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine
from backend.app.services.retrieval.online_context import OnlineContextIndex
from backend.app.services.retrieval.qa_evidence import QaEvidenceSearchEngine
from backend.app.services.retrieval.query_plan import build_query_plan
from backend.app.services.retrieval.retrieval_config import (
    OnlineRetrievalConfig,
    RetrievalRuntimeConfig,
)
from backend.app.services.retrieval import advanced_search, retrieval_manager


def _result(**changes) -> RetrievalResult:
    value = RetrievalResult(
        video_id="V1",
        frame_id="F1",
        timestamp=2.0,
        score=0.8,
        shot_id="S1",
        segment_id="S1",
        frame_index=50,
        keyframe_path="data/keyframes/V1/F1.jpg",
        caption="a man opens a car door beside the text CITY",
        objects=["person", "car"],
        modality_scores={"visual": 0.8},
    )
    return replace(value, **changes)


class _Provider:
    provider_name = "test-provider"
    model_name = "test/model"
    model_revision = "r1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def expand(self, query, _protected):
        self.calls.append(query)
        return ProviderResponse(
            {
                "paraphrases": ["a person opens the door of a car with text CITY"],
                "objects": ["person", "car"],
                "attributes": [],
                "actions": ["opening"],
                "relations": [],
                "ocr_literals": ["CITY"],
                "scene_terms": [],
            }
        )

    def close(self) -> None:
        return None


class _Visual:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        self.queries.append(query)
        return VisualSearchResponse(query, top_k or 1, 0.0, [_result()])


class _Text:
    def __init__(self, modality: str) -> None:
        self.modality = modality
        self.queries: list[str] = []

    def search_results(self, query: str, top_k: int | None = None):
        self.queries.append(query)
        return [
            _result(
                score=0.7,
                modality_scores={self.modality: 0.7},
            )
        ]


def _hybrid() -> tuple[HybridSearchEngine, _Visual, dict[str, _Text]]:
    visual = _Visual()
    text = {
        "caption": _Text("caption"),
        "ocr": _Text("ocr"),
        "objects": _Text("objects"),
    }
    return HybridSearchEngine(visual, text), visual, text


class _QaPipeline:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k, *, task_mode, expanded_queries):
        self.calls.append((query, top_k, task_mode, tuple(expanded_queries)))
        plan = build_query_plan(query, profile="qa")
        return {
            "query_plan": plan.to_dict(),
            "routing_trace": {"route": "qa"},
            "results": [_result().to_dict()],
            "evidence": [{"evidence_id": "E001"}],
            "answer": {"status": "disabled", "answer": None},
        }


class _TemporalEvidence:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k, *, task_mode, expanded_queries):
        self.calls.append((query, top_k, task_mode, tuple(expanded_queries)))
        plan = build_query_plan(query, profile="temporal")
        result = _result().to_dict()
        result["temporal_event_index"] = 0
        result["temporal_chain_id"] = "chain-1"
        return {
            "query_plan": plan.to_dict(),
            "routing_trace": {"route": "temporal"},
            "results": [result],
            "temporal_matches": [{"chain_id": "chain-1"}],
        }


class _TrakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k=100):
        self.calls.append((query, top_k))
        return {
            "schema_version": "trake.v1",
            "query": query,
            "task": "trake",
            "event_plan": {
                "original_query": query,
                "events": [
                    {"index": 0, "original_text": "enters"},
                    {"index": 1, "original_text": "sits"},
                ],
            },
            "hypotheses": [
                {
                    "rank": 1,
                    "video_id": "V1",
                    "frame_ids": [10, 20],
                    "score": 0.9,
                }
            ],
            "trace": {"route": "trake"},
            "latency_ms": 1.0,
        }


class _DenseEncoder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _DenseVisual:
    def __init__(self) -> None:
        self.encoder = _DenseEncoder()
        self.queries: list[str] = []
        self.vector_queries: list[np.ndarray] = []

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        self.queries.append(query)
        return self._response(query, top_k)

    def search_by_vector(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int | None = None,
    ) -> VisualSearchResponse:
        self.queries.append(query)
        self.vector_queries.append(np.asarray(query_vector, dtype=np.float32).copy())
        return self._response(query, top_k)

    @staticmethod
    def _response(query: str, top_k: int | None) -> VisualSearchResponse:
        coarse = _result(
            video_id="COARSE",
            frame_id="COARSE:F0",
            shot_id="S0",
            segment_id="S0",
            score=0.75,
            modality_scores={"visual": 0.75},
        )
        return VisualSearchResponse(query, top_k or 1, 0.0, [coarse])


class _DenseIndex:
    def __init__(self) -> None:
        self.records = [
            {
                "candidate_id": "COARSE:C0",
                "frame_id": "COARSE:F0",
                "video_id": "COARSE",
                "shot_id": "S0",
                "segment_id": "S0",
                "timestamp": 2.0,
                "frame_index": 50,
                "caption": "unrelated coarse frame",
                "ocr_text": "",
                "objects": [],
            },
            {
                "candidate_id": "RESCUED:C0",
                "frame_id": "RESCUED:F0",
                "video_id": "RESCUED",
                "shot_id": "S1",
                "segment_id": "S1",
                "timestamp": 4.0,
                "frame_index": 100,
                "caption": "a person opens a refrigerator and takes a bottle",
                "ocr_text": "",
                "objects": ["person", "refrigerator", "bottle"],
                "protected_event_ids": ["EVENT_BOTTLE"],
            },
        ]
        self.vectors = np.asarray(
            [[0.6, 0.8], [1.0, 0.0]],
            dtype=np.float32,
        )
        self.rows_by_clip = {
            ("COARSE", "S0"): [0],
            ("RESCUED", "S1"): [1],
        }
        self.search_calls: list[int] = []

    def search(self, _query_vector: np.ndarray, top_k: int):
        self.search_calls.append(top_k)
        return [(1, 1.0), (0, 0.6)][:top_k]


class OnlinePipelineTest(unittest.TestCase):
    def tearDown(self) -> None:
        retrieval_manager.clear_retrieval_caches()

    def test_kis_plans_expands_routes_and_normalizes_candidates(self) -> None:
        hybrid, visual, text = _hybrid()
        provider = _Provider()
        runtime = RetrievalRuntimeConfig(
            query_expansion=QueryExpansionConfig(enabled=True)
        )
        pipeline = OnlinePipeline(
            hybrid_engine=hybrid,
            runtime_config=runtime,
            query_expansion_provider=provider,
        )
        query = "a man opens a car door with the text CITY"

        response = pipeline.run(query, task="kis", top_k=3)

        self.assertEqual(provider.calls, [query])
        self.assertEqual(visual.queries[0], query)
        self.assertEqual(len(visual.queries), 2)
        self.assertEqual(text["caption"].queries, visual.queries)
        self.assertEqual(text["ocr"].queries, ["CITY"])
        self.assertEqual(text["objects"].queries, ["person car"])
        self.assertEqual(response["query_plan"]["original_query"], query)
        self.assertEqual(response["task"], "kis")
        candidate = response["candidates"][0]
        self.assertEqual(candidate["keyframe_id"], "F1")
        self.assertEqual(candidate["frame_id"], "F1")
        self.assertIsNotNone(candidate["visual_score"])
        self.assertIsNotNone(candidate["caption_score"])
        self.assertIsNotNone(candidate["fusion_score"])
        self.assertEqual(candidate["rerank_score"], candidate["score"])

    def test_context_reads_canonical_neighbor_and_segment_artifacts(self) -> None:
        hybrid, _visual, _text = _hybrid()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            neighbor_path = root / "neighbors_all.jsonl"
            segment_path = root / "segments_all.jsonl"
            frame_map_path = root / "frame_map.json"
            neighbor_path.write_text(
                json.dumps(
                    {
                        "video_id": "V1",
                        "frame_id": "F1",
                        "timestamp": 2.0,
                        "neighbors_before": [
                            {"frame_id": "F0", "delta_seconds": -1.0}
                        ],
                        "neighbors_after": [
                            {"frame_id": "F2", "delta_seconds": 1.5}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            segment_path.write_text(
                json.dumps(
                    {
                        "video_id": "V1",
                        "segment_id": "S1",
                        "start_time": 0.0,
                        "end_time": 5.0,
                        "keyframe_ids": ["F0", "F1", "F2"],
                        "captions_aggregated": "a man opens a car door",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            frame_map_path.write_text(
                json.dumps(
                    {
                        "0": {
                            "video_id": "V1",
                            "frame_id": "F0",
                            "timestamp": 1.0,
                            "segment_id": "S1",
                            "keyframe_path": "F0.jpg",
                        },
                        "1": {
                            "video_id": "V1",
                            "frame_id": "F2",
                            "timestamp": 3.5,
                            "segment_id": "S1",
                            "keyframe_path": "F2.jpg",
                        },
                    }
                ),
                encoding="utf-8",
            )
            context = OnlineContextIndex.from_artifacts(
                neighbor_path=neighbor_path,
                segment_path=segment_path,
                frame_map_path=frame_map_path,
            )
            pipeline = OnlinePipeline(
                hybrid_engine=hybrid,
                runtime_config=RetrievalRuntimeConfig(
                    query_expansion=QueryExpansionConfig(enabled=False)
                ),
                context_index=context,
                config=OnlinePipelineConfig(
                    include_neighbors=True,
                    include_segments=True,
                ),
            )

            candidate = pipeline.run("man opens car", top_k=1)["candidates"][0]

        self.assertEqual(
            [item["frame_id"] for item in candidate["neighbors"]],
            ["F0", "F2"],
        )
        self.assertEqual(candidate["segment_id"], "S1")
        self.assertEqual(candidate["segment_context"]["end_time"], 5.0)
        self.assertEqual(
            candidate["context_sources"],
            ["neighbors_all", "segments_all"],
        )

    def test_qa_and_temporal_routes_keep_task_specific_outputs(self) -> None:
        hybrid, _visual, _text = _hybrid()
        qa = _QaPipeline()
        temporal = _TemporalEvidence()
        pipeline = OnlinePipeline(
            hybrid_engine=hybrid,
            runtime_config=RetrievalRuntimeConfig(
                query_expansion=QueryExpansionConfig(enabled=False)
            ),
            qa_pipeline=qa,
            qa_evidence_engine=temporal,
        )

        qa_response = pipeline.run(
            "What is the man holding?",
            task="qa",
            expanded_queries=["man carries an object"],
        )
        temporal_response = pipeline.run(
            "a man enters then sits down",
            task="temporal",
            top_k=2,
            expanded_queries=["must be ignored"],
        )

        self.assertEqual(qa.calls[0][2], "qa")
        self.assertEqual(qa_response["query_plan"]["original_query"], "What is the man holding?")
        self.assertIn("answer", qa_response)
        self.assertEqual(temporal.calls[0][2:], ("temporal", ()))
        self.assertEqual(temporal_response["task"], "temporal")
        self.assertEqual(
            temporal_response["candidates"][0]["temporal"]["temporal_chain_id"],
            "chain-1",
        )
        self.assertEqual(temporal_response["temporal_matches"][0]["chain_id"], "chain-1")

    def test_trake_route_preserves_ranked_sequences_and_temporal_semantics(
        self,
    ) -> None:
        hybrid, _visual, _text = _hybrid()
        temporal = _TemporalEvidence()
        trake = _TrakePipeline()
        pipeline = OnlinePipeline(
            hybrid_engine=hybrid,
            runtime_config=RetrievalRuntimeConfig(
                query_expansion=QueryExpansionConfig(enabled=False)
            ),
            qa_evidence_engine=temporal,
            trake_pipeline=trake,
        )

        trake_response = pipeline.run(
            "a man enters then sits down",
            task="trake",
            top_k=500,
        )
        temporal_response = pipeline.run(
            "a man enters then sits down",
            task="temporal",
            top_k=2,
        )

        self.assertEqual(trake.calls, [("a man enters then sits down", 100)])
        self.assertEqual(trake_response["task"], "trake")
        self.assertEqual(trake_response["top_k"], 100)
        self.assertEqual(trake_response["hypotheses"][0]["frame_ids"], [10, 20])
        self.assertNotIn("candidates", trake_response)
        self.assertEqual(temporal.calls[0][2:], ("temporal", ()))
        self.assertEqual(temporal_response["task"], "temporal")

    def test_qa_temporal_matching_maps_canonical_segments_before_chain(self) -> None:
        hybrid, _visual, _text = _hybrid()
        context = OnlineContextIndex(
            segment_records=[
                {
                    "video_id": "V1",
                    "segment_id": "S1",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "keyframe_ids": ["F1"],
                }
            ]
        )
        engine = QaEvidenceSearchEngine(hybrid, context_index=context)

        response = engine.search(
            "a man enters then sits down",
            top_k=2,
            task_mode="temporal",
        )

        segment_trace = response["routing_trace"]["temporal_route"][
            "canonical_segment_context"
        ]
        self.assertTrue(segment_trace["enabled"])
        self.assertGreater(segment_trace["mapped_candidate_count"], 0)

    def test_context_override_fails_closed_when_artifacts_are_not_loaded(self) -> None:
        hybrid, _visual, _text = _hybrid()
        pipeline = OnlinePipeline(
            hybrid_engine=hybrid,
            runtime_config=RetrievalRuntimeConfig(
                query_expansion=QueryExpansionConfig(enabled=False)
            ),
        )

        with self.assertRaises(FileNotFoundError):
            pipeline.run("a red car", include_context=True)

    def test_real_search_online_factory_executes_dense_rescue_cses_and_rerank(
        self,
    ) -> None:
        visual = _DenseVisual()
        engine = HybridSearchEngine(visual, {})
        dense = _DenseIndex()
        runtime = RetrievalRuntimeConfig(
            query_expansion=QueryExpansionConfig(enabled=False),
            online=OnlineRetrievalConfig(
                coarse_top_n=1,
                dense_global_top_k=2,
                dense_rescue_clips=1,
                max_total_clips=2,
                dense_frames_per_clip=1,
            ),
        )
        corpus_key = retrieval_manager._CorpusCacheKey(
            manifest_path="test-manifest",
            bundle_generation=None,
            manifest_contract_sha256=None,
        )

        with (
            mock.patch.object(
                retrieval_manager,
                "_current_corpus_cache_key",
                return_value=corpus_key,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=runtime,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_hybrid_search_engine",
                return_value=engine,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_online_context_index",
                return_value=None,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_dense_candidate_index",
                return_value=dense,
            ) as dense_loader,
            mock.patch.object(
                advanced_search,
                "select_cses",
                wraps=advanced_search.select_cses,
            ) as cses_spy,
            mock.patch.object(
                retrieval_manager,
                "build_bge_candidate_reranker",
            ) as bge_reranker,
            mock.patch.object(
                retrieval_manager,
                "build_trake_bge_candidate_reranker",
            ) as trake_reranker,
            mock.patch.object(
                retrieval_manager,
                "get_qa_search_pipeline",
            ) as qa_pipeline,
            mock.patch.object(
                retrieval_manager,
                "get_trake_pipeline",
            ) as trake_pipeline,
            mock.patch(
                "backend.app.services.retrieval.vlm_reranker.build_local_vlm_runner"
            ) as vlm_reranker,
            mock.patch(
                "backend.app.services.retrieval.qa_answerer.build_local_qwen_runner"
            ) as qwen_answerer,
        ):
            response = retrieval_manager.search_online(
                "a person opens a refrigerator and takes a bottle",
                task="kis",
                top_k=2,
                debug=True,
            )

        self.assertEqual(dense.search_calls, [2])
        self.assertEqual(visual.encoder.queries, [
            "a person opens a refrigerator and takes a bottle"
        ])
        self.assertEqual(len(visual.vector_queries), 1)
        dense_loader.assert_called_once_with()
        self.assertGreaterEqual(cses_spy.call_count, 2)
        self.assertIn("RESCUED", {item["video_id"] for item in response["candidates"]})
        trace = response["routing_trace"]
        self.assertTrue(trace["coarse_to_dense"]["executed"])
        self.assertEqual(trace["dense_rescue_clip_count"], 1)
        self.assertGreater(trace["selected_row_count"], 0)
        for name in (
            "selected_visual_ms",
            "text_retrieval_ms",
            "fusion_ms",
            "dense_global_ms",
            "dense_rescue_ms",
            "cses_ms",
            "deterministic_rerank_ms",
            "context_attachment_ms",
            "total_ms",
        ):
            self.assertIn(name, trace["latency"])
        rescued = next(
            item for item in response["candidates"] if item["video_id"] == "RESCUED"
        )
        self.assertTrue(rescued["score_breakdown"])
        self.assertIsNotNone(rescued["cses_selection"])
        self.assertFalse(trace["heavy_rerankers"]["loaded_by_online_pipeline"])
        for heavy in (
            bge_reranker,
            trake_reranker,
            qa_pipeline,
            trake_pipeline,
            vlm_reranker,
            qwen_answerer,
        ):
            heavy.assert_not_called()

    def test_dense_missing_policy_is_explicit_for_fallback_and_error(self) -> None:
        hybrid, _visual, _text = _hybrid()
        fallback = OnlinePipeline(
            hybrid_engine=hybrid,
            runtime_config=RetrievalRuntimeConfig(
                query_expansion=QueryExpansionConfig(enabled=False),
                online=OnlineRetrievalConfig(dense_missing_behavior="fallback_sparse"),
            ),
        ).run("a red car", top_k=1)
        dense_trace = fallback["routing_trace"]["coarse_to_dense"]
        self.assertFalse(dense_trace["executed"])
        self.assertEqual(dense_trace["mode"], "selected_only_fallback")
        self.assertIn("dense_index_loader_unavailable", dense_trace["fallback_reason"])

        strict = OnlinePipeline(
            hybrid_engine=hybrid,
            runtime_config=RetrievalRuntimeConfig(
                query_expansion=QueryExpansionConfig(enabled=False),
                online=OnlineRetrievalConfig(dense_missing_behavior="error"),
            ),
        )
        with self.assertRaisesRegex(FileNotFoundError, "dense_index_loader_unavailable"):
            strict.run("a red car", top_k=1)


if __name__ == "__main__":
    unittest.main()
