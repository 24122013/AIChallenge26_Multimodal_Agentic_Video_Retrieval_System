from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

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
from backend.app.services.retrieval.retrieval_config import RetrievalRuntimeConfig


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


class OnlinePipelineTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
