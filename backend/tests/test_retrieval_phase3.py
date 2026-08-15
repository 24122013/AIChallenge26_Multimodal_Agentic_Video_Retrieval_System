from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.metadata.metadata_store import MetadataStore
from backend.app.services.retrieval.candidate_merger import merge_candidates
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine
from backend.app.services.retrieval.rerank import HybridReranker
from backend.app.services.retrieval.temporal_search import (
    decompose_temporal_query,
    match_ordered_events,
)


def _result(
    frame_id: str,
    *,
    video_id: str = "V001",
    timestamp: float = 0.0,
    score: float = 0.5,
    caption: str = "",
    shot_id: str = "",
    modality_scores: dict[str, float] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        video_id=video_id,
        frame_id=frame_id,
        timestamp=timestamp,
        score=score,
        caption=caption,
        shot_id=shot_id,
        timestamp_confidence=1.0,
        modality_scores=modality_scores or {},
    )


class FakeVisualEngine:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        results = self.results_by_query.get(query, [])
        return VisualSearchResponse(
            query=query,
            top_k=top_k or len(results),
            latency_ms=1.0,
            results=results[:top_k],
        )


class FakeTextEngine:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query

    def search_results(self, query: str, top_k: int | None = None):
        return self.results_by_query.get(query, [])[:top_k]


class Phase3RetrievalTest(unittest.TestCase):
    def test_metadata_store_reads_optional_multimodal_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_map_path = Path(tmp_dir) / "frame_map.json"
            frame_map_path.write_text(
                json.dumps(
                    {
                        "0": {
                            "frame_id": "F001",
                            "video_id": "V001",
                            "timestamp": 3.0,
                            "keyframe_path": "data/keyframes/V001/000001.jpg",
                            "caption": "a cashier at a shop counter",
                            "ocr": [{"text": "OPEN 24H"}],
                            "objects": [
                                {"class_name": "cashier"},
                                {"label": "counter"},
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = MetadataStore.from_frame_map(
                frame_map_path
            ).get_by_faiss_index(0)

            self.assertIsNotNone(result)
            self.assertEqual(result.caption, "a cashier at a shop counter")
            self.assertEqual(result.ocr_text, "OPEN 24H")
            self.assertEqual(result.objects, ["cashier", "counter"])

    def test_candidate_merger_preserves_modality_scores(self) -> None:
        visual = _result(
            "F001",
            score=0.4,
            caption="",
            modality_scores={"visual": 0.4},
        )
        caption = _result(
            "F001",
            score=0.9,
            caption="a person cooking",
            modality_scores={"caption": 0.9},
        )

        merged = merge_candidates([[visual], [caption]])

        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0].modality_scores,
            {"visual": 0.4, "caption": 0.9},
        )
        self.assertEqual(merged[0].caption, "a person cooking")

    def test_hybrid_engine_merges_text_candidate_before_rerank(self) -> None:
        query = "person cooking"
        visual_engine = FakeVisualEngine(
            {
                query: [
                    _result(
                        "F001",
                        score=0.55,
                        caption="orange sunset",
                        modality_scores={"visual": 0.55},
                    ),
                    _result(
                        "F002",
                        score=0.40,
                        modality_scores={"visual": 0.40},
                    ),
                ]
            }
        )
        caption_engine = FakeTextEngine(
            {
                query: [
                    _result(
                        "F002",
                        score=0.95,
                        caption="a person cooking in a kitchen",
                        modality_scores={"caption": 0.95},
                    )
                ]
            }
        )
        engine = HybridSearchEngine(
            visual_engine=visual_engine,
            text_engines={"caption": caption_engine},
            reranker=HybridReranker(),
        )

        response = engine.search(query, top_k=2)

        self.assertEqual(response.results[0].frame_id, "F002")
        self.assertEqual(
            engine.available_modalities,
            ("visual", "caption"),
        )

    def test_temporal_query_decomposition_supports_vietnamese(self) -> None:
        events = decompose_temporal_query(
            "người đàn ông vào bếp rồi bắt đầu nấu ăn"
        )

        self.assertEqual(
            [event.text for event in events],
            ["người đàn ông vào bếp", "bắt đầu nấu ăn"],
        )

    def test_match_ordered_events_requires_same_video_and_order(self) -> None:
        first_event = [
            _result("enter", video_id="V001", timestamp=10.0, score=0.8),
            _result("wrong_video", video_id="V002", timestamp=11.0, score=0.9),
        ]
        second_event = [
            _result("cook", video_id="V001", timestamp=30.0, score=0.7),
            _result("too_early", video_id="V001", timestamp=5.0, score=1.0),
        ]

        matches = match_ordered_events(
            [first_event, second_event],
            max_gap_seconds=60.0,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(
            [event.frame_id for event in matches[0].events],
            ["enter", "cook"],
        )

    def test_temporal_matching_prefers_complete_event_semantics(self) -> None:
        first_event = [
            _result(
                "wrong_enter",
                timestamp=10.0,
                score=0.9,
                caption="a person climbing a wall",
            ),
            _result(
                "correct_enter",
                timestamp=20.0,
                score=0.7,
                caption="a person entering a room",
            ),
        ]
        second_event = [
            _result(
                "wrong_sit",
                timestamp=30.0,
                score=0.9,
                caption="a man walking down the street",
            ),
            _result(
                "correct_sit",
                timestamp=40.0,
                score=0.7,
                caption="a man sitting down at a table",
            ),
        ]

        matches = match_ordered_events(
            [first_event, second_event],
            max_gap_seconds=60.0,
            event_queries=["a person enters", "a man sits down"],
        )

        self.assertEqual(
            [event.frame_id for event in matches[0].events],
            ["correct_enter", "correct_sit"],
        )

    def test_temporal_matching_does_not_reuse_a_frame_when_alternative_exists(
        self,
    ) -> None:
        first_event = [
            _result("same", timestamp=10.0, score=0.9),
        ]
        second_event = [
            _result("same", timestamp=11.0, score=1.0),
            _result("distinct", timestamp=12.0, score=0.8),
        ]

        matches = match_ordered_events(
            [first_event, second_event],
            max_gap_seconds=60.0,
        )

        self.assertEqual(
            [event.frame_id for event in matches[0].events],
            ["same", "distinct"],
        )

    def test_temporal_matching_keeps_best_effort_sparse_fallback(self) -> None:
        matches = match_ordered_events(
            [
                [_result("only", timestamp=10.0, score=0.8)],
                [_result("only", timestamp=10.0, score=0.7)],
            ],
            max_gap_seconds=60.0,
            event_queries=["person enters", "person sits"],
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(
            [event.frame_id for event in matches[0].events],
            ["only", "only"],
        )


if __name__ == "__main__":
    unittest.main()
