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
    video_id: str = "V001",
    timestamp: float = 0.0,
    score: float = 0.5,
    caption: str = "",
    ocr_text: str = "",
    objects: list[str] | None = None,
    shot_id: str = "",
) -> RetrievalResult:
    return RetrievalResult(
        video_id=video_id,
        frame_id=frame_id,
        timestamp=timestamp,
        score=score,
        caption=caption,
        ocr_text=ocr_text,
        objects=objects or [],
        shot_id=shot_id,
        timestamp_confidence=1.0,
    )


class FakeVisualEngine:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        results = self.results_by_query[query]
        return VisualSearchResponse(
            query=query,
            top_k=top_k or len(results),
            latency_ms=1.0,
            results=results[:top_k],
        )


class Phase3RetrievalTest(unittest.TestCase):
    def test_metadata_modal_fields_flow_to_retrieval_result(self) -> None:
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
                            "ocr": "OPEN 24H",
                            "objects": ["cashier", "counter"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            store = MetadataStore.from_frame_map(frame_map_path)

            result = store.get_by_faiss_index(0)

            self.assertEqual(result.caption, "a cashier at a shop counter")
            self.assertEqual(result.ocr_text, "OPEN 24H")
            self.assertEqual(result.objects, ["cashier", "counter"])

    def test_hybrid_reranker_promotes_metadata_match(self) -> None:
        reranker = HybridReranker()
        visual_only = _result("visual", score=0.90, caption="a beach")
        metadata_match = _result(
            "metadata",
            score=0.60,
            caption="a man cooking in a kitchen",
            objects=["man", "kitchen"],
        )

        results = reranker.rerank(
            query="man cooking kitchen",
            candidates=[visual_only, metadata_match],
        )

        self.assertEqual(results[0].frame_id, "metadata")
        self.assertGreater(results[0].score, results[1].score)

    def test_temporal_query_decomposition_handles_after(self) -> None:
        events = decompose_temporal_query("talks to cashier after man enters shop")

        self.assertEqual([event.text for event in events], ["man enters shop", "talks to cashier"])

    def test_match_ordered_events_requires_same_video_and_order(self) -> None:
        first_event = [
            _result("enter", video_id="V001", timestamp=10.0, score=0.8),
            _result("wrong_video", video_id="V002", timestamp=11.0, score=0.9),
        ]
        second_event = [
            _result("cashier", video_id="V001", timestamp=30.0, score=0.7),
            _result("too_early", video_id="V001", timestamp=5.0, score=1.0),
        ]

        matches = match_ordered_events([first_event, second_event], max_gap_seconds=60.0)

        self.assertEqual(len(matches), 1)
        self.assertEqual([event.frame_id for event in matches[0].events], ["enter", "cashier"])

    def test_hybrid_engine_temporal_search_runs_event_queries(self) -> None:
        visual_engine = FakeVisualEngine(
            {
                "man enters shop": [_result("enter", timestamp=10.0, score=0.9)],
                "talks to cashier": [_result("cashier", timestamp=40.0, score=0.8)],
            }
        )
        engine = HybridSearchEngine(visual_engine=visual_engine)

        matches = engine.temporal_search(
            "man enters shop then talks to cashier",
            top_k=5,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].video_id, "V001")

    def test_candidate_merger_keeps_best_duplicate_and_dedupes_shot(self) -> None:
        lower_duplicate = _result("F001", timestamp=10.0, score=0.4, shot_id="S001")
        higher_duplicate = _result("F001", timestamp=10.0, score=0.9, shot_id="S001")
        same_shot = _result("F002", timestamp=11.0, score=0.8, shot_id="S001")
        other_shot = _result("F003", timestamp=20.0, score=0.7, shot_id="S002")

        merged = merge_candidates(
            [[lower_duplicate, same_shot], [higher_duplicate, other_shot]],
            dedupe_same_shot=True,
        )

        self.assertEqual([item.frame_id for item in merged], ["F001", "F003"])
        self.assertEqual(merged[0].score, 0.9)


if __name__ == "__main__":
    unittest.main()
