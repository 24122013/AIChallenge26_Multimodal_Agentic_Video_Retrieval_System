from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.indexing.build_text_index import write_text_index
from competition.pipeline import (
    CorpusVideo,
    answers_from_results,
    build_competition_hybrid_engine,
)


class FakeVisualEngine:
    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        return VisualSearchResponse(
            query=query,
            top_k=int(top_k or 1),
            latency_ms=0.0,
            results=[
                RetrievalResult(
                    video_id="video0001",
                    frame_id="FRAME_video0001_000001",
                    segment_id="SHOT_video0001_000001",
                    shot_id="SHOT_video0001_000001",
                    timestamp=1.0,
                    frame_index=25,
                    score=0.8,
                    modality_scores={"visual": 0.8},
                )
            ],
        )


class CompetitionPipelineTest(unittest.TestCase):
    def test_hybrid_engine_enables_every_original_text_modality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_index = Path(temp_dir) / "retrieval_text_index.json"
            write_text_index(
                [
                    {
                        "video_id": "video0001",
                        "segment_id": "SHOT_video0001_000001",
                        "start_keyframe": "FRAME_video0001_000001",
                        "start_time": 1.0,
                        "start_frame": 25,
                        "captions_aggregated": "a person playing guitar",
                        "ocr": [{"text": "live music"}],
                        "asr": [{"text": "this is a song"}],
                        "objects": [{"class_name": "guitar"}],
                    }
                ],
                text_index,
            )
            engine = build_competition_hybrid_engine(
                FakeVisualEngine(),
                text_index_path=text_index,
                retrieval_config_path=Path("configs/retrieval.yaml"),
                search_depth=100,
            )

            self.assertEqual(
                engine.available_modalities,
                ("visual", "asr", "caption", "objects", "ocr"),
            )
            response = engine.search("person playing guitar", top_k=5)
            self.assertTrue(response.results)
            self.assertEqual(response.results[0].video_id, "video0001")
            self.assertIn("caption", response.results[0].modality_scores)
            self.assertIn("visual", response.results[0].modality_scores)

    def test_hybrid_segment_result_uses_its_start_frame_in_submission(self) -> None:
        corpus = [
            CorpusVideo(
                filename=f"video{index:04d}.mp4",
                relative_path=Path(f"videos/video{index:04d}.mp4"),
                fps=25.0,
                frame_count=500,
            )
            for index in range(1, 101)
        ]
        answers = answers_from_results(
            [
                RetrievalResult(
                    video_id="video0001",
                    frame_id="FRAME_video0001_000001",
                    segment_id="SHOT_video0001_000001",
                    timestamp=1.0,
                    frame_index=37,
                    score=0.9,
                    modality_scores={"caption": 0.9},
                )
            ],
            corpus=corpus,
        )

        self.assertEqual(len(answers), 100)
        self.assertEqual(answers[0], "video0001.mp4,37")
        self.assertEqual(len(set(answers)), 100)


if __name__ == "__main__":
    unittest.main()
