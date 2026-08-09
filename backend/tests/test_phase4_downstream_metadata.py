from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.services.indexing.build_faiss_index import frame_map_record
from backend.app.services.metadata.metadata_store import MetadataStore
from backend.app.services.retrieval.text_index import build_text_index
from src.indexing.build_segment_metadata import build_segment_metadata


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


class Phase4DownstreamMetadataTest(unittest.TestCase):
    def _selected_record(self) -> dict:
        return {
            "frame_id": "F0",
            "video_id": "V0",
            "shot_id": "SHOT_0",
            "segment_id": "SHOT_0",
            "shot_start": 0.0,
            "shot_end": 5.0,
            "timestamp": 1.0,
            "frame_index": 25,
            "keyframe_path": "keyframes/V0/F0.jpg",
            "candidate_id": "C0",
            "candidate_index": 0,
            "candidate_reasons": ["dense_interval"],
            "keyframe_strategy": "multimodal_coverage",
            "selection_phase": "protected",
            "selection_rank": 0,
            "selection_reasons": ["protected:ocr_new"],
            "covered_event_ids": ["OCR_0"],
            "selection_score": 0.87,
            "protected": True,
            "coverage_added": False,
            "importance_score": 0.81,
            "semantic_novelty": 0.64,
            "component_scores": {"ocr": 0.9, "transition": 0.4},
            "available_modalities": ["ocr", "transition"],
            "protected_event_ids": ["OCR_0"],
            "selection_provenance": {
                "strategy": "multimodal_coverage",
                "covered_event_ids": ["OCR_0"],
            },
        }

    def test_faiss_frame_map_and_metadata_store_keep_selection_provenance(self) -> None:
        record = self._selected_record()
        frame_map_value = frame_map_record(record)
        self.assertEqual(frame_map_value["importance_score"], 0.81)
        self.assertEqual(frame_map_value["semantic_novelty"], 0.64)
        self.assertEqual(frame_map_value["protected_event_ids"], ["OCR_0"])

        with tempfile.TemporaryDirectory() as temporary:
            frame_map_path = Path(temporary) / "frame_map.json"
            frame_map_path.write_text(
                json.dumps({"0": frame_map_value}),
                encoding="utf-8",
            )
            loaded = MetadataStore.from_frame_map(frame_map_path).get_by_faiss_index(0)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.selection_phase, "protected")
            self.assertEqual(loaded.covered_event_ids, ["OCR_0"])
            self.assertTrue(loaded.protected)
            self.assertEqual(loaded.importance_score, 0.81)
            self.assertEqual(loaded.semantic_novelty, 0.64)
            self.assertEqual(loaded.component_scores["ocr"], 0.9)
            self.assertEqual(loaded.to_dict()["candidate_id"], "C0")

    def test_segments_and_text_index_retain_keyframe_selection_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keyframes = root / "keyframes.jsonl"
            captions = root / "captions.jsonl"
            segments = root / "segments.jsonl"
            _write_jsonl(keyframes, [self._selected_record()])
            _write_jsonl(
                captions,
                [
                    {
                        "video_id": "V0",
                        "frame_id": "F0",
                        "timestamp": 1.0,
                        "status": "success",
                        "caption": "a menu appears",
                    }
                ],
            )
            build_segment_metadata(
                keyframes,
                segments,
                captions_path=captions,
            )
            segment = json.loads(segments.read_text(encoding="utf-8").strip())
            self.assertEqual(segment["covered_event_ids"], ["OCR_0"])
            self.assertTrue(segment["protected"])
            self.assertEqual(
                segment["keyframe_selection"][0]["semantic_novelty"],
                0.64,
            )

            payload = build_text_index([segment])
            documents = payload["modalities"]["caption"]["documents"]
            document = next(iter(documents.values()))
            self.assertEqual(document["covered_event_ids"], ["OCR_0"])
            self.assertTrue(document["protected"])
            self.assertEqual(
                document["keyframe_selection"][0]["importance_score"],
                0.81,
            )


if __name__ == "__main__":
    unittest.main()
