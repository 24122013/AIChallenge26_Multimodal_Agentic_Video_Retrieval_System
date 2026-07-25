from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.indexing.build_neighbor_index import build_neighbor_index
from src.indexing.build_segment_metadata import build_segment_metadata


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class NeighborIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_edges_short_video_and_empty_neighbor_sides(self) -> None:
        source = self.root / "keyframes.jsonl"
        output = self.root / "neighbors.jsonl"
        write_jsonl(
            source,
            [
                {
                    "video_id": "A",
                    "frame_id": "A0",
                    "frame_index": 0,
                    "timestamp": 0.0,
                },
                {
                    "video_id": "A",
                    "frame_id": "A1",
                    "frame_index": 30,
                    "timestamp": 1.0,
                },
                {
                    "video_id": "A",
                    "frame_id": "A2",
                    "frame_index": 60,
                    "timestamp": 2.0,
                },
            ],
        )
        build_neighbor_index(source, output, window_seconds=5.0)
        records = read_jsonl(output)

        self.assertEqual(records[0]["neighbors_before"], [])
        self.assertEqual(
            [item["frame_id"] for item in records[0]["neighbors_after"]],
            ["A1", "A2"],
        )
        self.assertEqual(
            [item["frame_id"] for item in records[-1]["neighbors_before"]],
            ["A0", "A1"],
        )
        self.assertEqual(records[-1]["neighbors_after"], [])
        self.assertTrue(
            all(
                record["frame_id"]
                not in {
                    item["frame_id"]
                    for item in record["neighbors_before"]
                    + record["neighbors_after"]
                }
                for record in records
            )
        )

    def test_per_video_fps_fallback_never_crosses_videos(self) -> None:
        source = self.root / "keyframes.jsonl"
        output = self.root / "neighbors.jsonl"
        write_jsonl(
            source,
            [
                {"video_id": "V30", "frame_id": "V30_2", "frame_index": 60, "fps": 30},
                {"video_id": "V25", "frame_id": "V25_1", "frame_index": 25, "fps": 25},
                {"video_id": "V30", "frame_id": "V30_1", "frame_index": 30, "fps": 30},
                {"video_id": "V25", "frame_id": "V25_3", "frame_index": 75, "fps": 25},
            ],
        )
        build_neighbor_index(source, output, window_seconds=2.1)
        records = read_jsonl(output)
        by_id = {record["frame_id"]: record for record in records}

        self.assertEqual(by_id["V30_1"]["timestamp"], 1.0)
        self.assertEqual(by_id["V25_3"]["timestamp"], 3.0)
        for record in records:
            prefix = record["frame_id"].split("_", 1)[0]
            neighbors = record["neighbors_before"] + record["neighbors_after"]
            self.assertTrue(
                all(item["frame_id"].startswith(prefix + "_") for item in neighbors)
            )

    def test_idempotent_duplicate_safe_and_deterministic_order(self) -> None:
        source = self.root / "keyframes.jsonl"
        reversed_source = self.root / "keyframes_reversed.jsonl"
        first_output = self.root / "neighbors_first.jsonl"
        second_output = self.root / "neighbors_second.jsonl"
        records = [
            {"video_id": "B", "frame_id": "B1", "timestamp": 1.0},
            {"video_id": "A", "frame_id": "A2", "timestamp": 2.0},
            {"video_id": "A", "frame_id": "A1", "timestamp": 1.0},
        ]
        write_jsonl(source, records + [records[2]])
        write_jsonl(reversed_source, list(reversed(records)))

        result = build_neighbor_index(source, first_output, window_seconds=3.0)
        build_neighbor_index(reversed_source, second_output, window_seconds=3.0)
        first_bytes = first_output.read_bytes()

        build_neighbor_index(source, first_output, window_seconds=3.0)
        self.assertEqual(first_output.read_bytes(), first_bytes)
        self.assertEqual(first_bytes, second_output.read_bytes())
        self.assertEqual(result["duplicate_input_count"], 1)
        self.assertEqual(len(read_jsonl(first_output)), 3)

    def test_zero_window_produces_no_neighbors(self) -> None:
        source = self.root / "keyframes.jsonl"
        output = self.root / "neighbors.jsonl"
        write_jsonl(
            source,
            [
                {"video_id": "A", "frame_id": "A1", "timestamp": 0.0},
                {"video_id": "A", "frame_id": "A2", "timestamp": 1.0},
            ],
        )
        build_neighbor_index(source, output, window_seconds=0.0)
        for record in read_jsonl(output):
            self.assertEqual(record["neighbors_before"], [])
            self.assertEqual(record["neighbors_after"], [])


class SegmentMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _keyframes(self) -> list[dict[str, Any]]:
        return [
            {
                "video_id": "A",
                "frame_id": "A0",
                "segment_id": "SHOT_A_001",
                "shot_id": "SHOT_A_001",
                "shot_start": 0.0,
                "shot_end": 5.0,
                "timestamp": 0.5,
                "frame_index": 15,
            },
            {
                "video_id": "A",
                "frame_id": "A1",
                "segment_id": "SHOT_A_001",
                "shot_id": "SHOT_A_001",
                "shot_start": 0.0,
                "shot_end": 5.0,
                "timestamp": 4.5,
                "frame_index": 135,
            },
            {
                "video_id": "A",
                "frame_id": "A2",
                "segment_id": "SHOT_A_002",
                "shot_id": "SHOT_A_002",
                "shot_start": 5.0,
                "shot_end": 8.0,
                "timestamp": 6.0,
                "frame_index": 180,
            },
            {
                "video_id": "B",
                "frame_id": "B0",
                "segment_id": "SHOT_B_001",
                "shot_id": "SHOT_B_001",
                "shot_start": 0.0,
                "shot_end": 2.0,
                "timestamp": 1.0,
                "frame_index": 25,
            },
        ]

    def test_boundary_segments_empty_modalities_and_no_video_mixing(self) -> None:
        keyframes = self.root / "keyframes.jsonl"
        output = self.root / "segments.jsonl"
        write_jsonl(keyframes, self._keyframes())

        build_segment_metadata(keyframes, output)
        records = read_jsonl(output)
        self.assertEqual(
            [record["segment_id"] for record in records],
            ["SHOT_A_001", "SHOT_A_002", "SHOT_B_001"],
        )
        first = records[0]
        self.assertEqual(first["start_time"], 0.0)
        self.assertEqual(first["end_time"], 5.0)
        self.assertEqual(first["start_frame"], 15)
        self.assertEqual(first["end_frame"], 135)
        self.assertEqual(first["keyframe_ids"], ["A0", "A1"])
        self.assertEqual(first["captions_aggregated"], "")
        self.assertEqual(first["ocr"], [])
        self.assertEqual(first["asr"], [])
        self.assertEqual(first["objects"], [])
        self.assertEqual(records[-1]["keyframe_ids"], ["B0"])

    def test_aggregates_duplicates_overlap_provenance_and_object_semantics(self) -> None:
        keyframes = self.root / "keyframes.jsonl"
        captions = self.root / "captions.jsonl"
        ocr = self.root / "ocr.jsonl"
        asr = self.root / "asr.jsonl"
        objects = self.root / "objects.jsonl"
        output = self.root / "segments.jsonl"
        write_jsonl(keyframes, self._keyframes())
        write_jsonl(
            captions,
            [
                {
                    "video_id": "A",
                    "frame_id": "A0",
                    "timestamp": 0.5,
                    "status": "success",
                    "caption": "A red car is parked.",
                },
                {
                    "video_id": "A",
                    "frame_id": "A1",
                    "timestamp": 4.5,
                    "status": "success",
                    "caption": "A red car is parked",
                },
                {
                    "video_id": "A",
                    "frame_id": "A2",
                    "timestamp": 6.0,
                    "status": "success",
                    "caption": "A person walks.",
                },
            ],
        )
        write_jsonl(
            ocr,
            [
                {
                    "video_id": "A",
                    "frame_id": "A0",
                    "timestamp": 0.5,
                    "status": "success",
                    "text_regions": [{"text": " Xin   chào ", "confidence": 0.7}],
                },
                {
                    "video_id": "A",
                    "frame_id": "A1",
                    "timestamp": 4.5,
                    "status": "success",
                    "text_regions": [{"text": "xin chào", "confidence": 0.95}],
                },
            ],
        )
        write_jsonl(
            asr,
            [
                {
                    "video_id": "A",
                    "transcript_segment_id": "ASR_1",
                    "start": 1.0,
                    "end": 3.0,
                    "text": "hello world",
                    "status": "success",
                },
                {
                    "video_id": "A",
                    "transcript_segment_id": "ASR_1_DUP",
                    "start": 2.0,
                    "end": 3.5,
                    "text": " hello  world ",
                    "status": "success",
                },
                {
                    "video_id": "A",
                    "transcript_segment_id": "ASR_PARTIAL",
                    "start": 4.8,
                    "end": 5.5,
                    "text": "crosses boundary",
                    "status": "success",
                },
                {
                    "video_id": "B",
                    "transcript_segment_id": "ASR_B",
                    "start": 0.0,
                    "end": 2.0,
                    "text": "other video",
                    "status": "success",
                },
            ],
        )
        write_jsonl(
            objects,
            [
                {
                    "video_id": "A",
                    "frame_id": "A0",
                    "status": "success",
                    "objects": [
                        {"class_name": "Car", "confidence": 0.8},
                        {"class_name": "Person", "confidence": 0.6, "track_id": 9},
                    ],
                },
                {
                    "video_id": "A",
                    "frame_id": "A1",
                    "status": "success",
                    "objects": [
                        {"class_name": " car ", "confidence": 0.9},
                        {"class_name": "Person", "confidence": 0.7, "track_id": 9},
                    ],
                },
            ],
        )

        build_segment_metadata(
            keyframes,
            output,
            captions_path=captions,
            ocr_path=ocr,
            asr_path=asr,
            objects_path=objects,
        )
        first = read_jsonl(output)[0]
        self.assertEqual(first["captions_aggregated"], "A red car is parked.")
        self.assertEqual(first["caption_source_ids"], ["A0", "A1"])
        self.assertEqual(len(first["ocr"]), 1)
        self.assertEqual(first["ocr"][0]["confidence"], 0.95)
        self.assertEqual(first["ocr"][0]["first_seen"], 0.5)
        self.assertEqual(first["ocr"][0]["last_seen"], 4.5)
        self.assertEqual(first["ocr"][0]["source_ids"], ["A0", "A1"])
        self.assertEqual(len(first["asr"]), 2)
        self.assertEqual(first["asr"][0]["source_ids"], ["ASR_1", "ASR_1_DUP"])
        self.assertEqual(len(first["asr"][0]["source_intervals"]), 2)
        self.assertEqual(first["asr"][1]["text"], "crosses boundary")
        by_label = {item["label"]: item for item in first["objects"]}
        self.assertEqual(by_label["car"]["occurrence_count"], 2)
        self.assertEqual(
            by_label["car"]["occurrence_count_semantics"],
            "detection_occurrence",
        )
        self.assertEqual(by_label["person"]["occurrence_count"], 1)
        self.assertEqual(
            by_label["person"]["occurrence_count_semantics"],
            "unique_track",
        )

    def test_fixed_windows_fps_fallback_and_deterministic_rerun(self) -> None:
        keyframes = self.root / "keyframes.jsonl"
        output = self.root / "segments.jsonl"
        write_jsonl(
            keyframes,
            [
                {"video_id": "A", "frame_id": "A2", "frame_index": 75, "fps": 25},
                {"video_id": "A", "frame_id": "A1", "frame_index": 25, "fps": 25},
                {"video_id": "A", "frame_id": "A3", "frame_index": 150, "fps": 25},
                {"video_id": "A", "frame_id": "A1", "frame_index": 25, "fps": 25},
            ],
        )
        build_segment_metadata(
            keyframes,
            output,
            strategy="fixed",
            fixed_duration_seconds=5.0,
        )
        first_bytes = output.read_bytes()
        records = read_jsonl(output)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["keyframe_ids"], ["A1", "A2"])
        self.assertEqual(records[0]["start_time"], 0.0)
        self.assertEqual(records[0]["end_time"], 5.0)

        build_segment_metadata(
            keyframes,
            output,
            strategy="fixed",
            fixed_duration_seconds=5.0,
        )
        self.assertEqual(output.read_bytes(), first_bytes)


if __name__ == "__main__":
    unittest.main()
