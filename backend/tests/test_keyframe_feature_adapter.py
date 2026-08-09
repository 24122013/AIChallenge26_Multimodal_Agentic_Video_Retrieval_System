from __future__ import annotations

import unittest
import unicodedata

import numpy as np

from backend.app.services.indexing.keyframe_feature_adapter import (
    FeatureAdapterConfig,
    adapt_feature_records,
)
from backend.app.services.indexing.keyframe_selection import (
    SelectionConfig,
    select_keyframes,
)


def base_candidate(
    index: int,
    timestamp: float,
    *,
    shot_index: int | None = None,
) -> dict[str, object]:
    shot_index = index if shot_index is None else shot_index
    return {
        "candidate_id": f"C{index}",
        "frame_id": f"F{index}",
        "video_id": "VIDEO",
        "timestamp": timestamp,
        "frame_index": index,
        "shot_index": shot_index,
        "candidate_reasons": ["dense_interval"],
    }


def ocr_record(
    index: int,
    text: str = "",
    *,
    confidence: float = 0.9,
    status: str = "success",
    polygon: list[list[float]] | None = None,
    use_legacy_id: bool = False,
) -> dict[str, object]:
    value: dict[str, object] = {
        "video_id": "VIDEO",
        "status": status,
        "image_size": [100, 100],
        "ocr_text": text,
        "text_regions": [],
    }
    value["frame_id" if use_legacy_id else "candidate_id"] = (
        f"F{index}" if use_legacy_id else f"C{index}"
    )
    if text and status == "success":
        value["text_regions"] = [
            {
                "text": text,
                "confidence": confidence,
                "polygon": polygon or [[10, 10], [70, 10], [70, 30], [10, 30]],
            }
        ]
    return value


def object_record(
    index: int,
    detections: list[dict[str, object]] | None = None,
    *,
    status: str = "success",
) -> dict[str, object]:
    return {
        "candidate_id": f"C{index}",
        "video_id": "VIDEO",
        "status": status,
        "image_size": [100, 100],
        "objects": detections or [],
    }


def detection(
    name: str,
    *,
    class_id: int,
    confidence: float = 0.8,
    bbox: list[float] | None = None,
) -> dict[str, object]:
    return {
        "class_id": class_id,
        "class_name": name,
        "confidence": confidence,
        "bbox_xyxy": bbox or [10, 10, 60, 60],
    }


def embedding_metadata(candidate_ids: list[str], dimension: int) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate_id,
            "video_id": "VIDEO",
            "embedding_index": index,
            "vector_dim": dimension,
            "normalized": True,
        }
        for index, candidate_id in enumerate(candidate_ids)
    ]


class KeyframeFeatureAdapterTest(unittest.TestCase):
    def test_join_and_output_are_order_independent_and_preserve_provenance(self) -> None:
        candidates = [base_candidate(2, 1.0), base_candidate(0, 0.0), base_candidate(1, 0.5)]
        ocr = [ocr_record(1, "Title"), ocr_record(0), ocr_record(2, "Title")]
        config = FeatureAdapterConfig(
            ocr_persistence_candidates=2,
            ocr_common_frame_fraction=1.0,
            ocr_common_shot_fraction=1.0,
        )

        first = adapt_feature_records(candidates, ocr_records=ocr, config=config)
        second = adapt_feature_records(
            reversed(candidates),
            ocr_records=reversed(ocr),
            config=config,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            [value.candidate_id for value in first.selection_candidates],
            ["C0", "C1", "C2"],
        )
        self.assertEqual(first.selection_candidates[0].source_reasons, ("dense_interval",))

        legacy = adapt_feature_records(
            candidates,
            ocr_records=[ocr_record(0, use_legacy_id=True)],
            config=config,
        )
        self.assertIn("ocr", legacy.candidate_scores[0].available_modalities)

    def test_identity_join_failures_are_strict(self) -> None:
        candidates = [base_candidate(0, 0.0), base_candidate(1, 0.5)]
        duplicate = [ocr_record(0), ocr_record(0)]
        with self.assertRaisesRegex(ValueError, "duplicate ocr"):
            adapt_feature_records(candidates, ocr_records=duplicate)
        unknown = ocr_record(0)
        unknown["candidate_id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            adapt_feature_records(candidates, ocr_records=[unknown])
        alias_in_candidate_field = ocr_record(0)
        alias_in_candidate_field["candidate_id"] = "F0"
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            adapt_feature_records(candidates, ocr_records=[alias_in_candidate_field])
        unknown_frame = ocr_record(0)
        unknown_frame["frame_id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            adapt_feature_records(candidates, ocr_records=[unknown_frame])
        mixed = ocr_record(0)
        mixed["video_id"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "does not match"):
            adapt_feature_records(candidates, ocr_records=[mixed])
        conflict = ocr_record(0)
        conflict["frame_id"] = "F1"
        with self.assertRaisesRegex(ValueError, "different candidates"):
            adapt_feature_records(candidates, ocr_records=[conflict])

    def test_embedding_contract_and_missing_embedding_are_explicit(self) -> None:
        candidates = [base_candidate(index, index * 0.5) for index in range(3)]
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = adapt_feature_records(
            candidates,
            embeddings=matrix,
            embedding_records=embedding_metadata(["C0", "C2"], 2),
        )

        self.assertEqual(result.report.missing_embedding_count, 1)
        self.assertEqual(result.selection_candidates[1].semantic_embedding, ())
        self.assertEqual(result.selection_candidates[2].semantic_embedding, (0.0, 1.0))

        with self.assertRaisesRegex(ValueError, "float32"):
            adapt_feature_records(
                candidates,
                embeddings=matrix.astype(np.float64),
                embedding_records=embedding_metadata(["C0", "C2"], 2),
            )
        bad_metadata = embedding_metadata(["C0", "C2"], 2)
        bad_metadata[1]["embedding_index"] = 0
        with self.assertRaisesRegex(ValueError, "embedding_index mismatch"):
            adapt_feature_records(
                candidates,
                embeddings=matrix,
                embedding_records=bad_metadata,
            )

    def test_availability_weighting_distinguishes_empty_error_and_missing(self) -> None:
        candidates = [base_candidate(index, index * 0.5) for index in range(3)]
        ocr = [ocr_record(0), ocr_record(1, status="error")]
        objects = [
            object_record(0, [detection("bicycle", class_id=1)]),
            object_record(1),
        ]
        config = FeatureAdapterConfig(
            ocr_weight=1.0,
            object_weight=1.0,
            transition_weight=0.0,
            caption_weight=0.0,
            asr_weight=0.0,
        )

        result = adapt_feature_records(
            candidates,
            ocr_records=ocr,
            object_records=objects,
            config=config,
        )
        by_id = {value.candidate_id: value for value in result.candidate_scores}

        self.assertEqual(by_id["C0"].available_modalities, ("ocr", "objects"))
        self.assertGreater(by_id["C0"].importance_score, 0.0)
        self.assertEqual(by_id["C1"].available_modalities, ("objects",))
        self.assertEqual(by_id["C1"].importance_score, 0.0)
        self.assertEqual(by_id["C2"].available_modalities, ())
        self.assertEqual(by_id["C2"].importance_score, 0.0)

    def test_ocr_normalization_persistence_and_weak_singleton(self) -> None:
        candidates = [base_candidate(index, index * 0.5, shot_index=1) for index in range(3)]
        decomposed = unicodedata.normalize("NFD", "thành phố")
        ocr = [
            ocr_record(0, " THÀNH   PHỐ! ", confidence=0.84),
            ocr_record(1, decomposed, confidence=0.86),
            ocr_record(2, "weak singleton", confidence=0.80),
        ]
        config = FeatureAdapterConfig(
            ocr_persistence_candidates=2,
            ocr_common_frame_fraction=1.0,
            ocr_common_shot_fraction=1.0,
        )

        result = adapt_feature_records(candidates, ocr_records=ocr, config=config)
        ocr_events = [event for event in result.protected_events if event.event_type == "ocr_new"]

        self.assertEqual(len(ocr_events), 1)
        self.assertEqual(ocr_events[0].candidate_ids, ("C0", "C1"))
        self.assertEqual(result.report.suppressed_ocr_weak_episodes, 1)

    def test_fixed_ocr_overlay_is_suppressed_but_rare_title_survives(self) -> None:
        candidates = [base_candidate(index, index * 0.5) for index in range(8)]
        ocr = [ocr_record(index, "Online") for index in range(6)]
        ocr.extend([ocr_record(6, "Rare title"), ocr_record(7, "Rare title")])
        config = FeatureAdapterConfig(
            ocr_common_min_frames=4,
            ocr_common_frame_fraction=0.5,
            ocr_common_shot_fraction=0.5,
            ocr_persistence_candidates=2,
        )

        result = adapt_feature_records(candidates, ocr_records=ocr, config=config)
        ocr_events = [event for event in result.protected_events if event.event_type == "ocr_new"]

        self.assertEqual(result.report.suppressed_ocr_common_tracks, 1)
        self.assertEqual(len(ocr_events), 1)
        self.assertEqual(ocr_events[0].candidate_ids, ("C6", "C7"))

    def test_object_episode_bridges_error_and_extreme_singleton_is_protected(self) -> None:
        candidates = [base_candidate(index, index * 0.5, shot_index=1) for index in range(4)]
        objects = [
            object_record(0, [detection("bicycle", class_id=1)]),
            object_record(1, status="error"),
            object_record(2, [detection("bicycle", class_id=1)]),
            object_record(
                3,
                [detection("horse", class_id=17, confidence=0.96, bbox=[0, 0, 80, 80])],
            ),
        ]

        result = adapt_feature_records(candidates, object_records=objects)
        events = [event for event in result.protected_events if event.event_type == "object_new"]

        self.assertEqual(len(events), 2)
        self.assertIn(("C0", "C2"), [event.candidate_ids for event in events])
        self.assertIn(("C3",), [event.candidate_ids for event in events])

        empty = object_record(0)
        empty["image_size"] = []
        empty_result = adapt_feature_records(
            [base_candidate(0, 0.0)],
            object_records=[empty],
        )
        self.assertIn("objects", empty_result.candidate_scores[0].available_modalities)
        self.assertEqual(
            dict(empty_result.candidate_scores[0].component_scores)["objects"],
            0.0,
        )

    def test_semantic_transition_uses_mad_floor_nms_and_two_sided_events(self) -> None:
        candidates = [base_candidate(index, index * 0.5, shot_index=1) for index in range(5)]
        matrix = np.asarray(
            [[1, 0], [-1, 0], [0, 1], [0, 1], [0, 1]],
            dtype=np.float32,
        )
        config = FeatureAdapterConfig(
            transition_absolute_floor=0.4,
            transition_mad_multiplier=0.0,
            transition_nms_seconds=1.0,
        )

        result = adapt_feature_records(
            candidates,
            embeddings=matrix,
            embedding_records=embedding_metadata([f"C{i}" for i in range(5)], 2),
            config=config,
        )
        transition_events = [
            event
            for event in result.protected_events
            if event.event_type.startswith("semantic_transition")
        ]

        self.assertEqual(result.report.transition_boundary_count, 1)
        self.assertEqual(len(transition_events), 2)
        self.assertEqual(
            {event.event_type: event.candidate_ids for event in transition_events},
            {
                "semantic_transition_pre": ("C0",),
                "semantic_transition_post": ("C1",),
            },
        )
        self.assertEqual(result.report.transition_threshold, 0.4)

    def test_transition_ignores_large_temporal_gap_and_adapter_feeds_selector(self) -> None:
        candidates = [
            base_candidate(0, 0.0, shot_index=1),
            base_candidate(1, 0.5, shot_index=1),
            base_candidate(2, 3.0, shot_index=1),
        ]
        matrix = np.asarray([[1, 0], [-1, 0], [1, 0]], dtype=np.float32)
        ocr = [ocr_record(0, "Title"), ocr_record(1, "Title"), ocr_record(2)]
        result = adapt_feature_records(
            candidates,
            ocr_records=ocr,
            embeddings=matrix,
            embedding_records=embedding_metadata(["C0", "C1", "C2"], 2),
            config=FeatureAdapterConfig(
                ocr_common_frame_fraction=1.0,
                ocr_common_shot_fraction=1.0,
                transition_absolute_floor=0.4,
                transition_max_pair_gap_seconds=0.75,
            ),
        )
        transition_events = [
            event
            for event in result.protected_events
            if event.event_type.startswith("semantic_transition")
        ]
        self.assertEqual(len(transition_events), 2)

        selected = select_keyframes(
            result.selection_candidates,
            result.protected_events,
            video_duration=3.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=0,
                protect_each_shot=False,
            ),
        )
        self.assertTrue(selected.constraints_satisfied)
        selected_ids = {item.candidate.candidate_id for item in selected.selected}
        self.assertTrue(
            all(selected_ids.intersection(event.candidate_ids) for event in result.protected_events)
        )

    def test_asr_is_half_open_and_caption_error_is_unavailable(self) -> None:
        candidates = [base_candidate(index, float(index), shot_index=1) for index in range(3)]
        captions = [
            {
                "candidate_id": "C0",
                "video_id": "VIDEO",
                "status": "success",
                "caption": "red car",
            },
            {
                "candidate_id": "C1",
                "video_id": "VIDEO",
                "status": "error",
                "caption": "",
            },
        ]
        asr = [
            {
                "video_id": "VIDEO",
                "status": "success",
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
                "confidence": 1.0,
                "no_speech_probability": 0.0,
            },
            {
                "candidate_id": "C2",
                "video_id": "VIDEO",
                "status": "success",
                "start": 2.0,
                "end": 2.5,
                "text": "",
            },
        ]
        result = adapt_feature_records(
            candidates,
            caption_records=captions,
            asr_records=asr,
            config=FeatureAdapterConfig(
                ocr_weight=0.0,
                object_weight=0.0,
                transition_weight=0.0,
                caption_weight=1.0,
                asr_weight=1.0,
            ),
        )
        by_id = {value.candidate_id: value for value in result.candidate_scores}

        self.assertIn("asr", by_id["C0"].available_modalities)
        self.assertGreater(dict(by_id["C0"].component_scores)["asr"], 0.0)
        self.assertEqual(dict(by_id["C1"].component_scores)["asr"], 0.0)
        self.assertIn("asr", by_id["C2"].available_modalities)
        self.assertEqual(dict(by_id["C2"].component_scores)["asr"], 0.0)
        self.assertNotIn("caption", by_id["C1"].available_modalities)


if __name__ == "__main__":
    unittest.main()
