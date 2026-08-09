from __future__ import annotations

import copy
import random
import unittest

import numpy as np

from backend.app.services.indexing.keyframe_feature_adapter import FeatureAdapterConfig
from backend.app.services.indexing.keyframe_multimodal_pipeline import (
    KEYFRAME_STRATEGY_MULTIMODAL_COVERAGE,
    MultimodalKeyframePipelineError,
    run_multimodal_keyframe_pipeline,
)
from backend.app.services.indexing.keyframe_selection import (
    PHASE_MMR,
    SelectionConfig,
)


VIDEO_ID = "L01_V001"


def candidate_records() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for index in range(8):
        shot_index = 0 if index < 4 else 1
        if index in {0, 4}:
            reason = "shot_boundary_start"
        elif index in {3, 7}:
            reason = "shot_boundary_end"
        else:
            reason = "dense_interval"
        values.append(
            {
                "candidate_id": f"C{index}",
                "frame_id": f"F{index}",
                "video_id": VIDEO_ID,
                "shot_id": f"SHOT_{VIDEO_ID}_{shot_index:06d}",
                "segment_id": f"SHOT_{VIDEO_ID}_{shot_index:06d}",
                "shot_index": shot_index,
                "frame_index": index * 15,
                "timestamp": index * 0.5,
                "candidate_index": index,
                "candidate_reasons": [reason],
                "keyframe_path": f"data/candidates/{VIDEO_ID}/F{index}.jpg",
                "keyframe_strategy": "dense_coverage",
                "selection_phase": "coverage_fill",
                "selection_rank": index + 1,
            }
        )
    return values


def embedding_artifacts() -> tuple[np.ndarray, list[dict[str, object]]]:
    matrix = np.asarray(
        [[1.0, 0.0]] * 4 + [[-1.0, 0.0]] * 4,
        dtype=np.float32,
    )
    records = [
        {
            "embedding_id": f"EMB_C{index}",
            "candidate_id": f"C{index}",
            "frame_id": f"F{index}",
            "video_id": VIDEO_ID,
            "embedding_index": index,
            "vector_dim": 2,
            "normalized": True,
            "model_family": "siglip2",
            "model_name": "fake/siglip2",
            "model_revision": "revision-1",
            "output_dtype": "float32",
        }
        for index in range(8)
    ]
    return matrix, records


def ocr_records() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for index in range(8):
        text = "Rare title" if index in {1, 2} else ""
        values.append(
            {
                "candidate_id": f"C{index}",
                "frame_id": f"F{index}",
                "video_id": VIDEO_ID,
                "status": "success",
                "image_size": [100, 100],
                "ocr_text": text,
                "text_regions": (
                    [
                        {
                            "text": text,
                            "confidence": 0.90,
                            "polygon": [[10, 10], [80, 10], [80, 35], [10, 35]],
                        }
                    ]
                    if text
                    else []
                ),
            }
        )
    return values


def object_records() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for index in range(8):
        detections: list[dict[str, object]] = []
        if index in {5, 6}:
            detections.append(
                {
                    "class_id": 1,
                    "class_name": "bicycle",
                    "confidence": 0.85,
                    "bbox_xyxy": [10, 10, 70, 70],
                }
            )
        values.append(
            {
                "candidate_id": f"C{index}",
                "frame_id": f"F{index}",
                "video_id": VIDEO_ID,
                "status": "success",
                "image_size": [100, 100],
                "objects": detections,
            }
        )
    return values


def caption_records() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "C1",
            "frame_id": "F1",
            "video_id": VIDEO_ID,
            "status": "success",
            "caption": "a title card",
        },
        {
            "candidate_id": "C6",
            "frame_id": "F6",
            "video_id": VIDEO_ID,
            "status": "success",
            "caption": "a bicycle",
        },
    ]


def asr_records() -> list[dict[str, object]]:
    return [
        {
            "video_id": VIDEO_ID,
            "status": "success",
            "start": 0.0,
            "end": 2.0,
            "text": "opening title",
            "confidence": 0.9,
        },
        {
            "video_id": VIDEO_ID,
            "status": "success",
            "start": 2.0,
            "end": 4.0,
            "text": "a bicycle appears",
            "confidence": 0.9,
        },
    ]


def adapter_config() -> FeatureAdapterConfig:
    return FeatureAdapterConfig(
        ocr_common_frame_fraction=1.0,
        ocr_common_shot_fraction=1.0,
        transition_absolute_floor=0.4,
        transition_mad_multiplier=0.0,
        transition_nms_seconds=0.0,
    )


def run_pipeline(
    *,
    candidates: list[dict[str, object]] | None = None,
    embeddings: np.ndarray | None = None,
    embedding_records: list[dict[str, object]] | None = None,
    ocr: list[dict[str, object]] | None = None,
    objects: list[dict[str, object]] | None = None,
    captions: list[dict[str, object]] | None = None,
    asr: list[dict[str, object]] | None = None,
    config: SelectionConfig | None = None,
    allow_partial_features: bool = False,
):
    default_embeddings, default_embedding_records = embedding_artifacts()
    return run_multimodal_keyframe_pipeline(
        candidates if candidates is not None else candidate_records(),
        embeddings=embeddings if embeddings is not None else default_embeddings,
        embedding_records=(
            embedding_records
            if embedding_records is not None
            else default_embedding_records
        ),
        ocr_records=ocr if ocr is not None else ocr_records(),
        object_records=objects if objects is not None else object_records(),
        caption_records=captions if captions is not None else caption_records(),
        asr_records=asr if asr is not None else asr_records(),
        video_duration=4.0,
        selection_config=config
        or SelectionConfig(max_gap_seconds=1.25, target_keyframes=None),
        adapter_config=adapter_config(),
        allow_partial_features=allow_partial_features,
    )


class MultimodalKeyframePipelineTest(unittest.TestCase):
    def test_protects_rare_events_transition_shots_and_all_temporal_gaps(self) -> None:
        result = run_pipeline()

        event_types = {record["event_type"] for record in result.event_ledger}
        self.assertIn("ocr_new", event_types)
        self.assertIn("object_new", event_types)
        self.assertIn("semantic_transition_pre", event_types)
        self.assertIn("semantic_transition_post", event_types)
        self.assertTrue(all(record["satisfied"] for record in result.event_ledger))

        guarantees = result.guarantee_report
        self.assertTrue(guarantees.constraints_satisfied)
        self.assertTrue(guarantees.event_recall_satisfied)
        self.assertTrue(guarantees.temporal_coverage_satisfied)
        self.assertTrue(guarantees.shot_coverage_satisfied)
        self.assertLessEqual(guarantees.observed_max_gap_seconds, 1.25)
        self.assertEqual(guarantees.missing_shot_indices, ())
        self.assertEqual(guarantees.missing_protected_event_ids, ())

        selected_event_ids = {
            event_id
            for record in result.final_records
            for event_id in record["protected_event_ids"]
        }
        detected_event_ids = {
            record["event_id"]
            for record in result.event_ledger
            if record["source"] == "feature_adapter"
        }
        self.assertTrue(detected_event_ids.issubset(selected_event_ids))
        self.assertTrue(
            all(
                record["keyframe_strategy"]
                == KEYFRAME_STRATEGY_MULTIMODAL_COVERAGE
                for record in result.final_records
            )
        )
        self.assertTrue(
            all(
                record["selection_phase"] != PHASE_MMR
                for record in result.final_records
            )
        )
        self.assertEqual(result.selection_result.soft_stop_reason, "target_not_configured")

    def test_subsets_embeddings_and_reindexes_all_frame_artifacts(self) -> None:
        source_matrix, _ = embedding_artifacts()
        result = run_pipeline()
        selected_ids = [record["candidate_id"] for record in result.final_records]
        source_row = {f"C{index}": index for index in range(8)}
        expected = source_matrix[[source_row[candidate_id] for candidate_id in selected_ids]]

        np.testing.assert_array_equal(result.final_embeddings, expected)
        self.assertEqual(
            [record["embedding_index"] for record in result.final_embedding_records],
            list(range(len(selected_ids))),
        )
        self.assertEqual(
            [record["candidate_id"] for record in result.final_embedding_records],
            selected_ids,
        )
        self.assertTrue(
            all(
                record["model_revision"] == "revision-1"
                for record in result.final_embedding_records
            )
        )
        self.assertEqual(
            [record["candidate_id"] for record in result.final_ocr_records],
            selected_ids,
        )
        self.assertEqual(
            [record["candidate_id"] for record in result.final_object_records],
            selected_ids,
        )
        self.assertEqual(
            [record["candidate_id"] for record in result.final_caption_records],
            [candidate_id for candidate_id in selected_ids if candidate_id in {"C1", "C6"}],
        )
        self.assertEqual(len(result.final_asr_records), 2)
        for record in result.final_records:
            provenance = record["selection_provenance"]
            self.assertEqual(provenance["selection_rank"], record["selection_rank"])
            self.assertEqual(provenance["covered_event_ids"], record["covered_event_ids"])
            self.assertEqual(provenance["component_scores"], record["component_scores"])
            self.assertEqual(
                provenance["semantic_novelty"],
                record["semantic_novelty"],
            )
            self.assertGreaterEqual(record["semantic_novelty"], 0.0)
            self.assertLessEqual(record["semantic_novelty"], 1.0)
        self.assertEqual(
            [record["semantic_novelty"] for record in result.final_embedding_records],
            [record["semantic_novelty"] for record in result.final_records],
        )

    def test_default_feature_completeness_is_fail_closed_but_can_be_opted_out(self) -> None:
        missing_ocr = ocr_records()[:-1]
        with self.assertRaisesRegex(
            MultimodalKeyframePipelineError,
            "OCR features are incomplete",
        ):
            run_pipeline(ocr=missing_ocr)

        failed_objects = object_records()
        failed_objects[0]["status"] = "error"
        with self.assertRaisesRegex(
            MultimodalKeyframePipelineError,
            "object features are incomplete",
        ):
            run_pipeline(objects=failed_objects)

        partial = run_pipeline(
            ocr=missing_ocr,
            objects=failed_objects,
            allow_partial_features=True,
        )
        self.assertTrue(partial.guarantee_report.constraints_satisfied)
        by_id = {
            record["candidate_id"]: record for record in partial.candidate_ledger
        }
        self.assertNotIn("ocr", by_id["C7"]["available_modalities"])
        self.assertNotIn("objects", by_id["C0"]["available_modalities"])

    def test_incomplete_embedding_alignment_and_hard_cap_produce_no_result(self) -> None:
        matrix, metadata = embedding_artifacts()
        with self.assertRaisesRegex(ValueError, "metadata count"):
            run_pipeline(
                embeddings=matrix,
                embedding_records=metadata[:-1],
            )

        with self.assertRaisesRegex(
            MultimodalKeyframePipelineError,
            "selector hard constraints are unsatisfied",
        ):
            run_pipeline(
                config=SelectionConfig(
                    max_gap_seconds=10.0,
                    target_keyframes=None,
                    hard_max_keyframes=1,
                    protect_each_shot=False,
                )
            )

    def test_output_is_deterministic_when_artifact_records_are_shuffled(self) -> None:
        first = run_pipeline()
        candidates = list(reversed(candidate_records()))
        ocr = list(reversed(ocr_records()))
        objects = list(reversed(object_records()))
        captions = list(reversed(caption_records()))
        asr = list(reversed(asr_records()))
        matrix, metadata = embedding_artifacts()
        order = list(range(len(metadata)))
        random.Random(42).shuffle(order)
        shuffled_matrix = np.ascontiguousarray(matrix[order])
        shuffled_metadata: list[dict[str, object]] = []
        for new_index, old_index in enumerate(order):
            record = copy.deepcopy(metadata[old_index])
            record["embedding_index"] = new_index
            shuffled_metadata.append(record)

        second = run_pipeline(
            candidates=candidates,
            embeddings=shuffled_matrix,
            embedding_records=shuffled_metadata,
            ocr=ocr,
            objects=objects,
            captions=captions,
            asr=asr,
        )

        self.assertEqual(first.final_records, second.final_records)
        self.assertEqual(first.final_embedding_records, second.final_embedding_records)
        self.assertEqual(first.final_ocr_records, second.final_ocr_records)
        self.assertEqual(first.final_object_records, second.final_object_records)
        self.assertEqual(first.final_caption_records, second.final_caption_records)
        self.assertEqual(first.final_asr_records, second.final_asr_records)
        self.assertEqual(first.candidate_ledger, second.candidate_ledger)
        self.assertEqual(first.event_ledger, second.event_ledger)
        self.assertEqual(first.guarantee_report, second.guarantee_report)
        np.testing.assert_array_equal(first.final_embeddings, second.final_embeddings)


if __name__ == "__main__":
    unittest.main()
