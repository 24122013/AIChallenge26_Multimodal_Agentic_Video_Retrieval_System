from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from competition.keyframe_phase3 import (
    PHASE3_CANDIDATE_CONTRACT_VERSION,
    atomic_save_npy,
    atomic_write_json,
    atomic_write_jsonl,
    candidate_run_contract,
    frame_ids_sha256,
    images_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    workspace_paths,
)
from competition.pipeline import (
    ARTIFACT_TAG,
    CorpusVideo,
    _load_valid_extraction_report,
    _phase3_candidate_config,
    _phase3_feature_config,
    _phase3_frame_modality_artifacts_current,
    _phase3_load_and_validate_features,
    _phase3_publish_video,
    _phase3_release_model_memory,
    _phase3_siglip_artifacts_current,
    _phase3_selection_config,
    _require_current_canonical_publish,
    _require_embedding_matches_extraction,
    build_parser,
)


VIDEO_ID = "video001"
RESOLVED_DEVICE = "cpu"
RESOLVED_MODEL_REVISION = "test-revision-1"


class Phase3PipelineIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.public_root = self.root / "public"
        self.output_root = self.root / "competition"
        self.video_relative_path = Path("videos") / f"{VIDEO_ID}.mp4"
        self.video_path = self.public_root / self.video_relative_path
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        # Publication only fingerprints the source in this focused test. Model
        # inference and video decoding belong to the preceding Phase 3 stages.
        self.video_path.write_bytes(b"synthetic-video-source")
        self.video = CorpusVideo(
            filename=f"{VIDEO_ID}.mp4",
            relative_path=self.video_relative_path,
            fps=2.0,
            frame_count=10,
        )
        self.args = build_parser().parse_args(
            [
                "keyframes",
                "--public-root",
                str(self.public_root),
                "--output-root",
                str(self.output_root),
                "--device",
                "cpu",
                "--max-gap-seconds",
                "1.25",
            ]
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _candidate_records(self, paths) -> list[dict[str, object]]:
        paths.candidate_images_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        for index in range(10):
            shot_index = 0 if index < 5 else 1
            if index in {0, 5}:
                reason = "shot_boundary_start"
            elif index in {4, 9}:
                reason = "shot_boundary_end"
            else:
                reason = "dense_interval"
            frame_id = f"FRAME_{VIDEO_ID}_{index:09d}"
            candidate_id = f"CAND_{VIDEO_ID}_{index:06d}"
            image_path = paths.candidate_images_dir / f"{frame_id}.jpg"
            image = np.full(
                (32, 32, 3),
                (index * 17) % 255,
                dtype=np.uint8,
            )
            self.assertTrue(cv2.imwrite(str(image_path), image))
            shot_start = 0.0 if shot_index == 0 else 2.5
            shot_end = 2.5 if shot_index == 0 else 5.0
            records.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                    "frame_id": frame_id,
                    "video_id": VIDEO_ID,
                    "shot_id": f"SHOT_{VIDEO_ID}_{shot_index:06d}",
                    "segment_id": f"SHOT_{VIDEO_ID}_{shot_index:06d}",
                    "shot_index": shot_index,
                    "shot_start": shot_start,
                    "shot_end": shot_end,
                    "frame_index": index,
                    "timestamp": index * 0.5,
                    "timestamp_source": "video_fps",
                    "timestamp_confidence": 1.0,
                    "candidate_reasons": [reason],
                    "selection_reason": reason,
                    "keyframe_path": image_path.as_posix(),
                    "thumbnail_path": image_path.as_posix(),
                    "source_video_path": self.video_path.as_posix(),
                    "video_path": self.video_path.as_posix(),
                    "keyframe_strategy": "dense_coverage",
                    "artifact_role": "dense_candidate",
                    "candidate_pool_run_id": paths.run_id,
                }
            )
        return records

    def _write_phase3_fixture(self):
        candidate_config = _phase3_candidate_config(self.args)
        contract = candidate_run_contract(
            video_path=self.video_path,
            video_id=VIDEO_ID,
            frame_count=self.video.frame_count,
            config=candidate_config,
        )
        paths = workspace_paths(self.output_root, VIDEO_ID, contract)
        candidates = self._candidate_records(paths)
        atomic_write_jsonl(paths.candidate_metadata, candidates)
        candidate_report = {
            "video_id": VIDEO_ID,
            "phase3_status": "passed",
            "phase3_candidate_contract_version": PHASE3_CANDIDATE_CONTRACT_VERSION,
            "phase3_candidate_contract": contract,
            "planned_candidate_count": len(candidates),
            "materialized_candidate_count": len(candidates),
            "candidate_metadata_sha256": sha256_file(paths.candidate_metadata),
            "candidate_frame_ids_sha256": frame_ids_sha256(candidates),
            "candidate_images_sha256": images_sha256(candidates),
            "shot_detector": "synthetic",
            "shot_count": 2,
            "frame_extractor": "synthetic",
        }
        atomic_write_json(paths.candidate_report, candidate_report)

        # A single sharp cosine boundary between the two shots is sufficient
        # to produce the protected pre/post semantic-transition pair.
        embeddings = np.asarray(
            [[1.0, 0.0]] * 5 + [[-1.0, 0.0]] * 5,
            dtype=np.float32,
        )
        embedding_records: list[dict[str, object]] = []
        caption_records: list[dict[str, object]] = []
        ocr_records: list[dict[str, object]] = []
        object_records: list[dict[str, object]] = []
        for index, candidate in enumerate(candidates):
            common = {
                "candidate_id": candidate["candidate_id"],
                "frame_id": candidate["frame_id"],
                "video_id": VIDEO_ID,
                "keyframe_path": candidate["keyframe_path"],
                "thumbnail_path": candidate["thumbnail_path"],
            }
            embedding_records.append(
                {
                    **common,
                    "embedding_id": f"EMB_{candidate['frame_id']}",
                    "embedding_index": index,
                    "vector_dim": 2,
                    "normalized": True,
                    "model_family": "siglip2",
                    "model_name": self.args.model_name,
                    "model_revision": RESOLVED_MODEL_REVISION,
                    "output_dtype": "float32",
                }
            )
            caption_records.append(
                {
                    **common,
                    "status": "success",
                    "caption": (
                        "opening title card" if index < 5 else "a bicycle outdoors"
                    ),
                }
            )
            ocr_records.append(
                {
                    **common,
                    "status": "success",
                    "image_size": [32, 32],
                    "ocr_text": "UNIQUE TITLE" if index == 1 else "",
                    "text_regions": (
                        [
                            {
                                "text": "UNIQUE TITLE",
                                "confidence": 0.99,
                                "polygon": [
                                    [2, 2],
                                    [24, 2],
                                    [24, 12],
                                    [2, 12],
                                ],
                            }
                        ]
                        if index == 1
                        else []
                    ),
                }
            )
            object_records.append(
                {
                    **common,
                    "status": "success",
                    "image_size": [32, 32],
                    "objects": (
                        [
                            {
                                "class_id": 1,
                                "class_name": "bicycle",
                                "confidence": 0.95,
                                "bbox_xyxy": [4, 4, 24, 24],
                            }
                        ]
                        if index == 7
                        else []
                    ),
                }
            )
        feature_config = _phase3_feature_config(
            self.args,
            resolved_device=RESOLVED_DEVICE,
            resolved_model_revision=RESOLVED_MODEL_REVISION,
        )
        atomic_save_npy(paths.embeddings, embeddings)
        atomic_write_jsonl(paths.embedding_metadata, embedding_records)
        atomic_write_jsonl(paths.captions, caption_records)
        atomic_write_jsonl(paths.ocr, ocr_records)
        atomic_write_jsonl(paths.objects, object_records)
        atomic_write_json(
            paths.caption_report,
            {
                "pipeline": "caption",
                "input_record_count": len(candidates),
                "error_count": 0,
                "model_name": self.args.caption_model_name,
                "requested_model_revision": self.args.caption_model_revision,
                "device": RESOLVED_DEVICE,
                "batch_size": self.args.caption_batch_size,
                "max_new_tokens": self.args.caption_max_new_tokens,
                "dtype": self.args.caption_dtype,
                "quantization": self.args.caption_quantization,
                "segment_caption_enabled": not self.args.no_segment_caption,
                "input_path": str(paths.candidate_metadata),
                "output_path": str(paths.captions),
            },
        )
        features = _phase3_load_and_validate_features(
            video=self.video,
            paths=paths,
            candidate_report=candidate_report,
            candidate_records=candidates,
            feature_config=feature_config,
            allow_partial_features=False,
            require_manifest=False,
        )
        atomic_write_json(paths.feature_manifest, features["manifest"])
        features = _phase3_load_and_validate_features(
            video=self.video,
            paths=paths,
            candidate_report=candidate_report,
            candidate_records=candidates,
            feature_config=feature_config,
            allow_partial_features=False,
            require_manifest=True,
        )
        return (
            paths,
            candidate_report,
            candidates,
            embeddings,
            caption_records,
            feature_config,
            features,
        )

    def test_keyframes_defaults_stop_at_hard_coverage_and_require_hard_features(self) -> None:
        args = build_parser().parse_args(["keyframes"])

        self.assertEqual(args.caption_model_name, "Qwen/Qwen3.5-4B")
        self.assertEqual(
            args.caption_model_revision,
            "c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97",
        )
        self.assertIsNone(args.target_keyframes)
        self.assertIsNone(args.hard_max_keyframes)
        self.assertFalse(args.allow_partial_features)
        selection = _phase3_selection_config(args)
        self.assertIsNone(selection.target_keyframes)
        self.assertIsNone(selection.hard_max_keyframes)
        self.assertTrue(selection.protect_each_shot)

    def test_model_cleanup_moves_and_drops_heavy_private_references(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.devices: list[str] = []

            def to(self, device: str) -> "FakeModel":
                self.devices.append(device)
                return self

        class FakeBackend:
            def __init__(self) -> None:
                self._model = FakeModel()
                self._processor = object()

        direct = FakeModel()
        backend = FakeBackend()
        nested = backend._model

        _phase3_release_model_memory(direct, backend)

        self.assertEqual(direct.devices, ["cpu"])
        self.assertEqual(nested.devices, ["cpu"])
        self.assertIsNone(backend._model)
        self.assertIsNone(backend._processor)

    def test_modality_checkpoints_require_exact_reports(self) -> None:
        (
            paths,
            _candidate_report,
            candidates,
            _embeddings,
            _caption_records,
            feature_config,
            _features,
        ) = self._write_phase3_fixture()

        self.assertTrue(
            _phase3_frame_modality_artifacts_current(
                paths=paths,
                candidate_records=candidates,
                output_path=paths.captions,
                report_path=paths.caption_report,
                pipeline="caption",
                expected_report={
                    "model_name": self.args.caption_model_name,
                    "requested_model_revision": self.args.caption_model_revision,
                    "device": RESOLVED_DEVICE,
                    "batch_size": self.args.caption_batch_size,
                    "max_new_tokens": self.args.caption_max_new_tokens,
                    "dtype": self.args.caption_dtype,
                    "quantization": self.args.caption_quantization,
                    "segment_caption_enabled": not self.args.no_segment_caption,
                },
                require_success=False,
            )
        )

    def test_siglip_checkpoint_resume_requires_exact_config_and_candidates(self) -> None:
        (
            paths,
            _candidate_report,
            candidates,
            embeddings,
            _caption_records,
            feature_config,
            _features,
        ) = self._write_phase3_fixture()
        atomic_write_jsonl(paths.embedding_skipped, [])
        atomic_write_json(
            paths.embedding_benchmark,
            {
                "model_family": "siglip2",
                "model_name": self.args.model_name,
                "model_revision": RESOLVED_MODEL_REVISION,
                "device": RESOLVED_DEVICE,
                "compute_dtype": "float32",
                "output_dtype": "float32",
                "normalized": True,
                "requested_batch_size": self.args.batch_size,
                "num_workers": self.args.num_workers,
                "prefetch_factor": self.args.prefetch_factor,
                "input_record_count": len(candidates),
                "encoded_count": len(candidates),
                "skipped_count": 0,
                "embedding_shape": list(embeddings.shape),
            },
        )

        self.assertTrue(
            _phase3_siglip_artifacts_current(
                paths=paths,
                candidate_records=candidates,
                feature_config=feature_config,
            )
        )
        changed_config = copy.deepcopy(feature_config)
        changed_config["siglip2"]["model_name"] = "different-model"
        self.assertFalse(
            _phase3_siglip_artifacts_current(
                paths=paths,
                candidate_records=candidates,
                feature_config=changed_config,
            )
        )

    def test_feature_loader_binds_exact_manifest_and_rejects_tampering(self) -> None:
        (
            paths,
            candidate_report,
            candidates,
            _embeddings,
            caption_records,
            feature_config,
            features,
        ) = self._write_phase3_fixture()

        self.assertEqual(read_json(paths.feature_manifest), features["manifest"])
        self.assertTrue(features["manifest"]["hard_feature_complete"])
        self.assertEqual(features["manifest"]["status"], "passed")

        tampered_manifest = copy.deepcopy(features["manifest"])
        tampered_manifest["artifacts"]["embeddings"]["sha256"] = "0" * 64
        atomic_write_json(paths.feature_manifest, tampered_manifest)
        with self.assertRaisesRegex(RuntimeError, "stale or was tampered"):
            _phase3_load_and_validate_features(
                video=self.video,
                paths=paths,
                candidate_report=candidate_report,
                candidate_records=candidates,
                feature_config=feature_config,
                allow_partial_features=False,
                require_manifest=True,
            )

        atomic_write_json(paths.feature_manifest, features["manifest"])
        changed_captions = copy.deepcopy(caption_records)
        changed_captions[0]["caption"] = "artifact content was changed"
        atomic_write_jsonl(paths.captions, changed_captions)
        with self.assertRaisesRegex(RuntimeError, "stale or was tampered"):
            _phase3_load_and_validate_features(
                video=self.video,
                paths=paths,
                candidate_report=candidate_report,
                candidate_records=candidates,
                feature_config=feature_config,
                allow_partial_features=False,
                require_manifest=True,
            )

    def test_publish_creates_valid_canonical_subset_and_commit_marker(self) -> None:
        (
            paths,
            candidate_report,
            candidates,
            dense_embeddings,
            _caption_records,
            feature_config,
            features,
        ) = self._write_phase3_fixture()

        manifest = _phase3_publish_video(
            self.args,
            video=self.video,
            paths=paths,
            candidate_report=candidate_report,
            candidate_records=candidates,
            features=features,
            feature_config=feature_config,
            resolved_device=RESOLVED_DEVICE,
            resolved_model_revision=RESOLVED_MODEL_REVISION,
        )

        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["candidate_count"], len(candidates))
        self.assertLess(manifest["selected_count"], len(candidates))
        guarantees = manifest["guarantees"]
        self.assertTrue(guarantees["constraints_satisfied"])
        self.assertTrue(guarantees["event_recall_satisfied"])
        self.assertTrue(guarantees["shot_coverage_satisfied"])
        self.assertTrue(guarantees["temporal_coverage_satisfied"])
        self.assertEqual(guarantees["missing_protected_event_ids"], [])
        self.assertEqual(guarantees["missing_shot_indices"], [])
        self.assertLessEqual(
            guarantees["observed_max_gap_seconds"],
            self.args.max_gap_seconds,
        )

        metadata_path = (
            self.output_root / "metadata" / f"keyframes_{VIDEO_ID}.jsonl"
        )
        final_records = read_jsonl(metadata_path)
        selected_ids = [record["candidate_id"] for record in final_records]
        self.assertEqual(selected_ids, manifest["selected_candidate_ids"])
        self.assertTrue(
            all(
                record["keyframe_strategy"] == "multimodal_coverage"
                and record["artifact_role"] == "selected_keyframe"
                and record["phase3_selection_run_id"] == manifest["selection_run_id"]
                for record in final_records
            )
        )
        self.assertTrue(
            all(Path(record["keyframe_path"]).is_file() for record in final_records)
        )

        embeddings_path = (
            self.output_root / "embeddings" / f"{ARTIFACT_TAG}_{VIDEO_ID}.npy"
        )
        embedding_metadata_path = (
            self.output_root
            / "metadata"
            / f"{ARTIFACT_TAG}_embeddings_{VIDEO_ID}.jsonl"
        )
        final_embeddings = np.load(embeddings_path, allow_pickle=False)
        final_embedding_records = read_jsonl(embedding_metadata_path)
        source_row = {
            candidate["candidate_id"]: index
            for index, candidate in enumerate(candidates)
        }
        np.testing.assert_array_equal(
            final_embeddings,
            dense_embeddings[[source_row[candidate_id] for candidate_id in selected_ids]],
        )
        self.assertEqual(
            [record["embedding_index"] for record in final_embedding_records],
            list(range(len(final_records))),
        )
        self.assertEqual(
            [record["candidate_id"] for record in final_embedding_records],
            selected_ids,
        )

        event_records = read_jsonl(paths.protected_events)
        event_types = {record["event_type"] for record in event_records}
        self.assertTrue(
            {
                "ocr_new",
                "object_new",
                "semantic_transition_pre",
                "semantic_transition_post",
            }.issubset(event_types)
        )
        self.assertTrue(all(record["satisfied"] for record in event_records))

        # The extraction report is intentionally the final atomic commit marker.
        extract_report_path = (
            self.output_root
            / "metadata"
            / f"keyframes_{VIDEO_ID}_extract_report.json"
        )
        extract_report = read_json(extract_report_path)
        phase3_manifest_path = (
            self.output_root
            / "metadata"
            / f"keyframes_{VIDEO_ID}_phase3_manifest.json"
        )
        self.assertEqual(
            extract_report["phase3_manifest_path"],
            phase3_manifest_path.as_posix(),
        )
        self.assertEqual(
            extract_report["phase3_manifest_sha256"],
            sha256_file(phase3_manifest_path),
        )
        self.assertEqual(read_json(paths.selection_report), manifest)

        validated_extract_report = _load_valid_extraction_report(
            video=self.video,
            video_path=self.video_path,
            metadata_path=metadata_path,
            report_path=extract_report_path,
        )
        _require_embedding_matches_extraction(
            video=self.video,
            extract_report=validated_extract_report,
            embeddings_path=embeddings_path,
            embedding_metadata_path=embedding_metadata_path,
            artifact_report_path=(
                self.output_root
                / "metadata"
                / f"{ARTIFACT_TAG}_artifacts_{VIDEO_ID}_validation.json"
            ),
        )
        _require_current_canonical_publish(
            [self.video],
            public_root=self.public_root,
            output_root=self.output_root,
        )

        # A per-file atomic replace is not a set-wide transaction.  The report
        # marker + manifest checksum guard must stop downstream consumers from
        # reading a mixed set after any canonical artifact changes.
        caption_path = self.output_root / "metadata" / f"captions_{VIDEO_ID}.jsonl"
        captions = read_jsonl(caption_path)
        captions[0]["caption"] = "tampered after commit"
        atomic_write_jsonl(caption_path, captions)
        with self.assertRaisesRegex(RuntimeError, "Canonical Phase 3 artifact is stale"):
            _require_current_canonical_publish(
                [self.video],
                public_root=self.public_root,
                output_root=self.output_root,
            )


if __name__ == "__main__":
    unittest.main()
