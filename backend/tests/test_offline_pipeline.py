from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.app.pipelines import offline_pipeline as pipeline
from backend.app.services.indexing.build_bge_m3_index import (
    load_canonical_keyframe_records,
)
from backend.app.services.indexing.extract_keyframes import Shot, VideoInfo
from backend.app.services.indexing.keyframe_candidates import KeyframeCandidate


class OfflinePipelineOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = pipeline.OfflinePipelineConfig(
            output_dir=self.root / "data",
            device="cpu",
            shot_device="cpu",
            bge_enabled=False,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _video(self, video_id: str) -> Path:
        path = self.root / "videos" / f"{video_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unit-test-video")
        return path

    def _shot_stage(self, video_id: str, count: int) -> pipeline.ShotStageResult:
        frame_count = max(25, count * 13)
        info = VideoInfo(video_id=video_id, fps=25.0, frame_count=frame_count)
        shots = (
            Shot(
                shot_index=0,
                start_frame=0,
                end_frame=frame_count - 1,
                fps=info.fps,
            ),
        )
        return pipeline.ShotStageResult(
            info=info,
            shots=shots,
            detector_name="fake-shot-detector",
            contract_sha256="shots-contract",
        )

    def _candidate_stage(
        self,
        video_id: str,
        count: int,
    ) -> pipeline.CandidateStageResult:
        candidates = tuple(
            KeyframeCandidate(
                candidate_id=f"{video_id}:C{index:04d}",
                video_id=video_id,
                shot_index=0,
                frame_index=index * 13,
                timestamp_sec=index * 0.5,
                shot_start_sec=0.0,
                shot_end_sec=max(0.5, count * 0.5),
                reasons=("dense_interval",),
            )
            for index in range(count)
        )
        return pipeline.CandidateStageResult(
            candidates=candidates,
            contract_sha256="candidates-contract",
        )

    def _dense_records(self, video_id: str, count: int) -> tuple[dict, ...]:
        return tuple(
            {
                "candidate_id": f"{video_id}:C{index:04d}",
                "frame_id": f"{video_id}:F{index:04d}",
                "video_id": video_id,
                "shot_id": f"SHOT_{video_id}_000000",
                "segment_id": f"SHOT_{video_id}_000000",
                "shot_index": 0,
                "frame_index": index * 13,
                "timestamp": index * 0.5,
                "candidate_reasons": ["dense_interval"],
                "keyframe_path": f"dense/{video_id}/F{index:04d}.jpg",
                "artifact_role": "dense_candidate",
            }
            for index in range(count)
        )

    def _materialized_stage(
        self,
        video_id: str,
        count: int,
    ) -> pipeline.MaterializedStageResult:
        return pipeline.MaterializedStageResult(
            records=self._dense_records(video_id, count),
            report={"candidate_count": count},
            contract_sha256="materialization-contract",
        )

    def _feature_bundle(
        self,
        video_id: str,
        count: int,
    ) -> pipeline.DenseFeatureArtifacts:
        dense = self._dense_records(video_id, count)
        embedding_records = tuple(
            {
                "candidate_id": record["candidate_id"],
                "frame_id": record["frame_id"],
                "video_id": video_id,
                "embedding_index": index,
                "vector_dim": 2,
                "normalized": True,
            }
            for index, record in enumerate(dense)
        )
        captions = tuple(
            {
                "candidate_id": record["candidate_id"],
                "frame_id": record["frame_id"],
                "video_id": video_id,
                "status": "success",
                "caption": f"caption {index}",
            }
            for index, record in enumerate(dense)
        )
        ocr = tuple(
            {
                "candidate_id": record["candidate_id"],
                "frame_id": record["frame_id"],
                "video_id": video_id,
                "status": "success",
                "ocr_text": "",
            }
            for record in dense
        )
        objects = tuple(
            {
                "candidate_id": record["candidate_id"],
                "frame_id": record["frame_id"],
                "video_id": video_id,
                "status": "success",
                "objects": [],
            }
            for record in dense
        )
        embeddings = np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            (count, 1),
        )
        return pipeline.DenseFeatureArtifacts(
            embeddings=embeddings,
            embedding_records=embedding_records,
            caption_records=captions,
            ocr_records=ocr,
            object_records=objects,
            contract_sha256="features-contract",
        )

    def _video_artifacts(
        self,
        video_path: Path,
        *,
        selected_count: int = 1,
        dense_count: int = 3,
        skipped: bool = False,
    ) -> pipeline.VideoArtifacts:
        paths = pipeline.PerVideoPaths.from_config(video_path.stem, self.config)
        return pipeline.VideoArtifacts(
            video_id=video_path.stem,
            video_path=video_path,
            paths=paths,
            selected_count=selected_count,
            dense_candidate_count=dense_count,
            skipped=skipped,
            validation={"status": "passed"},
        )

    @staticmethod
    def _recording_stage(events: list[str], name: str, value):
        def run(*_args, **_kwargs):
            events.append(name)
            return value

        return run

    def test_process_video_runs_canonical_stages_in_exact_order(self) -> None:
        video = self._video("video_A")
        shots = self._shot_stage(video.stem, 3)
        candidates = self._candidate_stage(video.stem, 3)
        materialized = self._materialized_stage(video.stem, 3)
        features = self._feature_bundle(video.stem, 3)
        selection = SimpleNamespace()
        artifacts = self._video_artifacts(video)
        events: list[str] = []

        with (
            patch.object(pipeline, "_try_load_complete_video", return_value=None),
            patch.object(
                pipeline,
                "_load_or_run_shot_detection",
                side_effect=self._recording_stage(events, "shots", shots),
            ),
            patch.object(
                pipeline,
                "_load_or_run_dense_candidate_generation",
                side_effect=self._recording_stage(events, "candidates", candidates),
            ),
            patch.object(
                pipeline,
                "_load_or_run_dense_materialization",
                side_effect=self._recording_stage(events, "materialization", materialized),
            ),
            patch.object(
                pipeline,
                "_extract_all_dense_features",
                side_effect=self._recording_stage(events, "features", features),
            ),
            patch.object(
                pipeline,
                "_run_multimodal_selection",
                side_effect=self._recording_stage(
                    events,
                    "selection",
                    (selection, {"contract_sha256": "selection-contract"}),
                ),
            ),
            patch.object(
                pipeline,
                "_persist_selected_artifacts",
                side_effect=self._recording_stage(events, "persistence", None),
            ),
            patch.object(
                pipeline,
                "_validate_and_commit_video",
                side_effect=self._recording_stage(events, "validation", artifacts),
            ),
            patch.object(pipeline, "_release_accelerator_memory"),
        ):
            result = pipeline.process_video(video, self.config)

        self.assertIs(result, artifacts)
        self.assertEqual(
            events,
            [
                "shots",
                "candidates",
                "materialization",
                "features",
                "selection",
                "persistence",
                "validation",
            ],
        )

    def test_feature_extraction_receives_the_entire_dense_pool(self) -> None:
        video = self._video("video_dense")
        count = 100
        shots = self._shot_stage(video.stem, count)
        candidates = self._candidate_stage(video.stem, count)
        materialized = self._materialized_stage(video.stem, count)
        features = self._feature_bundle(video.stem, count)
        artifacts = self._video_artifacts(video, dense_count=count)
        received_counts: list[int] = []

        def extract(_video_path, dense_pool, _config, _paths):
            received_counts.append(len(dense_pool.records))
            self.assertIs(dense_pool, materialized)
            self.assertEqual(
                [record["candidate_id"] for record in dense_pool.records],
                [candidate.candidate_id for candidate in candidates.candidates],
            )
            return features

        with (
            patch.object(pipeline, "_try_load_complete_video", return_value=None),
            patch.object(pipeline, "_load_or_run_shot_detection", return_value=shots),
            patch.object(
                pipeline,
                "_load_or_run_dense_candidate_generation",
                return_value=candidates,
            ),
            patch.object(
                pipeline,
                "_load_or_run_dense_materialization",
                return_value=materialized,
            ),
            patch.object(pipeline, "_extract_all_dense_features", side_effect=extract),
            patch.object(
                pipeline,
                "_run_multimodal_selection",
                return_value=(SimpleNamespace(), {"contract_sha256": "selection"}),
            ),
            patch.object(pipeline, "_persist_selected_artifacts"),
            patch.object(
                pipeline,
                "_validate_and_commit_video",
                return_value=artifacts,
            ),
            patch.object(pipeline, "_release_accelerator_memory"),
        ):
            pipeline.process_video(video, self.config)

        self.assertEqual(received_counts, [100])

    def test_selector_receives_full_aligned_dense_multimodal_pool(self) -> None:
        video_id = "video_selector"
        count = 100
        shots = self._shot_stage(video_id, count)
        materialized = self._materialized_stage(video_id, count)
        features = self._feature_bundle(video_id, count)
        selected = SimpleNamespace(
            final_records=(materialized.records[0],),
            guarantee_report=SimpleNamespace(constraints_satisfied=True),
            candidate_ledger=materialized.records,
        )

        with (
            patch.object(
                pipeline,
                "_selection_contract",
                return_value={"contract_sha256": "selection-contract"},
            ),
            patch.object(
                pipeline,
                "run_multimodal_keyframe_pipeline",
                return_value=selected,
            ) as selector,
        ):
            result, contract = pipeline._run_multimodal_selection(
                shots,
                materialized,
                features,
                self.config,
                pipeline.PerVideoPaths.from_config(video_id, self.config),
            )

        self.assertIs(result, selected)
        self.assertEqual(contract["contract_sha256"], "selection-contract")
        args, kwargs = selector.call_args
        self.assertIs(args[0], materialized.records)
        self.assertEqual(len(args[0]), 100)
        self.assertEqual(kwargs["embeddings"].shape[0], 100)
        expected_ids = [record["candidate_id"] for record in materialized.records]
        for key in (
            "embedding_records",
            "ocr_records",
            "object_records",
            "caption_records",
        ):
            self.assertEqual(
                [record["candidate_id"] for record in kwargs[key]],
                expected_ids,
            )
        self.assertFalse(kwargs["allow_partial_features"])

    def test_selector_rejects_a_missing_dense_feature_before_selection(self) -> None:
        video_id = "video_misaligned"
        materialized = self._materialized_stage(video_id, 4)
        features = self._feature_bundle(video_id, 4)
        misaligned = pipeline.DenseFeatureArtifacts(
            embeddings=features.embeddings,
            embedding_records=features.embedding_records,
            caption_records=features.caption_records,
            ocr_records=features.ocr_records[:-1],
            object_records=features.object_records,
            contract_sha256=features.contract_sha256,
        )

        with (
            patch.object(
                pipeline,
                "_selection_contract",
                return_value={"contract_sha256": "selection-contract"},
            ),
            patch.object(pipeline, "run_multimodal_keyframe_pipeline") as selector,
        ):
            with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                pipeline._run_multimodal_selection(
                    self._shot_stage(video_id, 4),
                    materialized,
                    misaligned,
                    self.config,
                    pipeline.PerVideoPaths.from_config(video_id, self.config),
                )

        selector.assert_not_called()

    def test_caption_stage_rejects_success_row_with_empty_caption(self) -> None:
        video = self._video("video_empty_caption")
        dense_image = self.root / "dense-empty-caption.jpg"
        dense_image.write_bytes(b"dense-image")
        dense_record = {
            **self._dense_records(video.stem, 1)[0],
            "keyframe_path": dense_image.as_posix(),
        }
        materialized = pipeline.MaterializedStageResult(
            records=(dense_record,),
            report={"candidate_count": 1},
            contract_sha256="materialization-contract",
        )
        paths = pipeline.PerVideoPaths.from_config(video.stem, self.config)
        paths.dense_metadata.parent.mkdir(parents=True, exist_ok=True)
        paths.dense_metadata.write_text(
            json.dumps(materialized.records[0]) + "\n",
            encoding="utf-8",
        )

        def write_empty_caption(*_args, output_path, report_path, **_kwargs):
            Path(output_path).write_text(
                json.dumps(
                    {
                        "candidate_id": materialized.records[0]["candidate_id"],
                        "frame_id": materialized.records[0]["frame_id"],
                        "video_id": video.stem,
                        "status": "success",
                        "caption": "   ",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            Path(report_path).write_text("{}\n", encoding="utf-8")
            return {"status": "passed"}

        with (
            patch.object(
                pipeline,
                "run_caption_file",
                side_effect=write_empty_caption,
            ) as caption_runner,
            patch.object(pipeline, "_release_accelerator_memory"),
        ):
            with self.assertRaisesRegex(ValueError, "caption"):
                pipeline._load_or_run_caption_features(
                    video,
                    materialized,
                    self.config,
                    paths,
                )

        caption_runner.assert_called_once()
        self.assertEqual(
            caption_runner.call_args.kwargs["model_name"],
            "florence-community/Florence-2-base-ft",
        )
        self.assertEqual(
            caption_runner.call_args.kwargs["task_prompt"],
            "<MORE_DETAILED_CAPTION>",
        )

    def test_ocr_worker_isolates_paddle_from_parent_torch_runtime(self) -> None:
        metadata_path = self.root / "dense.jsonl"
        output_path = self.root / "ocr.jsonl"
        report_path = self.root / "ocr_report.json"
        metadata_path.write_text("{}\n", encoding="utf-8")

        def run_worker(command, *, cwd, check):
            self.assertEqual(command[0], pipeline.sys.executable)
            self.assertIn("--isolate-paddle-runtime", command)
            self.assertIn("--overwrite", command)
            self.assertEqual(cwd, Path(pipeline.__file__).resolve().parents[3])
            self.assertFalse(check)
            report_path.write_text(
                json.dumps({"pipeline": "ocr", "success_count": 1}),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0)

        with patch.object(pipeline.subprocess, "run", side_effect=run_worker):
            report = pipeline._run_ocr_file_isolated(
                metadata_path=metadata_path,
                output_path=output_path,
                report_path=report_path,
                config=self.config,
                overwrite=True,
            )

        self.assertEqual(report["success_count"], 1)

    def test_empty_siglip_checkpoint_is_invalidated_and_rebuilt(self) -> None:
        video = self._video("video_empty_npy")
        materialized = self._materialized_stage(video.stem, 1)
        paths = pipeline.PerVideoPaths.from_config(video.stem, self.config)
        paths.dense_metadata.parent.mkdir(parents=True, exist_ok=True)
        paths.dense_metadata.write_text(
            json.dumps(materialized.records[0]) + "\n",
            encoding="utf-8",
        )
        paths.dense_embeddings.parent.mkdir(parents=True, exist_ok=True)
        paths.dense_embeddings.write_bytes(b"")
        contract = pipeline._siglip_contract(paths, materialized, self.config)
        paths.dense_embedding_report.write_text(
            json.dumps(pipeline._with_contract({"status": "passed"}, contract))
            + "\n",
            encoding="utf-8",
        )

        embedding_records = [
            {
                "candidate_id": materialized.records[0]["candidate_id"],
                "frame_id": materialized.records[0]["frame_id"],
                "video_id": video.stem,
                "embedding_index": 0,
                "vector_dim": 2,
                "normalized": True,
            }
        ]
        with (
            patch.object(
                pipeline,
                "encode_keyframes",
                return_value=(
                    np.asarray([[1.0, 0.0]], dtype=np.float32),
                    embedding_records,
                    [],
                    {},
                ),
            ) as encoder,
            patch.object(pipeline, "_release_accelerator_memory"),
        ):
            embeddings, records, returned_contract = (
                pipeline._load_or_run_siglip_features(
                    video,
                    materialized,
                    self.config,
                    paths,
                )
            )

        encoder.assert_called_once()
        self.assertEqual(embeddings.shape, (1, 2))
        self.assertEqual(
            records[0]["candidate_id"],
            materialized.records[0]["candidate_id"],
        )
        self.assertEqual(returned_contract, contract["contract_sha256"])

    def test_dataset_finishes_each_video_sequentially_in_sorted_order(self) -> None:
        videos = [
            self.root / "video_C.mp4",
            self.root / "video_A.mp4",
            self.root / "video_B.mp4",
        ]
        events: list[str] = []

        def process(video_path, _config):
            video_path = Path(video_path)
            events.append(f"start:{video_path.stem}")
            artifact = self._video_artifacts(video_path)
            events.append(f"finish:{video_path.stem}")
            return artifact

        with (
            patch.object(pipeline, "process_video", side_effect=process),
            patch.object(
                pipeline,
                "build_corpus_indexes",
                return_value={"status": "passed"},
            ),
        ):
            result = pipeline.process_dataset(videos, self.config)

        self.assertEqual(
            events,
            [
                "start:video_A",
                "finish:video_A",
                "start:video_B",
                "finish:video_B",
                "start:video_C",
                "finish:video_C",
            ],
        )
        self.assertEqual(
            [path.stem for path in result.requested_videos],
            ["video_A", "video_B", "video_C"],
        )

    def test_corpus_indexing_runs_once_and_only_after_all_videos(self) -> None:
        videos = [
            self.root / "video_C.mp4",
            self.root / "video_A.mp4",
            self.root / "video_B.mp4",
        ]
        events: list[str] = []

        def process(video_path, _config):
            video_path = Path(video_path)
            events.append(f"video:{video_path.stem}")
            return self._video_artifacts(video_path)

        def build(artifacts, _config):
            self.assertEqual(
                events,
                ["video:video_A", "video:video_B", "video:video_C"],
            )
            self.assertEqual(
                [artifact.video_id for artifact in artifacts],
                ["video_A", "video_B", "video_C"],
            )
            events.append("corpus")
            return {"status": "passed"}

        with (
            patch.object(pipeline, "process_video", side_effect=process),
            patch.object(pipeline, "build_corpus_indexes", side_effect=build) as builder,
        ):
            result = pipeline.process_dataset(videos, self.config)

        self.assertEqual(
            events,
            ["video:video_A", "video:video_B", "video:video_C", "corpus"],
        )
        builder.assert_called_once()
        self.assertTrue(result.complete)

    def test_quick_cli_preserves_existing_corpus_unless_explicitly_replaced(
        self,
    ) -> None:
        parser = pipeline.build_parser()
        quick = pipeline._config_from_args(
            parser.parse_args(
                ["--video-id", "video_A", "--output-dir", str(self.config.output_dir)]
            )
        )
        explicit = pipeline._config_from_args(
            parser.parse_args(
                [
                    "--video-id",
                    "video_A",
                    "--build-corpus",
                    "--output-dir",
                    str(self.config.output_dir),
                ]
            )
        )
        full = pipeline._config_from_args(
            parser.parse_args(["--output-dir", str(self.config.output_dir)])
        )

        self.assertFalse(quick.build_corpus)
        self.assertTrue(explicit.build_corpus)
        self.assertTrue(full.build_corpus)
        self.assertEqual(quick.caption_model_name, "florence-community/Florence-2-base-ft")
        self.assertEqual(
            quick.caption_model_revision,
            "0b03b6f15a4a211370fb204aee4e7dd48887ea37",
        )
        self.assertEqual(quick.caption_task_prompt, "<MORE_DETAILED_CAPTION>")

        video = self._video("video_A")
        artifact = self._video_artifacts(video)
        existing_index = quick.output_dir / "indexes" / "existing.marker"
        existing_index.parent.mkdir(parents=True, exist_ok=True)
        existing_index.write_text("keep", encoding="utf-8")
        with (
            patch.object(pipeline, "process_video", return_value=artifact),
            patch.object(pipeline, "build_corpus_indexes") as builder,
        ):
            result = pipeline.process_dataset((video,), quick)

        builder.assert_not_called()
        self.assertTrue(result.complete)
        self.assertTrue(result.corpus_skipped)
        self.assertEqual(existing_index.read_text(encoding="utf-8"), "keep")

    def test_empty_video_id_is_rejected_instead_of_expanding_to_full_dataset(
        self,
    ) -> None:
        video_dir = self.root / "videos"
        self._video("video_A")
        args = pipeline.build_parser().parse_args(
            ["--video-dir", str(video_dir), "--video-id", ""]
        )

        self.assertFalse(pipeline._config_from_args(args).build_corpus)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            pipeline._discover_videos(args)

    def test_source_hash_cache_detects_same_size_same_mtime_rewrite(self) -> None:
        video = self.root / "same-metadata.mp4"
        video.write_bytes(b"AAAA")
        original_stat = video.stat()
        first = pipeline._file_signature(video)

        time.sleep(0.02)
        video.write_bytes(b"BBBB")
        os.utime(
            video,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        second = pipeline._file_signature(video)

        self.assertEqual(first["size_bytes"], second["size_bytes"])
        self.assertEqual(first["mtime_ns"], second["mtime_ns"])
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_resume_skips_valid_video_and_reruns_invalid_video(self) -> None:
        video_a = self._video("video_A")
        video_b = self._video("video_B")
        shots = self._shot_stage(video_b.stem, 3)
        candidates = self._candidate_stage(video_b.stem, 3)
        materialized = self._materialized_stage(video_b.stem, 3)
        features = self._feature_bundle(video_b.stem, 3)
        rerun_artifact = self._video_artifacts(video_b)

        def validate_checkpoint(
            *,
            video_path,
            config,
            paths,
            require_completion,
            source_signature=None,
        ):
            self.assertTrue(require_completion)
            self.assertIs(config, self.config)
            self.assertIsInstance(source_signature, dict)
            if paths.video_id == video_a.stem:
                return (
                    {"dense_candidate_count": 3, "status": "passed"},
                    ({"frame_id": "F0"},),
                )
            raise ValueError("invalid completion checkpoint")

        with (
            patch.object(
                pipeline,
                "_validate_selected_bundle",
                side_effect=validate_checkpoint,
            ) as validator,
            patch.object(
                pipeline,
                "_load_or_run_shot_detection",
                return_value=shots,
            ) as shot_stage,
            patch.object(
                pipeline,
                "_load_or_run_dense_candidate_generation",
                return_value=candidates,
            ),
            patch.object(
                pipeline,
                "_load_or_run_dense_materialization",
                return_value=materialized,
            ),
            patch.object(
                pipeline,
                "_extract_all_dense_features",
                return_value=features,
            ),
            patch.object(
                pipeline,
                "_run_multimodal_selection",
                return_value=(SimpleNamespace(), {"contract_sha256": "selection"}),
            ),
            patch.object(pipeline, "_persist_selected_artifacts"),
            patch.object(
                pipeline,
                "_validate_and_commit_video",
                return_value=rerun_artifact,
            ),
            patch.object(pipeline, "_release_accelerator_memory"),
        ):
            skipped = pipeline.process_video(video_a, self.config)
            rerun = pipeline.process_video(video_b, self.config)

        self.assertTrue(skipped.skipped)
        self.assertEqual(skipped.video_id, "video_A")
        self.assertIs(rerun, rerun_artifact)
        self.assertFalse(rerun.skipped)
        self.assertEqual(validator.call_count, 2)
        shot_stage.assert_called_once()
        self.assertEqual(Path(shot_stage.call_args.args[0]).stem, "video_B")

    def test_failure_is_isolated_and_blocks_default_corpus_build(self) -> None:
        videos = [
            self.root / "video_C.mp4",
            self.root / "video_A.mp4",
            self.root / "video_B.mp4",
        ]
        artifact_a = self._video_artifacts(self.root / "video_A.mp4")
        artifact_c = self._video_artifacts(self.root / "video_C.mp4")
        completion = artifact_a.paths.completion_report
        completion.parent.mkdir(parents=True, exist_ok=True)
        completion.write_text("stable-completion-marker", encoding="utf-8")
        calls: list[str] = []

        def process(video_path, _config):
            video_id = Path(video_path).stem
            calls.append(video_id)
            if video_id == "video_B":
                raise pipeline.OfflineStageError(
                    video_id,
                    pipeline.STAGE_OCR,
                    RuntimeError("OCR exploded"),
                )
            return artifact_a if video_id == "video_A" else artifact_c

        with (
            patch.object(pipeline, "process_video", side_effect=process),
            patch.object(pipeline, "build_corpus_indexes") as builder,
        ):
            result = pipeline.process_dataset(videos, self.config)

        self.assertEqual(calls, ["video_A", "video_B", "video_C"])
        self.assertEqual(
            [artifact.video_id for artifact in result.successful_videos],
            ["video_A", "video_C"],
        )
        self.assertIs(result.successful_videos[0], artifact_a)
        self.assertEqual(completion.read_text(encoding="utf-8"), "stable-completion-marker")
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].video_id, "video_B")
        self.assertEqual(result.failures[0].stage, pipeline.STAGE_OCR)
        self.assertTrue(result.corpus_blocked)
        self.assertIsNone(result.corpus_result)
        builder.assert_not_called()

    def test_persisted_selected_metadata_loads_with_real_canonical_loader(self) -> None:
        video = self._video("video_metadata")
        paths = pipeline.PerVideoPaths.from_config(video.stem, self.config)
        dense_image = self.root / "dense.jpg"
        dense_image.write_bytes(b"fake-jpeg-content")
        candidate_id = f"{video.stem}:C0000"
        frame_id = f"{video.stem}:F0000"
        base = {
            "candidate_id": candidate_id,
            "frame_id": frame_id,
            "video_id": video.stem,
            "shot_id": f"SHOT_{video.stem}_000000",
            "segment_id": f"SHOT_{video.stem}_000000",
            "shot_index": 0,
            "frame_index": 25,
            "timestamp": 1.0,
            "candidate_reasons": ["dense_interval"],
            "keyframe_path": dense_image.as_posix(),
            "artifact_role": "dense_candidate",
        }
        result = SimpleNamespace(
            final_records=(base,),
            final_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            final_embedding_records=(
                {
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "embedding_index": 0,
                    "vector_dim": 2,
                    "normalized": True,
                },
            ),
            final_caption_records=(
                {
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "status": "success",
                    "caption": "a red bicycle",
                },
            ),
            final_ocr_records=(
                {
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "status": "success",
                    "ocr_text": "SALE",
                },
            ),
            final_object_records=(
                {
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "status": "success",
                    "objects": ["bicycle"],
                },
            ),
            candidate_ledger=(base,),
            event_ledger=(),
            to_report=lambda: {
                "candidate_count": 1,
                "selected_count": 1,
                "guarantees": {"constraints_satisfied": True},
            },
        )
        contract = pipeline._stage_contract("test_selection", fixture=True)

        pipeline._persist_selected_artifacts(
            video,
            result,
            contract,
            paths,
        )
        loaded = load_canonical_keyframe_records(
            paths.selected_metadata.parent,
            video_ids=(video.stem,),
        )

        self.assertEqual(len(loaded), 1)
        record = loaded[0]
        self.assertEqual(record["video_id"], video.stem)
        self.assertEqual(record["frame_id"], frame_id)
        self.assertEqual(record["artifact_role"], "selected_keyframe")
        self.assertEqual(record["caption"], "a red bicycle")
        self.assertEqual(record["ocr_text"], "SALE")
        self.assertEqual(record["objects"], ["bicycle"])
        self.assertEqual(record["bge_source_kind"], "canonical_selected_keyframe")
        self.assertEqual(
            Path(record["keyframe_path"]),
            paths.selected_images_dir / dense_image.name,
        )
        self.assertTrue(Path(record["image_path"]).is_file())

    def test_missing_or_corrupt_validation_report_never_skips_whole_video(self) -> None:
        video = self._video("video_validation_marker")
        paths = pipeline.PerVideoPaths.from_config(video.stem, self.config)
        dense_image = self.root / "validation-marker-source.jpg"
        dense_image.write_bytes(b"fake-jpeg-content")
        candidate_id = f"{video.stem}:C0000"
        frame_id = f"{video.stem}:F0000"
        base = {
            "candidate_id": candidate_id,
            "frame_id": frame_id,
            "video_id": video.stem,
            "shot_id": f"SHOT_{video.stem}_000000",
            "segment_id": f"SHOT_{video.stem}_000000",
            "shot_index": 0,
            "frame_index": 25,
            "timestamp": 1.0,
            "candidate_reasons": ["dense_interval"],
            "keyframe_path": dense_image.as_posix(),
            "artifact_role": "dense_candidate",
        }
        result = SimpleNamespace(
            final_records=(base,),
            final_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            final_embedding_records=(
                {
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "embedding_index": 0,
                    "vector_dim": 2,
                    "normalized": True,
                },
            ),
            final_caption_records=(
                {
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "status": "success",
                    "caption": "a validation marker frame",
                },
            ),
            final_ocr_records=(
                {
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "status": "success",
                    "ocr_text": "",
                },
            ),
            final_object_records=(
                {
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "status": "success",
                    "objects": [],
                },
            ),
            candidate_ledger=(base,),
            event_ledger=(),
            to_report=lambda: {
                "candidate_count": 1,
                "selected_count": 1,
                "guarantees": {"constraints_satisfied": True},
            },
        )
        pipeline._persist_selected_artifacts(
            video,
            result,
            pipeline._stage_contract("test_selection", fixture=True),
            paths,
        )
        pipeline._atomic_save_npy(paths.dense_embeddings, result.final_embeddings)
        pipeline._atomic_write_jsonl(
            paths.dense_embedding_metadata,
            result.final_embedding_records,
        )
        pipeline._atomic_write_jsonl(paths.dense_captions, result.final_caption_records)
        pipeline._atomic_write_jsonl(paths.dense_ocr, result.final_ocr_records)
        pipeline._atomic_write_jsonl(paths.dense_objects, result.final_object_records)
        shot_contract = pipeline._shot_contract(video, self.config)
        pipeline._atomic_write_json(
            paths.shot_report,
            pipeline._with_contract({"status": "passed"}, shot_contract),
        )

        with patch.object(
            pipeline,
            "validate_records",
            return_value={"valid": True, "errors": []},
        ):
            pipeline._validate_and_commit_video(video, self.config, paths)
            self.assertIsNotNone(
                pipeline._try_load_complete_video(video, self.config, paths)
            )

            paths.validation_report.write_text("{corrupt", encoding="utf-8")
            self.assertIsNone(
                pipeline._try_load_complete_video(video, self.config, paths)
            )

            paths.validation_report.unlink()
            self.assertIsNone(
                pipeline._try_load_complete_video(video, self.config, paths)
            )

            paths.completion_report.unlink()
            original_stat = video.stat()
            time.sleep(0.02)
            video.write_bytes(b"X" * original_stat.st_size)
            os.utime(
                video,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            with self.assertRaisesRegex(ValueError, "Source video changed"):
                pipeline._validate_and_commit_video(video, self.config, paths)
            self.assertFalse(paths.completion_report.exists())

    def test_canonical_loader_rejects_filename_row_video_mismatch_in_all_modes(
        self,
    ) -> None:
        metadata = self.config.output_dir / "metadata_mismatch"
        metadata.mkdir(parents=True, exist_ok=True)
        keyframe = {
            "candidate_id": "B:C0000",
            "video_id": "B",
            "frame_id": "B:F0000",
            "artifact_role": "selected_keyframe",
            "keyframe_path": "keyframes/B/B:F0000.jpg",
            "timestamp": 1.0,
        }
        caption = {
            "video_id": "B",
            "frame_id": "B:F0000",
            "caption": "wrong filename lineage",
        }
        (metadata / "keyframes_A.jsonl").write_text(
            json.dumps(keyframe) + "\n",
            encoding="utf-8",
        )
        (metadata / "captions_A.jsonl").write_text(
            json.dumps(caption) + "\n",
            encoding="utf-8",
        )

        for mode, video_ids in (("unscoped", None), ("scoped", ("A",))):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "video_id"):
                    load_canonical_keyframe_records(metadata, video_ids=video_ids)

    def test_offline_bge_validator_rejects_same_count_wrong_canonical_lineage(
        self,
    ) -> None:
        paths = pipeline.CorpusPaths.from_config(self.config)
        canonical = [
            {
                "candidate_id": "A:C0000",
                "video_id": "A",
                "frame_id": "A:F0000",
                "artifact_role": "selected_keyframe",
                "keyframe_path": "keyframes/A/A:F0000.jpg",
                "timestamp": 1.0,
                "caption": "canonical frame",
            }
        ]
        wrong_lineage = {
            **canonical[0],
            "candidate_id": "B:C0000",
            "video_id": "B",
            "frame_id": "B:F0000",
        }
        validated = SimpleNamespace(
            frame_records=({"row": 0, "metadata": wrong_lineage},),
            manifest={
                "model": {"name": self.config.bge_model_name},
                "source_contract": {
                    "canonical_only": True,
                    "source_kind": "selected_keyframes",
                },
            },
        )

        with patch.object(
            pipeline,
            "validate_bge_m3_artifacts",
            return_value=validated,
        ):
            with self.assertRaisesRegex(ValueError, "lineage|canonical|match"):
                pipeline._validate_bge_corpus_index(
                    paths,
                    canonical_records=canonical,
                    config=self.config,
                )

    def test_visual_corpus_validator_rejects_mixed_generation_at_same_count(
        self,
    ) -> None:
        paths = pipeline.CorpusPaths.from_config(self.config)
        for path in (
            paths.visual_index,
            paths.visual_metadata,
            paths.visual_frame_map,
            paths.visual_manifest,
            paths.visual_report,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        indexed_record = {
            "candidate_id": "A:C0000",
            "frame_id": "A:F0000",
            "video_id": "A",
            "shot_id": "SHOT_A_000000",
            "segment_id": "SHOT_A_000000",
            "frame_index": 25,
            "timestamp": 1.0,
            "keyframe_path": "keyframes/A/A:F0000.jpg",
            "embedding_id": "A:E0000",
            "embedding_index": 0,
        }
        frame_map = {
            "0": pipeline.frame_map_record(indexed_record),
        }
        original = {
            "index": b"FAISS-INDEX-A",
            "metadata": (
                json.dumps(indexed_record, ensure_ascii=False) + "\n"
            ).encode("utf-8"),
            "frame_map": json.dumps(frame_map, ensure_ascii=False).encode("utf-8"),
            "report": json.dumps({"status": "passed"}).encode("utf-8"),
        }
        artifact_paths = {
            "index": paths.visual_index,
            "metadata": paths.visual_metadata,
            "frame_map": paths.visual_frame_map,
            "report": paths.visual_report,
        }
        for label, payload in original.items():
            artifact_paths[label].write_bytes(payload)
        hashes = {
            label: hashlib.sha256(payload).hexdigest()
            for label, payload in original.items()
        }
        generation = hashlib.sha256(
            json.dumps(
                hashes,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        paths.visual_manifest.write_text(
            json.dumps(
                {
                    "vector_count": 1,
                    "artifacts": {
                        label: {
                            "filename": path.name,
                            "sha256": hashes[label],
                        }
                        for label, path in artifact_paths.items()
                    },
                    "bundle_generation": generation,
                }
            ),
            encoding="utf-8",
        )
        expected_vectors = np.array([[1.0, 0.0]], dtype=np.float32)

        def reconstruct_n(_start, _count, output) -> None:
            output[:] = expected_vectors

        fake_faiss = SimpleNamespace(
            read_index=lambda _path: SimpleNamespace(
                ntotal=1,
                reconstruct_n=reconstruct_n,
            ),
        )
        videos = (SimpleNamespace(video_id="A"),)

        with (
            patch.object(pipeline, "require_faiss", return_value=fake_faiss),
            patch.object(
                pipeline,
                "_expected_visual_index_records",
                return_value=([indexed_record], expected_vectors),
            ),
        ):
            valid = pipeline._validate_visual_corpus_index(paths, videos=videos)
            self.assertEqual(valid["bundle_generation"], generation)

            replacements = {
                "index": b"FAISS-INDEX-B",
                "frame_map": json.dumps(
                    {
                        "0": {
                            **frame_map["0"],
                            "timestamp": 2.0,
                        }
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            }
            for label, replacement in replacements.items():
                with self.subTest(swapped_artifact=label):
                    artifact_paths[label].write_bytes(replacement)
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"visual {label} checksum does not match manifest",
                    ):
                        pipeline._validate_visual_corpus_index(paths, videos=videos)
                    artifact_paths[label].write_bytes(original[label])

    def _write_canonical_family(self, video_id: str, caption: str) -> dict:
        metadata = self.config.output_dir / "metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        paths = pipeline.PerVideoPaths.from_config(video_id, self.config)
        frame_id = f"{video_id}:F0000"
        selected = {
            "candidate_id": f"{video_id}:C0000",
            "video_id": video_id,
            "frame_id": frame_id,
            "artifact_role": "selected_keyframe",
            "keyframe_path": f"keyframes/{video_id}/{frame_id}.jpg",
            "timestamp": 1.0,
            "timestamp_source": "video_fps",
            "frame_index": 25,
            "shot_id": f"SHOT_{video_id}_000000",
            "segment_id": f"SHOT_{video_id}_000000",
            "shot_start": 0.0,
            "shot_end": 2.0,
            "selection_phase": "protected",
            "protected": True,
            "covered_event_ids": [f"EVENT_{video_id}_000000"],
        }
        families = {
            f"keyframes_{video_id}.jsonl": selected,
            f"captions_{video_id}.jsonl": {
                "video_id": video_id,
                "frame_id": frame_id,
                "timestamp": 1.0,
                "status": "success",
                "caption": caption,
            },
            f"ocr_{video_id}.jsonl": {
                "video_id": video_id,
                "frame_id": frame_id,
                "timestamp": 1.0,
                "status": "success",
                "ocr_text": f"ocr-{video_id}",
            },
            f"objects_{video_id}.jsonl": {
                "video_id": video_id,
                "frame_id": frame_id,
                "timestamp": 1.0,
                "status": "success",
                "objects": [
                    {"class_name": f"object-{video_id}", "confidence": 0.9}
                ],
            },
        }
        for name, record in families.items():
            (metadata / name).write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        dense_vectors = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        paths.dense_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(paths.dense_embeddings, dense_vectors)
        dense_records = []
        dense_captions = []
        dense_ocr = []
        dense_objects = []
        dense_ledger = []
        for index in range(3):
            candidate_id = f"{video_id}:C{index:04d}"
            dense_frame_id = f"{video_id}:F{index:04d}"
            common = {
                "candidate_id": candidate_id,
                "frame_id": dense_frame_id,
                "video_id": video_id,
                "shot_id": f"SHOT_{video_id}_000000",
                "segment_id": f"SHOT_{video_id}_000000",
                "timestamp": float(index + 1),
                "frame_index": (index + 1) * 25,
                "keyframe_path": f"dense_keyframes/{video_id}/{dense_frame_id}.jpg",
            }
            dense_records.append(
                {
                    **common,
                    "embedding_id": f"EMB_{dense_frame_id}",
                    "embedding_index": index,
                    "model_family": "siglip2",
                    "model_name": "google/siglip2-so400m-patch16-384",
                    "model_revision": "unit-test-revision",
                    "processor_name": "google/siglip2-so400m-patch16-384",
                    "vector_dim": 2,
                    "input_resolution": 384,
                    "normalized": True,
                    "similarity": "cosine",
                    "output_dtype": "float32",
                }
            )
            dense_captions.append(
                {**common, "status": "success", "caption": f"{caption} {index}"}
            )
            dense_ocr.append(
                {**common, "status": "success", "ocr_text": f"ocr-{video_id}-{index}"}
            )
            dense_objects.append(
                {
                    **common,
                    "status": "success",
                    "objects": [
                        {"class_name": f"object-{video_id}-{index}", "confidence": 0.9}
                    ],
                }
            )
            dense_ledger.append(
                {
                    **common,
                    "selected": index == 0,
                    "importance_score": 1.0 - index * 0.1,
                    "semantic_novelty": index * 0.1,
                    "component_scores": {"caption": 0.8},
                    "available_modalities": ["caption", "ocr", "objects"],
                    "feature_protected_event_ids": (
                        [f"EVENT_{video_id}_000001"] if index == 1 else []
                    ),
                    "selection_rank": 1 if index == 0 else None,
                    "selection_phase": "protected" if index == 0 else None,
                    "selection_reasons": ["unit_test"] if index == 0 else [],
                }
            )
        for path, records in (
            (paths.dense_embedding_metadata, dense_records),
            (paths.dense_captions, dense_captions),
            (paths.dense_ocr, dense_ocr),
            (paths.dense_objects, dense_objects),
            (paths.candidate_ledger, dense_ledger),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
        return selected

    def test_neighbor_and_segment_stages_use_selected_keyframes_and_modalities(
        self,
    ) -> None:
        config = pipeline.replace(
            self.config,
            neighbor_window_seconds=5.0,
            segment_strategy="auto",
        )
        paths = pipeline.CorpusPaths.from_config(config)
        video_id = "video_temporal"
        records = [
            {
                "video_id": video_id,
                "frame_id": "F1",
                "artifact_role": "selected_keyframe",
                "timestamp": 1.0,
                "timestamp_source": "video_fps",
                "frame_index": 25,
                "shot_id": "SHOT_1",
                "segment_id": "SHOT_1",
                "shot_start": 0.0,
                "shot_end": 5.0,
                "selection_phase": "protected",
                "covered_event_ids": ["E1"],
            },
            {
                "video_id": video_id,
                "frame_id": "F2",
                "artifact_role": "selected_keyframe",
                "timestamp": 3.0,
                "timestamp_source": "video_fps",
                "frame_index": 75,
                "shot_id": "SHOT_1",
                "segment_id": "SHOT_1",
                "shot_start": 0.0,
                "shot_end": 5.0,
                "selection_phase": "coverage",
                "covered_event_ids": ["E2"],
            },
        ]
        modalities = {
            "captions": [
                {
                    "video_id": video_id,
                    "frame_id": "F1",
                    "timestamp": 1.0,
                    "status": "success",
                    "caption": "A red car is parked.",
                },
                {
                    "video_id": video_id,
                    "frame_id": "F2",
                    "timestamp": 3.0,
                    "status": "success",
                    "caption": "A person walks past the car.",
                },
            ],
            "ocr": [
                {
                    "video_id": video_id,
                    "frame_id": "F1",
                    "timestamp": 1.0,
                    "status": "success",
                    "ocr_text": "HELLO",
                },
                {
                    "video_id": video_id,
                    "frame_id": "F2",
                    "timestamp": 3.0,
                    "status": "success",
                    "ocr_text": "WORLD",
                },
            ],
            "objects": [
                {
                    "video_id": video_id,
                    "frame_id": "F1",
                    "timestamp": 1.0,
                    "status": "success",
                    "objects": [{"class_name": "car", "confidence": 0.9}],
                },
                {
                    "video_id": video_id,
                    "frame_id": "F2",
                    "timestamp": 3.0,
                    "status": "success",
                    "objects": [{"class_name": "person", "confidence": 0.8}],
                },
            ],
        }
        source_contract = {"contract_sha256": "temporal-contract"}

        neighbor_report = pipeline._build_neighbor_corpus_metadata(
            paths,
            records,
            source_contract,
            config,
        )
        segment_report = pipeline._build_segment_corpus_metadata(
            paths,
            records,
            modalities,
            source_contract,
            config,
        )

        neighbors = pipeline._read_jsonl(paths.neighbor_metadata)
        self.assertEqual(neighbor_report["record_count"], 2)
        self.assertEqual(
            neighbors[0]["neighbors_after"],
            [{"frame_id": "F2", "delta_seconds": 2.0}],
        )
        self.assertEqual(
            neighbors[1]["neighbors_before"],
            [{"frame_id": "F1", "delta_seconds": -2.0}],
        )
        segments = pipeline._read_jsonl(paths.segment_metadata)
        self.assertEqual(segment_report["record_count"], 1)
        self.assertEqual(segments[0]["keyframe_ids"], ["F1", "F2"])
        self.assertEqual(segments[0]["covered_event_ids"], ["E1", "E2"])
        self.assertIn("A red car is parked.", segments[0]["captions_aggregated"])
        self.assertEqual(
            {item["label"] for item in segments[0]["objects"]},
            {"car", "person"},
        )

    def test_corpus_uses_explicit_multi_video_inputs_and_excludes_stale_metadata(
        self,
    ) -> None:
        selected_by_video = {
            video_id: self._write_canonical_family(video_id, f"caption-{video_id}")
            for video_id in ("video_A", "video_B", "video_STALE")
        }
        videos = [self._video("video_B"), self._video("video_A")]
        artifacts: list[pipeline.VideoArtifacts] = []
        for video in videos:
            artifact = self._video_artifacts(video)
            artifact.paths.completion_report.parent.mkdir(parents=True, exist_ok=True)
            artifact.paths.completion_report.write_text(
                json.dumps({"status": "passed", "artifact_hashes": {}}) + "\n",
                encoding="utf-8",
            )
            artifacts.append(artifact)

        def validate_video(*, video_path, config, paths, require_completion):
            self.assertTrue(require_completion)
            self.assertIs(config, self.config)
            return (
                {"dense_candidate_count": 3, "status": "passed"},
                (selected_by_video[paths.video_id],),
            )

        def publish_report(staged, final, **_kwargs):
            final.validation_report.parent.mkdir(parents=True, exist_ok=True)
            final.validation_report.write_bytes(staged.validation_report.read_bytes())

        with (
            patch.object(
                pipeline,
                "_validate_selected_bundle",
                side_effect=validate_video,
            ),
            patch.object(
                pipeline,
                "load_canonical_keyframe_records",
                wraps=load_canonical_keyframe_records,
            ) as canonical_loader,
            patch.object(
                pipeline,
                "_build_visual_corpus_index",
                return_value={"status": "passed", "vector_count": 2},
            ),
            patch.object(
                pipeline,
                "_build_dense_corpus_index",
                return_value={"status": "passed", "vector_count": 6},
            ),
            patch.object(
                pipeline,
                "_build_text_corpus_index",
                return_value={"status": "passed", "input_record_count": 2},
            ) as text_builder,
            patch.object(
                pipeline,
                "_build_neighbor_corpus_metadata",
                return_value={"status": "passed", "record_count": 2},
            ) as neighbor_builder,
            patch.object(
                pipeline,
                "_build_segment_corpus_metadata",
                return_value={"status": "passed", "record_count": 2},
            ) as segment_builder,
            patch.object(pipeline, "_build_bge_corpus_index") as bge_builder,
            patch.object(pipeline, "_rebase_staged_visual_bundle"),
            patch.object(pipeline, "_rebase_staged_dense_bundle"),
            patch.object(
                pipeline,
                "_validate_visual_corpus_index",
                return_value={"status": "passed", "vector_count": 2},
            ),
            patch.object(
                pipeline,
                "_validate_dense_corpus_index",
                return_value={"status": "passed", "vector_count": 6},
            ),
            patch.object(
                pipeline,
                "_validate_text_corpus_index",
                return_value={"status": "passed", "input_record_count": 2},
            ),
            patch.object(
                pipeline,
                "_validate_neighbor_corpus_metadata",
                return_value={"status": "passed", "record_count": 2},
            ),
            patch.object(
                pipeline,
                "_validate_segment_corpus_metadata",
                return_value={"status": "passed", "record_count": 2},
            ),
            patch.object(
                pipeline,
                "_corpus_bundle_manifest_payload",
                return_value={"status": "passed"},
            ),
            patch.object(
                pipeline,
                "_publish_staged_corpus_bundle",
                side_effect=publish_report,
            ),
            patch.object(
                pipeline,
                "_validate_corpus_bundle_commit",
                return_value={"status": "passed"},
            ),
        ):
            report = pipeline.build_corpus_indexes(artifacts, self.config)

        canonical_loader.assert_called_once_with(
            self.config.output_dir / "metadata",
            video_ids=("video_A", "video_B"),
        )
        indexed_records = text_builder.call_args.args[1]
        self.assertEqual(
            [record["video_id"] for record in indexed_records],
            ["video_A", "video_B"],
        )
        self.assertNotIn(
            "video_STALE",
            {record["video_id"] for record in indexed_records},
        )
        neighbor_records = neighbor_builder.call_args.args[1]
        segment_records = segment_builder.call_args.args[1]
        segment_modalities = segment_builder.call_args.args[2]
        self.assertEqual(
            {record["video_id"] for record in neighbor_records},
            {"video_A", "video_B"},
        )
        self.assertEqual(
            {record["video_id"] for record in segment_records},
            {"video_A", "video_B"},
        )
        self.assertTrue(
            all(
                {record["video_id"] for record in records}
                == {"video_A", "video_B"}
                for records in segment_modalities.values()
            )
        )
        self.assertEqual(report["video_ids"], ["video_A", "video_B"])
        self.assertEqual(report["video_count"], 2)
        self.assertEqual(report["selected_keyframe_count"], 2)
        bge_builder.assert_not_called()

    def test_corpus_bundle_is_staged_committed_and_resumed_as_one_generation(
        self,
    ) -> None:
        video = self._video("video_atomic")
        selected = self._write_canonical_family(video.stem, "atomic caption")
        artifact = self._video_artifacts(video)
        paths = artifact.paths
        paths.selected_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            paths.selected_embeddings,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        embedding_record = {
            **selected,
            "embedding_id": f"EMB_{selected['frame_id']}",
            "embedding_index": 0,
            "model_family": "siglip2",
            "model_name": "google/siglip2-so400m-patch16-384",
            "model_revision": "unit-test-revision",
            "processor_name": "google/siglip2-so400m-patch16-384",
            "vector_dim": 2,
            "input_resolution": 384,
            "normalized": True,
            "similarity": "cosine",
            "output_dtype": "float32",
        }
        paths.selected_embedding_metadata.parent.mkdir(parents=True, exist_ok=True)
        paths.selected_embedding_metadata.write_text(
            json.dumps(embedding_record) + "\n",
            encoding="utf-8",
        )
        paths.completion_report.parent.mkdir(parents=True, exist_ok=True)
        paths.completion_report.write_text(
            json.dumps({"status": "passed", "artifact_hashes": {}}) + "\n",
            encoding="utf-8",
        )

        validation = {"dense_candidate_count": 3, "status": "passed"}
        with patch.object(
            pipeline,
            "_validate_selected_bundle",
            return_value=(validation, (selected,)),
        ):
            first = pipeline.build_corpus_indexes((artifact,), self.config)
            corpus_paths = pipeline.CorpusPaths.from_config(self.config)
            committed = pipeline._read_json(corpus_paths.corpus_manifest)
            self.assertEqual(committed["status"], "passed")
            self.assertEqual(committed["video_ids"], [video.stem])

            with patch.object(
                pipeline,
                "_build_visual_corpus_index",
                side_effect=AssertionError("valid bundle must resume as a whole"),
            ):
                resumed = pipeline.build_corpus_indexes((artifact,), self.config)

        self.assertEqual(first, resumed)
        self.assertEqual(first["selected_keyframe_count"], 1)
        self.assertTrue(corpus_paths.visual_index.is_file())
        self.assertTrue(corpus_paths.dense_index.is_file())
        self.assertTrue(corpus_paths.dense_metadata.is_file())
        self.assertTrue(corpus_paths.dense_frame_map.is_file())
        self.assertTrue(corpus_paths.dense_manifest.is_file())
        self.assertTrue(corpus_paths.text_index.is_file())
        self.assertTrue(corpus_paths.neighbor_metadata.is_file())
        self.assertTrue(corpus_paths.segment_metadata.is_file())
        neighbors = pipeline._read_jsonl(corpus_paths.neighbor_metadata)
        segments = pipeline._read_jsonl(corpus_paths.segment_metadata)
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0]["frame_id"], selected["frame_id"])
        self.assertEqual(neighbors[0]["neighbors_before"], [])
        self.assertEqual(neighbors[0]["neighbors_after"], [])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["keyframe_ids"], [selected["frame_id"]])
        self.assertEqual(segments[0]["captions_aggregated"], "atomic caption")
        self.assertEqual(
            {
                "dense_index",
                "dense_metadata",
                "dense_frame_map",
                "dense_manifest",
                "dense_report",
                "neighbor_metadata",
                "segment_metadata",
            }.issubset(
                committed["artifacts"]
            ),
            True,
        )
        from backend.app.services.retrieval import retrieval_manager

        runtime_environment = {
            "RETRIEVAL_CORPUS_MANIFEST_PATH": str(corpus_paths.corpus_manifest),
            "RETRIEVAL_INDEX_PATH": str(corpus_paths.visual_index),
            "RETRIEVAL_FRAME_MAP_PATH": str(corpus_paths.visual_frame_map),
            "RETRIEVAL_MANIFEST_PATH": str(corpus_paths.visual_manifest),
            "RETRIEVAL_DENSE_INDEX_PATH": str(corpus_paths.dense_index),
            "RETRIEVAL_DENSE_METADATA_PATH": str(corpus_paths.dense_metadata),
            "RETRIEVAL_DENSE_FRAME_MAP_PATH": str(corpus_paths.dense_frame_map),
            "RETRIEVAL_DENSE_MANIFEST_PATH": str(corpus_paths.dense_manifest),
            "RETRIEVAL_DENSE_REPORT_PATH": str(corpus_paths.dense_report),
        }
        with patch.dict(
            os.environ,
            runtime_environment,
        ):
            runtime_manifest = retrieval_manager.validate_runtime_corpus_bundle(
                required_roles=(
                    "visual_index",
                    "dense_index",
                    "dense_metadata",
                    "dense_frame_map",
                    "dense_manifest",
                    "dense_report",
                    "text_index",
                    "neighbor_metadata",
                    "segment_metadata",
                ),
                artifact_overrides={
                    "visual_index": corpus_paths.visual_index,
                    "dense_index": corpus_paths.dense_index,
                    "dense_metadata": corpus_paths.dense_metadata,
                    "dense_frame_map": corpus_paths.dense_frame_map,
                    "dense_manifest": corpus_paths.dense_manifest,
                    "dense_report": corpus_paths.dense_report,
                    "text_index": corpus_paths.text_index,
                    "neighbor_metadata": corpus_paths.neighbor_metadata,
                    "segment_metadata": corpus_paths.segment_metadata,
                },
            )
            self.assertEqual(
                runtime_manifest["bundle_generation"],
                committed["bundle_generation"],
            )
            retrieval_manager.clear_retrieval_caches()
            try:
                dense_index = retrieval_manager.get_dense_candidate_index()
                self.assertEqual(len(dense_index.records), 3)
                self.assertEqual(
                    dense_index.corpus_generation,
                    committed["bundle_generation"],
                )
            finally:
                retrieval_manager.clear_retrieval_caches()
            corpus_paths.text_index.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "committed corpus"):
                retrieval_manager.validate_runtime_corpus_bundle(
                    required_roles=("text_index",),
                    artifact_overrides={"text_index": corpus_paths.text_index},
                )

    def test_failed_staged_bge_never_partially_replaces_existing_corpus(self) -> None:
        config = pipeline.replace(self.config, bge_enabled=True)
        video = self._video("video_bge_failure")
        selected = self._write_canonical_family(video.stem, "stable caption")
        artifact = self._video_artifacts(video)
        artifact.paths.completion_report.parent.mkdir(parents=True, exist_ok=True)
        artifact.paths.completion_report.write_text(
            json.dumps({"status": "passed", "artifact_hashes": {}}) + "\n",
            encoding="utf-8",
        )
        corpus_paths = pipeline.CorpusPaths.from_config(config)
        corpus_paths.visual_index.parent.mkdir(parents=True, exist_ok=True)
        corpus_paths.text_index.parent.mkdir(parents=True, exist_ok=True)
        corpus_paths.visual_index.write_bytes(b"OLD-VISUAL")
        corpus_paths.dense_index.write_bytes(b"OLD-DENSE")
        corpus_paths.text_index.write_bytes(b"OLD-TEXT")

        with (
            patch.object(
                pipeline,
                "_validate_selected_bundle",
                return_value=(
                    {"dense_candidate_count": 3, "status": "passed"},
                    (selected,),
                ),
            ),
            patch.object(
                pipeline,
                "_build_visual_corpus_index",
                return_value={"status": "passed"},
            ),
            patch.object(
                pipeline,
                "_build_text_corpus_index",
                return_value={"status": "passed"},
            ),
            patch.object(
                pipeline,
                "_build_bge_corpus_index",
                side_effect=RuntimeError("BGE build failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "BGE build failed"):
                pipeline.build_corpus_indexes((artifact,), config)

        self.assertEqual(corpus_paths.visual_index.read_bytes(), b"OLD-VISUAL")
        self.assertEqual(corpus_paths.dense_index.read_bytes(), b"OLD-DENSE")
        self.assertEqual(corpus_paths.text_index.read_bytes(), b"OLD-TEXT")
        self.assertFalse(corpus_paths.corpus_manifest.exists())

    def test_failed_staged_segments_never_partially_replace_existing_corpus(
        self,
    ) -> None:
        video = self._video("video_segment_failure")
        selected = self._write_canonical_family(video.stem, "stable caption")
        artifact = self._video_artifacts(video)
        artifact.paths.completion_report.parent.mkdir(parents=True, exist_ok=True)
        artifact.paths.completion_report.write_text(
            json.dumps({"status": "passed", "artifact_hashes": {}}) + "\n",
            encoding="utf-8",
        )
        corpus_paths = pipeline.CorpusPaths.from_config(self.config)
        old_artifacts = {
            corpus_paths.visual_index: b"OLD-VISUAL",
            corpus_paths.dense_index: b"OLD-DENSE",
            corpus_paths.text_index: b"OLD-TEXT",
            corpus_paths.neighbor_metadata: b"OLD-NEIGHBORS",
            corpus_paths.segment_metadata: b"OLD-SEGMENTS",
        }
        for path, payload in old_artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        with (
            patch.object(
                pipeline,
                "_validate_selected_bundle",
                return_value=(
                    {"dense_candidate_count": 3, "status": "passed"},
                    (selected,),
                ),
            ),
            patch.object(
                pipeline,
                "_build_visual_corpus_index",
                return_value={"status": "passed"},
            ),
            patch.object(
                pipeline,
                "_build_text_corpus_index",
                return_value={"status": "passed"},
            ),
            patch.object(
                pipeline,
                "_build_neighbor_corpus_metadata",
                return_value={"status": "passed"},
            ),
            patch.object(
                pipeline,
                "_build_segment_corpus_metadata",
                side_effect=RuntimeError("segment build failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "segment build failed"):
                pipeline.build_corpus_indexes((artifact,), self.config)

        for path, payload in old_artifacts.items():
            self.assertEqual(path.read_bytes(), payload)
        self.assertFalse(corpus_paths.corpus_manifest.exists())


if __name__ == "__main__":
    unittest.main()
