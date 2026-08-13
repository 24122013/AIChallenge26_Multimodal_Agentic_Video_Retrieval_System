from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.indexing.build_text_index import write_text_index
from competition.pipeline import (
    ARTIFACT_TAG,
    CorpusVideo,
    _competition_extract_config,
    _embedding_lineage,
    _embedding_source_lineage,
    _frame_ids_sha256,
    _require_current_index_lineage,
    _sha256_file,
    answers_from_results,
    build_competition_hybrid_engine,
    build_parser,
    competition_index_paths,
    embed_command,
    extract_command,
    index_command,
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


def _dense_report(
    args,
    video: CorpusVideo,
    *,
    satisfied: bool = True,
    stop_reason: str = "constraints_satisfied",
) -> dict:
    return {
        "video_id": video.video_id,
        "keyframe_strategy": "dense_coverage",
        "status": "satisfied" if satisfied else "partial",
        "constraints_satisfied": satisfied,
        "coverage_satisfied": satisfied,
        "phash_threshold": args.phash_threshold,
        "phash_window_sec": args.phash_window_sec,
        "jpeg_quality": args.jpeg_quality,
        "shot_threshold": args.shot_threshold,
        "shot_device": args.device,
        "candidate_interval_sec": args.candidate_interval_sec,
        "boundary_guard_sec": args.boundary_guard_sec,
        "tiny_shot_max_sec": args.tiny_shot_max_sec,
        "selection_config": {
            "max_gap_seconds": args.max_gap_seconds,
            "gap_tolerance_seconds": args.gap_tolerance_seconds,
            "target_keyframes": args.target_keyframes,
            "hard_max_keyframes": args.hard_max_keyframes,
            "protect_each_shot": True,
        },
        "selection": {"stop_reason": stop_reason},
        "keyframe_count": 1 if satisfied else 0,
    }


def _write_mock_source(public_root: Path, video: CorpusVideo) -> Path:
    video_path = public_root / video.relative_path
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"mock-video")
    return video_path


def _write_mock_keyframe(output_root: Path, video: CorpusVideo) -> None:
    image_path = output_root / "keyframes" / video.video_id / "FRAME_video0001_000000001.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"mock-image")
    metadata_path = output_root / "metadata" / f"keyframes_{video.video_id}.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "frame_id": "FRAME_video0001_000000001",
                "video_id": video.video_id,
                "timestamp": 0.04,
                "frame_index": 1,
                "keyframe_path": image_path.as_posix(),
            }
        )
        + "\n",
        encoding="utf-8",
    )


class CompetitionPipelineTest(unittest.TestCase):
    def test_extract_defaults_to_dense_coverage_with_coverage_stop(self) -> None:
        args = build_parser().parse_args(["extract"])

        self.assertEqual(args.keyframe_strategy, "dense_coverage")
        self.assertEqual(args.candidate_interval_sec, 0.5)
        self.assertEqual(args.max_gap_seconds, 2.0)
        self.assertIsNone(args.target_keyframes)
        self.assertIsNone(args.hard_max_keyframes)

    def test_extract_keeps_explicit_legacy_rollback(self) -> None:
        args = build_parser().parse_args(
            ["extract", "--keyframe-strategy", "legacy"]
        )

        self.assertEqual(args.keyframe_strategy, "legacy")

    def test_extract_forwards_dense_configuration_and_persists_resume_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            public_root = root / "public"
            output_root = root / "output"
            args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(public_root),
                    "--output-root",
                    str(output_root),
                    "--candidate-interval-sec",
                    "0.4",
                    "--max-gap-seconds",
                    "4.0",
                ]
            )
            video = CorpusVideo(
                filename="video0001.mp4",
                relative_path=Path("videos/video0001.mp4"),
                fps=25.0,
                frame_count=250,
            )
            _write_mock_source(public_root, video)

            def extract_side_effect(**_kwargs):
                _write_mock_keyframe(output_root, video)
                return _dense_report(args, video)

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=extract_side_effect,
                ) as extract_mock,
            ):
                extract_command(args)

            forwarded = extract_mock.call_args.kwargs
            self.assertEqual(forwarded["strategy"], "dense_coverage")
            self.assertEqual(forwarded["candidate_interval_sec"], 0.4)
            self.assertEqual(forwarded["max_gap_seconds"], 4.0)
            self.assertIsNone(forwarded["target_keyframes"])
            report_path = (
                output_root / "metadata" / "keyframes_video0001_extract_report.json"
            )
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["competition_extract_config"],
                _competition_extract_config(args),
            )

    def test_dense_extract_fails_closed_when_constraints_are_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(root / "output"),
                ]
            )
            video = CorpusVideo(
                filename="video0001.mp4",
                relative_path=Path("videos/video0001.mp4"),
                fps=25.0,
                frame_count=250,
            )
            partial_report = {
                **_dense_report(
                    args,
                    video,
                    satisfied=False,
                    stop_reason="coverage_candidates_unavailable",
                )
            }
            _write_mock_source(args.public_root, video)

            def partial_extract_side_effect(**kwargs):
                metadata_path = kwargs["metadata_path"]
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text("", encoding="utf-8")
                return partial_report

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=partial_extract_side_effect,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "coverage_candidates_unavailable",
                ):
                    extract_command(args)
            saved_report = json.loads(
                (
                    args.output_root
                    / "metadata"
                    / "keyframes_video0001_extract_report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(saved_report["status"], "partial")
            self.assertIn("competition_extract_config", saved_report)

    def test_resume_only_skips_a_matching_satisfied_dense_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "output"
            args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                    "--resume",
                ]
            )
            video = CorpusVideo(
                filename="video0001.mp4",
                relative_path=Path("videos/video0001.mp4"),
                fps=25.0,
                frame_count=250,
            )
            _write_mock_source(args.public_root, video)

            def extract_side_effect(**_kwargs):
                _write_mock_keyframe(output_root, video)
                return _dense_report(args, video)

            args.resume = False
            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=extract_side_effect,
                ),
            ):
                extract_command(args)
            args.resume = True

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video"
                ) as extract_mock,
            ):
                extract_command(args)

            extract_mock.assert_not_called()

    def test_resume_reruns_when_dense_configuration_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "output"
            args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                    "--max-gap-seconds",
                    "4.0",
                    "--resume",
                ]
            )
            video = CorpusVideo(
                filename="video0001.mp4",
                relative_path=Path("videos/video0001.mp4"),
                fps=25.0,
                frame_count=250,
            )
            _write_mock_source(args.public_root, video)
            stale_args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                    "--max-gap-seconds",
                    "5.0",
                ]
            )

            def stale_extract_side_effect(**_kwargs):
                _write_mock_keyframe(output_root, video)
                return _dense_report(stale_args, video)

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=stale_extract_side_effect,
                ),
            ):
                extract_command(stale_args)

            def current_extract_side_effect(**_kwargs):
                _write_mock_keyframe(output_root, video)
                return _dense_report(args, video)

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=current_extract_side_effect,
                ) as extract_mock,
            ):
                extract_command(args)

            extract_mock.assert_called_once()

    def test_resume_reruns_when_keyframe_image_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "output"
            args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                ]
            )
            video = CorpusVideo(
                filename="video0001.mp4",
                relative_path=Path("videos/video0001.mp4"),
                fps=25.0,
                frame_count=250,
            )
            _write_mock_source(args.public_root, video)

            def extract_side_effect(**_kwargs):
                _write_mock_keyframe(output_root, video)
                return _dense_report(args, video)

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=extract_side_effect,
                ),
            ):
                extract_command(args)

            image_path = (
                output_root
                / "keyframes"
                / video.video_id
                / "FRAME_video0001_000000001.jpg"
            )
            image_path.write_bytes(b"changed-image-content")
            args.resume = True
            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=extract_side_effect,
                ) as extract_mock,
            ):
                extract_command(args)

            extract_mock.assert_called_once()

    def test_embed_rejects_partial_dense_extraction_before_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "output"
            extract_args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                ]
            )
            video = CorpusVideo(
                filename="video0001.mp4",
                relative_path=Path("videos/video0001.mp4"),
                fps=25.0,
                frame_count=250,
            )
            _write_mock_source(extract_args.public_root, video)

            def extract_side_effect(**_kwargs):
                _write_mock_keyframe(output_root, video)
                return _dense_report(extract_args, video)

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=extract_side_effect,
                ),
            ):
                extract_command(extract_args)

            report_path = (
                output_root / "metadata" / "keyframes_video0001_extract_report.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update(
                {
                    "status": "partial",
                    "constraints_satisfied": False,
                    "coverage_satisfied": False,
                }
            )
            report["selection"]["stop_reason"] = "coverage_candidates_unavailable"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            embed_args = build_parser().parse_args(
                [
                    "embed",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                    "--device",
                    "cpu",
                ]
            )

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.load_siglip2_model_processor"
                ) as load_model,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "coverage_candidates_unavailable",
                ):
                    embed_command(embed_args)

            load_model.assert_not_called()

    def test_embed_resume_checks_keyframe_lineage_before_skipping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "output"
            extract_args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                ]
            )
            embed_args = build_parser().parse_args(
                [
                    "embed",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                    "--device",
                    "cpu",
                    "--resume",
                ]
            )
            video = CorpusVideo(
                filename="video0001.mp4",
                relative_path=Path("videos/video0001.mp4"),
                fps=25.0,
                frame_count=250,
            )
            _write_mock_source(extract_args.public_root, video)

            def extract_side_effect(**_kwargs):
                _write_mock_keyframe(output_root, video)
                return _dense_report(extract_args, video)

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=extract_side_effect,
                ),
            ):
                extract_command(extract_args)

            extract_report_path = (
                output_root / "metadata" / "keyframes_video0001_extract_report.json"
            )
            extract_report = json.loads(
                extract_report_path.read_text(encoding="utf-8")
            )
            lineage = _embedding_lineage(
                embed_args,
                extract_report,
                resolved_device="cpu",
                resolved_model_revision="commit-test",
            )
            embeddings_path = (
                output_root / "embeddings" / f"{ARTIFACT_TAG}_video0001.npy"
            )
            embeddings_path.parent.mkdir(parents=True)
            vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
            np.save(embeddings_path, vectors)
            embedding_record = {
                "frame_id": "FRAME_video0001_000000001",
                "embedding_index": 0,
                "vector_dim": 2,
                "normalized": True,
                "model_name": embed_args.model_name,
                "model_revision": "commit-test",
            }
            embedding_metadata_path = (
                output_root
                / "metadata"
                / f"{ARTIFACT_TAG}_embeddings_video0001.jsonl"
            )
            embedding_metadata_path.write_text(
                json.dumps(embedding_record) + "\n",
                encoding="utf-8",
            )
            artifact_report_path = (
                output_root
                / "metadata"
                / f"{ARTIFACT_TAG}_artifacts_video0001_validation.json"
            )
            artifact_report_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "source_keyframe_count": 1,
                        "embedding_file_sha256": _sha256_file(embeddings_path),
                        "embedding_metadata_sha256": _sha256_file(
                            embedding_metadata_path
                        ),
                        "embedded_frame_ids_sha256": _frame_ids_sha256(
                            [embedding_record]
                        ),
                        "embedding_lineage": lineage,
                    }
                ),
                encoding="utf-8",
            )

            fake_model = mock.Mock()
            fake_model.config._commit_hash = "commit-test"
            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch("competition.pipeline.choose_device", return_value="cpu"),
                mock.patch(
                    "competition.pipeline.load_siglip2_model_processor",
                    return_value=(fake_model, object()),
                ) as load_model,
                mock.patch("competition.pipeline.encode_keyframes") as encode_mock,
            ):
                embed_command(embed_args)

            load_model.assert_called_once()
            encode_mock.assert_not_called()

            np.save(
                embeddings_path,
                np.asarray([[0.0, 1.0]], dtype=np.float32),
            )
            encoded_records = [embedding_record]
            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch("competition.pipeline.choose_device", return_value="cpu"),
                mock.patch(
                    "competition.pipeline.load_siglip2_model_processor",
                    return_value=(fake_model, object()),
                ) as load_model,
                mock.patch(
                    "competition.pipeline.validate_records",
                    return_value={"valid": True},
                ),
                mock.patch(
                    "competition.pipeline.encode_keyframes",
                    return_value=(vectors, encoded_records, [], {}),
                ) as encode_mock,
            ):
                embed_command(embed_args)

            load_model.assert_called_once()
            encode_mock.assert_called_once()
            rebuilt_report = json.loads(
                artifact_report_path.read_text(encoding="utf-8")
            )
            self.assertEqual(rebuilt_report["status"], "passed")
            self.assertEqual(rebuilt_report["source_keyframe_count"], 1)
            self.assertEqual(rebuilt_report["skipped_count"], 0)

            rebuilt_report["embedding_lineage"][
                "source_keyframe_metadata_sha256"
            ] = "stale-again"
            artifact_report_path.write_text(
                json.dumps(rebuilt_report),
                encoding="utf-8",
            )
            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch("competition.pipeline.choose_device", return_value="cpu"),
                mock.patch(
                    "competition.pipeline.load_siglip2_model_processor",
                    return_value=(fake_model, object()),
                ),
                mock.patch(
                    "competition.pipeline.validate_records",
                    return_value={"valid": True},
                ),
                mock.patch(
                    "competition.pipeline.encode_keyframes",
                    return_value=(
                        np.empty((0, 2), dtype=np.float32),
                        [],
                        [{"skip_reason": "image_load_error"}],
                        {},
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "completeness/contract failed",
                ):
                    embed_command(embed_args)
            partial_embedding_report = json.loads(
                artifact_report_path.read_text(encoding="utf-8")
            )
            self.assertEqual(partial_embedding_report["status"], "partial")
            self.assertEqual(partial_embedding_report["skipped_count"], 1)

    def test_index_rejects_embeddings_from_stale_keyframe_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "output"
            extract_args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                ]
            )
            video = CorpusVideo(
                filename="video0001.mp4",
                relative_path=Path("videos/video0001.mp4"),
                fps=25.0,
                frame_count=250,
            )
            _write_mock_source(extract_args.public_root, video)

            def extract_side_effect(**_kwargs):
                _write_mock_keyframe(output_root, video)
                return _dense_report(extract_args, video)

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=extract_side_effect,
                ),
            ):
                extract_command(extract_args)

            embeddings_path = (
                output_root / "embeddings" / f"{ARTIFACT_TAG}_video0001.npy"
            )
            embeddings_path.parent.mkdir(parents=True)
            np.save(
                embeddings_path,
                np.asarray([[1.0, 0.0]], dtype=np.float32),
            )
            embedding_metadata_path = (
                output_root
                / "metadata"
                / f"{ARTIFACT_TAG}_embeddings_video0001.jsonl"
            )
            embedding_metadata_path.write_text(
                json.dumps(
                    {
                        "embedding_index": 0,
                        "vector_dim": 2,
                        "normalized": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            artifact_report_path = (
                output_root
                / "metadata"
                / f"{ARTIFACT_TAG}_artifacts_video0001_validation.json"
            )
            artifact_report_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "embedding_lineage": {
                            "version": 1,
                            "source_keyframe_metadata_sha256": "stale",
                            "source_extractor_contract_version": 1,
                            "source_extract_config": _competition_extract_config(
                                extract_args
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            index_args = build_parser().parse_args(
                [
                    "index",
                    "--public-root",
                    str(root / "public"),
                    "--output-root",
                    str(output_root),
                ]
            )

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch("competition.pipeline.build_faiss_artifacts") as build_mock,
            ):
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    index_command(index_args)

            build_mock.assert_not_called()

    def test_runtime_index_lineage_rejects_a_stale_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            public_root = root / "public"
            output_root = root / "output"
            extract_args = build_parser().parse_args(
                [
                    "extract",
                    "--public-root",
                    str(public_root),
                    "--output-root",
                    str(output_root),
                ]
            )
            embed_args = build_parser().parse_args(
                [
                    "embed",
                    "--public-root",
                    str(public_root),
                    "--output-root",
                    str(output_root),
                    "--device",
                    "cpu",
                ]
            )
            video = CorpusVideo(
                filename="video0001.mp4",
                relative_path=Path("videos/video0001.mp4"),
                fps=25.0,
                frame_count=250,
            )
            _write_mock_source(public_root, video)

            def extract_side_effect(**_kwargs):
                _write_mock_keyframe(output_root, video)
                return _dense_report(extract_args, video)

            with (
                mock.patch("competition.pipeline.load_corpus", return_value=[video]),
                mock.patch(
                    "competition.pipeline.extract_keyframes_for_video",
                    side_effect=extract_side_effect,
                ),
            ):
                extract_command(extract_args)

            extract_report = json.loads(
                (
                    output_root
                    / "metadata"
                    / "keyframes_video0001_extract_report.json"
                ).read_text(encoding="utf-8")
            )
            lineage = _embedding_lineage(
                embed_args,
                extract_report,
                resolved_device="cpu",
                resolved_model_revision="commit-test",
            )
            embeddings_path = (
                output_root / "embeddings" / f"{ARTIFACT_TAG}_video0001.npy"
            )
            embeddings_path.parent.mkdir(parents=True)
            np.save(
                embeddings_path,
                np.asarray([[1.0, 0.0]], dtype=np.float32),
            )
            embedding_record = {
                "frame_id": "FRAME_video0001_000000001",
                "embedding_index": 0,
                "vector_dim": 2,
                "normalized": True,
                "model_name": embed_args.model_name,
                "model_revision": "commit-test",
            }
            embedding_metadata_path = (
                output_root
                / "metadata"
                / f"{ARTIFACT_TAG}_embeddings_video0001.jsonl"
            )
            embedding_metadata_path.write_text(
                json.dumps(embedding_record) + "\n",
                encoding="utf-8",
            )
            artifact_report_path = (
                output_root
                / "metadata"
                / f"{ARTIFACT_TAG}_artifacts_video0001_validation.json"
            )
            artifact_report_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "source_keyframe_count": 1,
                        "embedding_file_sha256": _sha256_file(embeddings_path),
                        "embedding_metadata_sha256": _sha256_file(
                            embedding_metadata_path
                        ),
                        "embedded_frame_ids_sha256": _frame_ids_sha256(
                            [embedding_record]
                        ),
                        "embedding_lineage": lineage,
                    }
                ),
                encoding="utf-8",
            )
            index_paths = competition_index_paths(output_root)
            index_paths["index"].parent.mkdir(parents=True, exist_ok=True)
            index_paths["index"].write_bytes(b"mock-faiss-index")
            index_paths["index_metadata"].write_text(
                "{}\n",
                encoding="utf-8",
            )
            index_paths["frame_map"].write_text("{}", encoding="utf-8")
            manifest_path = index_paths["manifest"]
            manifest_path.write_text(
                json.dumps(
                    {
                        "competition_index_lineage": {
                            "version": 1,
                            "artifacts": {
                                "index_sha256": _sha256_file(index_paths["index"]),
                                "index_metadata_sha256": _sha256_file(
                                    index_paths["index_metadata"]
                                ),
                                "frame_map_sha256": _sha256_file(
                                    index_paths["frame_map"]
                                ),
                            },
                            "sources": [
                                _embedding_source_lineage(
                                    video,
                                    artifact_report_path,
                                )
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            _require_current_index_lineage(
                corpus=[video],
                public_root=public_root,
                output_root=output_root,
                manifest_path=manifest_path,
            )

            index_paths["index"].write_bytes(b"changed-faiss-index")
            with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
                _require_current_index_lineage(
                    corpus=[video],
                    public_root=public_root,
                    output_root=output_root,
                    manifest_path=manifest_path,
                )
            index_paths["index"].write_bytes(b"mock-faiss-index")

            stale_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            stale_manifest["competition_index_lineage"]["sources"][0][
                "embedding_file_sha256"
            ] = "stale"
            manifest_path.write_text(
                json.dumps(stale_manifest),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "FAISS index is stale"):
                _require_current_index_lineage(
                    corpus=[video],
                    public_root=public_root,
                    output_root=output_root,
                    manifest_path=manifest_path,
                )

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
