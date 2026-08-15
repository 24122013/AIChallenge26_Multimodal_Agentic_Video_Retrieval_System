from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.indexing.keyframe_candidates import (
    REASON_VIDEO_END,
    REASON_VIDEO_START,
    generate_keyframe_candidates,
)
from backend.app.services.indexing.keyframe_selection import (
    PHASE_TEMPORAL_REPAIR,
    ProtectedEvent,
    SelectionCandidate,
    SelectionConfig,
    select_keyframes,
)
from backend.app.services.retrieval.advanced_rerank import AdvancedRankedFrame
from backend.app.services.retrieval.cses import CSESConfig, CSESSelection, select_cses
from backend.app.services.retrieval.query_plan import build_query_plan
from backend.app.services.retrieval.rank_fusion import weighted_rrf
from backend.app.services.retrieval.vlm_reranker import rerank_with_vlm
from competition.dense_index import build_dense_index, validate_dense_index
from competition.ensemble_submissions import ensemble_submissions
from competition.optimize_retrieval import load_experiment_config
from competition.retrieval_metrics import evaluate_submission


class _Shot:
    def __init__(self, shot_index: int, start_frame: int, end_frame: int) -> None:
        self.shot_index = shot_index
        self.start_frame = start_frame
        self.end_frame = end_frame


def _selection_candidate(
    candidate_id: str,
    timestamp: float,
    *,
    importance: float = 0.5,
    duplicate_group: str | None = None,
) -> SelectionCandidate:
    return SelectionCandidate(
        candidate_id=candidate_id,
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        shot_index=0,
        importance_score=importance,
        semantic_embedding=(1.0, 0.0),
        duplicate_group=duplicate_group,
    )


class QueryPlanTest(unittest.TestCase):
    def test_typo_temporal_and_modality_hints_are_deterministic(self) -> None:
        plan = build_query_plan(
            'Show the resturant sign "OPEN", then a person says hello'
        )
        self.assertIn("restaurant", plan.normalized_query)
        self.assertEqual(plan.profile, "temporal")
        self.assertEqual(plan.temporal_relation, "then")
        self.assertEqual(len(plan.temporal_events), 2)
        self.assertEqual(set(plan.modality_hints), {"ocr", "objects"})
        self.assertEqual(plan.query_for("ocr"), "OPEN")

    def test_explicit_profile_overrides_inference(self) -> None:
        plan = build_query_plan("find all scenes with cars", profile="kis")
        self.assertEqual(plan.profile, "kis")
        self.assertEqual(plan.profile_source, "explicit")


class RankFusionAndCSESTest(unittest.TestCase):
    def test_rrf_uses_rank_not_incomparable_raw_scores(self) -> None:
        first = RetrievalResult("v", "f1", 1.0, 99.0, shot_id="s1")
        second = RetrievalResult("v", "f2", 2.0, 0.01, shot_id="s2")
        plan = build_query_plan("a car")
        fused = weighted_rrf(
            {"visual": [first, second], "caption": [second, first]},
            plan=plan,
            k=60,
            weights={"visual": 1.0, "caption": 1.0},
        )
        self.assertEqual({item.result.frame_id for item in fused}, {"f1", "f2"})
        self.assertAlmostEqual(fused[0].rrf_score, fused[1].rrf_score)

    def test_rrf_accepts_bge_dense_text_as_an_additional_modality(self) -> None:
        visual = RetrievalResult("v", "visual", 1.0, 0.9, shot_id="s1")
        semantic = RetrievalResult(
            "v",
            "semantic",
            2.0,
            0.8,
            shot_id="s2",
            modality_scores={"dense_text": 0.8},
        )

        fused = weighted_rrf(
            {"visual": [visual], "dense_text": [semantic]},
            plan=build_query_plan("người đang cầm điện thoại"),
        )

        by_frame = {item.result.frame_id: item for item in fused}
        self.assertIn("dense_text", by_frame["semantic"].modality_ranks)
        self.assertGreater(
            by_frame["semantic"].modality_contributions["dense_text"],
            0.0,
        )

    def test_cses_is_deterministic_bounded_and_preserves_events(self) -> None:
        records = [
            {"candidate_id": "a", "timestamp": 0.0, "protected_event_ids": ["e1"]},
            {"candidate_id": "b", "timestamp": 1.0, "protected_event_ids": []},
            {"candidate_id": "c", "timestamp": 2.0, "protected_event_ids": ["e2"]},
            {"candidate_id": "d", "timestamp": 3.0, "protected_event_ids": []},
        ]
        vectors = np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        kwargs = dict(
            rows=[0, 1, 2, 3],
            records=records,
            vectors=vectors,
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
            profile="avs",
            config=CSESConfig(max_frames=3, similarity_threshold=0.92),
        )
        first = select_cses(**kwargs)
        second = select_cses(**kwargs)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 3)
        preserved = {event for item in first for event in item.preserved_event_ids}
        self.assertEqual(preserved, {"e1", "e2"})


class OfflineSelectorV2Test(unittest.TestCase):
    def test_exact_endpoints_exist_and_are_protected(self) -> None:
        generated = generate_keyframe_candidates(
            "v",
            [_Shot(0, 100, 102)],
            fps=10.0,
            include_video_endpoints=True,
        )
        self.assertEqual([item.frame_index for item in generated], [100, 101, 102])
        values = [SelectionCandidate.from_generated_candidate(item) for item in generated]
        result = select_keyframes(
            values,
            (),
            video_duration=10.3,
            config=SelectionConfig(
                max_gap_seconds=20.0,
                protect_each_shot=False,
                protect_video_endpoints=True,
                enable_event_aware_dedup=True,
            ),
        )
        selected = {item.candidate.candidate_id: item for item in result.selected}
        start = next(item for item in generated if REASON_VIDEO_START in item.reasons)
        end = next(item for item in generated if REASON_VIDEO_END in item.reasons)
        self.assertIn(start.candidate_id, selected)
        self.assertIn(end.candidate_id, selected)

    def test_protected_duplicate_override_and_repair_after_dedup(self) -> None:
        protected_values = (
            _selection_candidate("a", 1.0, duplicate_group="same"),
            _selection_candidate("b", 1.1, duplicate_group="same"),
        )
        protected = select_keyframes(
            protected_values,
            (
                ProtectedEvent("e1", "transition", ("a",)),
                ProtectedEvent("e2", "transition", ("b",)),
            ),
            video_duration=2.0,
            config=SelectionConfig(
                max_gap_seconds=5.0,
                target_keyframes=1,
                protect_each_shot=False,
                enable_event_aware_dedup=True,
            ),
        )
        self.assertEqual({item.candidate.candidate_id for item in protected.selected}, {"a", "b"})
        self.assertTrue(
            any("protected_override" in item.selection_reasons for item in protected.selected)
        )

        repaired = select_keyframes(
            (
                _selection_candidate("left", 1.0, importance=1.0, duplicate_group="same"),
                _selection_candidate("right", 3.0, importance=0.9, duplicate_group="same"),
            ),
            (),
            video_duration=4.0,
            config=SelectionConfig(
                max_gap_seconds=2.0,
                target_keyframes=2,
                protect_each_shot=False,
                enable_event_aware_dedup=True,
            ),
        )
        by_id = {item.candidate.candidate_id: item for item in repaired.selected}
        self.assertEqual(by_id["right"].selection_phase, PHASE_TEMPORAL_REPAIR)
        self.assertEqual(repaired.dedup_removed[0]["override_reason"], "temporal_repair")


class VLMFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "frame.jpg"
        self.image.write_bytes(b"not-decoded-by-fake-runner")
        selection = CSESSelection(0, 1, 1.0, 1.0, 0.0, 0.0, ())
        self.candidate = AdvancedRankedFrame(
            0,
            {"candidate_id": "c", "timestamp": 0.0},
            0.8,
            {"dense_visual": 0.8},
            selection,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_optional_timeout_falls_back_and_required_fails(self) -> None:
        def timeout(_query: str, _path: Path):
            raise TimeoutError("slow")

        ranked, report = rerank_with_vlm(
            [self.candidate],
            query="q",
            mode="optional",
            cache_root=self.root / "cache",
            image_resolver=lambda _item: self.image,
            runner=timeout,
        )
        self.assertEqual(ranked, [self.candidate])
        self.assertEqual(report.status, "fallback")
        with self.assertRaisesRegex(RuntimeError, "Required VLM"):
            rerank_with_vlm(
                [self.candidate],
                query="q",
                mode="required",
                cache_root=self.root / "cache-required",
                image_resolver=lambda _item: self.image,
                runner=timeout,
            )

    def test_vlm_semantic_query_is_original_and_candidate_image_is_evidence(self) -> None:
        original = "a red bus next to two cars"
        calls: list[tuple[str, Path]] = []

        def record(query: str, image_path: Path):
            calls.append((query, image_path))
            return {"score": 0.9}

        ranked, report = rerank_with_vlm(
            [self.candidate],
            query=original,
            mode="optional",
            cache_root=self.root / "cache-original-query",
            image_resolver=lambda _item: self.image,
            runner=record,
        )

        self.assertEqual(calls, [(original, self.image)])
        self.assertEqual(report.status, "passed")
        self.assertEqual(ranked[0].breakdown["vlm"], 0.9)


class DenseIndexContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_output = self.root / "competition"
        self.workspace_root = self.source_output / "work" / "keyframe_v3"
        self.run_root = self.root / "runs" / "r1"
        video_id = "video1"
        pool_id = "pool1"
        workspace = self.workspace_root / video_id / pool_id
        workspace.mkdir(parents=True)
        image = workspace / "frame.jpg"
        image.write_bytes(b"image")
        video = self.root / "video1.mp4"
        video.write_bytes(b"video")
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.save(workspace / "siglip2.npy", vectors)
        metadata = []
        for row in range(2):
            metadata.append(
                {
                    "embedding_index": row,
                    "candidate_id": f"c{row}",
                    "frame_id": f"f{row}",
                    "video_id": video_id,
                    "shot_id": "s1",
                    "segment_id": "s1",
                    "shot_index": 0,
                    "shot_start": 0.0,
                    "shot_end": 2.0,
                    "timestamp": float(row),
                    "frame_index": row,
                    "keyframe_path": image.as_posix(),
                    "source_video_path": video.as_posix(),
                }
            )
        self._write_jsonl(workspace / "siglip2_metadata.jsonl", metadata)
        feature = {
            "status": "passed",
            "hard_feature_complete": True,
            "feature_config": {
                "siglip2": {
                    "model_name": "siglip-test",
                    "resolved_model_revision": "revision-1",
                }
            },
        }
        (workspace / "feature_manifest.json").write_text(json.dumps(feature), encoding="utf-8")
        for name, records in {
            "captions.jsonl": [{"candidate_id": "c0", "caption": "cat"}],
            "ocr.jsonl": [{"candidate_id": "c0", "ocr_text": "OPEN"}],
            "objects.jsonl": [{"candidate_id": "c0", "object_classes": ["cat"]}],
            "candidate_scores.jsonl": [
                {"candidate_id": "c0", "importance_score": 0.8, "selected": True},
                {"candidate_id": "c1", "importance_score": None, "semantic_novelty": None},
            ],
            "protected_events.jsonl": [
                {"event_id": "e1", "candidate_ids": ["c0"]}
            ],
        }.items():
            self._write_jsonl(workspace / name, records)
        feature_sha = hashlib.sha256((workspace / "feature_manifest.json").read_bytes()).hexdigest()
        metadata_root = self.source_output / "metadata"
        metadata_root.mkdir(parents=True)
        phase3 = {
            "video_id": video_id,
            "candidate_pool_run_id": pool_id,
            "feature_manifest_sha256": feature_sha,
            "status": "passed",
            "degraded": False,
            "candidate_count": 2,
            "adapter_config": {"x": 1},
            "selection_config": {"max_gap_seconds": 5.0},
            "feature_config": feature["feature_config"],
        }
        (metadata_root / "keyframes_video1_phase3_manifest.json").write_text(
            json.dumps(phase3), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_build_validate_move_and_tamper_detection(self) -> None:
        manifest = build_dense_index(
            run_root=self.run_root,
            source_workspace=self.workspace_root,
            source_output_root=self.source_output,
        )
        self.assertEqual(manifest["candidate_count"], 2)
        report = validate_dense_index(self.run_root)
        self.assertEqual(report["faiss_ntotal"], 2)

        moved = self.root / "moved"
        shutil.copytree(self.root / "runs", moved / "runs")
        shutil.copytree(self.source_output, moved / "competition")
        # References remain relative when the run and source tree move together.
        moved_run = moved / "runs" / "r1"
        moved_manifest = json.loads(
            (moved_run / "dense" / "dense_manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(Path(moved_manifest["artifacts"]["vectors"]["path"]).is_absolute())
        self.assertEqual(validate_dense_index(moved_run)["candidate_count"], 2)

        vectors_path = self.run_root / "dense" / "dense_vectors.npy"
        vectors_path.write_bytes(vectors_path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_dense_index(self.run_root, verify_sources=False)


class EvaluationContractTest(unittest.TestCase):
    def test_vkis_tolerance_and_temporal_chain_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            submission = root / "submission.csv"
            submission.write_text(
                "query_id,answer_001,answer_002\n"
                'q1,"v.mp4,12","v.mp4,25"\n'
                'q2,"v.mp4,90","v.mp4,150"\n',
                encoding="utf-8",
            )
            labels = root / "labels.jsonl"
            labels.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "task": "VKIS",
                        "relevant": [{"video": "v.mp4", "frame": 20, "tolerance": 12}],
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "query_id": "q2",
                        "task": "TKIS",
                        "relevant": [
                            {"video": "v.mp4", "frame": 100, "tolerance": 15},
                            {"video": "v.mp4", "frame": 150, "tolerance": 0},
                        ],
                        "temporal_chain": [
                            {"frame": 100, "tolerance": 15},
                            {"frame": 150, "tolerance": 0},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = evaluate_submission(submission_path=submission, labels_path=labels)
            self.assertEqual(report["by_task"]["VKIS"]["VKIS_Hit@100"], 1.0)
            self.assertEqual(report["by_task"]["TKIS"]["temporal_Hit@20"], 1.0)

    def test_experiment_yaml_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.yaml"
            path.write_text(
                "version: 1\nversion: 1\nexperiments: []\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
                load_experiment_config(path)


class SubmissionEnsembleTest(unittest.TestCase):
    @mock.patch("competition.ensemble_submissions.validate_submission")
    def test_weighted_rrf_is_atomic_unique_and_has_checksum_manifest(
        self,
        validate_mock: mock.Mock,
    ) -> None:
        validate_mock.return_value = {
            "status": "passed",
            "query_count": 1,
            "answers_per_query": 100,
            "exact_duplicate_answers": 0,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            columns = [f"answer_{index:03d}" for index in range(1, 101)]
            header = "query_id," + ",".join(columns) + "\n"
            first_answers = [f'"v.mp4,{index}"' for index in range(100)]
            second_answers = [f'"v.mp4,{index}"' for index in range(99, -1, -1)]
            first = root / "first.csv"
            second = root / "second.csv"
            first.write_text(header + "q1," + ",".join(first_answers) + "\n", encoding="utf-8")
            second.write_text(header + "q1," + ",".join(second_answers) + "\n", encoding="utf-8")
            output = root / "ensemble.csv"

            report = ensemble_submissions(
                submissions=[first, second],
                weights=[2.0, 1.0],
                output_path=output,
                public_root=root,
            )

            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".csv.manifest.json").is_file())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(validate_mock.call_count, 3)
            values = output.read_text(encoding="utf-8").splitlines()[1].split(",")
            self.assertEqual(values[0], "q1")
            self.assertIn("v.mp4", values[1])


if __name__ == "__main__":
    unittest.main()
