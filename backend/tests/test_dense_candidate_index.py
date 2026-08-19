from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.app.pipelines import offline_pipeline as pipeline
from backend.app.services.indexing import build_faiss_index as faiss_builder
from backend.app.services.retrieval import dense_candidate_index as dense_module
from backend.app.services.retrieval.dense_candidate_index import (
    DENSE_ARCHITECTURE_VERSION,
    DENSE_ARTIFACT_ROLE,
    DENSE_LAYER,
    DENSE_MANIFEST_SCHEMA_VERSION,
    DENSE_MODALITIES,
    DENSE_RECORD_ARTIFACT_ROLE,
    DenseCandidateIndexConfig,
    FaissDenseCandidateIndex,
)


@unittest.skipIf(faiss_builder.faiss is None, "faiss is not installed")
class DenseCandidateIndexIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.offline_config = pipeline.OfflinePipelineConfig(
            output_dir=self.root / "data",
            device="cpu",
            shot_device="cpu",
            bge_enabled=False,
            resume=False,
        )
        self.video_id = "video_dense"
        self.video_path = self.root / f"{self.video_id}.mp4"
        self.video_path.write_bytes(b"unit-test-video")
        self.video_paths = pipeline.PerVideoPaths.from_config(
            self.video_id,
            self.offline_config,
        )
        self._write_dense_sources()
        self.video = pipeline.VideoArtifacts(
            video_id=self.video_id,
            video_path=self.video_path,
            paths=self.video_paths,
            selected_count=1,
            dense_candidate_count=3,
            validation={"status": "passed"},
        )
        self.corpus_paths = pipeline.CorpusPaths.from_config(self.offline_config)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def _write_dense_sources(self) -> None:
        vectors = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        self.video_paths.dense_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.video_paths.dense_embeddings, vectors)
        embeddings: list[dict] = []
        ledger: list[dict] = []
        for row in range(3):
            candidate_id = f"{self.video_id}:C{row:04d}"
            frame_id = f"{self.video_id}:F{row:04d}"
            shot_number = 0 if row < 2 else 1
            clip_id = f"SHOT_{self.video_id}_{shot_number:06d}"
            common = {
                "candidate_id": candidate_id,
                "frame_id": frame_id,
                "video_id": self.video_id,
                "shot_id": clip_id,
                "segment_id": clip_id,
                "timestamp": float(row),
                "frame_index": row * 25,
                "keyframe_path": f"dense_keyframes/{self.video_id}/{frame_id}.jpg",
            }
            embeddings.append(
                {
                    **common,
                    "embedding_id": f"EMB_{frame_id}",
                    "embedding_index": row,
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
            ledger.append(
                {
                    **common,
                    "selected": row == 0,
                    "feature_protected_event_ids": [],
                    "available_modalities": ["siglip2"],
                    "selection_rank": 1 if row == 0 else None,
                    "selection_phase": "visual_temporal" if row == 0 else None,
                    "selection_reasons": ["unit_test"] if row == 0 else [],
                }
            )
        self._write_jsonl(self.video_paths.dense_embedding_metadata, embeddings)
        self._write_jsonl(self.video_paths.candidate_ledger, ledger)
        self._write_jsonl(self.video_paths.selected_metadata, [ledger[0]])

    def _build(self) -> FaissDenseCandidateIndex:
        report = pipeline._build_dense_corpus_index(
            self.corpus_paths,
            (self.video,),
            {"contract_sha256": "dense-unit-test-contract"},
            self.offline_config,
        )
        self.assertEqual(report["vector_count"], 3)
        self.assertEqual(report["clip_count"], 2)
        return FaissDenseCandidateIndex(
            DenseCandidateIndexConfig(
                index_path=self.corpus_paths.dense_index,
                metadata_path=self.corpus_paths.dense_metadata,
                frame_map_path=self.corpus_paths.dense_frame_map,
                manifest_path=self.corpus_paths.dense_manifest,
                report_path=self.corpus_paths.dense_report,
            )
        )

    def test_offline_dense_sources_build_and_load_production_index(self) -> None:
        dense = self._build()

        self.assertEqual(len(dense.records), 3)
        self.assertEqual(dense.vectors.shape, (3, 2))
        self.assertEqual(
            dense.rows_by_clip,
            {
                (self.video_id, f"SHOT_{self.video_id}_000000"): (0, 1),
                (self.video_id, f"SHOT_{self.video_id}_000001"): (2,),
            },
        )
        self.assertEqual(
            dense.row_by_frame,
            {
                (self.video_id, f"{self.video_id}:F0000"): 0,
                (self.video_id, f"{self.video_id}:F0001"): 1,
                (self.video_id, f"{self.video_id}:F0002"): 2,
            },
        )
        self.assertEqual(dense.layer, DENSE_LAYER)
        self.assertEqual(dense.modalities, DENSE_MODALITIES)
        self.assertEqual(
            dense.records[1]["artifact_role"],
            DENSE_RECORD_ARTIFACT_ROLE,
        )
        self.assertEqual(dense.records[1]["layer"], DENSE_LAYER)
        for field_name in ("caption", "ocr_text", "objects"):
            self.assertNotIn(field_name, dense.records[1])
        self.assertEqual(dense.search(np.asarray([0.0, 2.0]), top_k=2)[0][0], 1)

        manifest = json.loads(self.corpus_paths.dense_manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], DENSE_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest["architecture_version"], DENSE_ARCHITECTURE_VERSION)
        self.assertEqual(manifest["artifact_role"], DENSE_ARTIFACT_ROLE)
        self.assertEqual(manifest["layer"], DENSE_LAYER)
        self.assertEqual(manifest["modalities"], list(DENSE_MODALITIES))

    def test_loader_rejects_metadata_tamper(self) -> None:
        self._build()
        with self.corpus_paths.dense_metadata.open("a", encoding="utf-8") as handle:
            handle.write("\n")

        with self.assertRaisesRegex(ValueError, "metadata checksum"):
            FaissDenseCandidateIndex(
                DenseCandidateIndexConfig(
                    index_path=self.corpus_paths.dense_index,
                    metadata_path=self.corpus_paths.dense_metadata,
                    frame_map_path=self.corpus_paths.dense_frame_map,
                    manifest_path=self.corpus_paths.dense_manifest,
                    report_path=self.corpus_paths.dense_report,
                )
            )

    def test_loader_rejects_manifest_count_drift(self) -> None:
        self._build()
        manifest = json.loads(self.corpus_paths.dense_manifest.read_text(encoding="utf-8"))
        manifest["vector_count"] = 4
        self.corpus_paths.dense_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "counts do not match"):
            FaissDenseCandidateIndex(
                DenseCandidateIndexConfig(
                    index_path=self.corpus_paths.dense_index,
                    metadata_path=self.corpus_paths.dense_metadata,
                    frame_map_path=self.corpus_paths.dense_frame_map,
                    manifest_path=self.corpus_paths.dense_manifest,
                    report_path=self.corpus_paths.dense_report,
                )
            )

    def test_loader_rejects_incompatible_architecture_version(self) -> None:
        self._build()
        manifest = json.loads(self.corpus_paths.dense_manifest.read_text(encoding="utf-8"))
        manifest["architecture_version"] = "legacy_dense_multimodal_v1"
        self.corpus_paths.dense_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "architecture version"):
            FaissDenseCandidateIndex(
                DenseCandidateIndexConfig(
                    index_path=self.corpus_paths.dense_index,
                    metadata_path=self.corpus_paths.dense_metadata,
                    frame_map_path=self.corpus_paths.dense_frame_map,
                    manifest_path=self.corpus_paths.dense_manifest,
                    report_path=self.corpus_paths.dense_report,
                )
            )

    def test_loader_rejects_stale_dense_manifest_schema(self) -> None:
        self._build()
        manifest = json.loads(self.corpus_paths.dense_manifest.read_text(encoding="utf-8"))
        manifest["schema_version"] = "1.2"
        self.corpus_paths.dense_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported.*schema"):
            FaissDenseCandidateIndex(
                DenseCandidateIndexConfig(
                    index_path=self.corpus_paths.dense_index,
                    metadata_path=self.corpus_paths.dense_metadata,
                    frame_map_path=self.corpus_paths.dense_frame_map,
                    manifest_path=self.corpus_paths.dense_manifest,
                    report_path=self.corpus_paths.dense_report,
                )
            )


class DenseCandidateIdentityValidationTests(unittest.TestCase):
    @staticmethod
    def _record(candidate_id: str, frame_id: str) -> dict:
        return {
            "faiss_index": 0,
            "artifact_role": DENSE_RECORD_ARTIFACT_ROLE,
            "layer": DENSE_LAYER,
            "candidate_id": candidate_id,
            "frame_id": frame_id,
            "video_id": "V1",
            "segment_id": "S1",
            "timestamp": 1.0,
            "frame_index": 25,
            "keyframe_path": "dense_keyframes/V1/F0.jpg",
            "vector_dim": 2,
            "normalized": True,
        }

    def test_dense_contract_rejects_empty_and_duplicate_frame_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_id"):
            dense_module._validate_records(
                [self._record("C0", "")],
                vector_dim=2,
            )

        first = self._record("C0", "F0")
        second = self._record("C1", "F0")
        second["faiss_index"] = 1
        with self.assertRaisesRegex(ValueError, "Duplicate dense frame identity"):
            dense_module._validate_records([first, second], vector_dim=2)

    def test_dense_contract_requires_visual_layer_identity_and_location(self) -> None:
        required = {
            "artifact_role": "artifact_role",
            "layer": "layer",
            "candidate_id": "candidate_id",
            "video_id": "video_id",
            "frame_id": "frame_id",
            "segment_id": "clip id",
            "timestamp": "timestamp",
            "frame_index": "frame_index",
            "keyframe_path": "keyframe_path",
        }
        for field_name, message in required.items():
            with self.subTest(field_name=field_name):
                record = self._record("C0", "F0")
                record.pop(field_name)
                with self.assertRaisesRegex(ValueError, message):
                    dense_module._validate_records([record], vector_dim=2)

        for field_name, invalid_value in (
            ("timestamp", float("nan")),
            ("timestamp", -0.1),
            ("frame_index", True),
            ("frame_index", -1),
        ):
            with self.subTest(field_name=field_name, value=invalid_value):
                record = self._record("C0", "F0")
                record[field_name] = invalid_value
                with self.assertRaisesRegex(ValueError, field_name):
                    dense_module._validate_records([record], vector_dim=2)

    def test_dense_contract_rejects_selected_semantic_keys_even_when_empty(self) -> None:
        for field_name, value in (
            ("caption", ""),
            ("ocr_text", ""),
            ("objects", []),
        ):
            with self.subTest(field_name=field_name):
                record = self._record("C0", "F0")
                record[field_name] = value
                with self.assertRaisesRegex(ValueError, field_name):
                    dense_module._validate_records([record], vector_dim=2)

    def test_dense_manifest_requires_exact_visual_only_layer_contract(self) -> None:
        dense_module._validate_layer_contract(
            {"layer": DENSE_LAYER, "modalities": list(DENSE_MODALITIES)}
        )
        for manifest, message in (
            ({"modalities": list(DENSE_MODALITIES)}, "layer"),
            ({"layer": DENSE_LAYER}, "modalities"),
            (
                {"layer": "selected_multimodal", "modalities": list(DENSE_MODALITIES)},
                "layer",
            ),
            (
                {"layer": DENSE_LAYER, "modalities": ["siglip2", "caption"]},
                "visual-only modalities",
            ),
        ):
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(ValueError, message):
                    dense_module._validate_layer_contract(manifest)


if __name__ == "__main__":
    unittest.main()
