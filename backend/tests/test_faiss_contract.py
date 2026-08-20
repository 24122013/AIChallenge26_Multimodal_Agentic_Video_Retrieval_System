from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import backend.app.services.indexing.build_faiss_index as faiss_builder


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def make_contract(model_name: str, vector_dim: int) -> dict:
    return {
        "model_family": "siglip2" if "siglip2" in model_name else "openclip",
        "model_name": model_name,
        "model_revision": "revision-1",
        "processor_name": model_name,
        "vector_dim": vector_dim,
        "input_resolution": 384,
        "normalized": True,
        "similarity": "cosine",
        "output_dtype": "float32",
    }


def make_records(video_id: str, count: int, contract: dict) -> list[dict]:
    return [
        {
            "embedding_id": f"EMB_FRAME_{video_id}_{offset:06d}",
            "frame_id": f"FRAME_{video_id}_{offset:06d}",
            "video_id": video_id,
            "shot_id": f"SHOT_{video_id}_{offset:06d}",
            "segment_id": f"SEG_{video_id}_{offset:06d}",
            "timestamp": float(offset),
            "keyframe_path": f"data/keyframes/{video_id}/{offset:06d}.jpg",
            "embedding_index": offset,
            "candidate_index": offset + 1,
            "candidate_id": f"CANDIDATE_{video_id}_{offset:09d}",
            "candidate_reasons": ["shot_boundary_start"],
            "keyframe_strategy": "dense_coverage",
            "selection_phase": "protected",
            "selection_rank": offset + 1,
            "selection_reasons": ["protected_event_cover"],
            "covered_event_ids": [f"__shot__:{offset}"],
            "selection_score": 0.91,
            "protected": True,
            "coverage_added": False,
            "importance_score": 0.81,
            "semantic_novelty": 0.64,
            "component_scores": {"ocr": 0.9},
            "available_modalities": ["ocr"],
            "protected_event_ids": [f"__shot__:{offset}"],
            "selection_provenance": {"strategy": "multimodal_coverage"},
            **contract,
        }
        for offset in range(count)
    ]


class FaissContractValidationTest(unittest.TestCase):
    def create_source(
        self,
        root: Path,
        video_id: str,
        vectors: np.ndarray,
        contract: dict,
    ) -> tuple[Path, Path, str]:
        embeddings_path = root / f"siglip2_so400m_patch16_384_{video_id}.npy"
        metadata_path = root / f"siglip2_so400m_patch16_384_embeddings_{video_id}.jsonl"
        np.save(embeddings_path, vectors.astype(np.float32))
        write_jsonl(metadata_path, make_records(video_id, len(vectors), contract))
        return embeddings_path, metadata_path, video_id

    def test_builder_rejects_mixed_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vectors = np.eye(2, dtype=np.float32)
            first = self.create_source(
                root,
                "L01_V001",
                vectors,
                make_contract("google/siglip2-so400m-patch16-384", 2),
            )
            second = self.create_source(
                root,
                "L01_V002",
                vectors,
                make_contract("openclip_ViT-B-16", 2),
            )
            with self.assertRaisesRegex(ValueError, "refusing to mix sources"):
                faiss_builder.build_faiss_artifacts(
                    [first, second],
                    root / "index.faiss",
                    root / "index.jsonl",
                    root / "frame_map.json",
                    root / "manifest.json",
                    root / "report.json",
                )

    def test_builder_rejects_different_vector_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = self.create_source(
                root,
                "L01_V001",
                np.eye(2, dtype=np.float32),
                make_contract("google/siglip2-so400m-patch16-384", 2),
            )
            second = self.create_source(
                root,
                "L01_V002",
                np.eye(3, dtype=np.float32),
                make_contract("google/siglip2-so400m-patch16-384", 3),
            )
            with self.assertRaisesRegex(ValueError, "vector_dim"):
                faiss_builder.build_faiss_artifacts(
                    [first, second],
                    root / "index.faiss",
                    root / "index.jsonl",
                    root / "frame_map.json",
                    root / "manifest.json",
                    root / "report.json",
                )

    def test_builder_rejects_metadata_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.create_source(
                root,
                "L01_V001",
                np.eye(2, dtype=np.float32),
                make_contract("google/siglip2-so400m-patch16-384", 3),
            )
            with self.assertRaisesRegex(ValueError, "shape\\[1\\]"):
                faiss_builder.validate_embedding_source(*source)

    @unittest.skipIf(faiss_builder.faiss is None, "faiss is not installed")
    def test_index_frame_map_and_manifest_share_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contract = make_contract("google/siglip2-so400m-patch16-384", 3)
            source = self.create_source(
                root,
                "L01_V001",
                np.eye(3, dtype=np.float32),
                contract,
            )
            result = faiss_builder.build_faiss_artifacts(
                [source],
                root / "index.faiss",
                root / "index.jsonl",
                root / "frame_map.json",
                root / "manifest.json",
                root / "report.json",
            )

            self.assertEqual(result["index"].ntotal, 3)
            self.assertEqual(list(result["frame_map"]), ["0", "1", "2"])
            self.assertEqual(
                [record["embedding_index"] for record in result["frame_map"].values()],
                [0, 1, 2],
            )
            self.assertEqual(result["manifest"]["schema_version"], "1.2")
            self.assertEqual(result["manifest"]["encoder"], contract)
            self.assertEqual(result["manifest"]["vector_count"], 3)
            self.assertEqual(result["manifest"]["metadata_record_count"], 3)
            self.assertTrue(
                all(
                    record["model_name"] == contract["model_name"]
                    and record["model_revision"] == contract["model_revision"]
                    and record["vector_dim"] == contract["vector_dim"]
                    for record in result["frame_map"].values()
                )
            )
            self.assertEqual(
                [
                    record["candidate_id"]
                    for record in result["frame_map"].values()
                ],
                [
                    "CANDIDATE_L01_V001_000000000",
                    "CANDIDATE_L01_V001_000000001",
                    "CANDIDATE_L01_V001_000000002",
                ],
            )
            self.assertTrue(
                all(
                    record["selection_phase"] == "protected"
                    and record["protected"] is True
                    for record in result["frame_map"].values()
                )
            )
            self.assertEqual(
                [
                    record["candidate_index"]
                    for record in result["frame_map"].values()
                ],
                [1, 2, 3],
            )
            self.assertTrue(
                all(
                    record["importance_score"] == 0.81
                    and record["semantic_novelty"] == 0.64
                    and record["component_scores"] == {"ocr": 0.9}
                    for record in result["frame_map"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
