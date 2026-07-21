"""Tests cho các file indexing P2 mới: vector_db, neighbor_index, extract_segments,
index_manager, embedding_factory (phần không cần torch/faiss)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.app.services.indexing.embedding_factory import (
    MODEL_REGISTRY,
    resolve_model_spec,
)
from backend.app.services.indexing.extract_segments import segments_from_keyframes
from backend.app.services.indexing.index_manager import build_index
from backend.app.services.indexing.neighbor_index import NeighborIndex, build_neighbor_index
from backend.app.services.indexing.vector_db import create_vector_db


def _keyframe_records(video_id: str = "L01_V001", n: int = 4) -> list[dict]:
    recs = []
    for i in range(1, n + 1):
        shot = (i + 1) // 2
        recs.append(
            {
                "frame_id": f"FRAME_{video_id}_{i:06d}",
                "video_id": video_id,
                "shot_id": f"SHOT_{video_id}_{shot:06d}",
                "segment_id": f"SEG_{video_id}_{shot:06d}",
                "shot_start": round((shot - 1) * 4.0, 3),
                "shot_end": round(shot * 4.0, 3),
                "timestamp": round(i * 1.5, 3),
                "timestamp_source": "video_fps",
                "timestamp_confidence": 1.0,
                "frame_index": i * 30,
                "keyframe_path": f"data/keyframes/{video_id}/{i}.jpg",
                "embedding_id": f"EMB_FRAME_{video_id}_{i:06d}",
                "embedding_index": i - 1,
            }
        )
    return recs


class VectorDBTest(unittest.TestCase):
    def test_numpy_ip_search_ranks_by_cosine(self) -> None:
        db = create_vector_db("numpy", metric="ip")
        db.add(np.eye(4, dtype="float32"))
        scores, idx = db.search(np.array([[1, 0, 0, 0]], dtype="float32"), top_k=2)
        self.assertEqual(idx[0][0], 0)
        self.assertAlmostEqual(float(scores[0][0]), 1.0, places=5)
        self.assertEqual(db.ntotal, 4)

    def test_empty_search_raises(self) -> None:
        db = create_vector_db("numpy")
        with self.assertRaises(ValueError):
            db.search(np.array([[1.0, 0.0]], dtype="float32"), top_k=1)

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = create_vector_db("numpy", metric="ip")
            db.add(np.eye(3, dtype="float32"))
            path = Path(td) / "vecs"
            db.save(path)
            from backend.app.services.indexing.vector_db import NumpyVectorDB

            loaded = NumpyVectorDB.load(path, metric="ip")
            self.assertEqual(loaded.ntotal, 3)


class NeighborIndexTest(unittest.TestCase):
    def test_same_shot_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            recs = _keyframe_records(n=4)
            frame_map = {str(i): {**r, "faiss_index": i} for i, r in enumerate(recs)}
            fm = tmp / "frame_map.json"
            fm.write_text(json.dumps(frame_map))

            out = tmp / "neighbor.json"
            report = build_neighbor_index(fm, out, max_neighbors=3, strategy="same_shot")
            self.assertEqual(report["total_frames"], 4)

            ni = NeighborIndex.load(out)
            # frame 0 và 1 cùng SHOT_...000001 -> là neighbor của nhau
            self.assertEqual(ni.neighbors(0), [1])
            self.assertEqual(ni.neighbors(2), [3])


class ExtractSegmentsTest(unittest.TestCase):
    def test_group_by_shot(self) -> None:
        segs = segments_from_keyframes(_keyframe_records(n=4))
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["segment_id"], "SEG_L01_V001_000001")
        self.assertEqual(segs[0]["start_time"], 0.0)
        self.assertEqual(segs[0]["end_time"], 4.0)

    def test_window_fallback_without_shot(self) -> None:
        recs = [
            {"video_id": "V", "timestamp": t}
            for t in (0.0, 1.0, 2.0, 10.0, 11.0)
        ]
        segs = segments_from_keyframes(recs, window_sec=5.0)
        # 0-2 rồi 10-11 -> 2 segment
        self.assertEqual(len(segs), 2)


class IndexManagerTest(unittest.TestCase):
    def test_build_index_numpy_backend(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            recs = _keyframe_records(n=4)
            (tmp / "openclip_vit_b16_L01_V001.npy")
            np.save(tmp / "openclip_vit_b16_L01_V001.npy", np.random.rand(4, 8).astype("float32"))
            meta = tmp / "openclip_vit_b16_embeddings_L01_V001.jsonl"
            meta.write_text("\n".join(json.dumps(r) for r in recs))

            report = build_index(
                embeddings_glob=str(tmp / "openclip_vit_b16_*.npy"),
                embedding_metadata_template=str(
                    tmp / "openclip_vit_b16_embeddings_{video_id}.jsonl"
                ),
                index_path=str(tmp / "idx"),
                frame_map_path=str(tmp / "frame_map.json"),
                metric="ip",
                backend="numpy",
                neighbor_index_path=str(tmp / "neighbor.json"),
            )
            self.assertEqual(report["ntotal"], 4)
            self.assertEqual(report["vector_dim"], 8)
            frame_map = json.loads((tmp / "frame_map.json").read_text())
            self.assertEqual(len(frame_map), 4)
            self.assertEqual(frame_map["0"]["faiss_index"], 0)
            self.assertIn("neighbor", report)


class EmbeddingFactoryTest(unittest.TestCase):
    def test_registry_resolution(self) -> None:
        self.assertIn("sigclip", MODEL_REGISTRY)
        self.assertEqual(resolve_model_spec("clip"), ("ViT-B-16", "openai"))

    def test_explicit_overrides_registry(self) -> None:
        self.assertEqual(
            resolve_model_spec("openclip", model_name="ViT-H-14", pretrained="x"),
            ("ViT-H-14", "x"),
        )

    def test_unknown_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_model_spec("does-not-exist")


if __name__ == "__main__":
    unittest.main()
