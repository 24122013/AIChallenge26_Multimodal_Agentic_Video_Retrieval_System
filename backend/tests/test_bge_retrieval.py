from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.bge_dense import (
    BGE_M3_DIMENSION,
    BGE_M3_FRAME_MAP_NAME,
    BGE_M3_INDEX_NAME,
    BGE_M3_MANIFEST_NAME,
    BgeM3DenseSearchEngine,
    build_bge_m3_index,
    field_tagged_document,
    validate_bge_m3_artifacts,
)
from backend.app.services.retrieval.bge_reranker import (
    rerank_with_bge,
    reranker_document,
)


class HashEncoder:
    """Deterministic injected encoder with no model/network dependency."""

    def __init__(self, overrides: dict[str, int] | None = None) -> None:
        self.overrides = dict(overrides or {})
        self.calls: list[list[str]] = []
        self.resolved_revision = "test-commit-123"

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        matrix = np.zeros((len(texts), BGE_M3_DIMENSION), dtype=np.float32)
        for row, text in enumerate(texts):
            slot = self.overrides.get(text)
            if slot is None:
                slot = int.from_bytes(
                    hashlib.sha256(text.encode("utf-8")).digest()[:2],
                    "big",
                ) % BGE_M3_DIMENSION
            matrix[row, slot] = 3.0
        return matrix


def result(
    frame_id: str,
    score: float,
    *,
    caption: str = "",
    ocr: str = "",
    objects: list[str] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        video_id="V1",
        frame_id=frame_id,
        timestamp=float(frame_id.removeprefix("F") or 0),
        score=score,
        caption=caption,
        ocr_text=ocr,
        objects=list(objects or []),
        modality_scores={"rrf": score},
    )


class BgeM3DenseTests(unittest.TestCase):
    def records(self) -> list[dict[str, object]]:
        return [
            {
                "video_id": "V2",
                "frame_id": "F2",
                "timestamp": 5.0,
                "caption": "blue bicycle",
                "ocr_text": "SHOP",
                "objects": ["bicycle"],
                "asr_text": "must never enter the document",
            },
            {
                "video_id": "V1",
                "frame_id": "F1",
                "timestamp": 1.0,
                "caption": "woman holding a phone",
                "ocr": [{"text": "HELLO"}],
                "objects": [{"class_name": "person"}, {"label": "phone"}],
            },
        ]

    def test_field_tagged_document_has_exact_modalities_and_no_asr(self) -> None:
        document = field_tagged_document(self.records()[0])
        self.assertEqual(
            document,
            "[CAPTION] blue bicycle\n[OCR] SHOP\n[OBJECTS] bicycle",
        )
        self.assertNotIn("must never", document)
        self.assertNotIn("ASR", document)

    def test_builds_exact_artifacts_with_canonical_order_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            report = build_bge_m3_index(
                self.records(),
                root,
                encoder=HashEncoder(),
                model_revision="requested-main",
            )

            self.assertEqual(
                {path.name for path in root.iterdir()},
                {BGE_M3_INDEX_NAME, BGE_M3_FRAME_MAP_NAME, BGE_M3_MANIFEST_NAME},
            )
            self.assertEqual(report.dimension, 1024)
            self.assertEqual(report.model_revision, "test-commit-123")
            validated = validate_bge_m3_artifacts(
                root,
                expected_model_revision="test-commit-123",
            )
            metadata = [item["metadata"] for item in validated.frame_records]
            self.assertEqual([item["video_id"] for item in metadata], ["V1", "V2"])
            manifest = validated.manifest
            self.assertEqual(manifest["index_type"], "IndexFlatIP")
            self.assertEqual(manifest["model"]["representation"], "dense")
            self.assertTrue(manifest["model"]["normalized"])
            self.assertEqual(len(manifest["source_hashes"]["records_sha256"]), 64)

    def test_injected_encoder_is_normalized_and_search_returns_dense_modality(self) -> None:
        records = self.records()
        documents = [field_tagged_document(record) for record in records]
        encoder = HashEncoder({documents[0]: 10, documents[1]: 20})
        with tempfile.TemporaryDirectory() as tmp_dir:
            build_bge_m3_index(records, tmp_dir, encoder=encoder)
            query_encoder = HashEncoder({"find the phone": 20})
            engine = BgeM3DenseSearchEngine(tmp_dir, encoder=query_encoder)

            hits = engine.search("find the phone", top_k=2)

            self.assertEqual(hits[0].frame_id, "F1")
            self.assertAlmostEqual(hits[0].score, 1.0, places=6)
            self.assertEqual(hits[0].modality_scores, {"dense_text": hits[0].score})
            self.assertEqual(hits[0].objects, ["person", "phone"])
            self.assertEqual(query_encoder.calls, [["find the phone"]])

    def test_rejects_wrong_encoder_dimension(self) -> None:
        def wrong(_: Sequence[str]) -> np.ndarray:
            return np.ones((2, 16), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                build_bge_m3_index(self.records(), tmp_dir, encoder=wrong)

    def test_detects_frame_map_tampering_before_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            build_bge_m3_index(self.records(), root, encoder=HashEncoder())
            frame_map = root / BGE_M3_FRAME_MAP_NAME
            payload = json.loads(frame_map.read_text(encoding="utf-8"))
            payload["records"][0]["metadata"]["caption"] = "tampered"
            frame_map.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_bge_m3_artifacts(root)

    def test_detects_manifest_lineage_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            build_bge_m3_index(self.records(), root, encoder=HashEncoder())
            manifest_path = root / BGE_M3_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_hashes"]["records_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "lineage hashes"):
                validate_bge_m3_artifacts(root)

    def test_detects_corrupted_faiss_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            build_bge_m3_index(self.records(), root, encoder=HashEncoder())
            index_path = root / BGE_M3_INDEX_NAME
            index_path.write_bytes(index_path.read_bytes() + b"corrupt")

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_bge_m3_artifacts(root)


class BgeRerankerTests(unittest.TestCase):
    def test_document_has_only_three_field_tags(self) -> None:
        candidate = result("F1", 0.5, caption="a phone", ocr="SALE", objects=["phone"])
        document = reranker_document(candidate)
        self.assertEqual(
            document,
            "[CAPTION] a phone\n[OCR] SALE\n[OBJECTS] phone",
        )
        self.assertNotIn("ASR", document)

    def test_blends_scores_and_keeps_candidate_without_text(self) -> None:
        candidates = [
            result("F1", 0.90),
            result("F2", 0.80, caption="wrong scene"),
            result("F3", 0.30, caption="woman holding a phone"),
        ]
        captured: list[tuple[str, str]] = []

        def runner(pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
            captured.extend(pairs)
            return [0.0, 1.0]

        reranked, report = rerank_with_bge(
            candidates,
            query="what is she holding",
            runner=runner,
            retrieval_alpha=0.5,
        )

        self.assertEqual([item.frame_id for item in reranked], ["F1", "F3", "F2"])
        self.assertEqual(reranked[0].score, 0.90)
        self.assertEqual(reranked[1].score, 0.65)
        self.assertEqual(reranked[1].modality_scores["bge_reranker"], 1.0)
        self.assertEqual(len(captured), 2)
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.scored_count, 2)

    def test_model_error_falls_back_to_phase5_order(self) -> None:
        candidates = [
            result("F1", 0.7, caption="one"),
            result("F2", 0.6, caption="two"),
        ]

        def broken(_: Sequence[tuple[str, str]]) -> Sequence[float]:
            raise RuntimeError("model offline")

        reranked, report = rerank_with_bge(
            candidates,
            query="query",
            runner=broken,
        )

        self.assertEqual(reranked, candidates)
        self.assertEqual(report.status, "fallback")
        self.assertIn("model offline", report.fallback_reason)

    def test_rejects_bad_score_and_uses_fallback(self) -> None:
        candidates = [result("F1", 0.7, caption="one")]
        reranked, report = rerank_with_bge(
            candidates,
            query="query",
            runner=lambda _: [float("nan")],
        )
        self.assertEqual(reranked, candidates)
        self.assertEqual(report.status, "fallback")

    def test_candidate_and_output_limits_are_enforced(self) -> None:
        candidates = [
            result(f"F{index}", 1.0 - index / 100.0, caption=f"frame {index}")
            for index in range(1, 31)
        ]
        seen: list[tuple[str, str]] = []

        def runner(pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
            seen.extend(pairs)
            return [0.5] * len(pairs)

        output, report = rerank_with_bge(
            candidates,
            query="query",
            runner=runner,
            candidate_limit=10,
            output_k=5,
        )

        self.assertEqual(len(seen), 10)
        self.assertEqual(len(output), 5)
        self.assertEqual(report.candidate_count, 10)
        self.assertEqual(report.output_count, 5)

    def test_alpha_validation_is_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "within"):
            rerank_with_bge(
                [result("F1", 0.5, caption="text")],
                query="query",
                runner=lambda _: [0.5],
                retrieval_alpha=1.1,
            )


if __name__ == "__main__":
    unittest.main()
