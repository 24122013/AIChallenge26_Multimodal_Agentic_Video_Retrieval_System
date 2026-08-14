from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.services.retrieval.qa_pipeline import (
    QaPipelineConfig,
    QaSearchPipeline,
)


def _evidence(
    index: int,
    *,
    temporal: bool,
    event_count: int,
) -> dict[str, object]:
    return {
        "evidence_id": f"E{index:03d}",
        "video_id": "V001",
        "frame_id": f"F{index:03d}",
        "shot_id": f"S{index:03d}",
        "timestamp": float(index),
        "image_path": "",
        "caption": f"event {index}",
        "ocr_text": "",
        "objects": ["person"],
        "source_modalities": ["visual", "dense_text"],
        "retrieval_score": 1.0 - index / 100.0,
        "base_retrieval_score": 1.0 - index / 100.0,
        "constraint_score": 0.5,
        "matched_constraints": ["person"],
        "temporal_event_index": index - 1 if temporal else None,
        "temporal_match_rank": 1 if temporal else None,
        "temporal_match_mode": "strict" if temporal else "",
        "temporal_chain_id": "TC-test" if temporal else "",
        "temporal_event_query": f"event {index}" if temporal else "",
        "temporal_event_role": (
            "answer_target" if temporal and index == event_count else (
                "context" if temporal else ""
            )
        ),
        "temporal_chain_score": 0.75 if temporal else None,
        "warnings": [],
    }


class _EvidenceEngine:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def search(self, *_: object, **__: object) -> dict[str, object]:
        return self.response


def _response(match_mode: str | None, *, count: int = 5) -> dict[str, object]:
    temporal = match_mode is not None
    eligible = not temporal or match_mode == "strict"
    evidence = [
        _evidence(index, temporal=temporal, event_count=count)
        for index in range(1, count + 1)
    ]
    if match_mode == "none":
        evidence = []
    return {
        "query_plan": {
            "answer_type": "action",
            "needs_temporal": temporal,
            "temporal_events": (
                [f"event {index}" for index in range(1, count + 1)]
                if temporal
                else []
            ),
            "answer_event_index": count - 1 if temporal else None,
        },
        "routing_trace": {
            "feature_flags": {"evidence_bundle": True},
            "temporal_route": {
                "executed": temporal,
                "match_mode": match_mode or "",
                "event_count": count if temporal else 0,
                "event_queries": (
                    [f"event {index}" for index in range(1, count + 1)]
                    if temporal
                    else []
                ),
            },
        },
        "answer_eligible": eligible,
        "preflight_block_reason": (
            None if eligible else f"temporal_match_not_strict:{match_mode}"
        ),
        "temporal_matches": (
            [
                {
                    "match_mode": match_mode,
                    "chain_id": "TC-test",
                    "score": 0.75,
                }
            ]
            if temporal and match_mode != "none"
            else []
        ),
        "evidence": evidence,
        "results": [],
    }


class QaPipelinePreflightTest(unittest.TestCase):
    def test_strict_temporal_sends_complete_five_event_chain(self) -> None:
        observed: list[dict[str, object]] = []

        def runner(_: str, evidence: object, __: str) -> dict[str, object]:
            observed.extend(dict(item) for item in evidence)  # type: ignore[arg-type]
            return {
                "status": "answered",
                "answer": "event 5",
                "answer_type": "action",
                "confidence": 0.9,
                "evidence_ids": ["E005"],
            }

        with tempfile.TemporaryDirectory() as temporary:
            pipeline = QaSearchPipeline(
                _EvidenceEngine(_response("strict")),  # type: ignore[arg-type]
                config=QaPipelineConfig(
                    answer_mode="required",
                    answer_cache_root=Path(temporary),
                ),
                answer_runner=runner,
            )
            response = pipeline.search("A rồi B rồi C rồi D rồi E?")

        self.assertEqual(len(observed), 5)
        self.assertEqual(
            [item["temporal_event_index"] for item in observed],
            [0, 1, 2, 3, 4],
        )
        self.assertTrue(response["answer_eligible"])
        self.assertIsNone(response["preflight_block_reason"])
        self.assertTrue(response["answer_report"]["model_invoked"])
        self.assertEqual(response["temporal_matches"][0]["match_mode"], "strict")

    def test_non_temporal_still_limits_answerer_to_three_evidence(self) -> None:
        observed_count = 0

        def runner(_: str, evidence: object, __: str) -> dict[str, object]:
            nonlocal observed_count
            observed_count = len(evidence)  # type: ignore[arg-type]
            return {
                "status": "answered",
                "answer": "event 1",
                "answer_type": "action",
                "confidence": 0.9,
                "evidence_ids": ["E001"],
            }

        with tempfile.TemporaryDirectory() as temporary:
            response = QaSearchPipeline(
                _EvidenceEngine(_response(None)),  # type: ignore[arg-type]
                config=QaPipelineConfig(
                    answer_mode="required",
                    answer_cache_root=Path(temporary),
                ),
                answer_runner=runner,
            ).search("What happens?")
        self.assertEqual(response["answer"]["status"], "answered")
        self.assertEqual(observed_count, 3)

    def test_non_strict_temporal_modes_abstain_without_model_or_503(self) -> None:
        for match_mode in ("relaxed_gap", "sparse_compat", "none"):
            with self.subTest(match_mode=match_mode):
                called = False

                def runner(*_: object) -> dict[str, object]:
                    nonlocal called
                    called = True
                    raise AssertionError("preflight must skip the model")

                response = QaSearchPipeline(
                    _EvidenceEngine(_response(match_mode)),  # type: ignore[arg-type]
                    config=QaPipelineConfig(answer_mode="required"),
                    answer_runner=runner,
                ).search("A rồi B?")
                self.assertFalse(called)
                self.assertFalse(response["answer_eligible"])
                self.assertEqual(
                    response["answer"]["status"],
                    "insufficient_evidence",
                )
                self.assertFalse(response["answer_report"]["model_invoked"])

    def test_strict_temporal_with_incomplete_lineage_abstains(self) -> None:
        raw = _response("strict", count=2)
        raw["evidence"] = raw["evidence"][:1]  # type: ignore[index]
        called = False

        def runner(*_: object) -> dict[str, object]:
            nonlocal called
            called = True
            raise AssertionError("incomplete chain must not reach the model")

        response = QaSearchPipeline(
            _EvidenceEngine(raw),  # type: ignore[arg-type]
            config=QaPipelineConfig(answer_mode="required"),
            answer_runner=runner,
        ).search("A rồi B?")

        self.assertFalse(called)
        self.assertFalse(response["answer_eligible"])
        self.assertEqual(response["preflight_block_reason"], "temporal_chain_incomplete")

    def test_strict_temporal_rejects_invalid_event_index_or_match_rank(self) -> None:
        for field, value, expected_reason in (
            ("temporal_event_index", True, "temporal_event_index_invalid"),
            ("temporal_match_rank", 2, "temporal_match_rank_invalid"),
        ):
            with self.subTest(field=field):
                raw = _response("strict", count=2)
                raw["evidence"][0][field] = value  # type: ignore[index]
                called = False

                def runner(*_: object) -> dict[str, object]:
                    nonlocal called
                    called = True
                    raise AssertionError("invalid lineage must not reach the model")

                response = QaSearchPipeline(
                    _EvidenceEngine(raw),  # type: ignore[arg-type]
                    config=QaPipelineConfig(answer_mode="required"),
                    answer_runner=runner,
                ).search("A rồi B?")

                self.assertFalse(called)
                self.assertFalse(response["answer_eligible"])
                self.assertEqual(
                    response["preflight_block_reason"],
                    expected_reason,
                )

    def test_temporal_manual_fallback_preserves_complete_lineage(self) -> None:
        raw = _response("strict", count=3)
        bundled = raw["evidence"]  # type: ignore[assignment]
        raw["evidence"] = []
        raw["results"] = [
            {
                **item,
                "keyframe_path": item["image_path"],
                "score": item["retrieval_score"],
            }
            for item in bundled  # type: ignore[union-attr]
        ]
        raw["routing_trace"]["feature_flags"]["evidence_bundle"] = False  # type: ignore[index]

        response = QaSearchPipeline(
            _EvidenceEngine(raw),  # type: ignore[arg-type]
            config=QaPipelineConfig(answer_mode="required"),
        ).search("A rồi B rồi C?")

        self.assertEqual(len(response["evidence"]), 3)
        self.assertEqual(
            [item["temporal_event_query"] for item in response["evidence"]],
            ["event 1", "event 2", "event 3"],
        )
        self.assertEqual(
            {item["temporal_chain_id"] for item in response["evidence"]},
            {"TC-test"},
        )
        self.assertEqual(
            [item["temporal_event_role"] for item in response["evidence"]],
            ["context", "context", "answer_target"],
        )
        self.assertEqual(response["answer"]["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
