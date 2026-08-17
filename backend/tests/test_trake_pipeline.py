from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.retrieval_config import TrakeConfig
from backend.app.services.trake.candidate_video import gate_candidate_videos
from backend.app.services.trake.event_retrieval import (
    EventRetriever,
    normalize_event_scores,
)
from backend.app.services.trake.models import (
    BoundaryType,
    EventCandidate,
    TemporalEvent,
    TemporalEventPlan,
    TemporalPath,
    VideoCandidate,
)
from backend.app.services.trake.pipeline import TrakePipeline
from backend.app.services.trake.ranking import rank_hypotheses
from backend.app.services.trake.temporal_alignment import align_candidate_video
from backend.app.services.trake.temporal_refinement import (
    DecodedFrame,
    LocalFrameHypothesis,
    RefinementVariant,
    TemporalRefiner,
    select_local_hypotheses,
)


def _result(
    internal_id: str,
    frame_index: int | None,
    *,
    video_id: str = "V1",
    score: float = 0.5,
    shot_id: str = "",
) -> RetrievalResult:
    return RetrievalResult(
        video_id=video_id,
        frame_id=internal_id,
        frame_index=frame_index,
        timestamp=float(frame_index or 0) / 25.0,
        score=score,
        shot_id=shot_id,
        modality_scores={"visual": score},
    )


def _event(index: int, text: str, boundary: BoundaryType = BoundaryType.UNKNOWN) -> TemporalEvent:
    return TemporalEvent(
        index=index,
        original_text=text,
        retrieval_query=f"scene {text}",
        boundary_type=boundary,
    )


def _candidate(
    event_index: int,
    frame_index: int,
    *,
    video_id: str = "V1",
    score: float = 1.0,
    internal_id: str | None = None,
    shot_id: str = "",
) -> EventCandidate:
    return EventCandidate(
        event_index=event_index,
        result=_result(
            internal_id or f"INTERNAL_{event_index}_{frame_index}",
            frame_index,
            video_id=video_id,
            shot_id=shot_id,
        ),
        normalized_score=score,
        rank=1,
    )


class FakeRetrievalEngine:
    def __init__(self, values: dict[str, list[RetrievalResult]]) -> None:
        self.values = values
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        self.calls.append((query, top_k))
        results = list(self.values.get(query, ()))[:top_k]
        return VisualSearchResponse(query, top_k or len(results), 0.1, results)


class StaticParser:
    def __init__(self, plan: TemporalEventPlan) -> None:
        self.plan = plan

    def parse(self, query: str) -> TemporalEventPlan:
        return self.plan


class ContextFailEngine(FakeRetrievalEngine):
    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        if query == "scene":
            raise RuntimeError("sensitive implementation detail")
        return super().search(query, top_k=top_k)


class WidePoolEngine(FakeRetrievalEngine):
    def __init__(self, values: dict[str, list[RetrievalResult]]) -> None:
        super().__init__(values)
        self.pool_calls: list[tuple[str, int]] = []

    def search_pool(self, query: str, top_k: int) -> VisualSearchResponse:
        self.pool_calls.append((query, top_k))
        results = list(self.values.get(query, ()))[:top_k]
        return VisualSearchResponse(query, top_k, 0.1, results)


class FailingBgeEngine(FakeRetrievalEngine):
    def __init__(self) -> None:
        super().__init__({})
        self.attempts = 0

    def search(self, query: str, top_k: int | None = None) -> VisualSearchResponse:
        self.attempts += 1
        raise RuntimeError(f"secret-token-for {query}")


class ReverseEventReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | None]] = []
        self.last_report: dict[str, object] | None = None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, len(candidates), top_k))
        self.last_report = {
            "status": "success",
            "candidate_count": len(candidates),
            "scored_count": len(candidates),
            "output_count": len(candidates),
            "retrieval_alpha": 0.5,
        }
        return [
            replace(
                item,
                score=0.95 - index / 100.0,
                modality_scores={**item.modality_scores, "bge_reranker": 0.9},
            )
            for index, item in enumerate(reversed(candidates))
        ]


class FallbackEventReranker:
    def __init__(self) -> None:
        self.last_report: dict[str, object] | None = None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.last_report = {
            "status": "fallback",
            "candidate_count": len(candidates),
            "scored_count": 0,
            "output_count": len(candidates),
            "fallback_reason": f"private model error; query={query}",
        }
        return list(candidates)


class InjectedTailEventReranker:
    def __init__(self, injected: RetrievalResult) -> None:
        self.injected = injected

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return [self.injected, *candidates]


class NonFiniteEventReranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return [replace(candidates[0], score=float("nan"))]


class FakeDecoder:
    def decode(
        self,
        video_path: Path,
        *,
        start_frame: int,
        end_frame: int,
        stride: int,
    ) -> list[DecodedFrame]:
        return [
            DecodedFrame(index, object())
            for index in range(start_frame, end_frame + 1, stride)
        ]


class SequenceScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def score(
        self,
        event: TemporalEvent,
        frames: list[DecodedFrame],
    ) -> list[float]:
        return self.scores[: len(frames)]


class TrakeCorePipelineTest(unittest.TestCase):
    def test_event_retrieval_is_independent_context_only_scores_video(self) -> None:
        plan = TemporalEventPlan(
            original_query="scene; 1. enter; 2. sit",
            context="scene",
            events=(_event(0, "enter"), _event(1, "sit")),
        )
        engine = FakeRetrievalEngine(
            {
                "scene enter": [
                    _result("F1", 10, score=1000.0, shot_id="S1"),
                    _result("F2", 11, score=900.0, shot_id="S1"),
                    _result("BAD", None, score=9999.0),
                ],
                "scene sit": [_result("F3", 20, score=0.001)],
                "scene": [_result("CTX", None, video_id="V1")],
            }
        )
        batch = EventRetriever(
            engine,
            TrakeConfig(
                event_top_k=10,
                max_candidates_per_shot=1,
                refinement_enabled=False,
            ),
        ).retrieve(plan)

        self.assertEqual(
            [query for query, _ in engine.calls],
            ["scene enter", "scene sit", "scene"],
        )
        self.assertEqual(len(batch.event_candidates[0]), 1)
        self.assertEqual(len(batch.event_candidates[1]), 1)
        self.assertEqual(batch.context_scores, {"V1": 1.0})
        self.assertIn("event_0_missing_frame_lineage", batch.warnings)

    def test_rank_normalization_ignores_raw_scale(self) -> None:
        large = [_result("A", 1, score=1000000.0), _result("B", 2, score=1.0)]
        tiny = [_result("C", 3, score=0.0002), _result("D", 4, score=-5.0)]
        self.assertEqual(normalize_event_scores(large), normalize_event_scores(tiny))
        self.assertEqual(normalize_event_scores(large)[0], 1.0)

    def test_unbucketed_adjacent_frames_receive_temporal_nms(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        engine = FakeRetrievalEngine(
            {
                "scene event": [
                    _result(f"F{index}", 100 + index, score=1.0 - index / 100.0)
                    for index in range(20)
                ]
            }
        )
        batch = EventRetriever(engine).retrieve(plan)
        frames = [item.result.frame_index for item in batch.event_candidates[0]]

        self.assertLess(len(frames), 20)
        self.assertTrue(
            all(abs(left - right) > 2 for left, right in zip(frames, frames[1:]))
        )

    def test_optional_context_failure_keeps_event_candidates(self) -> None:
        plan = TemporalEventPlan(
            "scene; event",
            "scene",
            (_event(0, "event"),),
        )
        batch = EventRetriever(
            ContextFailEngine({"scene event": [_result("F", 10)]})
        ).retrieve(plan)

        self.assertEqual(len(batch.event_candidates[0]), 1)
        self.assertEqual(batch.context_scores, {})
        self.assertIn("context_retrieval_failed", batch.warnings)
        self.assertEqual(batch.trace["context"]["status"], "failed_optional")

    def test_event_retrieval_uses_wide_canonical_pool_when_available(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        engine = WidePoolEngine({"scene event": [_result("F", 10)]})
        batch = EventRetriever(
            engine,
            TrakeConfig(event_top_k=300),
        ).retrieve(plan)

        self.assertEqual(engine.pool_calls, [("scene event", 300)])
        self.assertEqual(batch.trace["wide_pool_method"], "search_pool")

    def test_bge_dense_overlap_uses_weighted_rrf_and_preserves_canonical_result(
        self,
    ) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        hybrid_result = replace(
            _result("HYBRID_ID", 10, score=0.7),
            keyframe_path="canonical/frame.jpg",
        )
        dense_result = replace(
            _result("DENSE_ID", 10, score=0.9),
            caption="dense caption enriches the canonical result",
            modality_scores={"bge_dense": 0.9},
        )
        batch = EventRetriever(
            FakeRetrievalEngine({"scene event": [hybrid_result]}),
            TrakeConfig(
                bge_dense_enabled=True,
                bge_dense_top_k=25,
                refinement_enabled=False,
            ),
            dense_event_engine=FakeRetrievalEngine(
                {"scene event": [dense_result]}
            ),
        ).retrieve(plan)

        self.assertEqual(len(batch.event_candidates[0]), 1)
        fused = batch.event_candidates[0][0].result
        self.assertEqual(fused.frame_id, "HYBRID_ID")
        self.assertEqual(fused.keyframe_path, "canonical/frame.jpg")
        self.assertEqual(fused.caption, dense_result.caption)
        self.assertGreater(fused.modality_scores["trake_rrf_hybrid"], 0.0)
        self.assertGreater(fused.modality_scores["trake_rrf_bge_dense"], 0.0)
        self.assertEqual(batch.trace["events"][0]["fusion"]["overlap_count"], 1)
        self.assertEqual(
            batch.trace["events"][0]["sources"]["bge_dense"]["status"],
            "success",
        )

    def test_bge_dense_only_candidate_recovers_full_video_coverage(self) -> None:
        plan = TemporalEventPlan(
            "enter then sit",
            "",
            (_event(0, "enter"), _event(1, "sit")),
            parser_source="test",
            confidence=1.0,
        )
        hybrid = FakeRetrievalEngine(
            {
                "scene enter": [_result("H_ENTER", 10, video_id="RECOVERED")],
                "scene sit": [_result("H_SIT", 30, video_id="PARTIAL")],
            }
        )
        dense = FakeRetrievalEngine(
            {
                "scene enter": [],
                "scene sit": [_result("D_SIT", 20, video_id="RECOVERED")],
            }
        )
        response = TrakePipeline(
            retrieval_engine=hybrid,
            dense_event_engine=dense,
            config=TrakeConfig(
                bge_dense_enabled=True,
                bge_dense_top_k=50,
                refinement_enabled=False,
            ),
            parser=StaticParser(plan),
        ).search("enter then sit")

        self.assertTrue(response["hypotheses"])
        self.assertEqual(response["hypotheses"][0]["video_id"], "RECOVERED")
        self.assertEqual(response["hypotheses"][0]["frame_ids"], [10, 20])
        self.assertEqual(response["trace"]["event_retrieval"]["bge"]["dense_calls"], 2)

    def test_rrf_normalizes_only_against_non_empty_sources(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        batch = EventRetriever(
            FakeRetrievalEngine({"scene event": [_result("HYBRID", 10)]}),
            TrakeConfig(bge_dense_enabled=True),
            dense_event_engine=FakeRetrievalEngine({"scene event": []}),
        ).retrieve(plan)

        fused = batch.event_candidates[0][0].result
        fusion_trace = batch.trace["events"][0]["fusion"]
        self.assertEqual(fused.score, 1.0)
        self.assertEqual(fusion_trace["active_weight_sum"], 1.0)

    def test_optional_bge_dense_failure_falls_back_without_sensitive_trace(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        batch = EventRetriever(
            FakeRetrievalEngine({"scene event": [_result("H", 10)]}),
            TrakeConfig(bge_dense_enabled=True, bge_required=False),
            dense_event_engine=FailingBgeEngine(),
        ).retrieve(plan)

        self.assertEqual([item.result.frame_index for item in batch.event_candidates[0]], [10])
        self.assertIn("event_0_bge_dense_failed_optional", batch.warnings)
        audit_text = repr({"trace": batch.trace, "warnings": batch.warnings})
        self.assertNotIn("secret-token", audit_text)
        self.assertNotIn("scene event", audit_text)
        self.assertEqual(
            batch.trace["events"][0]["sources"]["bge_dense"]["fallback"],
            "canonical_hybrid",
        )

    def test_optional_bge_failure_opens_one_request_circuit(self) -> None:
        plan = TemporalEventPlan(
            "three events",
            "",
            (_event(0, "one"), _event(1, "two"), _event(2, "three")),
        )
        hybrid = FakeRetrievalEngine(
            {
                "scene one": [_result("H1", 10)],
                "scene two": [_result("H2", 20)],
                "scene three": [_result("H3", 30)],
            }
        )
        dense = FailingBgeEngine()
        batch = EventRetriever(
            hybrid,
            TrakeConfig(bge_dense_enabled=True),
            dense_event_engine=dense,
        ).retrieve(plan)

        self.assertEqual(dense.attempts, 1)
        self.assertTrue(batch.trace["bge"]["dense_circuit_open"])
        self.assertEqual(
            batch.trace["events"][1]["sources"]["bge_dense"]["status"],
            "skipped_circuit_open",
        )
        self.assertEqual(
            batch.trace["events"][2]["sources"]["bge_dense"]["status"],
            "skipped_circuit_open",
        )

    def test_required_bge_dense_failure_is_fail_closed_and_sanitized(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        retriever = EventRetriever(
            FakeRetrievalEngine({"scene event": [_result("H", 10)]}),
            TrakeConfig(bge_dense_enabled=True, bge_required=True),
            dense_event_engine=FailingBgeEngine(),
        )

        with self.assertRaises(RuntimeError) as captured:
            retriever.retrieve(plan)
        self.assertIn("required TRAKE BGE dense_retrieval", str(captured.exception))
        self.assertNotIn("secret-token", str(captured.exception))
        self.assertNotIn("scene event", str(captured.exception))

    def test_bge_reranker_applies_before_diversity_and_reports_counts(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        reranker = ReverseEventReranker()
        batch = EventRetriever(
            FakeRetrievalEngine(
                {
                    "scene event": [
                        _result("FIRST", 10),
                        _result("SECOND", 20),
                    ]
                }
            ),
            TrakeConfig(
                bge_reranker_enabled=True,
                bge_reranker_top_k=10,
                refinement_enabled=False,
            ),
            event_reranker=reranker,
        ).retrieve(plan)

        self.assertEqual(
            [item.result.frame_index for item in batch.event_candidates[0]],
            [20, 10],
        )
        self.assertEqual(
            batch.event_candidates[0][0].result.modality_scores["bge_reranker"],
            0.9,
        )
        reranker_trace = batch.trace["events"][0]["reranker"]
        self.assertEqual(reranker_trace["status"], "success")
        self.assertEqual(reranker_trace["report"]["scored_count"], 2)
        self.assertEqual(reranker.calls, [("scene event", 2, 2)])

    def test_bge_reranker_reported_fallback_is_safe_and_keeps_recall(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        batch = EventRetriever(
            FakeRetrievalEngine(
                {"scene event": [_result("FIRST", 10), _result("SECOND", 20)]}
            ),
            TrakeConfig(bge_reranker_enabled=True, bge_required=False),
            event_reranker=FallbackEventReranker(),
        ).retrieve(plan)

        self.assertEqual(
            [item.result.frame_index for item in batch.event_candidates[0]],
            [10, 20],
        )
        self.assertIn("event_0_bge_reranker_failed_optional", batch.warnings)
        reranker_trace = batch.trace["events"][0]["reranker"]
        self.assertEqual(reranker_trace["status"], "reported_fallback")
        self.assertEqual(
            reranker_trace["report"]["fallback_code"],
            "reranker_reported_fallback",
        )
        audit_text = repr({"trace": batch.trace, "warnings": batch.warnings})
        self.assertNotIn("private model error", audit_text)
        self.assertNotIn("scene event", audit_text)

    def test_required_bge_reranker_fallback_is_fail_closed_and_sanitized(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        retriever = EventRetriever(
            FakeRetrievalEngine({"scene event": [_result("FIRST", 10)]}),
            TrakeConfig(bge_reranker_enabled=True, bge_required=True),
            event_reranker=FallbackEventReranker(),
        )

        with self.assertRaises(RuntimeError) as captured:
            retriever.retrieve(plan)
        message = str(captured.exception)
        self.assertIn("required TRAKE BGE reranker", message)
        self.assertNotIn("private model error", message)
        self.assertNotIn("scene event", message)

    def test_bge_reranker_cannot_promote_unscored_tail_candidate(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        head = _result("HEAD", 10)
        tail = _result("TAIL", 20)
        batch = EventRetriever(
            FakeRetrievalEngine({"scene event": [head, tail]}),
            TrakeConfig(
                bge_reranker_enabled=True,
                bge_reranker_top_k=1,
            ),
            event_reranker=InjectedTailEventReranker(tail),
        ).retrieve(plan)

        self.assertEqual(
            [item.result.frame_id for item in batch.event_candidates[0]],
            ["HEAD", "TAIL"],
        )
        reranker_trace = batch.trace["events"][0]["reranker"]
        self.assertEqual(reranker_trace["scored_pool_count"], 1)
        self.assertEqual(reranker_trace["rejected_count"], 1)

    def test_bge_reranker_non_finite_score_fails_open(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        batch = EventRetriever(
            FakeRetrievalEngine({"scene event": [_result("HEAD", 10)]}),
            TrakeConfig(bge_reranker_enabled=True, bge_required=False),
            event_reranker=NonFiniteEventReranker(),
        ).retrieve(plan)

        self.assertEqual(
            [item.result.frame_id for item in batch.event_candidates[0]],
            ["HEAD"],
        )
        self.assertIn("event_0_bge_reranker_failed_optional", batch.warnings)
        self.assertEqual(
            batch.trace["events"][0]["reranker"]["status"],
            "failed_optional",
        )

    def test_bge_dense_rejects_missing_lineage_before_fusion(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))
        batch = EventRetriever(
            FakeRetrievalEngine({"scene event": [_result("H", 10)]}),
            TrakeConfig(bge_dense_enabled=True),
            dense_event_engine=FakeRetrievalEngine(
                {"scene event": [_result("BAD_DENSE", None)]}
            ),
        ).retrieve(plan)

        self.assertEqual([item.result.frame_id for item in batch.event_candidates[0]], ["H"])
        self.assertIn("event_0_bge_dense_missing_frame_lineage", batch.warnings)
        dense_trace = batch.trace["events"][0]["sources"]["bge_dense"]
        self.assertEqual(dense_trace["missing_lineage_count"], 1)
        self.assertEqual(dense_trace["valid_lineage_count"], 0)

    def test_weighted_rrf_equal_scores_have_deterministic_source_aware_order(self) -> None:
        plan = TemporalEventPlan("event", "", (_event(0, "event"),))

        def run_once() -> list[tuple[str, int | None]]:
            batch = EventRetriever(
                FakeRetrievalEngine(
                    {"scene event": [_result("HYBRID", 20, video_id="Z_VIDEO")]}
                ),
                TrakeConfig(bge_dense_enabled=True),
                dense_event_engine=FakeRetrievalEngine(
                    {"scene event": [_result("DENSE", 10, video_id="A_VIDEO")]}
                ),
            ).retrieve(plan)
            return [
                (item.result.video_id, item.result.frame_index)
                for item in batch.event_candidates[0]
            ]

        first = run_once()
        second = run_once()
        self.assertEqual(first, second)
        self.assertEqual(first, [("Z_VIDEO", 20), ("A_VIDEO", 10)])

    def test_video_gating_prefers_complete_coverage(self) -> None:
        candidates = {
            0: (
                _candidate(0, 10, video_id="COMPLETE", score=0.45),
                _candidate(0, 11, video_id="ONE_EVENT", score=1.0),
            ),
            1: (_candidate(1, 20, video_id="COMPLETE", score=0.45),),
            2: (_candidate(2, 30, video_id="COMPLETE", score=0.45),),
            3: (_candidate(3, 40, video_id="COMPLETE", score=0.45),),
        }
        gated = gate_candidate_videos(candidates, event_count=4)

        self.assertEqual([video.video_id for video in gated.videos], ["COMPLETE"])
        self.assertEqual(gated.videos[0].coverage, 1.0)

    def test_alignment_is_ordered_deterministic_and_has_no_hard_gap(self) -> None:
        video = VideoCandidate(
            video_id="V1",
            coverage=1.0,
            event_support=0.8,
            total_score=0.9,
            event_candidates={
                0: (_candidate(0, 10, score=0.9), _candidate(0, 20, score=0.8)),
                1: (_candidate(1, 500_000, score=0.9),),
                2: (_candidate(2, 500_010, score=0.9),),
            },
        )
        config = TrakeConfig(k_best_paths_per_video=5, beam_width=20)

        first = align_candidate_video(video, event_count=3, config=config)
        second = align_candidate_video(video, event_count=3, config=config)

        self.assertEqual([path.to_dict() for path in first], [path.to_dict() for path in second])
        self.assertTrue(first)
        self.assertEqual(len(first[0].frame_ids), 3)
        self.assertEqual(list(first[0].frame_ids), sorted(first[0].frame_ids))
        self.assertEqual(first[0].score_breakdown["gap_units"], "original_frames")

    def test_alignment_beam_does_not_prune_all_future_reachable_paths(self) -> None:
        first = tuple(
            _candidate(0, frame, score=0.9)
            for frame in range(20)
        )
        second = tuple(
            [
                *(_candidate(1, frame, score=0.1) for frame in range(20, 30)),
                *(_candidate(1, frame, score=1.0) for frame in range(1000, 1010)),
            ]
        )
        video = VideoCandidate(
            video_id="V1",
            coverage=1.0,
            event_support=0.8,
            total_score=0.8,
            event_candidates={
                0: first,
                1: second,
                2: (_candidate(2, 30, score=0.9),),
            },
        )
        paths = align_candidate_video(
            video,
            event_count=3,
            config=TrakeConfig(beam_width=200),
        )

        self.assertTrue(paths)
        self.assertLess(paths[0].frame_ids[1], paths[0].frame_ids[2])

    def test_boundary_selection_transition_and_peak(self) -> None:
        frames = [DecodedFrame(index, object()) for index in range(5)]
        transition = select_local_hypotheses(
            _event(0, "first opens", BoundaryType.FIRST_TRANSITION),
            frames,
            [0.1, 0.2, 0.8, 0.9, 0.7],
            limit=2,
        )
        peak = select_local_hypotheses(
            _event(0, "cao nhất", BoundaryType.PEAK),
            frames,
            [0.1, 0.6, 0.3, 0.9, 0.2],
            limit=2,
        )

        self.assertEqual(transition[0].frame_index, 2)
        self.assertEqual(transition[0].strategy, "first_positive_transition")
        self.assertEqual(peak[0].frame_index, 3)
        self.assertEqual(peak[0].strategy, "local_peak")

    def test_flat_local_scores_preserve_coarse_frame_without_false_confidence(self) -> None:
        frames = [DecodedFrame(index, object()) for index in range(40, 161)]
        selected = select_local_hypotheses(
            _event(0, "first contact", BoundaryType.FIRST_CONTACT),
            frames,
            [0.5] * len(frames),
            limit=3,
            coarse_frame_index=100,
        )

        self.assertEqual([item.frame_index for item in selected], [100])
        self.assertEqual(selected[0].confidence, 0.0)
        self.assertEqual(selected[0].strategy, "flat_local_score_fallback")
        self.assertEqual(selected[0].source, "canonical_metadata")

    def test_refinement_missing_video_falls_back_to_coarse_lineage(self) -> None:
        event = _event(0, "peak", BoundaryType.PEAK)
        path = TemporalPath(
            video_id="V1",
            event_candidates=(_candidate(0, 123),),
            score=0.8,
        )
        plan = TemporalEventPlan("peak", "", (event,))
        with tempfile.TemporaryDirectory() as temporary:
            variants = TemporalRefiner(
                config=TrakeConfig(window_before_frames=2, window_after_frames=2),
                video_root=temporary,
                decoder=FakeDecoder(),
                scorer=SequenceScorer([0.1, 0.4, 0.9, 0.3, 0.2]),
            ).refine(path, plan)

        self.assertEqual(variants[0].frame_indices, (123,))
        self.assertIn("local_refinement_video_unavailable", variants[0].warnings)
        self.assertEqual(variants[0].event_refinements[0].source, "canonical_metadata")

    def test_ranking_deduplicates_sequences_and_caps_at_100(self) -> None:
        paths = []
        for index in range(160):
            candidates = (
                _candidate(0, index * 5, video_id=f"V{index % 4}"),
                _candidate(1, index * 5 + 10, video_id=f"V{index % 4}"),
            )
            paths.append(
                TemporalPath(
                    video_id=f"V{index % 4}",
                    event_candidates=candidates,
                    score=1.0 - index / 1000.0,
                    path_id=f"P{index}",
                )
            )
        paths.append(paths[0])
        ranked, trace = rank_hypotheses(paths, max_answers=100, expected_event_count=2)

        identities = {(item.video_id, item.frame_ids) for item in ranked}
        self.assertLessEqual(len(ranked), 100)
        self.assertEqual(len(identities), len(ranked))
        self.assertGreaterEqual(trace.exact_duplicate_count, 1)

    def test_ranking_drops_refiner_variant_with_contradictory_lineage(self) -> None:
        path = TemporalPath(
            video_id="V1",
            event_candidates=(_candidate(0, 10), _candidate(1, 20)),
            score=0.8,
            path_id="P1",
        )
        malformed = RefinementVariant(
            coarse_path=path,
            frame_indices=(11, 21),
            score=0.9,
            event_refinements=(
                LocalFrameHypothesis(11, 0.9, "fake", 1.0, "local_refinement"),
            ),
        )
        ranked, trace = rank_hypotheses(
            [malformed],
            max_answers=100,
            expected_event_count=2,
        )

        self.assertEqual(ranked, [])
        self.assertEqual(trace.valid_count, 0)

    def test_pipeline_response_uses_original_frame_index_not_internal_id(self) -> None:
        plan = TemporalEventPlan(
            original_query="enter then sit",
            context="",
            events=(_event(0, "enter"), _event(1, "sit")),
            parser_source="test",
            confidence=1.0,
        )
        engine = FakeRetrievalEngine(
            {
                "scene enter": [_result("INTERNAL_ENTER", 101, score=0.9)],
                "scene sit": [_result("INTERNAL_SIT", 203, score=0.8)],
            }
        )
        contract = {
            "corpus_generation": "test-generation",
            "dense": {"resolved_revision": "commit-1"},
        }
        pipeline = TrakePipeline(
            retrieval_engine=engine,
            config=TrakeConfig(refinement_enabled=False, event_top_k=10),
            parser=StaticParser(plan),
            bge_contract=contract,
        )
        contract["dense"]["resolved_revision"] = "caller-mutated"
        response = pipeline.search("enter then sit", top_k=100)

        self.assertEqual(response["task"], "trake")
        self.assertEqual(response["hypotheses"][0]["frame_ids"], [101, 203])
        self.assertNotIn("INTERNAL_ENTER", response["hypotheses"][0]["frame_ids"])
        self.assertEqual(response["hypotheses"][0]["lineage"][0]["internal_frame_id"], "INTERNAL_ENTER")
        self.assertEqual(len(response["candidates"]), len(response["hypotheses"]))
        self.assertEqual(len(response["candidates"][0]["frame_ids"]), 2)
        self.assertEqual(
            response["trace"]["bge_contract"]["corpus_generation"],
            "test-generation",
        )
        self.assertEqual(
            response["trace"]["bge_contract"]["dense"]["resolved_revision"],
            "commit-1",
        )
        response["trace"]["bge_contract"]["dense"]["resolved_revision"] = "response-mutated"
        second = pipeline.search("enter then sit", top_k=100)
        self.assertEqual(
            second["trace"]["bge_contract"]["dense"]["resolved_revision"],
            "commit-1",
        )


if __name__ == "__main__":
    unittest.main()
