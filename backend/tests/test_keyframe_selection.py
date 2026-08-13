from __future__ import annotations

import itertools
import math
import random
import unittest

from backend.app.services.indexing.keyframe_selection import (
    PHASE_COVERAGE,
    PHASE_MMR,
    PHASE_PROTECTED,
    ProtectedEvent,
    SelectionCandidate,
    SelectionConfig,
    select_keyframes,
)
from backend.app.services.indexing.keyframe_candidates import KeyframeCandidate


def candidate(
    candidate_id: str,
    timestamp: float,
    *,
    importance: float = 0.5,
    embedding: tuple[float, ...] = (),
    duplicate_group: str | None = None,
    shot_index: int = 1,
    source_reasons: tuple[str, ...] = (),
) -> SelectionCandidate:
    return SelectionCandidate(
        candidate_id=candidate_id,
        timestamp=timestamp,
        frame_index=int(round(timestamp * 100)),
        shot_index=shot_index,
        importance_score=importance,
        semantic_embedding=embedding,
        duplicate_group=duplicate_group,
        source_reasons=source_reasons,
    )


def hard_constraints_hold(
    values: tuple[SelectionCandidate, ...],
    events: tuple[ProtectedEvent, ...],
    *,
    duration: float,
    max_gap: float,
) -> bool:
    selected_ids = {value.candidate_id for value in values}
    if any(not selected_ids.intersection(event.candidate_ids) for event in events):
        return False
    times = [
        0.0,
        *(value.timestamp for value in sorted(values, key=lambda item: item.timestamp)),
        duration,
    ]
    return all(right - left <= max_gap + 1e-12 for left, right in zip(times, times[1:]))


class KeyframeSelectionTest(unittest.TestCase):
    def test_one_candidate_can_cover_multiple_protected_events(self) -> None:
        values = (
            candidate("A", 2.0, importance=0.9),
            candidate("B", 5.0, importance=0.2),
            candidate("C", 8.0, importance=0.8),
        )
        events = (
            ProtectedEvent("ocr-title", "ocr_new", ("A", "B")),
            ProtectedEvent("object-bike", "object_new", ("B", "C")),
        )

        result = select_keyframes(
            values,
            events,
            video_duration=10.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=0,
                protect_each_shot=False,
            ),
        )

        self.assertTrue(result.constraints_satisfied)
        self.assertEqual([item.candidate.candidate_id for item in result.selected], ["B"])
        self.assertEqual(result.selected[0].selection_phase, PHASE_PROTECTED)
        self.assertEqual(result.selected[0].covered_event_ids, ("object-bike", "ocr-title"))

    def test_gap_fill_uses_minimum_cardinality_farthest_reachable_rule(self) -> None:
        values = tuple(candidate(str(time), float(time)) for time in (3, 5, 7))

        result = select_keyframes(
            values,
            (),
            video_duration=10.0,
            config=SelectionConfig(
                max_gap_seconds=4.0,
                target_keyframes=0,
                protect_each_shot=False,
            ),
        )

        self.assertTrue(result.coverage_satisfied)
        self.assertEqual(
            [item.candidate.timestamp for item in result.selected],
            [3.0, 7.0],
        )
        self.assertTrue(all(item.selection_phase == PHASE_COVERAGE for item in result.selected))

    def test_temporal_coverage_includes_head_and_tail(self) -> None:
        values = tuple(candidate(str(time), float(time)) for time in (1, 4, 7, 9))

        result = select_keyframes(
            values,
            (),
            video_duration=10.0,
            config=SelectionConfig(
                max_gap_seconds=3.0,
                target_keyframes=0,
                protect_each_shot=False,
            ),
        )

        self.assertTrue(result.coverage_satisfied)
        self.assertEqual([item.candidate.timestamp for item in result.selected], [1.0, 4.0, 7.0])
        self.assertEqual(result.max_gap_after, 3.0)

    def test_gap_equal_to_limit_passes_but_epsilon_over_limit_is_filled(self) -> None:
        exact = select_keyframes(
            (candidate("middle", 3.0),),
            (ProtectedEvent("shot", "shot", ("middle",)),),
            video_duration=6.0,
            config=SelectionConfig(
                max_gap_seconds=3.0,
                target_keyframes=0,
                protect_each_shot=False,
            ),
        )
        over = select_keyframes(
            (candidate("left", 3.0), candidate("right", 6.0)),
            (ProtectedEvent("shot", "shot", ("left",)),),
            video_duration=6.001,
            config=SelectionConfig(
                max_gap_seconds=3.0,
                target_keyframes=0,
                protect_each_shot=False,
            ),
        )

        self.assertEqual(len(exact.selected), 1)
        self.assertTrue(exact.coverage_satisfied)
        self.assertEqual([item.candidate.candidate_id for item in over.selected], ["left", "right"])
        self.assertTrue(over.coverage_satisfied)

    def test_exact_fallback_finds_joint_event_cover_under_hard_cap(self) -> None:
        values = (
            candidate("A", 0.5, importance=1.0),
            candidate("B", 0.2),
            candidate("C", 0.8),
        )
        events = (
            ProtectedEvent("e1", "test", ("A", "B")),
            ProtectedEvent("e2", "test", ("A", "B")),
            ProtectedEvent("e3", "test", ("A", "C")),
            ProtectedEvent("e4", "test", ("B",)),
            ProtectedEvent("e5", "test", ("C",)),
            ProtectedEvent("e6", "test", ("C",)),
        )

        result = select_keyframes(
            values,
            events,
            video_duration=1.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=0,
                hard_max_keyframes=2,
                protect_each_shot=False,
            ),
        )

        self.assertTrue(result.constraints_satisfied)
        self.assertEqual(result.selection_method, "exact_hard_constraint_fallback")
        self.assertEqual({item.candidate.candidate_id for item in result.selected}, {"B", "C"})

    def test_event_representative_lookahead_can_also_satisfy_coverage(self) -> None:
        values = (
            candidate("high-importance", 1.0, importance=1.0),
            candidate("coverage-friendly", 4.0, importance=0.1),
        )
        event = ProtectedEvent("title", "ocr_new", ("high-importance", "coverage-friendly"))

        result = select_keyframes(
            values,
            (event,),
            video_duration=8.0,
            config=SelectionConfig(
                max_gap_seconds=4.0,
                target_keyframes=0,
                hard_max_keyframes=1,
                protect_each_shot=False,
            ),
        )

        self.assertTrue(result.constraints_satisfied)
        self.assertEqual(
            [item.candidate.candidate_id for item in result.selected],
            ["coverage-friendly"],
        )

    def test_candidate_exhaustion_reports_explicit_violating_gap(self) -> None:
        result = select_keyframes(
            (candidate("only", 2.0),),
            (),
            video_duration=10.0,
            config=SelectionConfig(
                max_gap_seconds=3.0,
                target_keyframes=0,
                protect_each_shot=False,
            ),
        )

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.stop_reason, "coverage_candidates_unavailable")
        self.assertTrue(result.infeasibility_proven)
        self.assertEqual(len(result.violating_gaps), 1)
        self.assertEqual(
            (result.violating_gaps[0].start, result.violating_gaps[0].end),
            (2.0, 10.0),
        )

    def test_large_pool_hard_cap_failure_is_not_claimed_as_proven_infeasible(self) -> None:
        values = tuple(candidate(str(time), float(time)) for time in range(1, 10))

        result = select_keyframes(
            values,
            (ProtectedEvent("mandatory", "synthetic", ("1",)),),
            video_duration=10.0,
            config=SelectionConfig(
                max_gap_seconds=2.0,
                target_keyframes=0,
                hard_max_keyframes=1,
                exact_search_candidate_limit=0,
                protect_each_shot=False,
            ),
        )

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.stop_reason, "hard_cap_reached")
        self.assertFalse(result.infeasibility_proven)

    def test_default_protects_one_boundary_candidate_per_shot(self) -> None:
        values = (
            candidate(
                "shot-0-boundary",
                0.2,
                importance=0.1,
                shot_index=0,
                source_reasons=("shot_boundary_start",),
            ),
            candidate("shot-0-interior", 1.0, importance=1.0, shot_index=0),
            candidate(
                "shot-1-boundary",
                4.8,
                importance=0.1,
                shot_index=1,
                source_reasons=("shot_boundary_end",),
            ),
            candidate("shot-1-interior", 4.0, importance=1.0, shot_index=1),
        )

        result = select_keyframes(
            values,
            (),
            video_duration=5.0,
            config=SelectionConfig(max_gap_seconds=10.0, target_keyframes=0),
        )

        self.assertTrue(result.constraints_satisfied)
        self.assertEqual(
            {item.candidate.candidate_id for item in result.selected},
            {"shot-0-boundary", "shot-1-boundary"},
        )
        self.assertEqual(
            {event_id for item in result.selected for event_id in item.covered_event_ids},
            {"__shot__:0", "__shot__:1"},
        )

    def test_default_never_returns_zero_for_a_nonempty_short_video(self) -> None:
        result = select_keyframes(
            (candidate("representative", 1.0, shot_index=0),),
            (),
            video_duration=2.0,
            config=SelectionConfig(max_gap_seconds=5.0),
        )

        self.assertTrue(result.constraints_satisfied)
        self.assertEqual(
            [item.candidate.candidate_id for item in result.selected],
            ["representative"],
        )

    def test_exact_fallback_ignores_irrelevant_dense_candidates(self) -> None:
        values = (
            candidate("A", 0.5, importance=1.0),
            candidate("B", 0.2),
            candidate("C", 0.8),
            *(
                candidate(f"irrelevant-{index}", 0.01 * (index + 1))
                for index in range(16)
            ),
        )
        events = (
            ProtectedEvent("e1", "test", ("A", "B")),
            ProtectedEvent("e2", "test", ("A", "B")),
            ProtectedEvent("e3", "test", ("A", "C")),
            ProtectedEvent("e4", "test", ("B",)),
            ProtectedEvent("e5", "test", ("C",)),
            ProtectedEvent("e6", "test", ("C",)),
        )

        result = select_keyframes(
            values,
            events,
            video_duration=1.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=0,
                hard_max_keyframes=2,
                protect_each_shot=False,
            ),
        )

        self.assertTrue(result.constraints_satisfied)
        self.assertEqual(result.selection_method, "exact_hard_constraint_fallback")
        self.assertEqual({item.candidate.candidate_id for item in result.selected}, {"B", "C"})

    def test_integer_fields_and_event_candidate_sequence_are_strict(self) -> None:
        invalid_configs = (
            {"target_keyframes": 1.5},
            {"target_keyframes": True},
            {"hard_max_keyframes": 1.5},
            {"hard_max_keyframes": False},
            {"exact_search_candidate_limit": 1.5},
            {"exact_search_candidate_limit": True},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    SelectionConfig(max_gap_seconds=5.0, **kwargs)

        for field_name, value in (
            ("frame_index", 1.5),
            ("frame_index", True),
            ("shot_index", 1.5),
            ("shot_index", False),
        ):
            kwargs = {"frame_index": 1, "shot_index": 1, field_name: value}
            with self.subTest(field=field_name, value=value):
                with self.assertRaises(TypeError):
                    SelectionCandidate("candidate", 1.0, **kwargs)

        for priority in (1.5, True):
            with self.subTest(priority=priority):
                with self.assertRaises(TypeError):
                    ProtectedEvent("event", "ocr", ("candidate",), priority=priority)
        with self.assertRaisesRegex(TypeError, "sequence, not a string"):
            ProtectedEvent("event", "ocr", "candidate")

    def test_soft_stop_reasons_distinguish_pool_exhaustion_and_hard_cap(self) -> None:
        values = tuple(candidate(f"c{index}", float(index + 1)) for index in range(3))

        exhausted = select_keyframes(
            values,
            (),
            video_duration=4.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=5,
                protect_each_shot=False,
            ),
        )
        capped = select_keyframes(
            values,
            (),
            video_duration=4.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=5,
                hard_max_keyframes=2,
                protect_each_shot=False,
            ),
        )

        self.assertEqual(exhausted.soft_stop_reason, "candidate_pool_exhausted")
        self.assertFalse(exhausted.soft_target_reached)
        self.assertEqual(capped.soft_stop_reason, "hard_cap_reached")
        self.assertFalse(capped.soft_target_reached)

    def test_generated_candidate_adapter_preserves_generation_provenance(self) -> None:
        generated = KeyframeCandidate(
            candidate_id="CANDIDATE_VIDEO_000000025",
            video_id="VIDEO",
            shot_index=2,
            frame_index=25,
            timestamp_sec=1.0,
            shot_start_sec=0.5,
            shot_end_sec=1.5,
            reasons=("dense_interval", "shot_boundary_end"),
        )

        adapted = SelectionCandidate.from_generated_candidate(
            generated,
            importance_score=0.75,
            semantic_embedding=(1.0, 0.0),
        )

        self.assertEqual(adapted.candidate_id, generated.candidate_id)
        self.assertEqual(adapted.timestamp, generated.timestamp_sec)
        self.assertEqual(adapted.frame_index, generated.frame_index)
        self.assertEqual(adapted.shot_index, generated.shot_index)
        self.assertEqual(adapted.source_reasons, generated.reasons)
        self.assertEqual(adapted.shot_start_sec, generated.shot_start_sec)
        self.assertEqual(adapted.shot_end_sec, generated.shot_end_sec)

    def test_mmr_normalization_handles_extreme_finite_embedding_scales(self) -> None:
        values = (
            candidate("tiny", 1.0, embedding=(1e-50, 0.0)),
            candidate("huge", 2.0, embedding=(0.0, 1e100)),
        )

        result = select_keyframes(
            values,
            (),
            video_duration=3.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=2,
                protect_each_shot=False,
            ),
        )

        self.assertEqual(len(result.selected), 2)
        self.assertTrue(
            all(
                item.selection_score is not None
                and math.isfinite(item.selection_score)
                for item in result.selected
            )
        )

    def test_hard_events_override_duplicate_group_soft_suppression(self) -> None:
        values = (
            candidate("pre", 1.0, duplicate_group="same"),
            candidate("post", 2.0, duplicate_group="same"),
        )
        events = (
            ProtectedEvent("transition-pre", "transition", ("pre",)),
            ProtectedEvent("transition-post", "transition", ("post",)),
        )

        result = select_keyframes(
            values,
            events,
            video_duration=3.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=1,
                protect_each_shot=False,
            ),
        )

        self.assertTrue(result.constraints_satisfied)
        self.assertEqual({item.candidate.candidate_id for item in result.selected}, {"pre", "post"})
        self.assertTrue(result.soft_budget_exceeded)

    def test_empty_pool_reports_supplied_events_and_video_representative(self) -> None:
        result = select_keyframes(
            (),
            (ProtectedEvent("ocr-title", "ocr_new", ("missing",)),),
            video_duration=2.0,
            config=SelectionConfig(max_gap_seconds=5.0),
        )

        self.assertEqual(result.status, "partial")
        self.assertEqual(
            set(result.unsatisfied_event_ids),
            {"ocr-title", "__video__:representative"},
        )

    def test_protected_constraints_can_exceed_soft_target(self) -> None:
        values = (candidate("left", 2.0), candidate("right", 8.0))
        events = (
            ProtectedEvent("left-shot", "shot", ("left",)),
            ProtectedEvent("right-shot", "shot", ("right",)),
        )

        result = select_keyframes(
            values,
            events,
            video_duration=10.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=1,
                protect_each_shot=False,
            ),
        )

        self.assertTrue(result.constraints_satisfied)
        self.assertTrue(result.soft_budget_exceeded)
        self.assertTrue(result.soft_target_reached)
        self.assertEqual(len(result.selected), 2)

    def test_mmr_prefers_semantically_diverse_candidate(self) -> None:
        values = (
            candidate("best", 1.0, importance=0.9, embedding=(1.0, 0.0)),
            candidate("near-duplicate", 2.0, importance=0.85, embedding=(0.99, 0.1)),
            candidate("diverse", 3.0, importance=0.8, embedding=(0.0, 1.0)),
        )

        result = select_keyframes(
            values,
            (),
            video_duration=4.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=2,
                importance_weight=0.5,
                novelty_weight=0.5,
                protect_each_shot=False,
            ),
        )

        self.assertEqual(
            [item.candidate.candidate_id for item in result.selected],
            ["best", "diverse"],
        )
        self.assertTrue(all(item.selection_phase == PHASE_MMR for item in result.selected))

    def test_missing_embedding_is_not_treated_as_maximally_novel(self) -> None:
        values = (
            candidate("embedded", 1.0, importance=0.5, embedding=(1.0, 0.0)),
            candidate("missing", 2.0, importance=0.9),
        )

        result = select_keyframes(
            values,
            (),
            video_duration=3.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=1,
                importance_weight=0.2,
                novelty_weight=0.8,
                protect_each_shot=False,
            ),
        )

        self.assertEqual([item.candidate.candidate_id for item in result.selected], ["embedded"])

    def test_duplicate_group_receives_zero_novelty(self) -> None:
        values = (
            candidate("seed", 1.0, importance=1.0, embedding=(1.0, 0.0), duplicate_group="same"),
            candidate(
                "duplicate",
                2.0,
                importance=0.95,
                embedding=(0.0, 1.0),
                duplicate_group="same",
            ),
            candidate("other", 3.0, importance=0.8, embedding=(0.0, 1.0), duplicate_group="other"),
        )

        result = select_keyframes(
            values,
            (),
            video_duration=4.0,
            config=SelectionConfig(
                max_gap_seconds=10.0,
                target_keyframes=2,
                importance_weight=0.5,
                novelty_weight=0.5,
                protect_each_shot=False,
            ),
        )

        self.assertEqual(
            [item.candidate.candidate_id for item in result.selected],
            ["seed", "other"],
        )

    def test_output_is_deterministic_unique_and_sorted_by_timestamp(self) -> None:
        original = (
            candidate("late", 4.0, importance=0.7),
            candidate("early-b", 1.0, importance=0.5),
            candidate("early-a", 1.0, importance=0.5),
        )
        config = SelectionConfig(
            max_gap_seconds=10.0,
            target_keyframes=3,
            protect_each_shot=False,
        )

        first = select_keyframes(original, (), video_duration=5.0, config=config)
        second = select_keyframes(reversed(original), (), video_duration=5.0, config=config)

        first_ids = [item.candidate.candidate_id for item in first.selected]
        self.assertEqual(first_ids, [item.candidate.candidate_id for item in second.selected])
        self.assertEqual(first_ids, ["early-a", "early-b", "late"])
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertEqual(sorted(item.selection_rank for item in first.selected), [1, 2, 3])

    def test_invalid_candidates_events_and_embeddings_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_id values must be unique"):
            select_keyframes(
                (candidate("same", 1.0), candidate("same", 2.0)),
                (),
                video_duration=3.0,
                config=SelectionConfig(max_gap_seconds=3.0),
            )
        with self.assertRaisesRegex(ValueError, "unknown candidates"):
            select_keyframes(
                (candidate("known", 1.0),),
                (ProtectedEvent("event", "ocr", ("missing",)),),
                video_duration=3.0,
                config=SelectionConfig(max_gap_seconds=3.0),
            )
        with self.assertRaisesRegex(ValueError, "shared dimension"):
            select_keyframes(
                (
                    candidate("two", 1.0, embedding=(1.0, 0.0)),
                    candidate("three", 2.0, embedding=(1.0, 0.0, 0.0)),
                ),
                (),
                video_duration=3.0,
                config=SelectionConfig(max_gap_seconds=3.0),
            )
        with self.assertRaisesRegex(ValueError, "zero vector"):
            candidate("zero", 1.0, embedding=(0.0, 0.0))

    def test_small_pool_randomized_results_match_brute_force_feasibility(self) -> None:
        rng = random.Random(42)
        for case_number in range(200):
            count = rng.randint(0, 8)
            values = tuple(
                candidate(
                    f"c{index}",
                    float(rng.randint(0, 8)),
                    importance=rng.random(),
                )
                for index in range(count)
            )
            event_count = rng.randint(0, min(3, count))
            events = tuple(
                ProtectedEvent(
                    f"e{event_index}",
                    "synthetic",
                    tuple(
                        value.candidate_id
                        for value in values
                        if rng.random() < 0.5
                    )
                    or (values[rng.randrange(count)].candidate_id,),
                )
                for event_index in range(event_count)
            )
            hard_cap = rng.randint(0, count)
            max_gap = float(rng.randint(2, 5))
            config = SelectionConfig(
                max_gap_seconds=max_gap,
                target_keyframes=0,
                hard_max_keyframes=hard_cap,
                protect_each_shot=False,
            )

            feasible = any(
                hard_constraints_hold(subset, events, duration=8.0, max_gap=max_gap)
                for size in range(hard_cap + 1)
                for subset in itertools.combinations(values, size)
            )
            result = select_keyframes(values, events, video_duration=8.0, config=config)

            with self.subTest(case=case_number, feasible=feasible):
                self.assertEqual(result.constraints_satisfied, feasible)
                if not feasible:
                    self.assertTrue(result.infeasibility_proven)


if __name__ == "__main__":
    unittest.main()
