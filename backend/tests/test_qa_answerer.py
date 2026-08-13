from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.services.retrieval.qa_answerer import (
    GroundedAnswer,
    LazyQwenGroundedRunner,
    RequiredQaAnswerError,
    answer_question,
    build_grounded_prompt,
)


def _evidence(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "evidence_id": "E001",
        "video_id": "video-1",
        "frame_id": "frame-10",
        "shot_id": "shot-2",
        "timestamp": 4.25,
        "image_path": "",
        "caption": "A woman in a red shirt holds a phone.",
        "ocr_text": "",
        "objects": ["person", "phone"],
        "source_modalities": ["visual", "caption", "objects"],
        "retrieval_score": 0.91,
    }
    value.update(overrides)
    return value


def _answered_payload(evidence_id: str = "E001") -> dict[str, object]:
    return {
        "status": "answered",
        "answer": "một chiếc điện thoại",
        "answer_type": "object",
        "confidence": 0.86,
        "evidence_ids": [evidence_id],
    }


class GroundedAnswerContractTest(unittest.TestCase):
    def test_answered_requires_answer_and_citation(self) -> None:
        answer = GroundedAnswer(
            status="ANSWERED",
            answer="  một   chiếc điện thoại ",
            answer_type="object",
            confidence=0.8,
            evidence_ids=("E001", "E001"),
        )
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.answer, "một chiếc điện thoại")
        self.assertEqual(answer.evidence_ids, ("E001",))
        with self.assertRaisesRegex(ValueError, "non-empty answer"):
            GroundedAnswer(
                status="answered",
                answer=None,
                answer_type="object",
                confidence=0.5,
                evidence_ids=("E001",),
            )
        with self.assertRaisesRegex(ValueError, "evidence_id"):
            GroundedAnswer(
                status="answered",
                answer="phone",
                answer_type="object",
                confidence=0.5,
            )

    def test_non_answer_status_requires_null_answer(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires answer=null"):
            GroundedAnswer(
                status="insufficient_evidence",
                answer="guess",
                answer_type="object",
                confidence=0.1,
            )


class QaAnswererTest(unittest.TestCase):
    def test_off_mode_never_calls_runner(self) -> None:
        called = False

        def runner(*_: object) -> dict[str, object]:
            nonlocal called
            called = True
            return _answered_payload()

        with tempfile.TemporaryDirectory() as temporary:
            answer, report = answer_question(
                "Người phụ nữ cầm gì?",
                [_evidence()],
                answer_type="object",
                mode="off",
                cache_root=Path(temporary),
                runner=runner,
            )
            self.assertEqual(answer.status, "disabled")
            self.assertEqual(report.status, "disabled")
            self.assertTrue(report.manual_evidence_available)
            self.assertFalse(called)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_no_evidence_abstains_without_loading_model(self) -> None:
        def runner(*_: object) -> dict[str, object]:
            raise AssertionError("runner must not be called")

        with tempfile.TemporaryDirectory() as temporary:
            answer, report = answer_question(
                "What is she holding?",
                [],
                answer_type="object",
                mode="required",
                cache_root=Path(temporary),
                runner=runner,
            )
            self.assertEqual(answer.status, "insufficient_evidence")
            self.assertEqual(report.status, "insufficient_evidence")
            self.assertFalse(report.manual_evidence_available)

    def test_optional_success_is_cached_by_question_evidence_and_revision(self) -> None:
        calls = 0

        def runner(*_: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return _answered_payload()

        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            first, first_report = answer_question(
                "Người phụ nữ cầm gì?",
                [_evidence()],
                answer_type="object",
                mode="optional",
                cache_root=cache_root,
                runner=runner,
            )
            second, second_report = answer_question(
                "Người phụ nữ cầm gì?",
                [_evidence()],
                answer_type="object",
                mode="optional",
                cache_root=cache_root,
                runner=runner,
            )
            third, third_report = answer_question(
                "Người phụ nữ cầm gì?",
                [_evidence(caption="A woman holds a cup.")],
                answer_type="object",
                mode="optional",
                cache_root=cache_root,
                runner=runner,
            )
            fourth, fourth_report = answer_question(
                "Người phụ nữ cầm gì?",
                [_evidence()],
                answer_type="object",
                mode="optional",
                cache_root=cache_root,
                runner=runner,
                model_revision="different-revision",
            )

        self.assertEqual(first.status, "answered")
        self.assertEqual(second, first)
        self.assertFalse(first_report.cache_hit)
        self.assertTrue(second_report.cache_hit)
        self.assertFalse(third_report.cache_hit)
        self.assertFalse(fourth_report.cache_hit)
        self.assertEqual(third.status, "answered")
        self.assertEqual(fourth.status, "answered")
        self.assertEqual(calls, 3)

    def test_unknown_evidence_citation_falls_back_in_optional_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            answer, report = answer_question(
                "What is she holding?",
                [_evidence()],
                answer_type="object",
                mode="optional",
                cache_root=Path(temporary),
                runner=lambda *_: _answered_payload("E999"),
            )
        self.assertEqual(answer.status, "error")
        self.assertEqual(report.status, "fallback")
        self.assertIn("unknown evidence_ids", report.fallback_reason)
        self.assertTrue(report.manual_evidence_available)

    def test_cache_identity_failure_preserves_optional_manual_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "backend.app.services.retrieval.qa_answerer._cache_key",
            side_effect=OSError("image cannot be read"),
        ):
            answer, report = answer_question(
                "What is she holding?",
                [_evidence()],
                answer_type="object",
                mode="optional",
                cache_root=Path(temporary),
                runner=lambda *_: _answered_payload(),
            )
        self.assertEqual(answer.status, "error")
        self.assertEqual(report.status, "fallback")
        self.assertTrue(report.manual_evidence_available)
        self.assertIn("image cannot be read", report.fallback_reason)

    def test_model_answer_type_must_match_parser(self) -> None:
        payload = _answered_payload()
        payload["answer_type"] = "color"
        with tempfile.TemporaryDirectory() as temporary:
            answer, report = answer_question(
                "What is she holding?",
                [_evidence()],
                answer_type="object",
                mode="optional",
                cache_root=Path(temporary),
                runner=lambda *_: payload,
            )
        self.assertEqual(answer.status, "error")
        self.assertIn("does not match", report.fallback_reason)

    def test_required_failure_raises_with_manual_fallback_contract(self) -> None:
        def oom(*_: object) -> dict[str, object]:
            raise RuntimeError("CUDA out of memory")

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RequiredQaAnswerError) as caught:
                answer_question(
                    "What is she holding?",
                    [_evidence()],
                    answer_type="object",
                    mode="required",
                    cache_root=Path(temporary),
                    runner=oom,
                )
        self.assertEqual(caught.exception.answer.status, "error")
        self.assertEqual(caught.exception.report.status, "failed")
        self.assertTrue(caught.exception.report.manual_evidence_available)
        self.assertIn("CUDA out of memory", caught.exception.report.fallback_reason)

    def test_timeout_returns_quick_optional_fallback(self) -> None:
        def slow(*_: object) -> dict[str, object]:
            time.sleep(0.25)
            return _answered_payload()

        started = time.perf_counter()
        with tempfile.TemporaryDirectory() as temporary:
            answer, report = answer_question(
                "What is she holding?",
                [_evidence()],
                answer_type="object",
                mode="optional",
                cache_root=Path(temporary),
                runner=slow,
                timeout_seconds=0.02,
            )
        elapsed = time.perf_counter() - started
        self.assertEqual(answer.status, "error")
        self.assertEqual(report.status, "fallback")
        self.assertIn("TimeoutError", report.fallback_reason)
        self.assertLess(elapsed, 0.2)

    def test_runner_receives_only_top_three_and_no_asr_fields(self) -> None:
        observed: list[dict[str, object]] = []

        def runner(
            _: str,
            evidence: object,
            __: str,
        ) -> dict[str, object]:
            observed.extend(dict(item) for item in evidence)  # type: ignore[arg-type]
            return _answered_payload()

        items = [
            _evidence(
                evidence_id=f"E{index:03d}",
                asr_text="secret transcript",
                transcript="also excluded",
            )
            for index in range(1, 6)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            answer, _ = answer_question(
                "What is she holding?",
                items,
                answer_type="object",
                mode="optional",
                cache_root=Path(temporary),
                runner=runner,
            )
        self.assertEqual(answer.status, "answered")
        self.assertEqual(len(observed), 3)
        self.assertTrue(all("asr_text" not in item for item in observed))
        self.assertTrue(all("transcript" not in item for item in observed))

    def test_injected_generation_is_serialized(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def runner(*_: object) -> dict[str, object]:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return _answered_payload()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            errors: list[BaseException] = []

            def invoke(question: str) -> None:
                try:
                    answer_question(
                        question,
                        [_evidence()],
                        answer_type="object",
                        mode="optional",
                        cache_root=root,
                        runner=runner,
                    )
                except BaseException as exc:  # pragma: no cover - assertion aid
                    errors.append(exc)

            threads = [
                threading.Thread(target=invoke, args=(f"Question {index}",))
                for index in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(max_active, 1)


class PromptAndLazyRunnerTest(unittest.TestCase):
    def test_prompt_is_grounded_json_only_and_has_no_transcript(self) -> None:
        prompt = build_grounded_prompt(
            "Người phụ nữ cầm gì?",
            [_evidence(asr_text="must not appear")],
            "object",
        )
        self.assertIn("using only the supplied visual evidence", prompt)
        self.assertIn('"evidence_id": "E001"', prompt)
        self.assertNotIn("must not appear", prompt)

    def test_lazy_runner_validates_4bit_cpu_before_model_load(self) -> None:
        runner = LazyQwenGroundedRunner(device="cpu", quantization="4bit")
        self.assertIsNone(runner._model)
        self.assertEqual(runner.requested_quantization, "4bit")


if __name__ == "__main__":
    unittest.main()
