from __future__ import annotations

import tempfile
import threading
import time
import unittest
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

from PIL import Image

from backend.app.services.retrieval.qa_answerer import (
    GroundedAnswer,
    LazyQwenGroundedRunner,
    DEFAULT_QA_MODEL,
    DEFAULT_QA_MODEL_REVISION,
    QA_PROMPT_REVISION,
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
            self.assertFalse(report.model_invoked)
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
            self.assertFalse(report.model_invoked)

    def test_failed_preflight_abstains_in_required_mode_without_model(self) -> None:
        called = False

        def runner(*_: object) -> dict[str, object]:
            nonlocal called
            called = True
            return _answered_payload()

        with tempfile.TemporaryDirectory() as temporary:
            answer, report = answer_question(
                "A rồi B?",
                [_evidence()],
                answer_type="yes_no",
                mode="required",
                cache_root=Path(temporary),
                runner=runner,
                answer_eligible=False,
                preflight_block_reason="temporal_match_not_strict:relaxed_gap",
            )
        self.assertFalse(called)
        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertEqual(
            answer.reason,
            "temporal_match_not_strict:relaxed_gap",
        )
        self.assertEqual(report.status, "insufficient_evidence")
        self.assertFalse(report.model_invoked)

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
        self.assertTrue(first_report.model_invoked)
        self.assertTrue(second_report.cache_hit)
        self.assertFalse(second_report.model_invoked)
        self.assertFalse(third_report.cache_hit)
        self.assertFalse(fourth_report.cache_hit)
        self.assertEqual(third.status, "answered")
        self.assertEqual(fourth.status, "answered")
        self.assertEqual(calls, 3)

    def test_temporal_lineage_is_part_of_cache_identity(self) -> None:
        calls = 0

        def runner(*_: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            payload = _answered_payload()
            payload["answer"] = "waves"
            payload["answer_type"] = "action"
            return payload

        common = {
            "temporal_event_index": 0,
            "temporal_match_rank": 1,
            "temporal_match_mode": "strict",
            "temporal_chain_id": "TC-test",
            "temporal_chain_score": 0.75,
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            _, first_report = answer_question(
                "What did he do before leaving?",
                [
                    _evidence(
                        **common,
                        temporal_event_query="the man",
                        temporal_event_role="answer_target",
                    )
                ],
                answer_type="action",
                mode="optional",
                cache_root=cache_root,
                runner=runner,
            )
            _, second_report = answer_question(
                "What did he do before leaving?",
                [
                    _evidence(
                        **common,
                        temporal_event_query="he left",
                        temporal_event_role="context",
                    )
                ],
                answer_type="action",
                mode="optional",
                cache_root=cache_root,
                runner=runner,
            )

        self.assertFalse(first_report.cache_hit)
        self.assertFalse(second_report.cache_hit)
        self.assertEqual(calls, 2)

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
        self.assertTrue(report.model_invoked)

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
        self.assertTrue(caught.exception.report.model_invoked)

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
    def test_qwen35_2b_multimodal_loader_json_and_citation_contract(self) -> None:
        import torch

        transformers = ModuleType("transformers")
        model_calls: list[tuple[str, dict[str, Any]]] = []
        processor_calls: list[tuple[str, dict[str, Any]]] = []
        messages_seen: list[list[dict[str, object]]] = []
        quantization_calls: list[dict[str, Any]] = []

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs: Any) -> None:
                quantization_calls.append(kwargs)

        class FakeModel:
            device = torch.device("cpu")

            def eval(self) -> None:
                return None

            def generate(self, **kwargs: Any) -> Any:
                self.generate_kwargs = kwargs
                return torch.tensor([[10, 11, 12]])

        class FakeModelFactory:
            @classmethod
            def from_pretrained(cls, name: str, **kwargs: Any) -> FakeModel:
                model_calls.append((name, kwargs))
                return FakeModel()

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, name: str, **kwargs: Any) -> "FakeProcessor":
                processor_calls.append((name, kwargs))
                return cls()

            def apply_chat_template(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
                messages_seen.extend(messages)
                return {
                    "input_ids": torch.tensor([[10, 11]]),
                    "pixel_values": torch.zeros((1, 3, 2, 2)),
                }

            def batch_decode(self, *_: Any, **__: Any) -> list[str]:
                return [
                    '{"status":"answered","answer":"muỗng","answer_type":'
                    '"object","confidence":0.9,"evidence_ids":["E001"]}'
                ]

        transformers.AutoModelForMultimodalLM = FakeModelFactory  # type: ignore[attr-defined]
        transformers.AutoProcessor = FakeProcessor  # type: ignore[attr-defined]
        transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "evidence.jpg"
            Image.new("RGB", (8, 8), color="white").save(image_path)
            runner = LazyQwenGroundedRunner(
                device="cuda",
                quantization="auto",
                cache_dir=root / "qa-model-cache",
            )
            with (
                patch.dict(sys.modules, {"transformers": transformers}),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "is_bf16_supported", return_value=False),
            ):
                answer, report = answer_question(
                    "Dụng cụ nào đang được dùng?",
                    [_evidence(image_path=str(image_path))],
                    answer_type="object",
                    mode="required",
                    cache_root=root / "answer-cache",
                    runner=runner,
                )

        self.assertEqual(DEFAULT_QA_MODEL, "Qwen/Qwen3.5-2B")
        self.assertEqual(
            DEFAULT_QA_MODEL_REVISION,
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
        )
        self.assertEqual(model_calls[0][0], DEFAULT_QA_MODEL)
        self.assertEqual(model_calls[0][1]["revision"], DEFAULT_QA_MODEL_REVISION)
        self.assertIn("dtype", model_calls[0][1])
        self.assertNotIn("torch_dtype", model_calls[0][1])
        self.assertEqual(processor_calls[0][0], DEFAULT_QA_MODEL)
        self.assertTrue(quantization_calls[0]["load_in_4bit"])
        self.assertEqual(runner._quantization, "4bit")
        content = messages_seen[0]["content"]  # type: ignore[index]
        self.assertTrue(any(item.get("type") == "image" for item in content))  # type: ignore[union-attr]
        joined_text = " ".join(str(item.get("text", "")) for item in content)  # type: ignore[union-attr]
        self.assertIn("Dụng cụ nào đang được dùng?", joined_text)
        self.assertIn("E001", joined_text)
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.evidence_ids, ("E001",))
        self.assertEqual(report.model_name, "Qwen/Qwen3.5-2B")

    def test_prompt_is_grounded_json_only_and_has_no_transcript(self) -> None:
        prompt = build_grounded_prompt(
            "Người phụ nữ cầm gì?",
            [_evidence(asr_text="must not appear")],
            "object",
        )
        self.assertIn("using only the supplied visual evidence", prompt)
        self.assertIn('"evidence_id": "E001"', prompt)
        self.assertNotIn("must not appear", prompt)
        self.assertEqual(QA_PROMPT_REVISION, "grounded-qa-v2")

    def test_prompt_preserves_temporal_chain_lineage(self) -> None:
        prompt = build_grounded_prompt(
            "What happened before he left?",
            [
                _evidence(
                    temporal_event_index=0,
                    temporal_match_rank=1,
                    temporal_match_mode="strict",
                    temporal_chain_id="TC-test",
                    temporal_event_query="what happened",
                    temporal_event_role="answer_target",
                    temporal_chain_score=0.75,
                )
            ],
            "action",
        )

        self.assertIn('"temporal_chain_id": "TC-test"', prompt)
        self.assertIn('"temporal_event_role": "answer_target"', prompt)
        self.assertIn('"temporal_chain_score": 0.75', prompt)

    def test_lazy_runner_validates_4bit_cpu_before_model_load(self) -> None:
        runner = LazyQwenGroundedRunner(device="cpu", quantization="4bit")
        self.assertIsNone(runner._model)
        self.assertEqual(runner.requested_quantization, "4bit")


if __name__ == "__main__":
    unittest.main()
