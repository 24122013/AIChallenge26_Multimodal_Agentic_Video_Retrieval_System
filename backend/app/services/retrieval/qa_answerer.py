"""Evidence-grounded QA answering with lazy Qwen3.5 inference.

The answerer is deliberately isolated from candidate generation.  Callers keep
the evidence bundle even when generation is disabled or fails, so manual
inspection remains a deterministic fallback.  Only visual, caption, OCR and
object evidence is accepted; transcript/ASR fields are never forwarded to the
model or included in its cache key.
"""
from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


QA_ANSWER_MODES = ("off", "optional", "required")
QA_ANSWER_STATUSES = ("answered", "insufficient_evidence", "disabled", "error")
DEFAULT_QA_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_QA_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DEFAULT_QA_MODEL_CACHE_DIR = Path("data/model_cache/qa_answer")
QA_PROMPT_REVISION = "grounded-qa-v2"


class QaAnswerRunner(Protocol):
    """Injected/local inference contract used by :func:`answer_question`."""

    def __call__(
        self,
        question: str,
        evidence: Sequence[Mapping[str, object]],
        answer_type: str,
    ) -> Mapping[str, object] | str:
        ...


@dataclass(frozen=True)
class GroundedAnswer:
    """Validated public answer contract returned by the online QA pipeline."""

    status: str
    answer: str | None
    answer_type: str
    confidence: float
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        status = str(self.status).casefold().strip()
        if status not in QA_ANSWER_STATUSES:
            raise ValueError(f"Unsupported QA answer status: {self.status}")
        answer_type = str(self.answer_type).strip() or "unknown"
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        evidence_ids = tuple(
            dict.fromkeys(str(value).strip() for value in self.evidence_ids if str(value).strip())
        )
        answer = None if self.answer is None else " ".join(str(self.answer).split()).strip()
        reason = " ".join(str(self.reason).split()).strip()

        if status == "answered":
            if not answer:
                raise ValueError("answered status requires a non-empty answer")
            if not evidence_ids:
                raise ValueError("answered status requires at least one evidence_id")
        elif answer is not None:
            raise ValueError(f"{status} status requires answer=null")

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "answer_type", answer_type)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            **({"reason": self.reason} if self.reason else {}),
        }


@dataclass(frozen=True)
class QaAnswerReport:
    """Operational metadata kept separate from the user-facing answer."""

    mode: str
    status: str
    model_name: str
    model_revision: str
    prompt_revision: str
    evidence_count: int
    cache_hit: bool
    latency_ms: float
    manual_evidence_available: bool
    model_invoked: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RequiredQaAnswerError(RuntimeError):
    """Raised when ``qa_answer_mode=required`` cannot produce an answer.

    ``answer`` and ``report`` let an API layer expose the failure while still
    returning the already-retrieved manual evidence bundle.
    """

    def __init__(self, message: str, answer: GroundedAnswer, report: QaAnswerReport) -> None:
        super().__init__(message)
        self.answer = answer
        self.report = report


_GENERATION_LOCK = threading.RLock()
_RUNNER_CACHE_LOCK = threading.Lock()
_RUNNER_CACHE: dict[tuple[str, str], "LazyQwenGroundedRunner"] = {}


def answer_question(
    question: str,
    evidence: Sequence[Mapping[str, object] | object],
    *,
    answer_type: str = "unknown",
    mode: str = "off",
    cache_root: Path = Path("data/cache/qa_answers"),
    runner: QaAnswerRunner | None = None,
    model_name: str = DEFAULT_QA_MODEL,
    model_revision: str = DEFAULT_QA_MODEL_REVISION,
    max_evidence: int = 3,
    timeout_seconds: float = 120.0,
    answer_eligible: bool = True,
    preflight_block_reason: str = "",
) -> tuple[GroundedAnswer, QaAnswerReport]:
    """Answer one question from the top grounded evidence items.

    ``optional`` mode turns timeout/OOM/model/JSON failures into an ``error``
    answer and leaves manual evidence available.  ``required`` mode raises a
    :class:`RequiredQaAnswerError` carrying the same answer/report so the API
    can fail loudly without discarding evidence.
    """

    normalized_mode = str(mode).casefold().strip()
    if normalized_mode not in QA_ANSWER_MODES:
        raise ValueError(f"Unsupported QA answer mode: {mode}")
    cleaned_question = " ".join(str(question).split()).strip()
    if not cleaned_question:
        raise ValueError("question must not be empty")
    cleaned_answer_type = str(answer_type).strip() or "unknown"
    if int(max_evidence) < 1:
        raise ValueError("max_evidence must be >= 1")
    if not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0:
        raise ValueError("timeout_seconds must be finite and > 0")

    started = time.perf_counter()

    if not bool(answer_eligible):
        block_reason = (
            " ".join(str(preflight_block_reason).split()).strip()
            or "Answer generation was blocked by evidence preflight."
        )
        try:
            evidence_count = len(evidence[: int(max_evidence)])
        except Exception:
            evidence_count = 0
        answer = GroundedAnswer(
            status="insufficient_evidence",
            answer=None,
            answer_type=cleaned_answer_type,
            confidence=0.0,
            reason=block_reason,
        )
        return answer, _report(
            mode=normalized_mode,
            status="insufficient_evidence",
            model_name=model_name,
            model_revision=model_revision,
            evidence_count=evidence_count,
            cache_hit=False,
            started=started,
            manual_evidence_available=bool(evidence_count),
            model_invoked=False,
        )

    if normalized_mode == "off":
        try:
            evidence_count = len(evidence[: int(max_evidence)])
        except Exception:  # The disabled answerer must never break evidence retrieval.
            evidence_count = 0
        answer = GroundedAnswer(
            status="disabled",
            answer=None,
            answer_type=cleaned_answer_type,
            confidence=0.0,
            reason="qa_answer_mode=off",
        )
        return answer, _report(
            mode=normalized_mode,
            status="disabled",
            model_name=model_name,
            model_revision=model_revision,
            evidence_count=evidence_count,
            cache_hit=False,
            started=started,
            manual_evidence_available=bool(evidence_count),
            model_invoked=False,
        )

    selected: tuple[dict[str, object], ...] = ()
    raw_evidence_count = 0
    try:
        raw_evidence = evidence[: int(max_evidence)]
        raw_evidence_count = len(raw_evidence)
        selected = tuple(
            _sanitize_evidence(item, index)
            for index, item in enumerate(raw_evidence, start=1)
        )
        if not selected:
            answer = GroundedAnswer(
                status="insufficient_evidence",
                answer=None,
                answer_type=cleaned_answer_type,
                confidence=0.0,
                reason="No evidence was retrieved.",
            )
            return answer, _report(
                mode=normalized_mode,
                status="insufficient_evidence",
                model_name=model_name,
                model_revision=model_revision,
                evidence_count=0,
                cache_hit=False,
                started=started,
                manual_evidence_available=False,
                model_invoked=False,
            )

        known_evidence_ids = {str(item["evidence_id"]) for item in selected}
        cache_key = _cache_key(
            question=cleaned_question,
            answer_type=cleaned_answer_type,
            evidence=selected,
            model_name=model_name,
            model_revision=model_revision,
        )
        cache_path = Path(cache_root) / f"{cache_key}.json"
        cached = _load_cached_answer(
            cache_path,
            known_evidence_ids=known_evidence_ids,
            expected_answer_type=cleaned_answer_type,
        )
        if cached is not None:
            return cached, _report(
                mode=normalized_mode,
                status=cached.status,
                model_name=model_name,
                model_revision=model_revision,
                evidence_count=len(selected),
                cache_hit=True,
                started=started,
                manual_evidence_available=True,
                model_invoked=False,
            )

        scorer = runner or _shared_local_qwen_runner(model_name, model_revision)
        model_invoked = True
        payload = _call_with_timeout(
            scorer,
            cleaned_question,
            selected,
            cleaned_answer_type,
            timeout_seconds=float(timeout_seconds),
        )
        answer = _parse_answer_payload(
            payload,
            known_evidence_ids=known_evidence_ids,
            expected_answer_type=cleaned_answer_type,
        )
        _write_cached_answer(cache_path, answer)
        return answer, _report(
            mode=normalized_mode,
            status=answer.status,
            model_name=model_name,
            model_revision=model_revision,
            evidence_count=len(selected),
            cache_hit=False,
            started=started,
            manual_evidence_available=True,
            model_invoked=True,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        answer = GroundedAnswer(
            status="error",
            answer=None,
            answer_type=cleaned_answer_type,
            confidence=0.0,
            reason=reason,
        )
        report = _report(
            mode=normalized_mode,
            status="fallback" if normalized_mode == "optional" else "failed",
            model_name=model_name,
            model_revision=model_revision,
            evidence_count=len(selected) or raw_evidence_count,
            cache_hit=False,
            started=started,
            manual_evidence_available=bool(raw_evidence_count),
            model_invoked=locals().get("model_invoked", False),
            fallback_reason=reason,
        )
        if normalized_mode == "required":
            raise RequiredQaAnswerError(
                f"Required grounded QA failed: {reason}",
                answer,
                report,
            ) from exc
        return answer, report


def _report(
    *,
    mode: str,
    status: str,
    model_name: str,
    model_revision: str,
    evidence_count: int,
    cache_hit: bool,
    started: float,
    manual_evidence_available: bool,
    model_invoked: bool,
    fallback_reason: str = "",
) -> QaAnswerReport:
    return QaAnswerReport(
        mode=mode,
        status=status,
        model_name=model_name,
        model_revision=model_revision,
        prompt_revision=QA_PROMPT_REVISION,
        evidence_count=evidence_count,
        cache_hit=cache_hit,
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        manual_evidence_available=manual_evidence_available,
        model_invoked=model_invoked,
        fallback_reason=fallback_reason,
    )


def _sanitize_evidence(item: Mapping[str, object] | object, index: int) -> dict[str, object]:
    if isinstance(item, Mapping):
        raw = dict(item)
    elif hasattr(item, "to_dict"):
        value = item.to_dict()
        if not isinstance(value, Mapping):
            raise TypeError("evidence.to_dict() must return a mapping")
        raw = dict(value)
    elif is_dataclass(item):
        raw = asdict(item)
    else:
        raise TypeError("evidence items must be mappings, dataclasses, or expose to_dict()")

    evidence_id = str(raw.get("evidence_id") or f"E{index:03d}").strip()
    if not evidence_id:
        raise ValueError("evidence_id must not be empty")
    objects_raw = raw.get("objects", ())
    if isinstance(objects_raw, str):
        objects = [objects_raw]
    elif isinstance(objects_raw, Sequence):
        objects = [str(value) for value in objects_raw if str(value).strip()]
    else:
        objects = []
    modalities_raw = raw.get("source_modalities", ())
    if isinstance(modalities_raw, str):
        modalities = [modalities_raw]
    elif isinstance(modalities_raw, Sequence):
        modalities = [str(value) for value in modalities_raw if str(value).strip()]
    else:
        modalities = []
    warnings_raw = raw.get("warnings", raw.get("evidence_warnings", ()))
    if isinstance(warnings_raw, str):
        warnings = [warnings_raw]
    elif isinstance(warnings_raw, Sequence):
        warnings = [str(value) for value in warnings_raw if str(value).strip()]
    else:
        warnings = []

    image_path = (
        raw.get("image_path")
        or raw.get("keyframe_path")
        or raw.get("thumbnail_path")
        or ""
    )
    timestamp = raw.get("timestamp", 0.0)
    retrieval_score = raw.get("retrieval_score", raw.get("score", 0.0))
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError):
        timestamp = 0.0
    try:
        retrieval_score = float(retrieval_score)
    except (TypeError, ValueError):
        retrieval_score = 0.0
    base_retrieval_score = raw.get("base_retrieval_score", retrieval_score)
    constraint_score = raw.get("constraint_score", 0.0)
    try:
        base_retrieval_score = float(base_retrieval_score)
    except (TypeError, ValueError):
        base_retrieval_score = retrieval_score
    try:
        constraint_score = float(constraint_score)
    except (TypeError, ValueError):
        constraint_score = 0.0

    matched_raw = raw.get("matched_constraints", ())
    if isinstance(matched_raw, Mapping):
        matched_constraints: object = {
            str(key): (
                [str(entry) for entry in value if str(entry).strip()]
                if isinstance(value, Sequence) and not isinstance(value, str)
                else str(value)
            )
            for key, value in matched_raw.items()
        }
    elif isinstance(matched_raw, Sequence) and not isinstance(matched_raw, str):
        matched_constraints = [
            str(value) for value in matched_raw if str(value).strip()
        ]
    else:
        matched_constraints = []

    def optional_int(name: str) -> int | None:
        value = raw.get(name)
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def optional_float(name: str) -> float | None:
        value = raw.get(name)
        if value is None or value == "":
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    # Keep this allowlist explicit.  In particular, ASR/transcript fields are
    # intentionally excluded from both prompting and cache identity.
    return {
        "evidence_id": evidence_id,
        "video_id": str(raw.get("video_id") or ""),
        "frame_id": str(raw.get("frame_id") or ""),
        "shot_id": str(raw.get("shot_id") or ""),
        "timestamp": timestamp,
        "image_path": str(image_path),
        "caption": " ".join(str(raw.get("caption") or "").split()),
        "ocr_text": " ".join(str(raw.get("ocr_text") or "").split()),
        "objects": objects,
        "source_modalities": modalities,
        "retrieval_score": retrieval_score,
        "base_retrieval_score": base_retrieval_score,
        "constraint_score": constraint_score,
        "matched_constraints": matched_constraints,
        "temporal_event_index": optional_int("temporal_event_index"),
        "temporal_match_rank": optional_int("temporal_match_rank"),
        "temporal_match_mode": str(raw.get("temporal_match_mode") or ""),
        "temporal_chain_id": str(raw.get("temporal_chain_id") or ""),
        "temporal_event_query": " ".join(
            str(raw.get("temporal_event_query") or "").split()
        ),
        "temporal_event_role": str(raw.get("temporal_event_role") or ""),
        "temporal_chain_score": optional_float("temporal_chain_score"),
        "warnings": warnings,
    }


def _cache_key(
    *,
    question: str,
    answer_type: str,
    evidence: Sequence[Mapping[str, object]],
    model_name: str,
    model_revision: str,
) -> str:
    digest = hashlib.sha256()
    for value in (
        question,
        answer_type,
        model_name,
        model_revision,
        QA_PROMPT_REVISION,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for item in evidence:
        digest.update(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        image_path = Path(str(item.get("image_path") or ""))
        if image_path.is_file():
            with image_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _load_cached_answer(
    path: Path,
    *,
    known_evidence_ids: set[str],
    expected_answer_type: str,
) -> GroundedAnswer | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _parse_answer_payload(
            payload,
            known_evidence_ids=known_evidence_ids,
            expected_answer_type=expected_answer_type,
        )
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_cached_answer(path: Path, answer: GroundedAnswer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
    temporary.write_text(
        json.dumps(answer.to_dict(), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_answer_payload(
    payload: Mapping[str, object] | str,
    *,
    known_evidence_ids: set[str],
    expected_answer_type: str,
) -> GroundedAnswer:
    if isinstance(payload, str):
        value = _extract_json_object(payload)
    elif isinstance(payload, Mapping):
        value = dict(payload)
    else:
        raise TypeError("QA model output must be a JSON object or JSON string")
    missing = [
        field
        for field in ("status", "answer", "answer_type", "confidence", "evidence_ids")
        if field not in value
    ]
    if missing:
        raise ValueError("QA model output is missing: " + ", ".join(missing))
    if not isinstance(value["status"], str):
        raise TypeError("status must be a string")
    if str(value["status"]).casefold().strip() not in {
        "answered",
        "insufficient_evidence",
    }:
        raise ValueError("QA model status must be answered or insufficient_evidence")
    if value["answer"] is not None and not isinstance(value["answer"], str):
        raise TypeError("answer must be a string or null")
    if not isinstance(value["answer_type"], str):
        raise TypeError("answer_type must be a string")
    if isinstance(value["confidence"], bool) or not isinstance(value["confidence"], (int, float)):
        raise TypeError("confidence must be a number")
    evidence_ids_raw = value["evidence_ids"]
    if not isinstance(evidence_ids_raw, list) or any(
        not isinstance(item, str) for item in evidence_ids_raw
    ):
        raise TypeError("evidence_ids must be a list of strings")
    answer = GroundedAnswer(
        status=str(value["status"]),
        answer=None if value["answer"] is None else str(value["answer"]),
        answer_type=str(value["answer_type"]),
        confidence=float(value["confidence"]),
        evidence_ids=tuple(evidence_ids_raw),
        reason=str(value.get("reason") or ""),
    )
    unknown = set(answer.evidence_ids).difference(known_evidence_ids)
    if unknown:
        raise ValueError("QA model cited unknown evidence_ids: " + ", ".join(sorted(unknown)))
    if (
        expected_answer_type != "unknown"
        and answer.answer_type.casefold() != expected_answer_type.casefold()
    ):
        raise ValueError(
            f"QA model answer_type {answer.answer_type!r} does not match "
            f"expected {expected_answer_type!r}"
        )
    return answer


def _extract_json_object(text: str) -> Mapping[str, object]:
    cleaned = str(text).strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("QA model returned malformed JSON")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, Mapping):
        raise TypeError("QA model JSON must be an object")
    return value


def _call_with_timeout(
    runner: QaAnswerRunner,
    question: str,
    evidence: Sequence[Mapping[str, object]],
    answer_type: str,
    *,
    timeout_seconds: float,
) -> Mapping[str, object] | str:
    output: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            with _GENERATION_LOCK:
                value = runner(question, evidence, answer_type)
            output.put((True, value))
        except BaseException as exc:  # propagate runner failures to caller
            output.put((False, exc))

    worker = threading.Thread(target=invoke, name="qa-answerer", daemon=True)
    worker.start()
    try:
        succeeded, value = output.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(f"Grounded QA exceeded {timeout_seconds:g}s") from exc
    if not succeeded:
        if isinstance(value, BaseException):
            raise value
        raise RuntimeError(str(value))
    return value  # type: ignore[return-value]


def build_grounded_prompt(
    question: str,
    evidence: Sequence[Mapping[str, object]],
    answer_type: str,
) -> str:
    """Build the deterministic JSON-only prompt used by the local runner."""

    evidence_text: list[str] = []
    has_temporal_lineage = any(
        item.get("temporal_event_index") is not None for item in evidence
    )
    for item in evidence:
        evidence_text.append(
            json.dumps(
                {
                    "evidence_id": item["evidence_id"],
                    "video_id": item.get("video_id", ""),
                    "frame_id": item.get("frame_id", ""),
                    "shot_id": item.get("shot_id", ""),
                    "timestamp": item.get("timestamp", 0.0),
                    "caption": item.get("caption", ""),
                    "ocr_text": item.get("ocr_text", ""),
                    "objects": item.get("objects", []),
                    "temporal_event_index": item.get("temporal_event_index"),
                    "temporal_match_rank": item.get("temporal_match_rank"),
                    "temporal_match_mode": item.get("temporal_match_mode", ""),
                    "temporal_chain_id": item.get("temporal_chain_id", ""),
                    "temporal_event_query": item.get("temporal_event_query", ""),
                    "temporal_event_role": item.get("temporal_event_role", ""),
                    "temporal_chain_score": item.get("temporal_chain_score"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    joined = "\n".join(evidence_text)
    temporal_instruction = (
        " Evidence entries form one strict temporal chain; respect ascending "
        "temporal_event_index, temporal_event_query, and temporal_event_role; "
        "answer from answer_target evidence (or the whole_chain proposition) "
        "without collapsing separate events."
        if has_temporal_lineage
        else ""
    )
    return (
        "Answer the question using only the supplied visual evidence. "
        "Never use outside knowledge. If the evidence does not directly support an answer, "
        "return status=insufficient_evidence and answer=null. Cite only supplied evidence_id "
        f"values.{temporal_instruction} Return strict JSON only with exactly these fields: "
        '{"status":"answered|insufficient_evidence","answer":"string|null",'
        '"answer_type":"string","confidence":0.0,"evidence_ids":["E001"]}.\n'
        f"Expected answer_type: {answer_type}\n"
        f"Question: {question}\n"
        f"Evidence:\n{joined}"
    )


class LazyQwenGroundedRunner:
    """Lazy local Qwen3.5 runner; defaults to 4-bit on CUDA and CPU otherwise."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QA_MODEL,
        model_revision: str = DEFAULT_QA_MODEL_REVISION,
        device: str | None = None,
        quantization: str = "auto",
        cache_dir: Path | None = DEFAULT_QA_MODEL_CACHE_DIR,
        local_files_only: bool = False,
        max_new_tokens: int = 192,
    ) -> None:
        if quantization not in {"auto", "none", "4bit"}:
            raise ValueError("quantization must be auto, none, or 4bit")
        if int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be >= 1")
        self.model_name = model_name
        self.model_revision = model_revision
        self.requested_device = device
        self.requested_quantization = quantization
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self.max_new_tokens = int(max_new_tokens)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._quantization = "none"
        self._dtype: Any | None = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise RuntimeError(
                    "Qwen3.5 grounded QA requires torch and Transformers with "
                    "AutoModelForMultimodalLM support"
                ) from exc

            requested_device = str(self.requested_device or "auto").casefold()
            device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            ) if requested_device == "auto" else requested_device
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            quantization = self.requested_quantization
            if quantization == "auto":
                quantization = "4bit" if device == "cuda" else "none"
            if quantization == "4bit" and device != "cuda":
                raise ValueError("4-bit QA inference requires CUDA")
            dtype = (
                torch.bfloat16
                if device == "cuda" and torch.cuda.is_bf16_supported()
                else (torch.float16 if device == "cuda" else torch.float32)
            )
            kwargs: dict[str, object] = {
                "revision": self.model_revision,
                # Transformers 5.x routes Qwen3.5 through the native
                # multimodal auto class and uses the unified ``dtype`` kwarg.
                "dtype": dtype,
            }
            processor_kwargs: dict[str, object] = {"revision": self.model_revision}
            kwargs["local_files_only"] = self.local_files_only
            processor_kwargs["local_files_only"] = self.local_files_only
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                kwargs["cache_dir"] = str(self.cache_dir)
                processor_kwargs["cache_dir"] = str(self.cache_dir)
            if quantization == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:  # pragma: no cover - optional dependency path
                    raise RuntimeError("4-bit QA inference requires bitsandbytes") from exc
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                )
                kwargs["device_map"] = "auto"
            processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)
            model = AutoModelForMultimodalLM.from_pretrained(self.model_name, **kwargs)
            if quantization == "none":
                model = model.to(device)
            model.eval()
            self._torch = torch
            self._device = device
            self._quantization = quantization
            self._dtype = dtype
            self._processor = processor
            self._model = model

    def __call__(
        self,
        question: str,
        evidence: Sequence[Mapping[str, object]],
        answer_type: str,
    ) -> str:
        self._load()
        from PIL import Image

        opened_images: list[Any] = []
        content: list[dict[str, object]] = []
        try:
            for item in evidence:
                path = Path(str(item.get("image_path") or ""))
                if not path.is_file():
                    continue
                image = Image.open(path).convert("RGB")
                opened_images.append(image)
                content.extend(
                    [
                        {"type": "text", "text": f"Image for {item['evidence_id']}:"},
                        {"type": "image", "image": image},
                    ]
                )
            if not opened_images:
                raise ValueError(
                    "Grounded QA requires at least one readable evidence image"
                )
            content.append(
                {
                    "type": "text",
                    "text": build_grounded_prompt(question, evidence, answer_type),
                }
            )
            messages = [{"role": "user", "content": content}]
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_device = getattr(self._model, "device", self._torch.device(self._device))
            inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            for key, value in list(inputs.items()):
                if (
                    self._device == "cuda"
                    and hasattr(value, "is_floating_point")
                    and value.is_floating_point()
                ):
                    inputs[key] = value.to(dtype=self._dtype)
            input_length = int(inputs["input_ids"].shape[1])
            with self._torch.inference_mode():
                tokens = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generated = tokens[:, input_length:]
            return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        finally:
            for image in opened_images:
                image.close()


def build_local_qwen_runner(
    *,
    model_name: str = DEFAULT_QA_MODEL,
    model_revision: str = DEFAULT_QA_MODEL_REVISION,
    device: str | None = None,
    quantization: str = "auto",
    cache_dir: Path | None = DEFAULT_QA_MODEL_CACHE_DIR,
    local_files_only: bool = False,
    max_new_tokens: int = 192,
) -> QaAnswerRunner:
    """Create a callable whose processor/model allocation occurs on first use."""

    return LazyQwenGroundedRunner(
        model_name=model_name,
        model_revision=model_revision,
        device=device,
        quantization=quantization,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        max_new_tokens=max_new_tokens,
    )


def _shared_local_qwen_runner(
    model_name: str,
    model_revision: str,
) -> LazyQwenGroundedRunner:
    """Reuse one lazily allocated model per pinned name/revision."""

    key = (model_name, model_revision)
    runner = _RUNNER_CACHE.get(key)
    if runner is not None:
        return runner
    with _RUNNER_CACHE_LOCK:
        runner = _RUNNER_CACHE.get(key)
        if runner is None:
            runner = LazyQwenGroundedRunner(
                model_name=model_name,
                model_revision=model_revision,
            )
            _RUNNER_CACHE[key] = runner
    return runner
