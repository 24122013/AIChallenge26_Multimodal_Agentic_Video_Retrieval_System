"""End-to-end QA orchestration: evidence retrieval then grounded answering."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.app.services.retrieval.qa_answerer import (
    DEFAULT_QA_MODEL,
    DEFAULT_QA_MODEL_REVISION,
    QaAnswerRunner,
    RequiredQaAnswerError,
    answer_question,
)
from backend.app.services.retrieval.qa_evidence import QaEvidenceSearchEngine


@dataclass(frozen=True)
class QaPipelineConfig:
    answer_mode: str = "off"
    model_name: str = DEFAULT_QA_MODEL
    model_revision: str = DEFAULT_QA_MODEL_REVISION
    answer_cache_root: Path = Path("data/cache/qa_answers")
    answer_timeout_seconds: float = 120.0
    experiment_id: str = "qa-parser-router-evidence-v1"


class RequiredQaPipelineError(RuntimeError):
    """Required answer failure with the already-retrieved evidence response."""

    def __init__(self, message: str, response: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.response = dict(response)


class QaSearchPipeline:
    """Keep answer generation optional while always preserving evidence."""

    def __init__(
        self,
        evidence_engine: QaEvidenceSearchEngine,
        *,
        config: QaPipelineConfig | None = None,
        answer_runner: QaAnswerRunner | None = None,
    ) -> None:
        self.evidence_engine = evidence_engine
        self.config = config or QaPipelineConfig()
        self.answer_runner = answer_runner

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        task_mode: str = "qa",
        expanded_queries: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        evidence_response = self.evidence_engine.search(
            query,
            top_k=top_k,
            task_mode=task_mode,
            expanded_queries=expanded_queries,
        )
        evidence = _manual_evidence(evidence_response)
        feature_flags = evidence_response.get("routing_trace", {}).get(
            "feature_flags",
            {},
        )
        evidence_bundle_enabled = bool(feature_flags.get("evidence_bundle", True))
        effective_answer_mode = (
            self.config.answer_mode if evidence_bundle_enabled else "off"
        )
        answer_error: RequiredQaAnswerError | None = None
        try:
            answer, answer_report = answer_question(
                query,
                evidence,
                answer_type=str(
                    evidence_response["query_plan"].get("answer_type", "unknown")
                ),
                mode=effective_answer_mode,
                cache_root=self.config.answer_cache_root,
                runner=self.answer_runner,
                model_name=self.config.model_name,
                model_revision=self.config.model_revision,
                max_evidence=3,
                timeout_seconds=self.config.answer_timeout_seconds,
            )
        except RequiredQaAnswerError as exc:
            answer_error = exc
            answer = exc.answer
            answer_report = exc.report
        answer_payload = answer.to_dict()
        answer_report_payload = answer_report.to_dict()
        if not evidence_bundle_enabled:
            answer_payload["reason"] = "qa_evidence_bundle_disabled"
            answer_report_payload["fallback_reason"] = (
                "qa_evidence_bundle_disabled; using legacy manual evidence"
            )
        response = {
            "query_plan": evidence_response["query_plan"],
            "routing_trace": evidence_response["routing_trace"],
            "answer": answer_payload,
            "answer_report": answer_report_payload,
            "evidence": evidence,
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "experiment_id": self.config.experiment_id,
        }
        if answer_error is not None:
            response["required_answer_error"] = str(answer_error)
            raise RequiredQaPipelineError(
                str(answer_error),
                response,
            ) from answer_error
        return response


def _manual_evidence(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the evidence bundle or an ASR-free legacy fallback when disabled."""

    bundled = response.get("evidence")
    if isinstance(bundled, list) and bundled:
        return [dict(item) for item in bundled if isinstance(item, Mapping)]

    legacy = response.get("results")
    if not isinstance(legacy, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(legacy, start=1):
        if not isinstance(item, Mapping):
            continue
        modality_scores = item.get("modality_scores", {})
        modalities = (
            sorted(
                str(name)
                for name, score in modality_scores.items()
                if str(name) != "rrf" and float(score) != 0.0
            )
            if isinstance(modality_scores, Mapping)
            else []
        )
        # Explicit allowlist: legacy ASR/transcript keys are never copied.
        normalized.append(
            {
                "evidence_id": f"E{index:03d}",
                "video_id": str(item.get("video_id") or ""),
                "frame_id": str(item.get("frame_id") or ""),
                "shot_id": str(item.get("shot_id") or item.get("segment_id") or ""),
                "timestamp": item.get("timestamp", 0.0),
                "image_path": str(
                    item.get("keyframe_path") or item.get("thumbnail_path") or ""
                ),
                "caption": str(item.get("caption") or ""),
                "ocr_text": str(item.get("ocr_text") or ""),
                "objects": list(item.get("objects") or []),
                "source_modalities": modalities,
                "retrieval_score": item.get("score", 0.0),
                "warnings": ["evidence_bundle_disabled"],
            }
        )
    return normalized
