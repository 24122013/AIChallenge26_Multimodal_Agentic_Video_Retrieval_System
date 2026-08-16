"""End-to-end QA orchestration: evidence retrieval then grounded answering."""
from __future__ import annotations

import math
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
        query_plan = evidence_response.get("query_plan", {})
        if not isinstance(query_plan, Mapping):
            query_plan = {}
        routing_trace = evidence_response.get("routing_trace", {})
        if not isinstance(routing_trace, Mapping):
            routing_trace = {}
        feature_flags = routing_trace.get(
            "feature_flags",
            {},
        )
        if not isinstance(feature_flags, Mapping):
            feature_flags = {}
        evidence_bundle_enabled = bool(feature_flags.get("evidence_bundle", True))
        effective_answer_mode = (
            self.config.answer_mode if evidence_bundle_enabled else "off"
        )
        needs_temporal = bool(query_plan.get("needs_temporal", False))
        answer_eligible, preflight_block_reason = _answer_preflight(
            evidence_response,
            evidence=evidence,
            query_plan=query_plan,
        )
        max_answer_evidence = 5 if needs_temporal else 3
        answer_error: RequiredQaAnswerError | None = None
        try:
            answer, answer_report = answer_question(
                query,
                evidence,
                answer_type=str(
                    query_plan.get("answer_type", "unknown")
                ),
                mode=effective_answer_mode,
                cache_root=self.config.answer_cache_root,
                runner=self.answer_runner,
                model_name=self.config.model_name,
                model_revision=self.config.model_revision,
                max_evidence=max_answer_evidence,
                timeout_seconds=self.config.answer_timeout_seconds,
                # Disabling the evidence bundle keeps the legacy disabled
                # answer contract.  Temporal/no-evidence preflight otherwise
                # returns insufficient_evidence without invoking the model,
                # including when answer mode is required.
                answer_eligible=(
                    answer_eligible if evidence_bundle_enabled else True
                ),
                preflight_block_reason=preflight_block_reason,
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
            "query_plan": dict(query_plan),
            "routing_trace": dict(routing_trace),
            "answer": answer_payload,
            "answer_report": answer_report_payload,
            "answer_eligible": answer_eligible,
            "preflight_block_reason": preflight_block_reason or None,
            "evidence": evidence,
            "temporal_matches": list(evidence_response.get("temporal_matches") or []),
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


def _answer_preflight(
    response: Mapping[str, Any],
    *,
    evidence: Sequence[Mapping[str, Any]],
    query_plan: Mapping[str, Any],
) -> tuple[bool, str]:
    """Fail closed before generation when evidence is absent or temporal is unsafe."""

    needs_temporal = bool(query_plan.get("needs_temporal", False))
    declared = response.get("answer_eligible")
    eligible = bool(evidence) if declared is None else bool(declared)
    reason = " ".join(str(response.get("preflight_block_reason") or "").split()).strip()

    if not evidence:
        return False, reason or "no_evidence"
    if not needs_temporal:
        return eligible, "" if eligible else (reason or "answer_not_eligible")
    if not eligible:
        return False, reason or "temporal_answer_not_eligible"

    trace = response.get("routing_trace")
    temporal_route = trace.get("temporal_route", {}) if isinstance(trace, Mapping) else {}
    if not isinstance(temporal_route, Mapping):
        temporal_route = {}
    if temporal_route.get("executed") is not True:
        return False, reason or "temporal_route_not_executed"
    match_mode = str(temporal_route.get("match_mode") or "").casefold().strip()
    if match_mode != "strict":
        return False, reason or (
            f"temporal_match_not_answerable:{match_mode or 'no_chain'}"
        )

    try:
        event_count = int(temporal_route.get("event_count"))
    except (TypeError, ValueError):
        return False, "temporal_event_count_invalid"
    if not 2 <= event_count <= 5:
        return False, "temporal_event_count_invalid"
    if len(evidence) != event_count:
        return False, "temporal_chain_incomplete"

    event_queries_raw = temporal_route.get("event_queries")
    if not isinstance(event_queries_raw, Sequence) or isinstance(
        event_queries_raw,
        (str, bytes),
    ):
        return False, "temporal_event_queries_missing"
    event_queries = [" ".join(str(value).split()) for value in event_queries_raw]
    if len(event_queries) != event_count or any(not value for value in event_queries):
        return False, "temporal_event_queries_invalid"

    indices: list[int] = []
    roles: list[str] = []
    chain_ids: set[str] = set()
    chain_scores: list[float] = []
    for item in evidence:
        raw_index = item.get("temporal_event_index")
        if isinstance(raw_index, bool):
            return False, "temporal_event_index_invalid"
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return False, "temporal_event_index_invalid"
        indices.append(index)
        raw_match_rank = item.get("temporal_match_rank")
        if (
            isinstance(raw_match_rank, bool)
            or not isinstance(raw_match_rank, int)
            or raw_match_rank != 1
        ):
            return False, "temporal_match_rank_invalid"
        if str(item.get("temporal_match_mode") or "").casefold() != "strict":
            return False, "temporal_evidence_not_strict"
        chain_id = str(item.get("temporal_chain_id") or "").strip()
        if not chain_id:
            return False, "temporal_chain_id_missing"
        chain_ids.add(chain_id)
        event_query = " ".join(str(item.get("temporal_event_query") or "").split())
        if not 0 <= index < event_count or event_query != event_queries[index]:
            return False, "temporal_event_query_mismatch"
        roles.append(str(item.get("temporal_event_role") or "").casefold().strip())
        try:
            chain_score = float(item.get("temporal_chain_score"))
        except (TypeError, ValueError):
            return False, "temporal_chain_score_invalid"
        if not math.isfinite(chain_score):
            return False, "temporal_chain_score_invalid"
        chain_scores.append(chain_score)

    if indices != list(range(event_count)):
        return False, "temporal_event_indices_incomplete"
    if len(chain_ids) != 1:
        return False, "temporal_chain_id_mismatch"
    if any(not math.isclose(score, chain_scores[0], abs_tol=1e-9) for score in chain_scores):
        return False, "temporal_chain_score_mismatch"

    answer_type = str(query_plan.get("answer_type") or "unknown").casefold().strip()
    answer_index = query_plan.get("answer_event_index")
    if answer_type == "yes_no":
        if answer_index is not None or any(role != "whole_chain" for role in roles):
            return False, "temporal_yes_no_role_invalid"
    else:
        if (
            isinstance(answer_index, bool)
            or not isinstance(answer_index, int)
            or not 0 <= answer_index < event_count
        ):
            return False, "temporal_answer_target_missing"
        expected_roles = [
            "answer_target" if index == answer_index else "context"
            for index in range(event_count)
        ]
        if roles != expected_roles:
            return False, "temporal_event_role_mismatch"

    matches = response.get("temporal_matches")
    first_match = matches[0] if isinstance(matches, list) and matches else None
    if not isinstance(first_match, Mapping):
        return False, "temporal_match_lineage_missing"
    if str(first_match.get("chain_id") or "").strip() not in chain_ids:
        return False, "temporal_match_chain_id_mismatch"
    try:
        match_score = float(first_match.get("score"))
    except (TypeError, ValueError):
        return False, "temporal_match_score_invalid"
    if not math.isfinite(match_score) or not math.isclose(
        match_score,
        chain_scores[0],
        abs_tol=1e-9,
    ):
        return False, "temporal_match_score_mismatch"
    return True, ""


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
                "frame_index": item.get("frame_index"),
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
                "base_retrieval_score": item.get("score", 0.0),
                "constraint_score": 0.0,
                "matched_constraints": [],
                "temporal_event_index": item.get("temporal_event_index"),
                "temporal_match_rank": item.get("temporal_match_rank"),
                "temporal_match_mode": str(item.get("temporal_match_mode") or ""),
                "temporal_chain_id": str(item.get("temporal_chain_id") or ""),
                "temporal_event_query": str(item.get("temporal_event_query") or ""),
                "temporal_event_role": str(item.get("temporal_event_role") or ""),
                "temporal_chain_score": item.get("temporal_chain_score"),
                "warnings": ["evidence_bundle_disabled"],
            }
        )
    return normalized
