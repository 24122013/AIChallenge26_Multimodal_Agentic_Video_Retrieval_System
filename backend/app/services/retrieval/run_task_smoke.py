"""Run bounded canonical online-pipeline checks on existing artifacts.

This command never builds a submission.  It is the post-submission verification
surface used by the Colab launcher and README examples.  Every task enters via
``search_online``; only QA invokes the configured grounded answerer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from backend.app.services.retrieval.bge_dense import BGE_M3_SCHEMA_VERSION
from backend.app.services.retrieval.qa_pipeline import RequiredQaPipelineError
from backend.app.services.retrieval.retrieval_config import load_project_env
from backend.app.services.retrieval.retrieval_manager import (
    clear_retrieval_caches,
    get_qa_runtime_lineage,
    search_online,
)


DEFAULT_QUERIES = {
    "kis": "người phụ nữ mặc áo đỏ đang cầm điện thoại",
    "avs": "tất cả các cảnh có xe máy đi qua đường",
    "qa": "Người phụ nữ mặc áo đỏ đang cầm vật gì?",
}


class SmokeValidationError(RuntimeError):
    """Raised when a task smoke used a fallback or unverifiable artifact."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("kis", "avs", "temporal", "qa", "all"),
        default="all",
    )
    parser.add_argument("--kis-query", default=DEFAULT_QUERIES["kis"])
    parser.add_argument("--avs-query", default=DEFAULT_QUERIES["avs"])
    parser.add_argument(
        "--temporal-query",
        default="a person enters a room, then sits down",
    )
    parser.add_argument("--qa-query", default=DEFAULT_QUERIES["qa"])
    parser.add_argument("--expanded-query", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _bounded_evidence(items: object, limit: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    output: list[dict[str, Any]] = []
    for item in items[:limit]:
        if not isinstance(item, Mapping):
            continue
        output.append(
            {
                key: item.get(key)
                for key in (
                    "evidence_id",
                    "video_id",
                    "frame_id",
                    "shot_id",
                    "timestamp",
                    "image_path",
                    "caption",
                    "ocr_text",
                    "objects",
                    "source_modalities",
                    "retrieval_score",
                    "base_retrieval_score",
                    "constraint_score",
                    "matched_constraints",
                    "temporal_event_index",
                    "temporal_match_rank",
                    "temporal_match_mode",
                    "temporal_chain_id",
                    "temporal_event_query",
                    "temporal_event_role",
                    "temporal_chain_score",
                    "warnings",
                )
            }
        )
    return output


def _task_payload(
    task: str,
    query: str,
    top_k: int,
    expanded: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    response = search_online(
        query=query,
        task=task,
        top_k=top_k,
        expanded_queries=expanded if task == "qa" else (),
    )
    query_plan = response.get("query_plan", {})
    needs_temporal = bool(
        query_plan.get("needs_temporal")
        if isinstance(query_plan, Mapping)
        else False
    )
    evidence_limit = 5 if needs_temporal else top_k
    evidence_source = response.get("evidence")
    if not evidence_source and task in {"kis", "avs"}:
        evidence_source = _candidate_evidence(response.get("candidates"), top_k)
    payload = {
        "task": task,
        "query": query,
        "query_plan": query_plan,
        "routing_trace": response.get("routing_trace", {}),
        "answer": response.get("answer") if task == "qa" else None,
        "answer_report": response.get("answer_report") if task == "qa" else None,
        "answer_eligible": response.get("answer_eligible") if task == "qa" else None,
        "preflight_block_reason": (
            response.get("preflight_block_reason") if task == "qa" else None
        ),
        "temporal_matches": response.get("temporal_matches", []),
        "candidates": list(response.get("candidates") or []),
        "evidence": _bounded_evidence(evidence_source, evidence_limit),
        # Snapshot after retrieval so the reranker report proves this exact
        # request reached the model rather than merely having it configured.
        "runtime_lineage": get_qa_runtime_lineage(),
        "latency_ms": response.get(
            "latency_ms",
            round((time.perf_counter() - started) * 1000.0, 3),
        ),
    }
    return payload


def _candidate_evidence(items: object, limit: int) -> list[dict[str, Any]]:
    """Adapt canonical KIS/AVS candidates to the smoke evidence validator."""

    if not isinstance(items, list):
        return []
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(items[:limit], start=1):
        if not isinstance(item, Mapping):
            continue
        modality_scores = item.get("modality_scores")
        source_modalities = (
            sorted(
                str(name)
                for name, score in modality_scores.items()
                if float(score) != 0.0 and name not in {"fusion", "rrf"}
            )
            if isinstance(modality_scores, Mapping)
            else []
        )
        evidence.append(
            {
                "evidence_id": f"E{index:03d}",
                "video_id": item.get("video_id"),
                "frame_id": item.get("frame_id") or item.get("keyframe_id"),
                "shot_id": item.get("shot_id"),
                "timestamp": item.get("timestamp"),
                "image_path": item.get("keyframe_path") or item.get("thumbnail_path"),
                "caption": item.get("caption"),
                "ocr_text": item.get("ocr_text"),
                "objects": item.get("objects") or [],
                "source_modalities": source_modalities,
                "retrieval_score": item.get("rerank_score", item.get("score")),
                "base_retrieval_score": item.get("fusion_score"),
                "constraint_score": 0.0,
                "matched_constraints": [],
                "warnings": [],
            }
        )
    return evidence


def _validate_runtime_lineage(lineage: object) -> list[str]:
    if not isinstance(lineage, Mapping):
        return ["runtime_lineage_missing"]
    issues: list[str] = []
    dense = lineage.get("dense_text")
    if not isinstance(dense, Mapping) or not dense.get("enabled"):
        issues.append("bge_dense_not_enabled")
    else:
        if not str(dense.get("model_name") or "").strip():
            issues.append("bge_dense_model_name_missing")
        if not str(dense.get("model_revision") or "").strip():
            issues.append("bge_dense_model_revision_missing")
        if dense.get("index_schema_version") != BGE_M3_SCHEMA_VERSION:
            issues.append("bge_dense_schema_version_invalid")
        if not _positive_int(dense.get("vector_count")):
            issues.append("bge_dense_vector_count_invalid")
        source_contract = dense.get("source_contract")
        if not isinstance(source_contract, Mapping):
            issues.append("bge_dense_source_contract_missing")
        else:
            if source_contract.get("canonical_only") is not True:
                issues.append("bge_dense_source_not_canonical_only")
            if source_contract.get("source_kind") not in {
                "canonical_segments",
                "selected_keyframes",
            }:
                issues.append("bge_dense_source_kind_invalid")
        checksums = dense.get("artifact_checksums")
        if not isinstance(checksums, Mapping):
            issues.append("bge_dense_artifact_checksums_missing")
        else:
            for artifact in ("index", "frame_map"):
                contract = checksums.get(artifact)
                sha256 = contract.get("sha256") if isinstance(contract, Mapping) else None
                if not isinstance(sha256, str) or re.fullmatch(
                    r"[0-9a-fA-F]{64}",
                    sha256,
                ) is None:
                    issues.append(f"bge_dense_{artifact}_checksum_missing")

    reranker = lineage.get("reranker")
    if not isinstance(reranker, Mapping):
        issues.append("reranker_lineage_missing")
    elif reranker.get("enabled") is not False:
        issues.append("model_reranker_must_be_disabled")

    answer_model = lineage.get("answer_model")
    if not isinstance(answer_model, Mapping):
        issues.append("qa_model_lineage_missing")
    else:
        for field in ("name", "revision", "prompt_revision"):
            if not str(answer_model.get(field) or "").strip():
                issues.append(f"qa_model_{field}_missing")
    return issues


def _positive_int(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value > 0
    )


def _validate_task_payload(payload: Mapping[str, Any]) -> list[str]:
    task = str(payload.get("task") or "")
    issues = (
        _validate_runtime_lineage(payload.get("runtime_lineage"))
        if task == "qa"
        else []
    )
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        issues.append("evidence_empty")
        evidence = []
    evidence_ids: set[str] = set()
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, Mapping):
            issues.append(f"evidence_{index}_invalid")
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        if evidence_id:
            if evidence_id in evidence_ids:
                issues.append(f"evidence_{index}_id_duplicate")
            evidence_ids.add(evidence_id)
        else:
            issues.append(f"evidence_{index}_id_missing")
        image_path = Path(str(item.get("image_path") or ""))
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            issues.append(f"evidence_{index}_image_unreadable")

    trace = payload.get("routing_trace")
    if not isinstance(trace, Mapping):
        issues.append("routing_trace_missing")
        trace = {}
    if trace.get("fallback_used") or list(trace.get("fallback_reasons") or []):
        issues.append("retrieval_fallback_used")
    if task == "qa":
        if trace.get("reranker") != "off":
            issues.append("model_reranker_applied")
        modality_queries = trace.get("modality_queries")
        dense_calls = [
            item
            for item in modality_queries
            if isinstance(item, Mapping) and item.get("modality") == "dense_text"
        ] if isinstance(modality_queries, list) else []
        if (
            not dense_calls
            or any(item.get("error") for item in dense_calls)
            or any(not _positive_int(item.get("candidate_count")) for item in dense_calls)
        ):
            issues.append("routing_bge_dense_not_applied")

    query_plan = payload.get("query_plan")
    original_query = (
        " ".join(str(query_plan.get("original_query") or "").split())
        if isinstance(query_plan, Mapping)
        else ""
    )
    if original_query and original_query != " ".join(
        str(payload.get("query") or "").split()
    ):
        issues.append("original_query_anchor_missing")
    needs_temporal = bool(
        query_plan.get("needs_temporal")
        if isinstance(query_plan, Mapping)
        else False
    )
    if needs_temporal:
        issues.extend(
            _validate_temporal_payload(
                payload,
                evidence=evidence,
                query_plan=query_plan if isinstance(query_plan, Mapping) else {},
                trace=trace,
            )
        )

    if task != "qa":
        return list(dict.fromkeys(issues))

    lineage = payload.get("runtime_lineage")
    answer_model = lineage.get("answer_model", {}) if isinstance(lineage, Mapping) else {}
    if (
        not isinstance(answer_model, Mapping)
        or answer_model.get("enabled") is not True
        or answer_model.get("mode") != "required"
    ):
        issues.append("qa_answer_mode_not_required")
    report = payload.get("answer_report")
    if not isinstance(report, Mapping):
        issues.append("qa_answer_report_missing")
    else:
        if report.get("cache_hit") is not False:
            issues.append("qa_answer_cache_not_miss")
        if report.get("model_invoked") is not True:
            issues.append("qa_model_not_invoked")
        if report.get("mode") != "required":
            issues.append("qa_answer_report_mode_not_required")
        if report.get("manual_evidence_available") is not True:
            issues.append("qa_manual_evidence_not_available")
        expected_evidence_count = (
            len(evidence) if needs_temporal else min(3, len(evidence))
        )
        if report.get("evidence_count") != expected_evidence_count:
            issues.append("qa_answer_evidence_count_mismatch")
        lineage_fields = {
            "model_name": "name",
            "model_revision": "revision",
            "prompt_revision": "prompt_revision",
        }
        for field, lineage_field in lineage_fields.items():
            value = str(report.get(field) or "").strip()
            if not value:
                issues.append(f"qa_answer_report_{field}_missing")
            elif isinstance(answer_model, Mapping) and value != str(
                answer_model.get(lineage_field) or ""
            ).strip():
                issues.append(f"qa_answer_report_{field}_mismatch")

    if payload.get("answer_eligible") is not True:
        issues.append("qa_answer_not_eligible")
    if payload.get("preflight_block_reason") not in {None, ""}:
        issues.append("qa_preflight_blocked")

    answer = payload.get("answer")
    if not isinstance(answer, Mapping):
        issues.append("qa_answer_missing")
    else:
        status = answer.get("status")
        answer_value = answer.get("answer")
        cited = answer.get("evidence_ids")
        if status != "answered":
            issues.append("qa_answer_not_answered")
        if not isinstance(cited, list) or any(
            not isinstance(value, str) for value in cited
        ):
            issues.append("qa_citations_invalid")
        elif set(cited).difference(evidence_ids):
            issues.append("qa_citation_unknown")
        elif status == "answered" and not cited:
            issues.append("qa_answer_citation_missing")
        if status == "answered" and not str(answer_value or "").strip():
            issues.append("qa_answer_text_missing")
        if isinstance(report, Mapping) and report.get("status") != status:
            issues.append("qa_answer_report_status_mismatch")
    return list(dict.fromkeys(issues))


def _validate_temporal_payload(
    payload: Mapping[str, Any],
    *,
    evidence: list[object],
    query_plan: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    temporal = trace.get("temporal_route")
    if (
        not isinstance(temporal, Mapping)
        or temporal.get("executed") is not True
        or temporal.get("match_mode") != "strict"
    ):
        return ["temporal_route_not_strict"]
    event_count = temporal.get("event_count")
    if not _positive_int(event_count) or not 2 <= int(event_count) <= 5:
        return ["temporal_event_count_invalid"]
    event_count = int(event_count)
    if len(evidence) != event_count:
        issues.append("temporal_evidence_incomplete")

    queries_raw = temporal.get("event_queries")
    queries = (
        [" ".join(str(value).split()) for value in queries_raw]
        if isinstance(queries_raw, list)
        else []
    )
    if len(queries) != event_count or any(not value for value in queries):
        issues.append("temporal_event_queries_invalid")

    indices: list[int] = []
    roles: list[str] = []
    chain_ids: set[str] = set()
    scores: list[float] = []
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        raw_event_index = item.get("temporal_event_index")
        if isinstance(raw_event_index, bool):
            issues.append("temporal_event_index_invalid")
            continue
        try:
            event_index = int(raw_event_index)
        except (TypeError, ValueError):
            issues.append("temporal_event_index_invalid")
            continue
        indices.append(event_index)
        raw_match_rank = item.get("temporal_match_rank")
        if (
            isinstance(raw_match_rank, bool)
            or not isinstance(raw_match_rank, int)
            or raw_match_rank != 1
        ):
            issues.append("temporal_match_rank_invalid")
        event_query = " ".join(str(item.get("temporal_event_query") or "").split())
        if (
            not 0 <= event_index < len(queries)
            or event_query != queries[event_index]
        ):
            issues.append("temporal_event_query_mismatch")
        roles.append(str(item.get("temporal_event_role") or "").casefold().strip())
        chain_id = str(item.get("temporal_chain_id") or "").strip()
        if not chain_id:
            issues.append("temporal_chain_id_missing")
        else:
            chain_ids.add(chain_id)
        if item.get("temporal_match_mode") != "strict":
            issues.append("temporal_evidence_not_strict")
        try:
            score = float(item.get("temporal_chain_score"))
        except (TypeError, ValueError):
            issues.append("temporal_chain_score_invalid")
        else:
            if not math.isfinite(score):
                issues.append("temporal_chain_score_invalid")
            else:
                scores.append(score)

    if indices != list(range(event_count)):
        issues.append("temporal_event_indices_incomplete")
    if len(chain_ids) != 1:
        issues.append("temporal_chain_id_mismatch")
    if len(scores) != event_count or any(
        not math.isclose(score, scores[0], abs_tol=1e-9) for score in scores
    ):
        issues.append("temporal_chain_score_mismatch")

    answer_type = str(query_plan.get("answer_type") or "unknown").casefold().strip()
    answer_index = query_plan.get("answer_event_index")
    if answer_type == "yes_no":
        if answer_index is not None or roles != ["whole_chain"] * event_count:
            issues.append("temporal_yes_no_role_invalid")
    elif (
        isinstance(answer_index, bool)
        or not isinstance(answer_index, int)
        or not 0 <= answer_index < event_count
    ):
        issues.append("temporal_answer_target_missing")
    else:
        expected_roles = [
            "answer_target" if index == answer_index else "context"
            for index in range(event_count)
        ]
        if roles != expected_roles:
            issues.append("temporal_event_role_mismatch")

    matches = payload.get("temporal_matches")
    first_match = matches[0] if isinstance(matches, list) and matches else None
    if not isinstance(first_match, Mapping):
        issues.append("temporal_match_lineage_missing")
    elif chain_ids:
        if str(first_match.get("chain_id") or "").strip() not in chain_ids:
            issues.append("temporal_match_chain_id_mismatch")
        try:
            match_score = float(first_match.get("score"))
        except (TypeError, ValueError):
            issues.append("temporal_match_score_invalid")
        else:
            if not scores or not math.isclose(match_score, scores[0], abs_tol=1e-9):
                issues.append("temporal_match_score_mismatch")
    return list(dict.fromkeys(issues))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _isolated_answer_cache(enabled: bool):
    """Force a real model invocation without deleting a caller-owned cache."""

    if not enabled:
        yield
        return
    previous = os.environ.get("QA_ANSWER_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="qa-smoke-answer-cache-") as temporary:
        os.environ["QA_ANSWER_CACHE_DIR"] = temporary
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("QA_ANSWER_CACHE_DIR", None)
            else:
                os.environ["QA_ANSWER_CACHE_DIR"] = previous
            clear_retrieval_caches()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= int(args.top_k) <= 20:
        raise ValueError("--top-k must be within [1, 20]")
    tasks = (
        ("kis", "avs", "temporal", "qa")
        if args.task == "all"
        else (args.task,)
    )
    queries = {
        "kis": args.kis_query,
        "avs": args.avs_query,
        "temporal": args.temporal_query,
        "qa": args.qa_query,
    }
    with _isolated_answer_cache("qa" in tasks):
        clear_retrieval_caches()
        return _run_tasks(args, tasks=tasks, queries=queries)


def _run_tasks(
    args: argparse.Namespace,
    *,
    tasks: tuple[str, ...],
    queries: Mapping[str, str],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    try:
        for task in tasks:
            results.append(
                _task_payload(
                    task,
                    queries[task],
                    int(args.top_k),
                    list(args.expanded_query),
                )
            )
    except RequiredQaPipelineError as exc:
        failed_qa = {
            "task": "qa",
            "query": queries["qa"],
            **dict(exc.response),
            "runtime_lineage": get_qa_runtime_lineage(),
        }
        failure = {
            "status": "failed",
            "reason": str(exc),
            "results": [*results, failed_qa],
        }
        if args.output is not None:
            _atomic_json(args.output, failure)
        raise
    issues = [
        {"task": item["task"], "issues": task_issues}
        for item in results
        if (task_issues := _validate_task_payload(item))
    ]
    if issues:
        failure = {
            "status": "failed",
            "reason": "task_smoke_validation_failed",
            "validation_errors": issues,
            "results": results,
        }
        if args.output is not None:
            _atomic_json(args.output, failure)
        raise SmokeValidationError(json.dumps(issues, ensure_ascii=False))
    payload = {"status": "passed", "results": results}
    if args.output is not None:
        _atomic_json(args.output, payload)
    return payload


def main() -> int:
    load_project_env()
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
