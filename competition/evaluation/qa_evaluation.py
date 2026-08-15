"""Strict Phase-0 evaluation utilities for evidence-grounded QA.

The checked-in dataset beside this module is deliberately an example fixture.  It
exercises the schema and evaluation workflow; it is not a quality benchmark and
must not be used to claim production or competition performance.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


QA_ANSWER_TYPES = frozenset(
    {
        "object",
        "color",
        "ocr",
        "action",
        "count",
        "location",
        "yes_no",
        "identity",
    }
)
QA_CONSTRAINT_CATEGORIES = frozenset(
    {"subject", "objects", "attributes", "actions", "locations", "ocr_terms"}
)
REQUIRED_LABEL_FIELDS = frozenset(
    {
        "question",
        "task_mode",
        "answer_type",
        "known_constraints",
        "gold_evidence",
        "gold_answer",
        "answerable",
    }
)
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
MAX_TEMPORAL_EVENTS = 5


class LockedTestReuseError(RuntimeError):
    """Raised when a single-use locked-test slot has already been consumed."""


def load_qa_dataset(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate QA labels from JSONL, keyed by ``query_id``."""

    rows = _load_jsonl(path)
    labels: dict[str, dict[str, Any]] = {}
    for line_number, row in rows:
        query_id = str(row.get("query_id") or "").strip()
        missing = sorted(REQUIRED_LABEL_FIELDS - set(row))
        if missing:
            raise ValueError(f"Missing fields {missing} at {path}:{line_number}")
        if not query_id or query_id in labels:
            raise ValueError(f"Missing/duplicate query_id at {path}:{line_number}")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"question must be non-empty at {path}:{line_number}")
        if str(row["task_mode"]) != "qa":
            raise ValueError(f"task_mode must be 'qa' at {path}:{line_number}")
        answer_type = str(row["answer_type"])
        if answer_type not in QA_ANSWER_TYPES:
            raise ValueError(f"Invalid answer_type at {path}:{line_number}: {answer_type}")
        if not isinstance(row["known_constraints"], Mapping):
            raise ValueError(f"known_constraints must be an object at {path}:{line_number}")
        unknown_constraints = sorted(
            set(str(key) for key in row["known_constraints"])
            - QA_CONSTRAINT_CATEGORIES
        )
        if unknown_constraints:
            raise ValueError(
                f"Invalid constraint categories {unknown_constraints} at {path}:{line_number}"
            )
        for category, values in row["known_constraints"].items():
            if not isinstance(values, list) or not all(
                isinstance(value, str) and bool(value.strip())
                for value in values
            ):
                raise ValueError(
                    "known_constraints values must be non-empty string lists "
                    f"at {path}:{line_number} ({category})"
                )
        _flatten_constraints(row["known_constraints"], context=f"{path}:{line_number}")
        if not isinstance(row["gold_evidence"], list):
            raise ValueError(f"gold_evidence must be a list at {path}:{line_number}")
        for evidence in row["gold_evidence"]:
            _evidence_key(
                evidence,
                context=f"{path}:{line_number}",
                require_lineage=True,
            )
        _validate_gold_temporal_chains(
            row.get("gold_temporal_chains"),
            context=f"{path}:{line_number}",
        )
        answerable = row["answerable"]
        if not isinstance(answerable, bool):
            raise ValueError(f"answerable must be boolean at {path}:{line_number}")
        gold_answer = row["gold_answer"]
        if answerable and not _answer_variants(gold_answer):
            raise ValueError(f"answerable row needs gold_answer at {path}:{line_number}")
        if not answerable and gold_answer not in (None, [], ""):
            raise ValueError(f"unanswerable row needs null gold_answer at {path}:{line_number}")
        labels[query_id] = dict(row)
    if not labels:
        raise ValueError(f"No QA labels found: {path}")
    return labels


def load_qa_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL while rejecting duplicate or missing IDs."""

    predictions: dict[str, dict[str, Any]] = {}
    for line_number, row in _load_jsonl(path):
        query_id = str(row.get("query_id") or "").strip()
        if not query_id or query_id in predictions:
            raise ValueError(f"Missing/duplicate query_id at {path}:{line_number}")
        predictions[query_id] = dict(row)
    if not predictions:
        raise ValueError(f"No QA predictions found: {path}")
    return predictions


def load_split_manifest(path: Path) -> dict[str, Any]:
    """Load a split manifest and enforce disjoint dev/locked-test IDs."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Split manifest must be an object: {path}")
    if value.get("quality_claim_allowed") is not False:
        raise ValueError("Example manifest must explicitly disable quality claims")
    splits = value.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Split manifest needs a splits object")
    dev = _manifest_ids(splits, "dev")
    locked = _manifest_ids(splits, "locked_test")
    overlap = sorted(set(dev) & set(locked))
    if overlap:
        raise ValueError(f"QA evaluation splits overlap: {overlap}")
    if not dev or not locked:
        raise ValueError("Both dev and locked_test splits must be non-empty")
    policy = value.get("locked_test_policy")
    if not isinstance(policy, Mapping) or policy.get("reuse") != "single_use":
        raise ValueError("locked_test_policy.reuse must be 'single_use'")
    return dict(value)


def evaluate_qa_predictions(
    *,
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    query_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate parser, evidence, answer, abstention, and optional resource stats.

    This pure function is appropriate for development evaluation.  Locked-test
    evaluation should go through :func:`evaluate_locked_test_once`.
    """

    selected_ids = list(query_ids) if query_ids is not None else sorted(labels)
    if not selected_ids:
        raise ValueError("No query IDs selected for QA evaluation")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("QA evaluation query IDs must be unique")
    unknown = sorted(set(selected_ids) - set(labels))
    if unknown:
        raise ValueError(f"Unknown label query IDs: {unknown}")
    missing = sorted(set(selected_ids) - set(predictions))
    if missing:
        raise ValueError(f"Predictions are missing labelled queries: {missing}")

    rows = [
        _evaluate_row(query_id, labels[query_id], predictions[query_id])
        for query_id in selected_ids
    ]
    answerable_rows = [row for row in rows if row["answerable"]]
    evidence_rows = [row for row in answerable_rows if row["has_gold_evidence"]]
    temporal_rows = [row for row in rows if row["has_gold_temporal"]]
    parser = _aggregate_parser(rows)
    evidence = _average_metrics(
        evidence_rows,
        ("Evidence Hit@1", "Evidence Hit@5", "Evidence Hit@10", "MRR", "nDCG@10"),
    )
    answer = _average_metrics(answerable_rows, ("answer_EM", "answer_F1"))
    abstention = _aggregate_abstention(rows)
    temporal = {
        "query_count": len(temporal_rows),
        "Temporal Chain Hit@5": (
            statistics.fmean(
                float(row["Temporal Chain Hit@5"]) for row in temporal_rows
            )
            if temporal_rows
            else 0.0
        ),
    }
    validity = {
        "valid_response_rate": statistics.fmean(
            float(row["response_valid"]) for row in rows
        ),
        "invalid_response_count": sum(
            int(row["invalid_response"]) for row in rows
        ),
        "pipeline_error_count": sum(int(row["pipeline_error"]) for row in rows),
        "invalid_citation_count": sum(
            int(row["invalid_citation"]) for row in rows
        ),
    }
    integrity = {
        # This is a deliberately conservative gold-lineage/citation proxy.  It
        # is not a semantic entailment judge; see README_QA_EVALUATION.md.
        "unsupported_answer_rate": statistics.fmean(
            float(row["unsupported_answer"]) for row in rows
        ),
        "pipeline_error_rate": statistics.fmean(
            float(row["pipeline_error"]) for row in rows
        ),
        "invalid_response_rate": statistics.fmean(
            float(row["invalid_response"]) for row in rows
        ),
        "invalid_citation_rate": statistics.fmean(
            float(row["invalid_citation"]) for row in rows
        ),
    }
    latency = _distribution(
        float(value)
        for value in (row.get("latency_ms") for row in rows)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    vram = _distribution(
        float(value)
        for value in (row.get("peak_vram_mb") for row in rows)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return {
        "status": "evaluated",
        "query_count": len(rows),
        "answerable_count": len(answerable_rows),
        "unanswerable_count": len(rows) - len(answerable_rows),
        "parser": parser,
        "evidence": evidence,
        "temporal": temporal,
        "answer": answer,
        "abstention": abstention,
        "validity": validity,
        "integrity": integrity,
        "query_ids_sha256": _stable_hash(sorted(selected_ids)),
        "dataset_sha256": _stable_hash(
            {query_id: labels[query_id] for query_id in selected_ids}
        ),
        "evidence_sha256": _stable_hash(
            {
                query_id: predictions[query_id].get("evidence", [])
                for query_id in selected_ids
            }
        ),
        "latency_ms": latency,
        "peak_vram_mb": vram,
        "queries": rows,
    }


def evaluate_answerer_only_predictions(
    *,
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    query_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate answers only after proving every supplied evidence item is gold."""

    selected_ids = list(query_ids) if query_ids is not None else sorted(labels)
    report = evaluate_qa_predictions(
        labels=labels,
        predictions=predictions,
        query_ids=selected_ids,
    )
    gold_lineage = {
        query_id: _lineage_payload(labels[query_id].get("gold_evidence", []))
        for query_id in selected_ids
    }
    predicted_lineage = {
        query_id: _lineage_payload(predictions[query_id].get("evidence", []))
        for query_id in selected_ids
    }
    if predicted_lineage != gold_lineage:
        raise ValueError(
            "answerer-only evaluation requires evidence identical to gold lineage"
        )
    lineage_hash = _stable_hash(gold_lineage)
    report.update(
        {
            "evaluation_mode": "answerer_only_gold_evidence",
            "evidence_source": "gold",
            "gold_evidence_sha256": lineage_hash,
            "evaluated_evidence_lineage_sha256": lineage_hash,
        }
    )
    return report


def evaluate_locked_test_once(
    *,
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    """Consume a locked-test slot exactly once and write an auditable receipt.

    The caller must deliberately remove/archive the receipt to authorize another
    locked-test run.  The function never silently replays even identical output.
    """

    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        raise LockedTestReuseError(
            "Locked-test evaluation was already consumed; archive the receipt "
            f"before an explicitly authorized rerun (prediction_sha256={prior.get('prediction_sha256')})."
        )
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Manifest needs splits")
    locked_ids = _manifest_ids(splits, "locked_test")
    policy = manifest.get("locked_test_policy")
    if not isinstance(policy, Mapping) or policy.get("reuse") != "single_use":
        raise ValueError("Locked evaluation requires single_use policy")
    report = evaluate_qa_predictions(
        labels=labels,
        predictions=predictions,
        query_ids=locked_ids,
    )
    receipt = {
        "schema_version": 1,
        "split": "locked_test",
        "query_ids_sha256": _stable_hash(locked_ids),
        "label_sha256": _stable_hash({key: labels[key] for key in locked_ids}),
        "prediction_sha256": _stable_hash({key: predictions[key] for key in locked_ids}),
        "query_count": len(locked_ids),
        "reuse_policy": "single_use",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise LockedTestReuseError("Locked-test receipt was created concurrently") from exc
    report["locked_test_receipt"] = receipt
    return report


def _evaluate_row(
    query_id: str,
    label: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    plan = prediction.get("query_plan")
    if not isinstance(plan, Mapping):
        plan = {}
    task_mode = str(plan.get("task_mode") or prediction.get("task_mode") or "")
    answer_type = str(plan.get("answer_type") or prediction.get("answer_type") or "")
    constraints = plan.get("known_constraints", prediction.get("known_constraints", {}))
    if not isinstance(constraints, Mapping):
        constraints = {}
    gold_constraints = _flatten_constraints(label["known_constraints"])
    predicted_constraints = _flatten_constraints(constraints)
    constraint_precision, constraint_recall, constraint_f1 = _set_prf(
        predicted_constraints,
        gold_constraints,
    )

    gold_evidence = list(label["gold_evidence"])
    for value in gold_evidence:
        _evidence_keys(value, require_lineage=True)
    raw_evidence = prediction.get("evidence", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    for value in raw_evidence:
        _evidence_keys(value)
    matched_gold: set[int] = set()
    relevance: list[bool] = []
    for predicted_evidence in raw_evidence:
        gold_index = next(
            (
                index
                for index, gold_value in enumerate(gold_evidence)
                if index not in matched_gold
                and _evidence_matches(predicted_evidence, gold_value)
            ),
            None,
        )
        relevance.append(gold_index is not None)
        if gold_index is not None:
            matched_gold.add(gold_index)
    first_rank = next((rank for rank, hit in enumerate(relevance, start=1) if hit), None)
    ideal_count = min(10, len(gold_evidence))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, hit in enumerate(relevance[:10], start=1)
        if hit
    )

    answer_payload = prediction.get("answer")
    if isinstance(answer_payload, Mapping):
        answer_status = str(answer_payload.get("status") or "").casefold()
        answer_field_present = "answer" in answer_payload
        predicted_answer = answer_payload.get("answer")
        cited_ids = answer_payload.get("evidence_ids")
    else:
        answer_status = str(prediction.get("status") or "").casefold()
        answer_field_present = "answer" in prediction
        predicted_answer = prediction.get("answer")
        cited_ids = prediction.get("evidence_ids")
    answered_valid = (
        answer_status == "answered"
        and answer_field_present
        and _has_answer(predicted_answer)
    )
    abstention_valid = (
        answer_status == "insufficient_evidence"
        and answer_field_present
        and predicted_answer is None
    )
    response_valid = answered_valid or abstention_valid
    predicted_abstention = abstention_valid
    local_evidence_ids = {
        str(value.get("evidence_id")).strip()
        for value in raw_evidence
        if isinstance(value, Mapping) and str(value.get("evidence_id") or "").strip()
    }
    citation_valid = True
    evidence_by_local_id = {
        str(value.get("evidence_id")).strip(): value
        for value in raw_evidence
        if isinstance(value, Mapping)
        and str(value.get("evidence_id") or "").strip()
    }
    if answered_valid:
        citation_valid = (
            isinstance(cited_ids, list)
            and bool(cited_ids)
            and all(
                isinstance(value, str)
                and bool(value.strip())
                and value.strip() in local_evidence_ids
                for value in cited_ids
            )
        )
    cited_evidence = (
        [evidence_by_local_id[str(value).strip()] for value in cited_ids]
        if citation_valid and answered_valid and isinstance(cited_ids, list)
        else []
    )
    cited_gold_lineage = any(
        _evidence_matches(predicted_value, gold_value)
        for predicted_value in cited_evidence
        for gold_value in gold_evidence
    )
    unsupported_answer = bool(
        answered_valid
        and (
            not citation_valid
            or not bool(label["answerable"])
            or not gold_evidence
            or not cited_gold_lineage
        )
    )
    gold_temporal_chains = _gold_temporal_chains(label)
    temporal_chain_hit = _temporal_chain_hit_at_5(
        prediction.get("temporal_matches"),
        gold_temporal_chains,
    )
    raw_answer_report = prediction.get("answer_report")
    answer_report = raw_answer_report if isinstance(raw_answer_report, Mapping) else {}
    gold_answers = _answer_variants(label["gold_answer"])
    scored_answer = predicted_answer if answered_valid else None
    em = max((_exact_match(scored_answer, answer) for answer in gold_answers), default=0.0)
    f1 = max((_token_f1(scored_answer, answer) for answer in gold_answers), default=0.0)
    row: dict[str, Any] = {
        "query_id": query_id,
        "answerable": bool(label["answerable"]),
        "has_gold_evidence": bool(gold_evidence),
        "has_gold_temporal": bool(gold_temporal_chains),
        "task_mode_correct": float(task_mode == str(label["task_mode"])),
        "gold_answer_type": str(label["answer_type"]),
        "predicted_answer_type": answer_type,
        "constraint_precision": constraint_precision,
        "constraint_recall": constraint_recall,
        "constraint_F1": constraint_f1,
        "Evidence Hit@1": float(any(relevance[:1])),
        "Evidence Hit@5": float(any(relevance[:5])),
        "Evidence Hit@10": float(any(relevance[:10])),
        "MRR": 1.0 / first_rank if first_rank else 0.0,
        "nDCG@10": dcg / ideal_dcg if ideal_dcg else 0.0,
        "Temporal Chain Hit@5": temporal_chain_hit,
        "answer_EM": em,
        "answer_F1": f1,
        "answer_status": answer_status,
        "response_valid": response_valid,
        "invalid_response": not response_valid,
        "pipeline_error": answer_status == "error",
        "citation_valid": citation_valid,
        "invalid_citation": answered_valid and not citation_valid,
        "unsupported_answer": unsupported_answer,
        "support_verdict_source": "gold_lineage_proxy",
        "model_invoked": answer_report.get("model_invoked"),
        "cache_hit": answer_report.get("cache_hit"),
        "predicted_abstention": predicted_abstention,
        "abstention_correct": float(
            response_valid
            and predicted_abstention == (not bool(label["answerable"]))
        ),
    }
    for metric_name in ("latency_ms", "peak_vram_mb"):
        value = prediction.get(metric_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{metric_name} cannot be negative for {query_id}")
            row[metric_name] = float(value)
    return row


def _aggregate_parser(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = sorted({str(row["gold_answer_type"]) for row in rows})
    per_type: dict[str, dict[str, float]] = {}
    for answer_type in labels:
        true_positive = sum(
            row["gold_answer_type"] == answer_type
            and row["predicted_answer_type"] == answer_type
            for row in rows
        )
        false_positive = sum(
            row["gold_answer_type"] != answer_type
            and row["predicted_answer_type"] == answer_type
            for row in rows
        )
        false_negative = sum(
            row["gold_answer_type"] == answer_type
            and row["predicted_answer_type"] != answer_type
            for row in rows
        )
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        per_type[answer_type] = {
            "precision": precision,
            "recall": recall,
            "F1": _harmonic(precision, recall),
            "support": float(sum(row["gold_answer_type"] == answer_type for row in rows)),
        }
    return {
        "task_mode_accuracy": statistics.fmean(float(row["task_mode_correct"]) for row in rows),
        "answer_type_macro_F1": statistics.fmean(value["F1"] for value in per_type.values()),
        "constraint_precision": statistics.fmean(float(row["constraint_precision"]) for row in rows),
        "constraint_recall": statistics.fmean(float(row["constraint_recall"]) for row in rows),
        "constraint_F1": statistics.fmean(float(row["constraint_F1"]) for row in rows),
        "answer_type_by_class": per_type,
    }


def _aggregate_abstention(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    unanswerable = [row for row in rows if not row["answerable"]]
    answerable = [row for row in rows if row["answerable"]]
    return {
        "accuracy": statistics.fmean(float(row["abstention_correct"]) for row in rows),
        "unanswerable_recall": (
            statistics.fmean(float(row["predicted_abstention"]) for row in unanswerable)
            if unanswerable
            else 0.0
        ),
        "answerable_response_rate": (
            statistics.fmean(
                float(row["response_valid"] and not row["predicted_abstention"])
                for row in answerable
            )
            if answerable
            else 0.0
        ),
        "invalid_response_rate": statistics.fmean(
            float(row["invalid_response"]) for row in rows
        ),
    }


def _average_metrics(
    rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> dict[str, float]:
    return {
        name: statistics.fmean(float(row[name]) for row in rows) if rows else 0.0
        for name in names
    }


def _distribution(values: Iterable[float]) -> dict[str, float | int] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 50.0),
        "p95": _percentile(ordered, 95.0),
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _set_prf(predicted: set[tuple[str, str]], gold: set[tuple[str, str]]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    intersection = len(predicted & gold)
    precision = _safe_div(intersection, len(predicted))
    recall = _safe_div(intersection, len(gold))
    return precision, recall, _harmonic(precision, recall)


def _flatten_constraints(
    constraints: Mapping[str, Any],
    *,
    context: str = "constraints",
) -> set[tuple[str, str]]:
    flattened: set[tuple[str, str]] = set()
    for raw_key, raw_values in constraints.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"Empty constraint key at {context}")
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        for value in values:
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Invalid constraint value at {context}: {value!r}")
            normalized = _normalize_text(value)
            if normalized:
                flattened.add((key.casefold(), normalized))
    return flattened


def _validate_gold_temporal_chains(value: Any, *, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"gold_temporal_chains must be a non-empty list at {context}"
        )
    for chain_index, chain in enumerate(value):
        if not isinstance(chain, list) or not 2 <= len(chain) <= MAX_TEMPORAL_EVENTS:
            raise ValueError(
                "each gold temporal chain must contain 2 to "
                f"{MAX_TEMPORAL_EVENTS} ordered events at {context}"
            )
        for event_index, event in enumerate(chain):
            _evidence_key(
                event,
                context=f"{context}.gold_temporal_chains[{chain_index}]"
                f"[{event_index}]",
                require_lineage=True,
            )


def _lineage_payload(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        raise ValueError("evidence must be a list for lineage hashing")
    return sorted(
        [sorted(_evidence_keys(item, require_lineage=True)) for item in value]
    )


def _gold_temporal_chains(label: Mapping[str, Any]) -> list[list[Any]]:
    value = label.get("gold_temporal_chains")
    if value is None:
        return []
    _validate_gold_temporal_chains(value, context="label")
    return [list(chain) for chain in value]


def _temporal_chain_hit_at_5(
    predicted_matches: Any,
    gold_chains: Sequence[Sequence[Any]],
) -> float:
    if not gold_chains:
        return 0.0
    if not isinstance(predicted_matches, list):
        return 0.0
    for predicted in predicted_matches[:5]:
        if not isinstance(predicted, Mapping):
            continue
        # Relaxed/sparse chains are manual-inspection output and must never
        # inflate the answer-quality gate.
        if str(predicted.get("match_mode") or "").casefold() != "strict":
            continue
        events = predicted.get("events")
        if not isinstance(events, list):
            continue
        for gold_chain in gold_chains:
            if len(events) != len(gold_chain):
                continue
            if all(
                _evidence_matches(predicted_event, gold_event)
                for predicted_event, gold_event in zip(events, gold_chain)
            ):
                return 1.0
    return 0.0


def _evidence_matches(predicted: Any, gold: Any) -> bool:
    """Match stable lineage, including point-in-window and interval overlap."""

    if not isinstance(predicted, Mapping) or not isinstance(gold, Mapping):
        return False
    predicted_video = str(predicted.get("video_id") or "").strip()
    gold_video = str(gold.get("video_id") or "").strip()
    if not predicted_video or predicted_video != gold_video:
        return False

    predicted_frames = _frame_ids(predicted, context="prediction evidence")
    gold_frames = _frame_ids(gold, context="gold evidence")
    if predicted_frames and gold_frames and predicted_frames & gold_frames:
        return True

    predicted_shot = str(predicted.get("shot_id") or "").strip()
    gold_shot = str(gold.get("shot_id") or "").strip()
    if predicted_shot and gold_shot and predicted_shot == gold_shot:
        return True

    predicted_interval = _evidence_interval(
        predicted,
        context="prediction evidence",
        include_point=True,
    )
    gold_interval = _evidence_interval(
        gold,
        context="gold evidence",
        include_point=True,
    )
    return bool(
        predicted_interval
        and gold_interval
        and predicted_interval[0] <= gold_interval[1]
        and gold_interval[0] <= predicted_interval[1]
    )


def _frame_ids(value: Mapping[str, Any], *, context: str) -> set[str]:
    frames: set[str] = set()
    frame_id = str(value.get("frame_id") or "").strip()
    if frame_id:
        frames.add(frame_id)
    raw_many = value.get("frame_ids")
    if raw_many is not None:
        if not isinstance(raw_many, list) or not raw_many or not all(
            isinstance(item, (str, int))
            and not isinstance(item, bool)
            and bool(str(item).strip())
            for item in raw_many
        ):
            raise ValueError(f"frame_ids must be a non-empty ID list at {context}")
        frames.update(str(item).strip() for item in raw_many)
    return frames


def _evidence_interval(
    value: Mapping[str, Any],
    *,
    context: str,
    include_point: bool,
) -> tuple[float, float] | None:
    start = value.get("start_time", value.get("start_timestamp"))
    end = value.get("end_time", value.get("end_timestamp"))
    if start is not None or end is not None:
        if not _is_finite_number(start) or not _is_finite_number(end):
            raise ValueError(f"Evidence time window must be finite at {context}")
        if float(end) < float(start):
            raise ValueError(f"Evidence time window is reversed at {context}")
        return float(start), float(end)
    if include_point:
        point = value.get("timestamp", value.get("timestamp_seconds"))
        if point is not None:
            if not _is_finite_number(point):
                raise ValueError(f"Evidence timestamp must be finite at {context}")
            return float(point), float(point)
    return None


def _evidence_key(
    value: Any,
    *,
    context: str = "evidence",
    require_lineage: bool = False,
) -> str:
    return sorted(
        _evidence_keys(
            value,
            context=context,
            require_lineage=require_lineage,
        )
    )[0]


def _evidence_keys(
    value: Any,
    *,
    context: str = "evidence",
    require_lineage: bool = False,
) -> frozenset[str]:
    if isinstance(value, str) and value.strip():
        if require_lineage:
            raise ValueError(
                f"Gold evidence needs stable video lineage at {context}; "
                "local evidence IDs are prediction-only"
            )
        return frozenset({f"local::{value.strip()}"})
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid evidence at {context}: {value!r}")
    video_id = str(value.get("video_id") or "").strip()
    lineage: set[str] = set()
    for frame_id in _frame_ids(value, context=context):
        if video_id:
            lineage.add(f"video::{video_id}::frame::{frame_id}")
    shot_id = str(value.get("shot_id") or "").strip()
    if video_id and shot_id:
        lineage.add(f"video::{video_id}::shot::{shot_id}")
    interval = _evidence_interval(value, context=context, include_point=False)
    if video_id and interval is not None:
        start, end = interval
        lineage.add(
            f"video::{video_id}::time::{_format_time(start)}"
            f"::{_format_time(end)}"
        )
    if lineage:
        return frozenset(lineage)
    if require_lineage:
        raise ValueError(
            "Gold evidence needs stable video lineage: video_id plus shot_id, "
            f"frame_id, or a time window at {context}"
        )
    evidence_id = str(value.get("evidence_id") or "").strip()
    if evidence_id:
        return frozenset({f"local::{evidence_id}"})
    raise ValueError(
        f"Evidence needs a local evidence_id or stable video lineage at {context}"
    )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _format_time(value: float) -> str:
    return format(value, ".9g")


def _has_answer(value: Any) -> bool:
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return False
    return bool(str(value).strip())


def _answer_variants(value: Any) -> list[str]:
    raw = value if isinstance(value, list) else [value]
    return [str(item) for item in raw if item is not None and str(item).strip()]


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(_TOKEN_PATTERN.findall(text))


def _exact_match(predicted: Any, gold: Any) -> float:
    return float(_normalize_text(predicted) == _normalize_text(gold))


def _token_f1(predicted: Any, gold: Any) -> float:
    predicted_tokens = _normalize_text(predicted).split()
    gold_tokens = _normalize_text(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    predicted_counts: defaultdict[str, int] = defaultdict(int)
    gold_counts: defaultdict[str, int] = defaultdict(int)
    for token in predicted_tokens:
        predicted_counts[token] += 1
    for token in gold_tokens:
        gold_counts[token] += 1
    overlap = sum(min(count, gold_counts[token]) for token, count in predicted_counts.items())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return _harmonic(precision, recall)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _harmonic(first: float, second: float) -> float:
    return 2.0 * first * second / (first + second) if first + second else 0.0


def _manifest_ids(splits: Mapping[str, Any], name: str) -> list[str]:
    split = splits.get(name)
    if not isinstance(split, Mapping):
        raise ValueError(f"Missing split: {name}")
    ids = split.get("query_ids")
    if not isinstance(ids, list) or not all(isinstance(value, str) and value for value in ids):
        raise ValueError(f"{name}.query_ids must be a non-empty string list")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {name} split")
    return list(ids)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append((line_number, value))
    return rows
