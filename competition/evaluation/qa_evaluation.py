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
        "unanswerable",
    }
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
        _flatten_constraints(row["known_constraints"], context=f"{path}:{line_number}")
        if not isinstance(row["gold_evidence"], list):
            raise ValueError(f"gold_evidence must be a list at {path}:{line_number}")
        for evidence in row["gold_evidence"]:
            _evidence_key(evidence, context=f"{path}:{line_number}")
        answerable = row["answerable"]
        if not isinstance(answerable, bool):
            raise ValueError(f"answerable must be boolean at {path}:{line_number}")
        gold_answer = row["gold_answer"]
        if answerable and not _answer_variants(gold_answer):
            raise ValueError(f"answerable row needs gold_answer at {path}:{line_number}")
        if not answerable and gold_answer not in (None, [], ""):
            raise ValueError(f"unanswerable row needs null gold_answer at {path}:{line_number}")
        if answerable and answer_type == "unanswerable":
            raise ValueError(f"answerable row cannot use unanswerable type at {path}:{line_number}")
        if not answerable and answer_type != "unanswerable":
            raise ValueError(f"unanswerable row must use unanswerable type at {path}:{line_number}")
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
    parser = _aggregate_parser(rows)
    evidence = _average_metrics(
        evidence_rows,
        ("Evidence Hit@1", "Evidence Hit@5", "Evidence Hit@10", "MRR", "nDCG@10"),
    )
    answer = _average_metrics(answerable_rows, ("answer_EM", "answer_F1"))
    abstention = _aggregate_abstention(rows)
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
        "answer": answer,
        "abstention": abstention,
        "latency_ms": latency,
        "peak_vram_mb": vram,
        "queries": rows,
    }


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

    gold_evidence = {_evidence_key(value) for value in label["gold_evidence"]}
    raw_evidence = prediction.get("evidence", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    ranked_evidence = [_evidence_key(value) for value in raw_evidence]
    matched_gold: set[str] = set()
    relevance: list[bool] = []
    for key in ranked_evidence:
        is_new_match = key in gold_evidence and key not in matched_gold
        relevance.append(is_new_match)
        if is_new_match:
            matched_gold.add(key)
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
        answer_status = str(answer_payload.get("status") or "")
        predicted_answer = answer_payload.get("answer")
    else:
        answer_status = str(prediction.get("status") or "")
        predicted_answer = prediction.get("answer")
    predicted_abstention = answer_status == "insufficient_evidence" or predicted_answer is None
    gold_answers = _answer_variants(label["gold_answer"])
    em = max((_exact_match(predicted_answer, answer) for answer in gold_answers), default=0.0)
    f1 = max((_token_f1(predicted_answer, answer) for answer in gold_answers), default=0.0)
    row: dict[str, Any] = {
        "query_id": query_id,
        "answerable": bool(label["answerable"]),
        "has_gold_evidence": bool(gold_evidence),
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
        "answer_EM": em,
        "answer_F1": f1,
        "predicted_abstention": predicted_abstention,
        "abstention_correct": float(predicted_abstention == (not bool(label["answerable"]))),
    }
    for metric_name in ("latency_ms", "peak_vram_mb"):
        value = prediction.get(metric_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if float(value) < 0:
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
            statistics.fmean(float(not row["predicted_abstention"]) for row in answerable)
            if answerable
            else 0.0
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


def _evidence_key(value: Any, *, context: str = "evidence") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid evidence at {context}: {value!r}")
    video_id = str(value.get("video_id") or "").strip()
    frame_id = str(value.get("frame_id") or "").strip()
    if video_id and frame_id:
        return f"{video_id}::{frame_id}"
    evidence_id = str(value.get("evidence_id") or "").strip()
    if evidence_id:
        return evidence_id
    raise ValueError(f"Evidence needs evidence_id or video_id/frame_id at {context}")


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
