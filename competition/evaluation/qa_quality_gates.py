"""Pre-registered quality gates for adjudicated grounded-QA datasets.

Synthetic fixtures must never pass these gates.  Missing metrics fail closed so
that a partial smoke run cannot be presented as a quality evaluation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


REAL_SPLIT_QUERY_COUNT = 80
REAL_SPLIT_LANGUAGE_COUNT = 40
MIN_UNANSWERABLE_COUNT = 16
MIN_TEMPORAL_COUNT = 16
MIN_CONSTRAINT_HEAVY_COUNT = 24
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def evaluate_real_dev_gates(
    report: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen real-dev gates from the approved QA plan."""

    _validate_real_split_manifest(dataset_manifest, split="dev")
    _validate_report_binding(report, dataset_manifest, split="dev")
    parser = _mapping(report, "parser")
    evidence = _mapping(report, "evidence")
    answer = _mapping(report, "answer")
    abstention = _mapping(report, "abstention")
    temporal = _mapping(report, "temporal")
    integrity = _mapping(report, "integrity")
    gates = {
        "parser_answer_type_macro_F1": _rate(parser, "answer_type_macro_F1") >= 0.90,
        "parser_constraint_F1": _rate(parser, "constraint_F1") >= 0.80,
        "evidence_Hit@5": _rate(evidence, "Evidence Hit@5") >= 0.75,
        "evidence_nDCG@10": _rate(evidence, "nDCG@10") >= 0.65,
        "temporal_chain_Hit@5": _rate(temporal, "Temporal Chain Hit@5") >= 0.70,
        "answer_token_F1": _rate(answer, "answer_F1") >= 0.65,
        "unanswerable_recall": _rate(abstention, "unanswerable_recall") >= 0.90,
        "unsupported_answer_rate": _rate(integrity, "unsupported_answer_rate") <= 0.05,
        "pipeline_error_rate": _rate(integrity, "pipeline_error_rate") == 0.0,
        "invalid_response_rate": _rate(integrity, "invalid_response_rate") == 0.0,
        "invalid_citation_rate": _rate(integrity, "invalid_citation_rate") == 0.0,
    }
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "passed": all(gates.values()),
        "split": "dev",
        "pre_registered": True,
        "gates": gates,
    }


def evaluate_answerer_only_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    """Gate Qwen on gold evidence, independently from retrieval quality."""

    if report.get("evaluation_mode") != "answerer_only_gold_evidence":
        raise ValueError(
            "answerer-only report must use evaluation_mode=answerer_only_gold_evidence"
        )
    if report.get("evidence_source") != "gold":
        raise ValueError("answerer-only report must declare evidence_source=gold")
    gold_hash = _sha256(report, "gold_evidence_sha256")
    evaluated_hash = _sha256(report, "evaluated_evidence_lineage_sha256")
    if gold_hash.casefold() != evaluated_hash.casefold():
        raise ValueError("answerer-only evaluated evidence does not match gold lineage")
    answer = _mapping(report, "answer")
    abstention = _mapping(report, "abstention")
    integrity = _mapping(report, "integrity")
    gates = {
        "answer_token_F1": _rate(answer, "answer_F1") >= 0.80,
        "answerable_response_rate": (
            _rate(abstention, "answerable_response_rate") >= 0.90
        ),
        "invalid_response_rate": _rate(integrity, "invalid_response_rate") == 0.0,
        "invalid_citation_rate": _rate(integrity, "invalid_citation_rate") == 0.0,
    }
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "passed": all(gates.values()),
        "evidence_source": "gold",
        "gates": gates,
    }


def validate_locked_split_manifest(dataset_manifest: Mapping[str, Any]) -> None:
    """Validate hidden locked-set quotas before its single authorized use."""

    _validate_real_split_manifest(dataset_manifest, split="locked_test")
    policy = dataset_manifest.get("locked_test_policy")
    if not isinstance(policy, Mapping) or policy.get("reuse") != "single_use":
        raise ValueError("locked_test_policy.reuse must be 'single_use'")
    if policy.get("labels_hidden") is not True:
        raise ValueError("locked_test labels must remain hidden")
    splits = _mapping(dataset_manifest, "splits")
    dev_ids = set(_query_ids(_mapping(splits, "dev"), split="dev"))
    locked_ids = set(
        _query_ids(_mapping(splits, "locked_test"), split="locked_test")
    )
    if dev_ids & locked_ids:
        raise ValueError("real QA dev and locked_test query IDs must be disjoint")


def _validate_real_split_manifest(
    manifest: Mapping[str, Any],
    *,
    split: str,
) -> None:
    if manifest.get("example_only") is not False:
        raise ValueError("real QA manifest must set example_only=false")
    if manifest.get("labels_adjudicated") is not True:
        raise ValueError("real QA labels must be adjudicated")
    _sha256(manifest, "dataset_sha256")
    splits = _mapping(manifest, "splits")
    summary = _mapping(splits, split)
    query_count = _count(summary, "query_count")
    if query_count != REAL_SPLIT_QUERY_COUNT:
        raise ValueError(f"{split} must contain exactly {REAL_SPLIT_QUERY_COUNT} queries")
    query_ids = _query_ids(summary, split=split)
    if len(query_ids) != query_count:
        raise ValueError(f"{split}.query_ids must match query_count")
    declared_query_hash = _sha256(summary, "query_ids_sha256")
    actual_query_hash = hashlib.sha256(
        json.dumps(
            sorted(query_ids),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if declared_query_hash.casefold() != actual_query_hash:
        raise ValueError(f"{split}.query_ids_sha256 does not match query_ids")
    languages = _mapping(summary, "languages")
    for language in ("vi", "en"):
        if _count(languages, language) != REAL_SPLIT_LANGUAGE_COUNT:
            raise ValueError(
                f"{split} must contain exactly {REAL_SPLIT_LANGUAGE_COUNT} {language} queries"
            )
    if sum(_count(languages, language) for language in ("vi", "en")) != query_count:
        raise ValueError(f"{split} language counts must sum to query_count")
    unanswerable_count = _count(summary, "unanswerable_count")
    temporal_count = _count(summary, "temporal_count")
    constraint_heavy_count = _count(summary, "constraint_heavy_count")
    for key, value in (
        ("unanswerable_count", unanswerable_count),
        ("temporal_count", temporal_count),
        ("constraint_heavy_count", constraint_heavy_count),
    ):
        if value > query_count:
            raise ValueError(f"{split}.{key} cannot exceed query_count")
    if unanswerable_count < MIN_UNANSWERABLE_COUNT:
        raise ValueError(f"{split} needs at least {MIN_UNANSWERABLE_COUNT} unanswerable queries")
    if temporal_count < MIN_TEMPORAL_COUNT:
        raise ValueError(f"{split} needs at least {MIN_TEMPORAL_COUNT} temporal queries")
    if constraint_heavy_count < MIN_CONSTRAINT_HEAVY_COUNT:
        raise ValueError(
            f"{split} needs at least {MIN_CONSTRAINT_HEAVY_COUNT} constraint-heavy queries"
        )


def _validate_report_binding(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    split: str,
) -> None:
    summary = _mapping(_mapping(manifest, "splits"), split)
    report_count = _count(report, "query_count")
    manifest_count = _count(summary, "query_count")
    if report_count != manifest_count:
        raise ValueError("report query_count does not match dataset manifest")
    report_hash = _sha256(report, "query_ids_sha256")
    manifest_hash = _sha256(summary, "query_ids_sha256")
    if report_hash.casefold() != manifest_hash.casefold():
        raise ValueError("report query_ids_sha256 does not match dataset manifest")
    report_dataset_hash = _sha256(report, "dataset_sha256")
    manifest_dataset_hash = (
        _sha256(summary, "dataset_sha256")
        if "dataset_sha256" in summary
        else _sha256(manifest, "dataset_sha256")
    )
    if report_dataset_hash.casefold() != manifest_dataset_hash.casefold():
        raise ValueError("report dataset_sha256 does not match dataset manifest")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"missing object: {key}")
    return nested


def _query_ids(value: Mapping[str, Any], *, split: str) -> list[str]:
    raw = value.get("query_ids")
    if not isinstance(raw, list) or not raw or not all(
        isinstance(item, str) and bool(item.strip()) for item in raw
    ):
        raise ValueError(f"{split}.query_ids must be a non-empty string list")
    normalized = [item.strip() for item in raw]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{split}.query_ids must be unique")
    return normalized


def _number(value: Mapping[str, Any], key: str) -> float:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"missing numeric metric: {key}")
    number = float(raw)
    if not math.isfinite(number):
        raise ValueError(f"numeric metric must be finite: {key}")
    return number


def _rate(value: Mapping[str, Any], key: str) -> float:
    number = _number(value, key)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"metric must be within [0, 1]: {key}")
    return number


def _count(value: Mapping[str, Any], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"missing numeric metric: {key}")
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError(f"count must be an exact nonnegative integer: {key}")
        return raw
    if not math.isfinite(raw) or raw < 0.0 or not raw.is_integer():
        raise ValueError(f"count must be an exact nonnegative integer: {key}")
    return int(raw)


def _sha256(value: Mapping[str, Any], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or _SHA256.fullmatch(raw) is None:
        raise ValueError(f"{key} must be a 64-character hexadecimal SHA256")
    return raw
