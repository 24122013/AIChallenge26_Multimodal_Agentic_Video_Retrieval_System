"""Fail-closed comparison gates for grounded-QA answer models.

The module deliberately does not run either model.  It compares two auditable
evaluation reports produced from identical evidence and refuses to make a
promotion recommendation when the run manifests are not paired correctly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


BASELINE_MODEL = "Qwen/Qwen3.5-9B"
BASELINE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CANDIDATE_MODEL = "Qwen/Qwen3.5-4B"
CANDIDATE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

_PAIRED_FIELDS = (
    "dataset_sha256",
    "evidence_sha256",
    "prompt_revision",
    "decoding",
    "hardware",
    "software",
    "quantization",
    "cache_policy",
    "split",
    "labels_adjudicated",
    "languages",
    "unanswerable_count",
    "temporal_count",
    "constraint_heavy_count",
    "query_count",
    "query_ids_sha256",
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_COMPARISON_EPSILON = 1e-12
REAL_BENCHMARK_QUERY_COUNT = 80
REAL_BENCHMARK_LANGUAGE_COUNT = 40
MIN_GPU_MEMORY_GB = 16.0


def evaluate_4b_eligibility(
    *,
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    bootstrap_samples: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare pinned 9B and 4B reports using the approved promotion gates."""

    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 100
    ):
        raise ValueError("bootstrap_samples must be >= 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _validate_manifest_pair(baseline_manifest, candidate_manifest)
    baseline_rows, candidate_rows = _paired_rows(baseline_report, candidate_report)
    _validate_report_binding(
        baseline_report,
        baseline_manifest,
        baseline_rows,
        label="baseline",
    )
    _validate_report_binding(
        candidate_report,
        candidate_manifest,
        candidate_rows,
        label="candidate",
    )
    answerable_ids = [
        query_id
        for query_id, row in baseline_rows.items()
        if bool(row.get("answerable"))
    ]
    unanswerable_ids = [
        query_id
        for query_id, row in baseline_rows.items()
        if not bool(row.get("answerable"))
    ]
    if not answerable_ids or not unanswerable_ids:
        raise ValueError("benchmark needs answerable and unanswerable examples")

    answer_f1_ci = _paired_bootstrap_ci(
        answerable_ids,
        baseline_rows,
        candidate_rows,
        metric="answer_F1",
        samples=bootstrap_samples,
        seed=seed,
    )
    balanced_abstention_ci = _balanced_abstention_ci(
        answerable_ids,
        unanswerable_ids,
        baseline_rows,
        candidate_rows,
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    answer_f1_delta = _mean_delta(
        answerable_ids,
        baseline_rows,
        candidate_rows,
        metric="answer_F1",
    )
    balanced_abstention_delta = _balanced_abstention_delta(
        answerable_ids,
        unanswerable_ids,
        baseline_rows,
        candidate_rows,
    )

    baseline_unsupported = _boolean_rate(baseline_rows, "unsupported_answer")
    candidate_unsupported = _boolean_rate(candidate_rows, "unsupported_answer")
    unsupported_delta = candidate_unsupported - baseline_unsupported
    response_schema_valid = _all_true(candidate_rows, "response_valid")
    citations_valid = _all_true(candidate_rows, "citation_valid")

    latency_reduction = _resource_reduction(
        baseline_report,
        candidate_report,
        section="latency_ms",
        field="p95",
    )
    vram_reduction = _resource_reduction(
        baseline_report,
        candidate_report,
        section="peak_vram_mb",
        field="max",
    )
    quality_pass = (
        answer_f1_delta >= -0.02 - _COMPARISON_EPSILON
        and balanced_abstention_delta >= -0.02 - _COMPARISON_EPSILON
    )
    integrity_pass = (
        unsupported_delta <= 0.01 + _COMPARISON_EPSILON
        and response_schema_valid
        and citations_valid
    )
    efficiency_pass = latency_reduction >= 0.20 or vram_reduction >= 0.25
    gates = {
        "quality_non_inferiority": quality_pass,
        "unsupported_answer_delta": unsupported_delta <= 0.01 + _COMPARISON_EPSILON,
        "response_schema_valid": response_schema_valid,
        "citations_valid": citations_valid,
        "latency_or_vram_improved": efficiency_pass,
    }
    return {
        "status": "evaluated",
        "eligible": all(gates.values()),
        "default_model_changed": False,
        "baseline_model": f"{BASELINE_MODEL}@{BASELINE_REVISION}",
        "candidate_model": f"{CANDIDATE_MODEL}@{CANDIDATE_REVISION}",
        "query_count": len(baseline_rows),
        "answer_F1_delta": answer_f1_delta,
        "answer_F1_delta_CI95": answer_f1_ci,
        "balanced_abstention_delta": balanced_abstention_delta,
        "balanced_abstention_delta_CI95": balanced_abstention_ci,
        "unsupported_answer_rate": {
            "baseline": baseline_unsupported,
            "candidate": candidate_unsupported,
            "delta": unsupported_delta,
        },
        "latency_p95_reduction": latency_reduction,
        "peak_vram_reduction": vram_reduction,
        "gates": gates,
        "note": "Eligibility is advisory; Qwen3.5-9B remains the runtime default.",
    }


def _validate_manifest_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    expected_models = (
        (baseline, BASELINE_MODEL, BASELINE_REVISION, "baseline"),
        (candidate, CANDIDATE_MODEL, CANDIDATE_REVISION, "candidate"),
    )
    for manifest, expected_name, expected_revision, label in expected_models:
        if manifest.get("model_name") != expected_name:
            raise ValueError(f"{label} model_name must be {expected_name}")
        if manifest.get("model_revision") != expected_revision:
            raise ValueError(f"{label} model_revision must be pinned to {expected_revision}")
        missing = [field for field in _PAIRED_FIELDS if field not in manifest]
        if missing:
            raise ValueError(f"{label} manifest missing paired fields: {missing}")
        _sha256(manifest.get("dataset_sha256"), field=f"{label}.dataset_sha256")
        _sha256(manifest.get("evidence_sha256"), field=f"{label}.evidence_sha256")
        if not isinstance(manifest.get("prompt_revision"), str) or not str(
            manifest["prompt_revision"]
        ).strip():
            raise ValueError(f"{label}.prompt_revision must be a non-empty string")
        for field in ("decoding", "hardware", "software"):
            if not isinstance(manifest.get(field), Mapping):
                raise ValueError(f"{label}.{field} must be an object")
        if manifest.get("split") != "real_dev":
            raise ValueError(f"{label}.split must be real_dev")
        if manifest.get("labels_adjudicated") is not True:
            raise ValueError(f"{label} benchmark labels must be adjudicated")
        if _count(manifest["query_count"], field=f"{label}.query_count") != REAL_BENCHMARK_QUERY_COUNT:
            raise ValueError(
                f"{label} benchmark must contain exactly {REAL_BENCHMARK_QUERY_COUNT} queries"
            )
        _sha256(
            manifest["query_ids_sha256"],
            field=f"{label}.query_ids_sha256",
        )
        languages = manifest.get("languages")
        if not isinstance(languages, Mapping) or any(
            _count(languages.get(language), field=f"{label}.languages.{language}")
            != REAL_BENCHMARK_LANGUAGE_COUNT
            for language in ("vi", "en")
        ):
            raise ValueError(f"{label} benchmark needs exactly 40 vi and 40 en queries")
        for field, minimum in (
            ("unanswerable_count", 16),
            ("temporal_count", 16),
            ("constraint_heavy_count", 24),
        ):
            count = _count(manifest[field], field=f"{label}.{field}")
            if not minimum <= count <= REAL_BENCHMARK_QUERY_COUNT:
                raise ValueError(f"{label}.{field} must be within [{minimum}, 80]")
        hardware = manifest["hardware"]
        assert isinstance(hardware, Mapping)
        gpu_memory = _finite_number(
            hardware.get("gpu_memory_gb"),
            field=f"{label}.hardware.gpu_memory_gb",
        )
        if gpu_memory < MIN_GPU_MEMORY_GB:
            raise ValueError(
                f"{label} benchmark GPU must provide at least {MIN_GPU_MEMORY_GB:g} GB"
            )
    mismatched = [
        field for field in _PAIRED_FIELDS if baseline[field] != candidate[field]
    ]
    if mismatched:
        raise ValueError(f"benchmark manifests are not paired: {mismatched}")
    if baseline["quantization"] != "4bit":
        raise ValueError("benchmark quantization must be exactly '4bit'")
    if baseline["cache_policy"] != "miss_only":
        raise ValueError("benchmark cache_policy must be exactly 'miss_only'")


def _paired_rows(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    baseline = _rows_by_id(baseline_report)
    candidate = _rows_by_id(candidate_report)
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate reports must contain identical query IDs")
    for query_id in baseline:
        if baseline[query_id]["answerable"] is not candidate[query_id]["answerable"]:
            raise ValueError(f"answerable label differs for {query_id}")
    return baseline, candidate


def _rows_by_id(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = report.get("queries")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("QA report needs a queries list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("QA report query rows must be objects")
        raw_query_id = row.get("query_id")
        if not isinstance(raw_query_id, str):
            raise ValueError("QA report query_id must be a string")
        query_id = raw_query_id.strip()
        if not query_id or query_id in indexed:
            raise ValueError("QA report has missing or duplicate query_id")
        for field in (
            "answerable",
            "answer_F1",
            "abstention_correct",
            "unsupported_answer",
            "response_valid",
            "citation_valid",
            "model_invoked",
            "cache_hit",
        ):
            if field not in row:
                raise ValueError(f"QA report row {query_id} missing {field}")
        if not isinstance(row["answerable"], bool):
            raise ValueError(f"QA report row {query_id}.answerable must be boolean")
        _rate(row["answer_F1"], field=f"{query_id}.answer_F1")
        _rate(
            row["abstention_correct"],
            field=f"{query_id}.abstention_correct",
        )
        for field in (
            "unsupported_answer",
            "response_valid",
            "citation_valid",
            "model_invoked",
            "cache_hit",
        ):
            if not isinstance(row[field], bool):
                raise ValueError(f"QA report row {query_id}.{field} must be boolean")
        if row["model_invoked"] is not True or row["cache_hit"] is not False:
            raise ValueError(
                f"QA report row {query_id} must prove model_invoked=true and cache_hit=false"
            )
        indexed[query_id] = row
    if not indexed:
        raise ValueError("QA report cannot be empty")
    return indexed


def _validate_report_binding(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    for field in ("dataset_sha256", "evidence_sha256"):
        if field not in report:
            raise ValueError(f"{label} report missing manifest-bound {field}")
        report_hash = _sha256(
            report[field],
            field=f"{label}.report.{field}",
        )
        manifest_hash = _sha256(
            manifest[field],
            field=f"{label}.manifest.{field}",
        )
        if report_hash.casefold() != manifest_hash.casefold():
            raise ValueError(f"{label} report {field} does not match manifest")
    if "query_count" in report:
        report_count = _count(report["query_count"], field=f"{label}.report.query_count")
        if report_count != len(rows):
            raise ValueError(f"{label} report query_count does not match query rows")
    if "query_count" in manifest:
        manifest_count = _count(
            manifest["query_count"],
            field=f"{label}.manifest.query_count",
        )
        if "query_count" not in report:
            raise ValueError(f"{label} report missing manifest-bound query_count")
        if manifest_count != _count(
            report["query_count"],
            field=f"{label}.report.query_count",
        ):
            raise ValueError(f"{label} report query_count does not match manifest")

    if "query_ids_sha256" in report:
        report_hash = _sha256(
            report["query_ids_sha256"],
            field=f"{label}.report.query_ids_sha256",
        )
        actual_hash = _query_ids_sha256(rows)
        if report_hash.casefold() != actual_hash:
            raise ValueError(f"{label} report query_ids_sha256 does not match query rows")
    if "query_ids_sha256" in manifest:
        manifest_hash = _sha256(
            manifest["query_ids_sha256"],
            field=f"{label}.manifest.query_ids_sha256",
        )
        if "query_ids_sha256" not in report:
            raise ValueError(f"{label} report missing manifest-bound query_ids_sha256")
        report_hash = _sha256(
            report["query_ids_sha256"],
            field=f"{label}.report.query_ids_sha256",
        )
        if manifest_hash.casefold() != report_hash.casefold():
            raise ValueError(f"{label} report query_ids_sha256 does not match manifest")


def _paired_bootstrap_ci(
    query_ids: Sequence[str],
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        drawn = [rng.choice(query_ids) for _ in query_ids]
        deltas.append(
            statistics.fmean(
                float(candidate[query_id][metric]) - float(baseline[query_id][metric])
                for query_id in drawn
            )
        )
    return _ci_summary(deltas)


def _mean_delta(
    query_ids: Sequence[str],
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
) -> float:
    return statistics.fmean(
        float(candidate[query_id][metric]) - float(baseline[query_id][metric])
        for query_id in query_ids
    )


def _balanced_abstention_delta(
    answerable_ids: Sequence[str],
    unanswerable_ids: Sequence[str],
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> float:
    baseline_score = 0.5 * (
        statistics.fmean(
            float(baseline[key]["abstention_correct"])
            for key in answerable_ids
        )
        + statistics.fmean(
            float(baseline[key]["abstention_correct"])
            for key in unanswerable_ids
        )
    )
    candidate_score = 0.5 * (
        statistics.fmean(
            float(candidate[key]["abstention_correct"])
            for key in answerable_ids
        )
        + statistics.fmean(
            float(candidate[key]["abstention_correct"])
            for key in unanswerable_ids
        )
    )
    return candidate_score - baseline_score


def _balanced_abstention_ci(
    answerable_ids: Sequence[str],
    unanswerable_ids: Sequence[str],
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        answerable = [rng.choice(answerable_ids) for _ in answerable_ids]
        unanswerable = [rng.choice(unanswerable_ids) for _ in unanswerable_ids]
        baseline_score = 0.5 * (
            statistics.fmean(float(baseline[key]["abstention_correct"]) for key in answerable)
            + statistics.fmean(float(baseline[key]["abstention_correct"]) for key in unanswerable)
        )
        candidate_score = 0.5 * (
            statistics.fmean(float(candidate[key]["abstention_correct"]) for key in answerable)
            + statistics.fmean(float(candidate[key]["abstention_correct"]) for key in unanswerable)
        )
        deltas.append(candidate_score - baseline_score)
    return _ci_summary(deltas)


def _ci_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "lower_95": _percentile(ordered, 0.025),
        "median": _percentile(ordered, 0.5),
        "upper_95": _percentile(ordered, 0.975),
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _boolean_rate(rows: Mapping[str, Mapping[str, Any]], field: str) -> float:
    return statistics.fmean(float(bool(row[field])) for row in rows.values())


def _all_true(rows: Mapping[str, Mapping[str, Any]], field: str) -> bool:
    return all(row[field] is True for row in rows.values())


def _resource_reduction(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    section: str,
    field: str,
) -> float:
    baseline_section = baseline.get(section)
    candidate_section = candidate.get(section)
    if not isinstance(baseline_section, Mapping) or not isinstance(candidate_section, Mapping):
        raise ValueError(f"both reports need {section} distributions")
    baseline_value = _finite_number(
        baseline_section.get(field),
        field=f"baseline.{section}.{field}",
    )
    candidate_value = _finite_number(
        candidate_section.get(field),
        field=f"candidate.{section}.{field}",
    )
    if baseline_value <= 0.0 or candidate_value < 0.0:
        raise ValueError(f"invalid {section}.{field} values")
    return (baseline_value - candidate_value) / baseline_value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _rate(value: Any, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return number


def _count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field} must be an exact nonnegative integer")
        return value
    if not math.isfinite(value) or value < 0.0 or not value.is_integer():
        raise ValueError(f"{field} must be an exact nonnegative integer")
    return int(value)


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA256")
    return value


def _query_ids_sha256(rows: Mapping[str, Mapping[str, Any]]) -> str:
    payload = json.dumps(
        sorted(rows),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_4b_eligibility(
        baseline_report=_load_object(args.baseline_report),
        candidate_report=_load_object(args.candidate_report),
        baseline_manifest=_load_object(args.baseline_manifest),
        candidate_manifest=_load_object(args.candidate_manifest),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
