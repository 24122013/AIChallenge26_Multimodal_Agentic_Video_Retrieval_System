"""Deterministically ensemble validated competition submissions with weighted RRF."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from competition.pipeline import validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_submission(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Submission has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _atomic_submission(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensemble_submissions(
    *,
    submissions: Sequence[Path],
    weights: Sequence[float],
    output_path: Path,
    public_root: Path,
    rrf_k: int = 60,
) -> dict[str, object]:
    if len(submissions) < 2:
        raise ValueError("At least two submissions are required for an ensemble")
    if len(weights) != len(submissions):
        raise ValueError("Provide exactly one weight per submission")
    if any(weight <= 0 for weight in weights):
        raise ValueError("Ensemble weights must be positive")
    if rrf_k <= 0:
        raise ValueError("RRF k must be positive")

    inputs = []
    for path in submissions:
        resolved = path.resolve()
        validate_submission(resolved, public_root.resolve())
        inputs.append((resolved, *_read_submission(resolved)))
    canonical_header = inputs[0][1]
    query_ids = [row["query_id"] for row in inputs[0][2]]
    for path, header, rows in inputs[1:]:
        if header != canonical_header:
            raise ValueError(f"Submission header differs: {path}")
        if [row["query_id"] for row in rows] != query_ids:
            raise ValueError(f"Submission query order differs: {path}")

    answer_columns = canonical_header[1:]
    output_rows: list[dict[str, str]] = []
    for row_index, query_id in enumerate(query_ids):
        scores: dict[str, float] = {}
        tie_break: dict[str, tuple[int, int, str]] = {}
        for input_index, ((_, _, rows), weight) in enumerate(zip(inputs, weights)):
            seen: set[str] = set()
            for rank, column in enumerate(answer_columns, start=1):
                answer = (rows[row_index].get(column) or "").strip()
                if not answer or answer in seen:
                    continue
                seen.add(answer)
                scores[answer] = scores.get(answer, 0.0) + float(weight) / (rrf_k + rank)
                key = (rank, input_index, answer)
                tie_break[answer] = min(tie_break.get(answer, key), key)
        ranked = sorted(scores, key=lambda answer: (-scores[answer], tie_break[answer]))
        if len(ranked) < len(answer_columns):
            raise ValueError(
                f"Query {query_id} has only {len(ranked)} unique ensemble answers; "
                f"need {len(answer_columns)}"
            )
        output_rows.append(
            {
                "query_id": query_id,
                **{
                    column: answer
                    for column, answer in zip(answer_columns, ranked[: len(answer_columns)])
                },
            }
        )

    output_path = output_path.resolve()
    _atomic_submission(output_path, canonical_header, output_rows)
    validation = validate_submission(output_path, public_root.resolve())
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "weighted_reciprocal_rank_fusion",
        "rrf_k": rrf_k,
        "inputs": [
            {"path": path.as_posix(), "sha256": _sha256(path), "weight": float(weight)}
            for (path, _, _), weight in zip(inputs, weights)
        ],
        "output": {
            "path": output_path.name,
            "sha256": _sha256(output_path),
            "validation": validation,
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    _atomic_json(manifest_path, manifest)
    return {**validation, "submission_sha256": manifest["output"]["sha256"], "manifest": manifest_path.as_posix()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", action="append", type=Path, required=True)
    parser.add_argument("--weight", action="append", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, default=Path("data/public"))
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    weights = args.weight or [1.0] * len(args.submission)
    try:
        report = ensemble_submissions(
            submissions=args.submission,
            weights=weights,
            output_path=args.output,
            public_root=args.public_root,
            rrf_k=args.rrf_k,
        )
    except ValueError as exc:
        parser.exit(1, f"ensemble failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
