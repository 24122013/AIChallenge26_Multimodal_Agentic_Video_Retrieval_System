"""Run bounded KIS, AVS, and grounded-QA checks on existing artifacts.

This command never builds a submission.  It is the post-submission verification
surface used by the Colab launcher and README examples.  KIS/AVS stop at the
evidence bundle; only QA invokes the configured grounded answerer.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from backend.app.services.retrieval.qa_pipeline import RequiredQaPipelineError
from backend.app.services.retrieval.retrieval_manager import (
    clear_retrieval_caches,
    get_qa_evidence_search_engine,
    search_qa,
)


DEFAULT_QUERIES = {
    "kis": "người phụ nữ mặc áo đỏ đang cầm điện thoại",
    "avs": "tất cả các cảnh có xe máy đi qua đường",
    "qa": "Người phụ nữ mặc áo đỏ đang cầm vật gì?",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("kis", "avs", "qa", "all"), default="all")
    parser.add_argument("--kis-query", default=DEFAULT_QUERIES["kis"])
    parser.add_argument("--avs-query", default=DEFAULT_QUERIES["avs"])
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
                    "warnings",
                )
            }
        )
    return output


def _task_payload(task: str, query: str, top_k: int, expanded: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    if task == "qa":
        response = search_qa(
            query,
            top_k=top_k,
            task_mode="qa",
            expanded_queries=expanded,
        )
    else:
        response = get_qa_evidence_search_engine().search(
            query,
            top_k=top_k,
            task_mode=task,
        )
    payload = {
        "task": task,
        "query": query,
        "query_plan": response.get("query_plan", {}),
        "routing_trace": response.get("routing_trace", {}),
        "answer": response.get("answer") if task == "qa" else None,
        "answer_report": response.get("answer_report") if task == "qa" else None,
        "evidence": _bounded_evidence(response.get("evidence"), top_k),
        "latency_ms": response.get(
            "latency_ms",
            round((time.perf_counter() - started) * 1000.0, 3),
        ),
    }
    return payload


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= int(args.top_k) <= 20:
        raise ValueError("--top-k must be within [1, 20]")
    clear_retrieval_caches()
    tasks = ("kis", "avs", "qa") if args.task == "all" else (args.task,)
    queries = {
        "kis": args.kis_query,
        "avs": args.avs_query,
        "qa": args.qa_query,
    }
    results: list[dict[str, Any]] = []
    try:
        for task in tasks:
            results.append(
                _task_payload(task, queries[task], int(args.top_k), list(args.expanded_query))
            )
    except RequiredQaPipelineError as exc:
        failure = {
            "status": "failed",
            "reason": str(exc),
            "results": [*results, {"task": "qa", **dict(exc.response)}],
        }
        if args.output is not None:
            _atomic_json(args.output, failure)
        raise
    payload = {"status": "passed", "results": results}
    if args.output is not None:
        _atomic_json(args.output, payload)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
