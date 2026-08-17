"""Serialize ranked backend results to the AIC KIS/QA long-form CSV contract."""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, is_dataclass
from numbers import Integral
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from backend.app.services.submission.schemas import ExportRequest, ExportedCsv, SubmissionTask


class SubmissionExportError(RuntimeError):
    """A retrieval result cannot safely be represented as a submission."""


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    raise TypeError("submission candidates must be mappings or expose to_dict()")


def _video_id(candidate: Mapping[str, Any]) -> str:
    raw = str(candidate.get("video_id") or "").strip().replace("\\", "/")
    video_id = PurePosixPath(raw).stem
    if not video_id or video_id in {".", ".."}:
        raise ValueError("candidate has no valid video_id")
    return video_id


def _original_frame_index(candidate: Mapping[str, Any]) -> int:
    raw = candidate.get("frame_index")
    if raw is None:
        metadata = candidate.get("metadata")
        if isinstance(metadata, Mapping):
            raw = metadata.get("frame_index")
    if isinstance(raw, bool) or not isinstance(raw, (Integral, str)):
        raise ValueError("candidate has no original frame_index")
    if isinstance(raw, str) and not raw.strip().isdigit():
        raise ValueError("candidate frame_index is not an integer")
    try:
        frame_index = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate frame_index is not an integer") from exc
    if frame_index < 0:
        raise ValueError("candidate frame_index must be a non-negative integer")
    return frame_index


def _ranked_rows(candidates: Sequence[Any], top_k: int) -> list[tuple[str, int, Mapping[str, Any]]]:
    rows: list[tuple[str, int, Mapping[str, Any]]] = []
    seen: set[tuple[str, int]] = set()
    for raw_candidate in candidates:
        candidate = _mapping(raw_candidate)
        try:
            identity = (_video_id(candidate), _original_frame_index(candidate))
        except ValueError:
            continue
        if identity in seen:
            continue
        seen.add(identity)
        rows.append((*identity, candidate))
        if len(rows) >= top_k:
            break
    return rows


def _write_csv(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def serialize_kis_csv(candidates: Sequence[Any], *, top_k: int = 100) -> str:
    request = ExportRequest.parse("serialization", "kis", top_k)
    ranked = _ranked_rows(candidates, request.top_k)
    return _write_csv(("video_id", "frame_id"), [(video, frame) for video, frame, _ in ranked])


def _qa_candidates(response: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]]]:
    answer = response.get("answer")
    if not isinstance(answer, Mapping):
        raise SubmissionExportError("grounded QA returned no answer contract")
    status = str(answer.get("status") or "").casefold().strip()
    text = answer.get("answer")
    if status != "answered" or not isinstance(text, str) or not text.strip():
        reason = str(answer.get("reason") or status or "insufficient evidence")
        raise SubmissionExportError(f"grounded QA did not answer: {reason}")

    evidence_raw = response.get("evidence")
    evidence = [dict(item) for item in evidence_raw or [] if isinstance(item, Mapping)]
    by_id = {str(item.get("evidence_id") or ""): item for item in evidence}
    cited_ids = [str(value) for value in answer.get("evidence_ids") or []]
    if not cited_ids or any(value not in by_id for value in cited_ids):
        raise SubmissionExportError("grounded QA citations are missing or invalid")
    ordered = [by_id[value] for value in dict.fromkeys(cited_ids)]
    cited = set(cited_ids)
    ordered.extend(item for item in evidence if str(item.get("evidence_id") or "") not in cited)
    return text, ordered


def serialize_qa_csv(response: Mapping[str, Any], *, top_k: int = 100) -> str:
    request = ExportRequest.parse("serialization", "qa", top_k)
    answer, candidates = _qa_candidates(response)
    ranked = _ranked_rows(candidates, request.top_k)
    if not ranked:
        raise SubmissionExportError("grounded QA has no valid original-frame evidence")
    return _write_csv(
        ("video_id", "frame_id", "answer"),
        [(video, frame, answer) for video, frame, _ in ranked],
    )


def export_query_csv(
    query: str,
    task: str,
    top_k: int = 100,
    *,
    online_search: Callable[..., Mapping[str, Any]] | None = None,
) -> ExportedCsv:
    request = ExportRequest.parse(query, task, top_k)
    if online_search is None:
        from backend.app.services.retrieval.retrieval_manager import search_online

        online_search = search_online
    response = online_search(
        query=request.query,
        task=request.task.value,
        top_k=request.top_k,
    )
    if request.task is SubmissionTask.KIS:
        payload = _mapping(response)
        candidates = payload.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise SubmissionExportError("KIS retrieval returned no ranked results")
        content = serialize_kis_csv(candidates, top_k=request.top_k)
        row_count = max(0, content.count("\r\n") - 1)
    else:
        content = serialize_qa_csv(response, top_k=request.top_k)
        row_count = max(0, content.count("\r\n") - 1)
    return ExportedCsv(
        content=content,
        filename=f"{request.task.value}_result.csv",
        row_count=row_count,
        task=request.task,
    )
