"""Serialize ranked backend results to the AIC submission CSV contracts."""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, is_dataclass
from numbers import Integral
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from backend.app.services.submission.schemas import ExportRequest, ExportedCsv, SubmissionTask


# No official sample_submission.csv is checked into this repository.  Keep the
# provisional TRAKE header convention isolated here so it is trivial to replace
# without touching validation or ranking code.
TRAKE_VIDEO_COLUMN = "video_id"
TRAKE_FRAME_COLUMN_PREFIX = "frame_id_"


class SubmissionExportError(RuntimeError):
    """A retrieval result cannot safely be represented as a submission."""


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    if is_dataclass(value):
        return asdict(value)
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


def _frame_index_value(raw: Any, *, name: str) -> int:
    """Return an explicit original-frame integer without deriving one."""

    if isinstance(raw, bool) or not isinstance(raw, (Integral, str)):
        raise ValueError(f"{name} is not an integer")
    if isinstance(raw, str) and not raw.strip().isdigit():
        raise ValueError(f"{name} is not an integer")
    try:
        frame_index = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an integer") from exc
    if frame_index < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return frame_index


def _ranked_rows(
    candidates: Sequence[Any],
    top_k: int,
) -> list[tuple[str, int, Mapping[str, Any]]]:
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


def _trake_hypotheses(
    response: Mapping[str, Any] | Sequence[Any],
) -> tuple[Sequence[Any], int | None]:
    """Extract hypotheses and, when present, the parser-declared event count."""

    if isinstance(response, Mapping):
        hypotheses = response.get("hypotheses")
        if not isinstance(hypotheses, Sequence) or isinstance(hypotheses, (str, bytes)):
            raise SubmissionExportError("TRAKE retrieval returned no ranked hypotheses")
        event_count: int | None = None
        event_plan = response.get("event_plan")
        if event_plan is not None:
            plan = _mapping(event_plan)
            events = plan.get("events")
            if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
                event_count = len(events)
                if event_count < 1:
                    raise SubmissionExportError("TRAKE event plan must contain at least one event")
        return hypotheses, event_count
    if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
        return response, None
    raise TypeError("TRAKE response must be a mapping or a sequence of hypotheses")


def _declared_trake_event_count(hypotheses: Sequence[Any]) -> int | None:
    """Infer only the row width, never frame identity, from the first sequence."""

    for raw_hypothesis in hypotheses:
        try:
            hypothesis = _mapping(raw_hypothesis)
        except TypeError:
            continue
        raw_frame_ids = hypothesis.get("frame_ids")
        if isinstance(raw_frame_ids, Sequence) and not isinstance(
            raw_frame_ids,
            (str, bytes),
        ):
            return len(raw_frame_ids)
    return None


def _validated_trake_sequence(
    raw_hypothesis: Any,
    *,
    event_count: int,
) -> tuple[str, tuple[int, ...]]:
    """Validate the explicit original-frame lineage for one sequence.

    ``frame_ids`` is the public submission-shaped alias, but it is not trusted
    on its own.  Each value must be corroborated by the corresponding lineage
    item produced from ``RetrievalResult.frame_index``.  We intentionally do
    not fall back to internal frame IDs, timestamps, filenames, or row numbers.
    """

    hypothesis = _mapping(raw_hypothesis)
    video_id = _video_id(hypothesis)
    raw_frame_ids = hypothesis.get("frame_ids")
    if not isinstance(raw_frame_ids, Sequence) or isinstance(raw_frame_ids, (str, bytes)):
        raise ValueError("TRAKE hypothesis has no frame_ids sequence")
    if len(raw_frame_ids) != event_count:
        raise ValueError("TRAKE hypothesis event count does not match the event plan")
    frame_ids = tuple(
        _frame_index_value(value, name=f"frame_ids[{position}]")
        for position, value in enumerate(raw_frame_ids)
    )

    raw_lineage = hypothesis.get("lineage")
    if not isinstance(raw_lineage, Sequence) or isinstance(raw_lineage, (str, bytes)):
        raise ValueError("TRAKE hypothesis has no per-event original-frame lineage")
    if len(raw_lineage) != event_count:
        raise ValueError("TRAKE hypothesis lineage count does not match frame_ids")
    for position, raw_entry in enumerate(raw_lineage):
        entry = _mapping(raw_entry)
        raw_event_index = entry.get("event_index")
        if (
            isinstance(raw_event_index, bool)
            or not isinstance(raw_event_index, Integral)
            or int(raw_event_index) != position
        ):
            raise ValueError("TRAKE lineage event_index values must be ordered and contiguous")
        if _video_id(entry) != video_id:
            raise ValueError("TRAKE lineage contains a different video_id")
        lineage_frame = _frame_index_value(
            entry.get("original_frame_index"),
            name=f"lineage[{position}].original_frame_index",
        )
        if lineage_frame != frame_ids[position]:
            raise ValueError("TRAKE frame_ids do not match original-frame lineage")
        source = entry.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("TRAKE original-frame lineage has no source")
    return video_id, frame_ids


def serialize_trake_csv(
    response: Mapping[str, Any] | Sequence[Any],
    *,
    top_k: int = 100,
) -> str:
    """Serialize ranked TRAKE sequences using original zero-based frame indices.

    The provisional columns are ``video_id,frame_id_1,...,frame_id_N``.  Invalid
    hypotheses are omitted fail-closed.  Deduplication uses the complete
    ``(video_id, frame_ids)`` sequence, so two distinct event chains may reuse
    individual frames without consuming each other's submission slot.
    """

    request = ExportRequest.parse("serialization", "trake", top_k)
    hypotheses, planned_event_count = _trake_hypotheses(response)
    event_count = planned_event_count
    if event_count is None:
        event_count = _declared_trake_event_count(hypotheses)
    if event_count is None or event_count < 1:
        raise SubmissionExportError("cannot determine the TRAKE event count")

    rows: list[tuple[Any, ...]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for raw_hypothesis in hypotheses:
        try:
            video_id, frame_ids = _validated_trake_sequence(
                raw_hypothesis,
                event_count=event_count,
            )
        except (TypeError, ValueError):
            continue
        identity = (video_id, frame_ids)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append((video_id, *frame_ids))
        if len(rows) >= request.top_k:
            break

    header = (
        TRAKE_VIDEO_COLUMN,
        *(f"{TRAKE_FRAME_COLUMN_PREFIX}{index}" for index in range(1, event_count + 1)),
    )
    return _write_csv(header, rows)


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
    elif request.task is SubmissionTask.QA:
        content = serialize_qa_csv(response, top_k=request.top_k)
        row_count = max(0, content.count("\r\n") - 1)
    else:
        content = serialize_trake_csv(response, top_k=request.top_k)
        row_count = max(0, content.count("\r\n") - 1)
    return ExportedCsv(
        content=content,
        filename=f"{request.task.value}_result.csv",
        row_count=row_count,
        task=request.task,
    )
