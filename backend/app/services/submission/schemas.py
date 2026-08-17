"""Small, framework-independent schemas for AIC CSV export."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral


class SubmissionTask(str, Enum):
    KIS = "kis"
    QA = "qa"
    TRAKE = "trake"


@dataclass(frozen=True)
class ExportRequest:
    query: str
    task: SubmissionTask
    top_k: int = 100

    @classmethod
    def parse(cls, query: str, task: str, top_k: int = 100) -> "ExportRequest":
        cleaned_query = " ".join(str(query).split()).strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")
        normalized_task = str(task).casefold().strip()
        try:
            parsed_task = SubmissionTask(normalized_task)
        except ValueError as exc:
            raise ValueError("task must be 'kis', 'qa', or 'trake'") from exc
        if isinstance(top_k, bool) or not isinstance(top_k, Integral):
            raise ValueError("top_k must be an integer between 1 and 100")
        bounded_top_k = int(top_k)
        if not 1 <= bounded_top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        return cls(query=cleaned_query, task=parsed_task, top_k=bounded_top_k)


@dataclass(frozen=True)
class ExportedCsv:
    content: str
    filename: str
    row_count: int
    task: SubmissionTask
