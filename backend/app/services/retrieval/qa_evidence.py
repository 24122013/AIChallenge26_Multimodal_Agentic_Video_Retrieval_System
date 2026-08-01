"""Human-in-the-loop QA evidence retrieval.

This module does not generate a textual answer. It rewrites a question into
known visual constraints, retrieves relevant frames through the existing
hybrid engine, and returns images/timestamps for a user to inspect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Protocol

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.candidate_merger import (
    candidate_identity,
    merge_candidates,
)


class HybridSearchEngineLike(Protocol):
    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> VisualSearchResponse:
        ...


@dataclass(frozen=True)
class QaEvidencePlan:
    question: str
    answer_target: str
    retrieval_queries: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer_target": self.answer_target,
            "retrieval_queries": list(self.retrieval_queries),
        }


class QaEvidenceSearchEngine:
    """Retrieve diverse evidence frames without changing the index pipeline."""

    def __init__(
        self,
        hybrid_engine: HybridSearchEngineLike,
        *,
        candidate_multiplier: int = 4,
        consensus_bonus: float = 0.03,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        if consensus_bonus < 0:
            raise ValueError("consensus_bonus must be non-negative")
        self.hybrid_engine = hybrid_engine
        self.candidate_multiplier = candidate_multiplier
        self.consensus_bonus = consensus_bonus

    def search(self, question: str, top_k: int = 10) -> dict:
        bounded_top_k = max(1, min(int(top_k), 200))
        plan = plan_qa_question(question)
        candidate_top_k = min(
            200,
            max(bounded_top_k, bounded_top_k * self.candidate_multiplier),
        )
        candidate_groups = [
            self.hybrid_engine.search(
                query,
                top_k=candidate_top_k,
            ).results
            for query in plan.retrieval_queries
        ]
        results = _fuse_evidence(
            candidate_groups,
            top_k=bounded_top_k,
            consensus_bonus=self.consensus_bonus,
        )
        return {
            **plan.to_dict(),
            "top_k": bounded_top_k,
            "answer_mode": "manual_visual_inspection",
            "evidence_count": len(results),
            "results": [result.to_dict() for result in results],
        }


_VI_HELD_OBJECT_RE = re.compile(
    r"\b(?:đang\s+)?cầm\s+(?:cái\s+|vật\s+)?gì\b",
    re.IGNORECASE,
)
_EN_HELD_OBJECT_RE = re.compile(
    r"^\s*what\s+(?:is|was)\s+(.+?)\s+holding\s*[?.!]*$",
    re.IGNORECASE,
)
_EN_HOLD_OBJECT_RE = re.compile(
    r"^\s*what\s+(?:does|did)\s+(.+?)\s+hold\s*[?.!]*$",
    re.IGNORECASE,
)

_UNKNOWN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "held_object",
        re.compile(
            r"\b(?:cầm|holding|hold)\b.*\b(?:gì|what)\b"
            r"|\bwhat\b.*\b(?:holding|hold)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "color",
        re.compile(r"\b(?:màu\s+gì|what\s+colou?r)\b", re.IGNORECASE),
    ),
    (
        "location",
        re.compile(r"\b(?:ở\s+đâu|where)\b", re.IGNORECASE),
    ),
    (
        "person_identity",
        re.compile(r"\b(?:là\s+ai|who)\b", re.IGNORECASE),
    ),
    (
        "action",
        re.compile(r"\b(?:làm\s+gì|doing\s+what)\b", re.IGNORECASE),
    ),
)

_COLOR_TERMS = {
    "đỏ": "red",
    "xanh lá": "green",
    "xanh dương": "blue",
    "vàng": "yellow",
    "đen": "black",
    "trắng": "white",
    "cam": "orange",
    "hồng": "pink",
    "tím": "purple",
    "nâu": "brown",
}


def plan_qa_question(question: str) -> QaEvidencePlan:
    """Convert an answer-seeking question into evidence retrieval queries."""
    cleaned = " ".join(str(question).strip().split()).strip(" ?.!;,")
    if not cleaned:
        raise ValueError("question must not be empty")

    answer_target = _answer_target(cleaned)
    known_query = _known_constraint_query(cleaned, answer_target)
    queries = [known_query]
    for expansion in _english_expansions(cleaned, answer_target):
        if expansion.casefold() not in {query.casefold() for query in queries}:
            queries.append(expansion)
    return QaEvidencePlan(
        question=cleaned,
        answer_target=answer_target,
        retrieval_queries=tuple(queries),
    )


def _answer_target(question: str) -> str:
    for target, pattern in _UNKNOWN_PATTERNS:
        if pattern.search(question):
            return target
    return "unknown"


def _known_constraint_query(question: str, answer_target: str) -> str:
    if answer_target == "held_object":
        vietnamese = _VI_HELD_OBJECT_RE.sub("cầm một vật", question)
        if vietnamese != question:
            return vietnamese.strip(" ?.!;,")
        for pattern in (_EN_HELD_OBJECT_RE, _EN_HOLD_OBJECT_RE):
            match = pattern.match(question)
            if match is not None:
                return f"{match.group(1).strip()} holding an object"

    replacements = {
        "color": (
            re.compile(r"\b(?:có\s+)?màu\s+gì\b", re.IGNORECASE),
            "",
        ),
        "location": (
            re.compile(r"\b(?:đang\s+)?ở\s+đâu\b", re.IGNORECASE),
            "",
        ),
        "person_identity": (
            re.compile(r"\b(?:đó\s+)?là\s+ai\b", re.IGNORECASE),
            "",
        ),
        "action": (
            re.compile(r"\b(?:đang\s+)?làm\s+gì\b", re.IGNORECASE),
            "",
        ),
    }
    pattern_and_value = replacements.get(answer_target)
    if pattern_and_value is not None:
        pattern, value = pattern_and_value
        rewritten = " ".join(pattern.sub(value, question).split())
        if rewritten:
            return rewritten.strip(" ?.!;,")
    return question.strip(" ?.!;,")


def _english_expansions(
    question: str,
    answer_target: str,
) -> list[str]:
    """Build small visual English expansions for common Vietnamese QA cues."""
    normalized = question.casefold()
    parts: list[str] = []

    if any(value in normalized for value in ("người phụ nữ", "cô gái")):
        parts.append("a woman")
    elif any(value in normalized for value in ("người đàn ông", "chàng trai")):
        parts.append("a man")
    elif "đứa trẻ" in normalized or "trẻ em" in normalized:
        parts.append("a child")
    elif "người" in normalized:
        parts.append("a person")

    color = next(
        (
            english
            for vietnamese, english in _COLOR_TERMS.items()
            if vietnamese in normalized
        ),
        None,
    )
    if color:
        if "áo" in normalized:
            parts.append(f"wearing a {color} shirt")
        else:
            parts.append(f"wearing {color}")

    if "ngồi" in normalized:
        parts.append("sitting")
    elif "đứng" in normalized:
        parts.append("standing")
    elif "đi bộ" in normalized:
        parts.append("walking")
    elif "chạy" in normalized:
        parts.append("running")

    has_table = "bàn" in normalized
    if has_table:
        parts.append("at a table")
    if answer_target == "held_object" or "cầm" in normalized:
        parts.append("holding an object")

    if not parts:
        return []
    primary = " ".join(parts)
    expansions = [primary]
    if has_table and "trên bàn" in normalized:
        expansions.append(primary.replace("at a table", "on a table"))
    return list(dict.fromkeys(expansions))


def _fuse_evidence(
    candidate_groups: list[list[RetrievalResult]],
    *,
    top_k: int,
    consensus_bonus: float,
) -> list[RetrievalResult]:
    occurrences: dict[tuple[str, str], int] = {}
    for group in candidate_groups:
        seen_in_query: set[tuple[str, str]] = set()
        for candidate in group:
            identity = candidate_identity(candidate)
            if identity in seen_in_query:
                continue
            seen_in_query.add(identity)
            occurrences[identity] = occurrences.get(identity, 0) + 1

    merged = merge_candidates(
        candidate_groups,
        dedupe_same_shot=True,
    )
    boosted = [
        replace(
            candidate,
            score=round(
                min(
                    1.0,
                    candidate.score
                    + consensus_bonus
                    * max(0, occurrences.get(candidate_identity(candidate), 1) - 1),
                ),
                6,
            ),
        )
        for candidate in merged
    ]
    boosted.sort(
        key=lambda candidate: (
            candidate.score,
            candidate.timestamp_confidence,
        ),
        reverse=True,
    )
    return boosted[:top_k]
