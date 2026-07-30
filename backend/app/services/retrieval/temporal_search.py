"""Phase 3 temporal helpers for ordered multi-event video retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.models.retrieval import RetrievalResult


_THEN_RE = re.compile(
    r"\b(?:then|after that|next|followed by|sau đó|tiếp theo|rồi)\b",
    re.IGNORECASE,
)
_AFTER_RE = re.compile(r"\b(?:after|sau khi)\b", re.IGNORECASE)


@dataclass(frozen=True)
class EventQuery:
    text: str
    order: int


@dataclass(frozen=True)
class TemporalMatch:
    video_id: str
    events: list[RetrievalResult]
    score: float
    start_time: float
    end_time: float

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "score": self.score,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "events": [event.to_dict() for event in self.events],
        }


def decompose_temporal_query(query: str) -> list[EventQuery]:
    """Split English or Vietnamese temporal queries into ordered events."""
    cleaned = query.strip()
    if not cleaned:
        return []
    parts = [
        part.strip(" ,.;")
        for part in _THEN_RE.split(cleaned)
        if part.strip(" ,.;")
    ]
    if len(parts) == 1:
        after_match = _AFTER_RE.search(cleaned)
        if after_match is not None:
            left = cleaned[: after_match.start()].strip(" ,.;")
            right = cleaned[after_match.end() :].strip(" ,.;")
            if left and right:
                parts = [right, left]
    return [
        EventQuery(text=part, order=index)
        for index, part in enumerate(parts)
    ]


def match_ordered_events(
    event_results: list[list[RetrievalResult]],
    max_gap_seconds: float = 180.0,
    top_k: int = 20,
) -> list[TemporalMatch]:
    """Join event candidates from the same video in timestamp order."""
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be non-negative")
    if not event_results or any(not results for results in event_results):
        return []
    if len(event_results) == 1:
        return [
            TemporalMatch(
                video_id=result.video_id,
                events=[result],
                score=result.score,
                start_time=result.timestamp,
                end_time=result.timestamp,
            )
            for result in event_results[0][: max(0, int(top_k))]
        ]

    matches: list[TemporalMatch] = []
    for first in event_results[0]:
        chains = [[first]]
        for candidates in event_results[1:]:
            next_chains: list[list[RetrievalResult]] = []
            for chain in chains:
                previous = chain[-1]
                for candidate in candidates:
                    if candidate.video_id != previous.video_id:
                        continue
                    gap = candidate.timestamp - previous.timestamp
                    if 0.0 <= gap <= max_gap_seconds:
                        next_chains.append([*chain, candidate])
            chains = next_chains
            if not chains:
                break
        for chain in chains:
            if len(chain) == len(event_results):
                matches.append(
                    _to_match(chain, max_gap_seconds=max_gap_seconds)
                )

    matches.sort(
        key=lambda match: (match.score, -match.start_time),
        reverse=True,
    )
    return matches[: max(0, int(top_k))]


def _to_match(
    events: list[RetrievalResult],
    max_gap_seconds: float,
) -> TemporalMatch:
    duration = max(0.0, events[-1].timestamp - events[0].timestamp)
    avg_score = sum(event.score for event in events) / len(events)
    gap_penalty = min(
        0.20,
        duration / max(max_gap_seconds, 1.0) * 0.20,
    )
    return TemporalMatch(
        video_id=events[0].video_id,
        events=events,
        score=round(max(0.0, avg_score - gap_penalty), 6),
        start_time=events[0].timestamp,
        end_time=events[-1].timestamp,
    )
