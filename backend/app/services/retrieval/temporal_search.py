"""Phase 3 temporal helpers for multi-event video search."""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.models.retrieval import RetrievalResult


_THEN_RE = re.compile(r"\b(?:then|after that|next|followed by)\b", re.IGNORECASE)


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
    """Split a temporal query into ordered sub-events with simple rules."""
    parts = [part.strip(" ,.;") for part in _THEN_RE.split(query) if part.strip(" ,.;")]
    if len(parts) == 1 and " after " in query.lower():
        left, right = re.split(r"\bafter\b", query, maxsplit=1, flags=re.IGNORECASE)
        parts = [right.strip(" ,.;"), left.strip(" ,.;")]
    return [EventQuery(text=part, order=index) for index, part in enumerate(parts)]


def match_ordered_events(
    event_results: list[list[RetrievalResult]],
    max_gap_seconds: float = 180.0,
    top_k: int = 20,
) -> list[TemporalMatch]:
    """Join per-event candidates into same-video ordered temporal matches."""
    if not event_results:
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
            for result in event_results[0][:top_k]
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
            matches.append(_to_match(chain, max_gap_seconds=max_gap_seconds))

    matches.sort(key=lambda match: match.score, reverse=True)
    return matches[: max(0, int(top_k))]


def _to_match(events: list[RetrievalResult], max_gap_seconds: float) -> TemporalMatch:
    duration = max(0.0, events[-1].timestamp - events[0].timestamp)
    avg_score = sum(event.score for event in events) / len(events)
    gap_penalty = min(0.20, duration / max(max_gap_seconds, 1.0) * 0.20)
    score = round(max(0.0, avg_score - gap_penalty), 6)
    return TemporalMatch(
        video_id=events[0].video_id,
        events=events,
        score=score,
        start_time=events[0].timestamp,
        end_time=events[-1].timestamp,
    )
