"""Phase 3 temporal helpers for ordered multi-event video retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.query_terms import weighted_query_coverage


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
    event_queries: list[str] | None = None,
) -> list[TemporalMatch]:
    """Join ordered events with strict matching and best-effort fallback."""
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be non-negative")
    if not event_results or any(not results for results in event_results):
        return []
    if event_queries is not None and len(event_queries) != len(event_results):
        raise ValueError("event_queries must align with event_results")
    if len(event_results) == 1:
        return [
            TemporalMatch(
                video_id=result.video_id,
                events=[result],
                score=round(
                    _event_quality(
                        event_queries[0] if event_queries else "",
                        result,
                    ),
                    6,
                ),
                start_time=result.timestamp,
                end_time=result.timestamp,
            )
            for result in event_results[0][: max(0, int(top_k))]
        ]

    # First pass: preserve temporal meaning with a strict positive gap and
    # distinct event locations inside the configured time window.
    matches = _build_matches(
        event_results,
        join_gap_seconds=max_gap_seconds,
        scoring_gap_seconds=max_gap_seconds,
        top_k=top_k,
        event_queries=event_queries,
        allow_equal_timestamp=False,
        allow_same_location=False,
    )
    if matches:
        return matches

    # Best-effort pass: do not return an empty response solely because the
    # configured window is too narrow. The normal gap penalty still lowers
    # long-distance chains.
    matches = _build_matches(
        event_results,
        join_gap_seconds=None,
        scoring_gap_seconds=max_gap_seconds,
        top_k=top_k,
        event_queries=event_queries,
        allow_equal_timestamp=False,
        allow_same_location=False,
    )
    if matches:
        return matches

    # Final compatibility fallback for very sparse indexes. This preserves the
    # previous non-decreasing timestamp behavior while scoring it lower.
    return _build_matches(
        event_results,
        join_gap_seconds=None,
        scoring_gap_seconds=max_gap_seconds,
        top_k=top_k,
        event_queries=event_queries,
        allow_equal_timestamp=True,
        allow_same_location=True,
    )


def _build_matches(
    event_results: list[list[RetrievalResult]],
    *,
    join_gap_seconds: float | None,
    scoring_gap_seconds: float,
    top_k: int,
    event_queries: list[str] | None,
    allow_equal_timestamp: bool,
    allow_same_location: bool,
) -> list[TemporalMatch]:
    matches: list[TemporalMatch] = []
    beam_width = max(200, max(1, int(top_k)) * 50)
    for first in event_results[0]:
        chains = [[first]]
        for event_index, candidates in enumerate(event_results[1:], start=1):
            next_chains: list[list[RetrievalResult]] = []
            for chain in chains:
                previous = chain[-1]
                for candidate in candidates:
                    if candidate.video_id != previous.video_id:
                        continue
                    gap = candidate.timestamp - previous.timestamp
                    if allow_equal_timestamp:
                        if gap < 0.0:
                            continue
                    elif gap <= 0.0:
                        continue
                    if join_gap_seconds is not None and gap > join_gap_seconds:
                        continue
                    if not allow_same_location and any(
                        _same_event_location(existing, candidate)
                        for existing in chain
                    ):
                        continue
                    next_chains.append([*chain, candidate])
            if len(next_chains) > beam_width:
                next_chains.sort(
                    key=lambda chain: _partial_chain_rank(
                        chain,
                        event_queries[: event_index + 1]
                        if event_queries
                        else None,
                    ),
                    reverse=True,
                )
                next_chains = next_chains[:beam_width]
            chains = next_chains
            if not chains:
                break
        for chain in chains:
            if len(chain) == len(event_results):
                matches.append(
                    _to_match(
                        chain,
                        max_gap_seconds=scoring_gap_seconds,
                        event_queries=event_queries,
                    )
                )

    matches.sort(
        key=lambda match: (match.score, -match.start_time),
        reverse=True,
    )
    return matches[: max(0, int(top_k))]


def _to_match(
    events: list[RetrievalResult],
    max_gap_seconds: float,
    event_queries: list[str] | None = None,
) -> TemporalMatch:
    duration = max(0.0, events[-1].timestamp - events[0].timestamp)
    qualities = [
        _event_quality(
            event_queries[index] if event_queries else "",
            event,
        )
        for index, event in enumerate(events)
    ]
    avg_score = sum(qualities) / len(qualities)
    # A temporal chain is only as trustworthy as its weakest event. Blending
    # the bottleneck with the average prevents one excellent event from hiding
    # a semantically unrelated partner.
    sequence_score = 0.65 * min(qualities) + 0.35 * avg_score
    gap_penalty = min(
        0.20,
        duration / max(max_gap_seconds, 1.0) * 0.20,
    )
    return TemporalMatch(
        video_id=events[0].video_id,
        events=events,
        score=round(max(0.0, sequence_score - gap_penalty), 6),
        start_time=events[0].timestamp,
        end_time=events[-1].timestamp,
    )


def _event_quality(query: str, candidate: RetrievalResult) -> float:
    """Softly penalize incomplete event semantics without dropping results."""
    base_score = max(0.0, min(1.0, float(candidate.score)))
    if not query:
        return base_score

    metadata_texts = [
        candidate.caption,
        candidate.ocr_text,
        candidate.asr_text,
        " ".join(candidate.objects),
    ]
    populated = [text for text in metadata_texts if str(text).strip()]
    if not populated:
        # Visual-only candidates remain eligible so hybrid/temporal keeps a
        # best-effort result even when text metadata is absent.
        return 0.85 * base_score
    coverage = max(
        weighted_query_coverage(query, text)
        for text in populated
    )
    return 0.55 * base_score + 0.45 * coverage


def _same_event_location(
    left: RetrievalResult,
    right: RetrievalResult,
) -> bool:
    if left.video_id != right.video_id:
        return False
    if left.frame_id and right.frame_id:
        return left.frame_id == right.frame_id
    left_segment = left.segment_id or left.shot_id
    right_segment = right.segment_id or right.shot_id
    return bool(left_segment and left_segment == right_segment)


def _partial_chain_rank(
    chain: list[RetrievalResult],
    event_queries: list[str] | None,
) -> tuple[float, float, str]:
    qualities = [
        _event_quality(
            event_queries[index] if event_queries else "",
            event,
        )
        for index, event in enumerate(chain)
    ]
    duration = max(0.0, chain[-1].timestamp - chain[0].timestamp)
    identity = "|".join(
        event.frame_id or event.segment_id or event.shot_id
        for event in chain
    )
    return (sum(qualities) / len(qualities), -duration, identity)
