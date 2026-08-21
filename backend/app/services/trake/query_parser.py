"""Conservative deterministic parser for ordered TRAKE event queries.

Query text is untrusted data.  This module never executes instructions found in
that text and does not invoke an LLM.  Explicit list boundaries take precedence
over linguistic splitting, which prevents a connective inside one numbered
criterion from silently changing the requested event count.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from backend.app.services.trake.models import (
    BoundaryType,
    TemporalEvent,
    TemporalEventPlan,
)


_SEQUENTIAL = re.compile(
    r"(?<!\w)(?:and\s+then|then|after\s+that|next(?!\s+to\b)|"
    r"followed\s+by|và\s+sau\s+đó|sau\s+đó|tiếp\s+theo|rồi)(?!\w)",
    re.IGNORECASE,
)
_BEFORE_OR_AFTER = re.compile(
    r"(?<!\w)(?P<relation>before|after|trước\s+khi|sau\s+khi)(?!\w)",
    re.IGNORECASE,
)
_LINE_LIST_MARKER = re.compile(
    r"^[ \t]*(?P<marker>"
    # Official TRAKE prompts use E1:, E2:, ... .  The trailing whitespace is
    # optional for that form so pasted prompts such as ``E1:event`` also work.
    r"e\s*\d+\s*[:.)-][ \t]*"
    r"|(?:events?|sự\s+kiện)\s+\d+\s*[:.)-][ \t]+"
    r"|\(\d+\)[.:]?[ \t]+"
    r"|\d+[:.)-][ \t]+"
    r"|[-*•][ \t]+"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_INLINE_NUMBERED_MARKER = re.compile(
    r"(?<!\S)(?P<marker>"
    r"e\s*\d+\s*[:.)-][ \t]*"
    r"|(?:events?|sự\s+kiện)\s+\d+\s*[:.)-][ \t]+"
    r"|\(\d+\)[.:]?[ \t]+"
    r"|\d+[:.)-][ \t]+"
    r")",
    re.IGNORECASE,
)
_NUMBERED_MARKER_VALUE = re.compile(
    r"^(?:e|events?|sự\s+kiện)?\s*\(?(?P<number>\d+)\)?",
    re.IGNORECASE,
)
_EVENT_SECTION = re.compile(
    r"(?:^|[\r\n;.!?])[ \t]*"
    r"(?:ordered\s+events?|events?|event\s+sequence|"
    r"chuỗi\s+sự\s+kiện|các\s+sự\s+kiện|sự\s+kiện)"
    r"[ \t]*:[ \t]*",
    re.IGNORECASE,
)
_CONTEXT_LABEL = re.compile(
    r"^\s*(?:context|scenario|setting|bối\s+cảnh|ngữ\s+cảnh)\s*:\s*",
    re.IGNORECASE,
)
_TRAILING_EVENT_HEADER = re.compile(
    r"(?:^|[\r\n;,.!?])\s*(?:ordered\s+events?|events?|event\s+sequence|"
    r"chuỗi\s+sự\s+kiện|các\s+sự\s+kiện|sự\s+kiện|"
    r"(?:please\s+)?find\s+(?:the\s+)?following\s+(?:ordered\s+)?events?|"
    r"(?:hãy\s+)?tìm\s+(?:các\s+)?sự\s+kiện\s+sau(?:\s+đây)?)\s*:?\s*$",
    re.IGNORECASE,
)
_GENERIC_EVENT_PREAMBLE = re.compile(
    r"^\s*(?:please\s+)?(?:find|show|retrieve|return|parse)\s+"
    r"(?:the\s+)?(?:following\s+)?(?:ordered\s+)?events?\s*:?\s*$"
    r"|^\s*(?:hãy\s+)?(?:tìm|hiển\s+thị|trả\s+về|phân\s+tích)\s+"
    r"(?:các\s+)?(?:sự\s+kiện|chuỗi\s+sự\s+kiện)\s*:?\s*$",
    re.IGNORECASE,
)
_FIRST_EVENT_CUE = re.compile(
    r"^\s*(?:first(?:ly)?|đầu\s+tiên|ban\s+đầu)(?!\w)",
    re.IGNORECASE,
)
_INSTRUCTION_LIKE = re.compile(
    r"\b(?:ignore|disregard|forget)\b.{0,40}\b(?:instruction|prompt|rule)s?\b"
    r"|\b(?:bỏ\s+qua|quên)\b.{0,40}\b(?:chỉ\s+dẫn|hướng\s+dẫn|prompt|quy\s+tắc)\b"
    r"|\b(?:system|developer)\s+(?:message|prompt)\b"
    r"|\b(?:merge|drop|delete|renumber)\s+(?:the\s+)?events?\b"
    r"|\b(?:gộp|xóa|bỏ)\s+(?:các\s+)?sự\s+kiện\b",
    re.IGNORECASE | re.DOTALL,
)


# Longest alternatives are listed first so overlapping terms are emitted once
# and in their original textual order/casing.
_PROTECTED_TERM = re.compile(
    r"(?<!\w)(?:"
    r"rời\s+hoàn\s+toàn|hoàn\s+toàn\s+rời|"
    r"fully\s+(?:leaves?|left|exits?|outside)|"
    r"completely\s+(?:leaves?|left|exits?)|(?:leaves?|exits?)\s+completely|"
    r"lần\s+đầu\s+tiên|đầu\s+tiên|lần\s+đầu|bắt\s+đầu|"
    r"cao\s+nhất|cực\s+đại|đạt\s+đỉnh|"
    r"first|begins?|starts?|maximum|peak|highest|"
    r"tiếp\s+xúc|chạm|touch(?:es|ed|ing)?|(?:makes?\s+)?contact"
    r")(?!\w)",
    re.IGNORECASE,
)
_ADDITIONAL_PROTECTED_TERM = re.compile(
    r"(?<!\w)(?:sau\s+đó|khởi\s+đầu|chuyển\s+sang|chuyển\s+động|"
    r"cử\s+động\s+đầu|cử\s+động|xoay\s+vòng|quay\s+vòng|cúi\s+chào|"
    r"tiếp\s+đất|rời\s+cột|nhấc\s+chân|hạ\s+chân|đứng\s+dậy|"
    r"ngồi\s+xuống|tiến\s+lại|chạm\s+đất|hoàn\s+toàn|"
    r"cột\s+số\s+\d+|\d+\s+chân(?:\s+trước)?|con\s+rồng)(?!\w)",
    re.IGNORECASE,
)
_FULL_LEAVE = re.compile(
    r"(?<!\w)(?:rời\s+hoàn\s+toàn|hoàn\s+toàn\s+rời|"
    r"fully\s+(?:leaves?|left|exits?|outside)|"
    r"completely\s+(?:leaves?|left|exits?)|(?:leaves?|exits?)\s+completely)(?!\w)",
    re.IGNORECASE,
)
_PEAK = re.compile(
    r"(?<!\w)(?:cao\s+nhất|cực\s+đại|đạt\s+đỉnh|maximum|peak|highest)(?!\w)",
    re.IGNORECASE,
)
_CONTACT = re.compile(
    r"(?<!\w)(?:tiếp\s+xúc|chạm|touch(?:es|ed|ing)?|(?:makes?\s+)?contact)(?!\w)",
    re.IGNORECASE,
)
_FIRST_OR_START = re.compile(
    r"(?<!\w)(?:lần\s+đầu\s+tiên|đầu\s+tiên|lần\s+đầu|bắt\s+đầu|"
    r"first|begins?|starts?)(?!\w)",
    re.IGNORECASE,
)
_STRONG_START = re.compile(
    r"(?<!\w)(?:bắt\s+đầu|khởi\s+đầu|begins?|starts?)(?!\w)",
    re.IGNORECASE,
)
_TRANSITION_ACTION = re.compile(
    r"(?<!\w)(?:bắt\s+đầu|khởi\s+đầu|chuyển\s+sang|chuyển\s+động|"
    r"cử\s+động(?:\s+đầu)?|xoay(?:\s+vòng)?|quay(?:\s+vòng)?|"
    r"cúi(?:\s+chào)?|chào|tiến\s+lại|tiếp\s+đất|rời\s+cột|"
    r"nhấc\s+chân|hạ\s+chân|đứng\s+dậy|ngồi\s+xuống|"
    r"vào|rời(?:\s+khỏi)?|xuất\s+hiện|biến\s+mất|chuyển|đổi|mở|đóng|"
    r"enter(?:s|ed|ing)?|leave(?:s|d|ing)?|left|exit(?:s|ed|ing)?|"
    r"appear(?:s|ed|ing)?|disappear(?:s|ed|ing)?|become(?:s|ing)?|"
    r"change(?:s|d|ing)?|open(?:s|ed|ing)?|close(?:s|d|ing)?)(?!\w)",
    re.IGNORECASE,
)
_ONGOING_STATE = re.compile(
    r"(?<!\w)(?:đang|vẫn|remains?|stays?|still)(?!\w)"
    r"|(?<!\w)(?:is|are|was|were)\s+\w+ing(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ParsedText:
    context: str
    events: tuple[str, ...]
    source: str
    confidence: float
    warnings: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()


class TrakeQueryParser:
    """Parse a query without changing its explicit event cardinality/order."""

    def parse(self, query: str) -> TemporalEventPlan:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        original_query = query
        normalized_query = _normalize_for_parsing(query)
        if not normalized_query or re.search(r"\w", normalized_query, re.UNICODE) is None:
            raise ValueError("query must not be empty")

        parsed = self._parse_text(normalized_query)
        warnings = list(parsed.warnings)
        if _INSTRUCTION_LIKE.search(original_query):
            # The text is retained as event/context data; the apparent command
            # never controls parsing behavior.
            warnings.append("possible_instruction_text_treated_as_data")
        if len({event.casefold() for event in parsed.events}) != len(parsed.events):
            warnings.append("duplicate_event_text_preserved")

        events_list: list[TemporalEvent] = []
        for index, text in enumerate(parsed.events):
            normalized_text, normalization_warnings = _normalize_event_text(text)
            semantic = _analyze_event(normalized_text, parsed.context)
            event_warnings = list(normalization_warnings)
            if not text.strip():
                event_warnings.append("empty_event_marker")
            if semantic["boundary_type"] == BoundaryType.UNKNOWN:
                event_warnings.append("boundary_unknown")
            confidence = _semantic_confidence(
                text,
                semantic["boundary_type"],
                event_warnings,
                bool(semantic["target_text"]),
            )
            trace = {
                "matched_cues": semantic["matched_cues"],
                "matched_actions": semantic["matched_actions"],
                "selected_boundary": semantic["boundary_type"].value,
                "rejected_boundary_alternatives": semantic["rejected_boundaries"],
                "normalization_warnings": list(normalization_warnings),
                "fallback_reason": "insufficient_boundary_evidence"
                if semantic["boundary_type"] == BoundaryType.UNKNOWN else "",
                "semantic_confidence": confidence,
            }
            events_list.append(
                TemporalEvent(
                    index=index,
                    name=f"event_{index + 1}",
                    source_label=parsed.labels[index] if index < len(parsed.labels) else "",
                    original_text=text,
                    normalized_text=normalized_text,
                    event_context=semantic["event_context"],
                    target_text=semantic["target_text"],
                    retrieval_query=_coarse_retrieval_query(
                        parsed.context,
                        semantic["event_context"],
                        semantic["target_text"],
                        normalized_text,
                    ),
                    refinement_query=_refinement_query(
                        parsed.context,
                        semantic["target_text"],
                    ),
                    boundary_type=semantic["boundary_type"],
                    protected_terms=extract_protected_terms(normalized_text),
                    parser_warnings=tuple(dict.fromkeys(event_warnings)),
                    semantic_confidence=confidence,
                    parser_trace=trace,
                )
            )
        events = tuple(events_list)
        semantic_confidence = (
            sum(event.semantic_confidence for event in events) / len(events)
            if events else 0.0
        )
        structural_confidence = parsed.confidence
        overall_confidence = round(
            min(1.0, 0.55 * structural_confidence + 0.45 * semantic_confidence),
            6,
        )
        for event in events:
            warnings.extend(event.parser_warnings)
        return TemporalEventPlan(
            original_query=original_query,
            context=parsed.context,
            events=events,
            parser_source=parsed.source,
            confidence=overall_confidence,
            warnings=tuple(dict.fromkeys(warnings)),
            structural_confidence=structural_confidence,
            semantic_confidence=round(semantic_confidence, 6),
        )

    def _parse_text(self, query: str) -> _ParsedText:
        listed = _parse_explicit_list(query)
        if listed is not None:
            context, events, warnings, labels = listed
            return _ParsedText(
                context=context,
                events=events,
                source="deterministic_list",
                confidence=1.0,
                warnings=warnings,
                labels=labels,
            )

        context, body = _split_explicit_context(query)
        try:
            events, relations = _parse_temporal_expression(body)
            events = tuple(event for event in (_trim_clause(item) for item in events) if event)
        except (RecursionError, re.error):
            events, relations = (), ()

        if len(events) > 1 and relations:
            return _ParsedText(
                context=context,
                events=events,
                source="deterministic_connective",
                confidence=0.96,
            )

        fallback = body.strip() or query
        fallback_context = context if body.strip() else ""
        return _ParsedText(
            context=fallback_context,
            events=(fallback,),
            source="deterministic_fallback",
            confidence=0.70,
            warnings=("single_event_verbatim_fallback",),
        )


# A descriptive alias makes the safety property discoverable without creating
# a second implementation or behavior branch.
ConservativeQueryParser = TrakeQueryParser


def parse_trake_query(query: str) -> TemporalEventPlan:
    """Convenience entrypoint for the deterministic TRAKE parser."""

    return TrakeQueryParser().parse(query)


def parse_temporal_query(query: str) -> TemporalEventPlan:
    """Compatibility alias for callers using the generic temporal name."""

    return parse_trake_query(query)


def extract_protected_terms(text: str) -> tuple[str, ...]:
    """Return boundary-critical terms verbatim and in textual order."""

    terms: list[str] = []
    seen: set[str] = set()
    occupied: list[tuple[int, int]] = []
    matches = [*_PROTECTED_TERM.finditer(text), *_ADDITIONAL_PROTECTED_TERM.finditer(text)]
    # Boundary cues first, then action/constraint terms in source order.  This
    # keeps weak presentation words from hiding stronger later evidence.
    matches.sort(key=lambda match: (0 if _FIRST_OR_START.fullmatch(match.group(0)) else 1, match.start(), -len(match.group(0))))
    for match in matches:
        if any(match.start() >= start and match.end() <= end for start, end in occupied):
            continue
        value = match.group(0)
        folded = value.casefold()
        if folded not in seen:
            seen.add(folded)
            terms.append(value)
            occupied.append((match.start(), match.end()))
    return tuple(terms)


def infer_boundary_type(text: str) -> BoundaryType:
    """Infer only boundary semantics supported by explicit lexical evidence."""

    if _FULL_LEAVE.search(text):
        return BoundaryType.FIRST_LEAVE
    if _PEAK.search(text):
        return BoundaryType.PEAK
    contact = _CONTACT.search(text)
    first_matches = list(_FIRST_OR_START.finditer(text))
    first_or_start = first_matches[0] if first_matches else None
    if contact is not None:
        # An explicit ongoing contact is a state unless the criterion asks for
        # its beginning/first occurrence.  Other contact wording denotes the
        # instantaneous contact keyframe conservatively used by TRAKE.
        prefix = text[: contact.start()]
        ongoing_contact = re.search(
            r"(?<!\w)(?:đang|is|are|was|were)\s*$",
            prefix,
            flags=re.IGNORECASE,
        )
        if (
            (_ONGOING_STATE.search(prefix) or ongoing_contact is not None)
            and first_or_start is None
        ):
            return BoundaryType.STATE
        return BoundaryType.FIRST_CONTACT
    if _STRONG_START.search(text) is not None:
        return BoundaryType.FIRST_TRANSITION
    if first_or_start is not None:
        # "First" may merely enumerate a list.  Require a transition action
        # unless the stronger start/begin cue itself denotes the transition.
        cue = first_or_start.group(0).casefold()
        if cue in {"bắt đầu", "begin", "begins", "start", "starts"}:
            return BoundaryType.FIRST_TRANSITION
        if _TRANSITION_ACTION.search(text):
            return BoundaryType.FIRST_TRANSITION
    if _ONGOING_STATE.search(text):
        return BoundaryType.STATE
    return BoundaryType.UNKNOWN


def _parse_explicit_list(
    text: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    matches = list(_LINE_LIST_MARKER.finditer(text))
    if not matches:
        inline = list(_INLINE_NUMBERED_MARKER.finditer(text))
        if inline and (
            len(inline) > 1
            or inline[0].start() == 0
            or _CONTEXT_LABEL.search(text[: inline[0].start()])
        ):
            matches = inline
    if not matches:
        return None

    prefix = text[: matches[0].start()]
    context = _clean_context(prefix)
    events: list[str] = []
    labels: list[str] = []
    marker_numbers: list[int] = []
    for index, match in enumerate(matches):
        marker_number = _marker_number(match.group("marker"))
        labels.append(match.group("marker").strip().rstrip(":.)-"))
        if marker_number is not None:
            marker_numbers.append(marker_number)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        event = text[match.end() : end].strip()
        # Preserve the marker slot even when its payload is empty.  Downstream
        # support validation will fail closed instead of silently changing N.
        events.append(event)
    warnings: list[str] = []
    if any(not event for event in events):
        warnings.append("empty_event_marker_preserved")
    # Labels in benchmark statements are presentation metadata, not event
    # identity.  Preserve every non-empty event in source order even when the
    # prompt contains a typo such as E1, E2, E2, E4.
    if len(marker_numbers) == len(matches):
        if len(set(marker_numbers)) != len(marker_numbers):
            warnings.append("duplicate_event_label_preserved")
        if marker_numbers != list(range(1, len(marker_numbers) + 1)):
            warnings.append("event_labels_reindexed_by_appearance")
    return context, tuple(events), tuple(warnings), tuple(labels)


def _marker_number(marker: str) -> int | None:
    match = _NUMBERED_MARKER_VALUE.match(marker.strip())
    return int(match.group("number")) if match is not None else None


def _split_explicit_context(text: str) -> tuple[str, str]:
    section = _EVENT_SECTION.search(text)
    if section is not None:
        return _clean_context(text[: section.start()]), text[section.end() :].strip()

    label = _CONTEXT_LABEL.match(text)
    if label is not None:
        payload = text[label.end() :]
        for separator in re.finditer(r"[;\r\n]+|[.!?]\s+", payload):
            context = _clean_context(payload[: separator.start()])
            body = payload[separator.end() :].strip()
            if body and (
                _FIRST_EVENT_CUE.search(body)
                or _SEQUENTIAL.search(body)
                or _BEFORE_OR_AFTER.search(body)
            ):
                return context, body

    # A narrative prefix is context only when a visibly ordered first event
    # follows it and the remaining body contains another connective.
    separator = re.search(
        r"[.!?;]\s+(?=\s*(?:first(?:ly)?|đầu\s+tiên|ban\s+đầu)\b)",
        text,
        flags=re.IGNORECASE,
    )
    if separator is not None:
        body = text[separator.end() :].strip()
        if _FIRST_EVENT_CUE.search(body) and _SEQUENTIAL.search(body):
            return _clean_context(text[: separator.start()]), body
    return "", text


def _clean_context(value: str) -> str:
    context = value.strip()
    context = _TRAILING_EVENT_HEADER.sub("", context).strip()
    context = _CONTEXT_LABEL.sub("", context).strip()
    if _GENERIC_EVENT_PREAMBLE.fullmatch(context):
        return ""
    return " ".join(context.strip(" \t\r\n:;,.!").split())


def _retrieval_query(context: str, event_text: str) -> str:
    if not context:
        return event_text
    separator = " " if context.endswith((".", "!", "?", ":", ";")) else ". "
    return f"{context}{separator}{event_text}"


def _normalize_for_parsing(value: str) -> str:
    return unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n").strip()


_TYPO_ALLOWLIST = ((re.compile(r"(?<!\w)cuối\s+chào(?!\w)", re.IGNORECASE), "cúi chào", "cuối chào -> cúi chào"),)


def _normalize_event_text(value: str) -> tuple[str, tuple[str, ...]]:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    warnings: list[str] = []
    for pattern, replacement, description in _TYPO_ALLOWLIST:
        if pattern.search(normalized):
            normalized = pattern.sub(replacement, normalized)
            warnings.append(f"normalized_probable_typo: {description}")
    return normalized, tuple(warnings)


_TARGET_SENTENCE = re.compile(r"(?i)^\s*(?:khoả?nh|khoảng)\s+khắc\s+(?:đầu\s+tiên\s+)?(?:mà\s+)?")
_LOCATION = re.compile(r"(?i)(trên\s+cột\s+số\s+\d+)")


def _analyze_event(text: str, global_context: str) -> dict[str, Any]:
    sentences = [part.strip(" \t\n.") for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    target_index = next((i for i in range(len(sentences) - 1, -1, -1) if _TARGET_SENTENCE.match(sentences[i])), None)
    if target_index is None:
        event_context = ""
        target = text.strip()
    else:
        event_context = " ".join(sentences[:target_index]).strip()
        target = _TARGET_SENTENCE.sub("", sentences[target_index]).strip()
    event_context = re.sub(r"(?i)^sau\s+đó\s+", "", event_context).strip(" .")
    if event_context:
        event_context = event_context[0].upper() + event_context[1:]
    if target:
        if re.search(r"(?i)cử\s+động", target) and not _STRONG_START.search(target):
            target = re.sub(r"(?i)\bcử\s+động\b", "bắt đầu cử động", target, count=1)
        elif re.search(r"(?i)cúi\s+chào", target) and not _STRONG_START.search(target):
            target = re.sub(r"(?i)\bcúi\s+chào\b", "bắt đầu cúi chào", target, count=1)
        location = _LOCATION.search(event_context)
        if location and location.group(0).casefold() not in target.casefold():
            target = f"{target} {location.group(0)}"
        target = target[0].upper() + target[1:]
    boundary, boundary_trace = _boundary_analysis(text)
    return {
        "event_context": event_context,
        "target_text": target,
        "boundary_type": boundary,
        **boundary_trace,
    }


def _boundary_analysis(text: str) -> tuple[BoundaryType, dict[str, list[str]]]:
    cue_matches = [match.group(0) for match in _FIRST_OR_START.finditer(text)]
    for pattern in (_FULL_LEAVE, _PEAK, _ONGOING_STATE):
        cue_matches.extend(match.group(0) for match in pattern.finditer(text))
    transition_actions = [match.group(0) for match in _TRANSITION_ACTION.finditer(text)]
    actions = list(transition_actions)
    actions.extend(match.group(0) for match in _CONTACT.finditer(text))
    selected = infer_boundary_type(text)
    alternatives: list[str] = []
    if _FULL_LEAVE.search(text) and selected != BoundaryType.FIRST_LEAVE: alternatives.append(BoundaryType.FIRST_LEAVE.value)
    if _PEAK.search(text) and selected != BoundaryType.PEAK: alternatives.append(BoundaryType.PEAK.value)
    if _CONTACT.search(text) and selected != BoundaryType.FIRST_CONTACT: alternatives.append(BoundaryType.FIRST_CONTACT.value)
    if transition_actions and selected != BoundaryType.FIRST_TRANSITION: alternatives.append(BoundaryType.FIRST_TRANSITION.value)
    if _ONGOING_STATE.search(text) and selected != BoundaryType.STATE: alternatives.append(BoundaryType.STATE.value)
    return selected, {
        "matched_cues": list(dict.fromkeys(cue_matches)),
        "matched_actions": list(dict.fromkeys(actions)),
        "rejected_boundaries": list(dict.fromkeys(alternatives)),
    }


def _semantic_confidence(text: str, boundary: BoundaryType, warnings: list[str], has_target: bool) -> float:
    if not text.strip() or not has_target:
        return 0.0
    value = 0.92 if boundary != BoundaryType.UNKNOWN else 0.45
    if warnings:
        value -= 0.05 * len(warnings)
    return round(max(0.0, min(1.0, value)), 6)


def _coarse_retrieval_query(context: str, event_context: str, target: str, fallback: str) -> str:
    # A single target-only criterion remains verbatim for backwards-compatible
    # coarse recall (and retains every contact/count constraint).  Multi-state
    # criteria use the normalized context + isolated target representation.
    event_payload = (
        ". ".join(part for part in (event_context, target) if part)
        if event_context
        else fallback
    )
    return _retrieval_query(context, event_payload)


def _refinement_query(context: str, target: str) -> str:
    return _retrieval_query(context, target) if target else context


def _parse_temporal_expression(value: str) -> tuple[list[str], list[str]]:
    """Return chronological clauses and the relations used to derive them."""

    text = _trim_clause(value)
    if not text:
        return [], []

    leading = _BEFORE_OR_AFTER.match(text)
    if leading is not None:
        separator = re.search(r"[,;]", text[leading.end() :])
        if separator is not None:
            separator_start = leading.end() + separator.start()
            separator_end = leading.end() + separator.end()
            subordinate = text[leading.end() : separator_start]
            main = text[separator_end:]
            subordinate_events, subordinate_relations = _parse_temporal_expression(
                subordinate
            )
            main_events, main_relations = _parse_temporal_expression(main)
            if subordinate_events and main_events:
                relation = _canonical_relation(leading.group("relation"))
                if relation == "before":
                    return (
                        [*main_events, *subordinate_events],
                        [*main_relations, *subordinate_relations, relation],
                    )
                return (
                    [*subordinate_events, *main_events],
                    [*subordinate_relations, *main_relations, relation],
                )

    sequential_parts = _SEQUENTIAL.split(text)
    if len(sequential_parts) > 1:
        events: list[str] = []
        relations: list[str] = []
        for part in sequential_parts:
            child_events, child_relations = _parse_temporal_expression(part)
            events.extend(child_events)
            relations.extend(child_relations)
        if len(events) > 1:
            relations.append("then")
            return events, relations

    comparison = _BEFORE_OR_AFTER.search(text)
    if comparison is not None:
        left_events, left_relations = _parse_temporal_expression(
            text[: comparison.start()]
        )
        right_events, right_relations = _parse_temporal_expression(
            text[comparison.end() :]
        )
        if left_events and right_events:
            relation = _canonical_relation(comparison.group("relation"))
            if relation == "before":
                return (
                    [*left_events, *right_events],
                    [*left_relations, *right_relations, relation],
                )
            return (
                [*right_events, *left_events],
                [*right_relations, *left_relations, relation],
            )
    return [text], []


def _canonical_relation(value: str) -> str:
    folded = value.casefold()
    return "before" if folded == "before" or folded.startswith("trước") else "after"


def _trim_clause(value: str) -> str:
    clause = value.strip(" \t\r\n,;")
    clause = re.sub(r"^\s*(?:and|và)\s+", "", clause, flags=re.IGNORECASE)
    return clause.strip(" \t\r\n,;")


__all__ = [
    "ConservativeQueryParser",
    "TrakeQueryParser",
    "extract_protected_terms",
    "infer_boundary_type",
    "parse_temporal_query",
    "parse_trake_query",
]
