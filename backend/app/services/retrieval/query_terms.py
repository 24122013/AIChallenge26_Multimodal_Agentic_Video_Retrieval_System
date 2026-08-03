"""Lightweight query normalization and lexical quality helpers.

The project intentionally keeps Retrieval dependency-light.  These helpers
provide enough normalization for common English action queries without adding
an NLP runtime or changing the text-index artifact schema.
"""
from __future__ import annotations

import re
from collections.abc import Iterable


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_STOPWORDS = {
    # English function words.
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "having",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "she",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
    # Temporal connectors are structural, not event content.
    "then",
    "next",
    # Vietnamese function words.
    "bị",
    "các",
    "cho",
    "có",
    "của",
    "đã",
    "đang",
    "đó",
    "được",
    "kia",
    "là",
    "mà",
    "một",
    "này",
    "những",
    "rồi",
    "sẽ",
    "thì",
    "tiếp",
    "và",
    "với",
    "ở",
}

_STEM_OVERRIDES = {
    "children": "child",
    "gone": "go",
    "lay": "lie",
    "left": "leave",
    "lying": "lie",
    "men": "man",
    "people": "person",
    "ran": "run",
    "sat": "sit",
    "stood": "stand",
    "went": "go",
    "women": "woman",
}

_GENERIC_SUBJECT_TERMS = {
    "anh",
    "bà",
    "child",
    "chị",
    "cô",
    "đàn",
    "human",
    "man",
    "người",
    "ông",
    "person",
    "somebody",
    "someone",
    "woman",
}

_DIRECTION_OR_PARTICLE_TERMS = {
    "away",
    "down",
    "off",
    "on",
    "out",
    "ra",
    "up",
    "vào",
    "xuống",
    "lên",
}


def raw_tokens(text: str) -> list[str]:
    """Return lowercase Unicode word tokens without filtering."""
    return _TOKEN_RE.findall(str(text).casefold())


def stem_token(token: str) -> str:
    """Apply a conservative English stemmer while leaving Unicode words intact."""
    value = str(token).casefold().strip()
    if not value:
        return ""
    override = _STEM_OVERRIDES.get(value)
    if override is not None:
        return override
    if not value.isascii() or not value.isalpha():
        return value

    if len(value) > 5 and value.endswith("ies"):
        value = f"{value[:-3]}y"
    elif len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
        if (
            len(value) > 3
            and value[-1] == value[-2]
            and value[-1] not in {"a", "e", "i", "o", "u"}
        ):
            value = value[:-1]
    elif len(value) > 4 and value.endswith("ied"):
        value = f"{value[:-3]}y"
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
        if (
            len(value) > 3
            and value[-1] == value[-2]
            and value[-1] not in {"a", "e", "i", "o", "u"}
        ):
            value = value[:-1]
    elif len(value) > 4 and value.endswith(
        ("sses", "shes", "ches", "xes", "zes", "oes")
    ):
        value = value[:-2]
    elif len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]

    # Porter-style final-e reduction aligns prepare/preparing and ride/riding.
    if len(value) > 4 and value.endswith("e"):
        value = value[:-1]
    return value


def content_tokens(
    text: str,
    *,
    fallback_to_all: bool = False,
) -> list[str]:
    """Return stemmed content terms, optionally retaining all-term queries."""
    tokens = raw_tokens(text)
    all_terms = [stem_token(token) for token in tokens]
    all_terms = [term for term in all_terms if term]
    content = [
        stem_token(token)
        for token in tokens
        if token not in _STOPWORDS and stem_token(token)
    ]
    if content or not fallback_to_all:
        return content
    return all_terms


def term_weight(term: str) -> float:
    """Weight event-defining terms above generic subjects and particles."""
    normalized = stem_token(term)
    if normalized in _GENERIC_SUBJECT_TERMS:
        return 0.35
    if normalized in _DIRECTION_OR_PARTICLE_TERMS:
        return 0.55
    return 1.0


def weighted_term_coverage(
    query_terms: Iterable[str],
    document_terms: Iterable[str],
) -> float:
    """Return query-side weighted coverage in the inclusive range [0, 1]."""
    query: set[str] = set()
    document: set[str] = set()
    for term in query_terms:
        normalized = stem_token(term)
        if normalized:
            query.add(normalized)
    for term in document_terms:
        normalized = stem_token(term)
        if normalized:
            document.add(normalized)
    if not query:
        return 0.0
    denominator = sum(term_weight(term) for term in query)
    if denominator <= 0:
        return 0.0
    matched = sum(term_weight(term) for term in query if term in document)
    return max(0.0, min(1.0, matched / denominator))


def weighted_query_coverage(query: str, document: str) -> float:
    """Measure how much important query content appears in a document."""
    return weighted_term_coverage(
        content_tokens(query, fallback_to_all=True),
        content_tokens(document),
    )


def content_phrase_match(query: str, document: str) -> float:
    """Return 1 when all normalized query terms occur contiguously in order."""
    query_terms = content_tokens(query, fallback_to_all=True)
    document_terms = content_tokens(document)
    if len(query_terms) < 2 or len(document_terms) < len(query_terms):
        return 0.0
    width = len(query_terms)
    for offset in range(len(document_terms) - width + 1):
        if document_terms[offset : offset + width] == query_terms:
            return 1.0
    return 0.0
