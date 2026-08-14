"""Deterministic, dependency-light query planning."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from backend.app.services.agent.query_expansion import (
    QueryExpansionConfig,
    QueryExpansionPlan,
    QueryExpansionProvider,
    build_query_expansion_plan,
)


SUPPORTED_PROFILES = ("auto", "kis", "avs", "qa", "temporal")
_TEMPORAL_SPLIT = re.compile(
    r"\b(?:then|after\s+that|next(?!\s+to\b)|followed\s+by|sau\s+đó|tiếp\s+theo|rồi)\b",
    re.IGNORECASE,
)
_BEFORE = re.compile(r"\b(?:before|trước\s+khi)\b", re.IGNORECASE)
_AFTER = re.compile(r"\b(?:after|sau\s+khi)\b", re.IGNORECASE)
_QUOTED = re.compile(r'''["'“”‘’]([^"'“”‘’]+)["'“”‘’]''')
_OCR = re.compile(
    r"\b(?:text|written|read|subtitle|sign|signboard|plate|menu|logo|qr)\b"
    r"|\b(?:chữ|văn\s+bản|ghi\s+gì|viết\s+gì|phụ\s+đề|biển\s+hiệu|biển\s+báo|biển\s+số|thực\s+đơn|mã\s+qr)\b",
    re.IGNORECASE,
)
_OBJECT = re.compile(
    r"\b(?:holding|hold|wearing|object|person|people|car|vehicle)\b"
    r"|\b(?:cầm|mặc|vật|đồ\s+vật|người|xe)\b",
    re.IGNORECASE,
)
_AVS = re.compile(
    r"\b(?:all|every|many|different)\s+(?:scenes?|shots?|moments?|videos?)\b"
    r"|\b(?:tất\s+cả|mọi|nhiều)\s+(?:các\s+)?(?:cảnh|đoạn|khoảnh\s+khắc|video)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"^\s*(?:what|when|where|who|why|how|which)\b"
    r"|\b(?:gì|khi\s+nào|ở\s+đâu|là\s+ai|bao\s+nhiêu|màu\s+gì|làm\s+gì)\b",
    re.IGNORECASE,
)
_COMMON_TYPOS = {
    "resturant": "restaurant",
    "restaraunt": "restaurant",
    "motobike": "motorbike",
    "motorcyle": "motorcycle",
    "bulding": "building",
    "signbord": "signboard",
    "subtile": "subtitle",
    "licenceplate": "license plate",
}
_EXPANSIONS = {
    "store": ("shop", "retail"),
    "shop": ("store", "retail"),
    "motorbike": ("motorcycle", "scooter"),
    "motorcycle": ("motorbike", "scooter"),
    "car": ("vehicle", "automobile"),
    "sign": ("signboard", "text"),
    "biển hiệu": ("signboard", "store sign"),
    "cửa hàng": ("store", "shop"),
}


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    normalized_query: str
    retrieval_query: str
    requested_profile: str
    profile: str
    profile_source: str
    temporal_relation: str
    temporal_events: tuple[str, ...]
    modality_hints: tuple[str, ...]
    quoted_phrases: tuple[str, ...]
    expansions: tuple[str, ...]
    modality_queries: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]
    expansion_plan: QueryExpansionPlan

    def query_for(self, modality: str) -> str:
        return dict(self.modality_queries).get(modality, self.retrieval_query)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "retrieval_query": self.retrieval_query,
            "requested_profile": self.requested_profile,
            "profile": self.profile,
            "profile_source": self.profile_source,
            "temporal_relation": self.temporal_relation,
            "temporal_events": list(self.temporal_events),
            "modality_hints": list(self.modality_hints),
            "quoted_phrases": list(self.quoted_phrases),
            "expansions": list(self.expansions),
            "modality_queries": dict(self.modality_queries),
            "reasons": list(self.reasons),
            "query_expansion": self.expansion_plan.to_dict(),
        }


def build_query_plan(
    query: str,
    profile: str = "auto",
    *,
    expansion_provider: QueryExpansionProvider | None = None,
    expansion_config: QueryExpansionConfig | None = None,
    expansion_plan: QueryExpansionPlan | None = None,
) -> QueryPlan:
    original = " ".join(str(query).split())
    if not original or re.search(r"\w", original, re.UNICODE) is None:
        raise ValueError("query must not be empty")
    requested = str(profile or "auto").strip().casefold()
    if requested not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported retrieval profile {profile!r}; expected {SUPPORTED_PROFILES}"
        )

    normalized = _normalize_typos(original)
    quoted = tuple(value.strip() for value in _QUOTED.findall(normalized) if value.strip())
    events, relation = _temporal_events(normalized)
    classification = _QUOTED.sub(" ", normalized)
    hints: list[str] = []
    reasons: list[str] = []
    if _OCR.search(classification):
        hints.append("ocr")
        reasons.append("OCR/text cue")
    if _OBJECT.search(classification):
        hints.append("objects")
        reasons.append("object cue")

    if requested != "auto":
        resolved, source = requested, "explicit"
        reasons.append(f"explicit profile: {resolved}")
    elif len(events) > 1:
        resolved, source = "temporal", "inferred"
        reasons.append("ordered temporal event chain")
    elif _QUESTION.search(classification):
        resolved, source = "qa", "inferred"
        reasons.append("question/evidence query")
    elif _AVS.search(classification):
        resolved, source = "avs", "inferred"
        reasons.append("broad multi-result query")
    else:
        resolved, source = "kis", "default"
        reasons.append("default exact-instance search")

    # Temporal chaining is a TRAKE/profile-temporal concern. An explicit TKIS,
    # AVS, or QA profile must not silently execute temporal event retrieval just
    # because its natural-language query contains words such as "then".
    if resolved != "temporal":
        events = (normalized,)
        relation = "none"

    resolved_expansion = expansion_plan or build_query_expansion_plan(
        original,
        provider=expansion_provider,
        config=expansion_config,
    )
    accepted_paraphrases = tuple(
        value.text
        for value in resolved_expansion.accepted_variants
        if value.type == "paraphrase"
    )
    retrieval_query = normalized
    decomposition = resolved_expansion.decomposition
    object_query = " ".join(decomposition.objects)
    ocr_query = " ".join(decomposition.ocr_literals)
    if decomposition.ocr_literals and "ocr" not in hints:
        hints.append("ocr")
        reasons.append("protected OCR literal")
    if decomposition.objects and "objects" not in hints:
        hints.append("objects")
        reasons.append("decomposed object cue")
    modality_queries = (
        ("visual", retrieval_query),
        ("caption", retrieval_query),
        ("objects", object_query),
        ("ocr", ocr_query),
    )
    return QueryPlan(
        original_query=original,
        normalized_query=normalized,
        retrieval_query=retrieval_query,
        requested_profile=requested,
        profile=resolved,
        profile_source=source,
        temporal_relation=relation,
        temporal_events=events,
        modality_hints=tuple(dict.fromkeys(hints)),
        quoted_phrases=quoted,
        expansions=accepted_paraphrases,
        modality_queries=modality_queries,
        reasons=tuple(reasons),
        expansion_plan=resolved_expansion,
    )


def _normalize_typos(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query)
    for typo, replacement in _COMMON_TYPOS.items():
        normalized = re.sub(
            rf"(?<!\w){re.escape(typo)}(?!\w)",
            replacement,
            normalized,
            flags=re.IGNORECASE,
        )
    return " ".join(normalized.split())


def _expand(query: str) -> list[str]:
    additions: list[str] = []
    for phrase, values in _EXPANSIONS.items():
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", query, re.IGNORECASE):
            additions.extend(values)
    return list(dict.fromkeys(additions))


def _temporal_events(query: str) -> tuple[tuple[str, ...], str]:
    parts = [part.strip(" ,.;") for part in _TEMPORAL_SPLIT.split(query)]
    parts = [part for part in parts if part]
    if len(parts) > 1:
        return tuple(parts), "then"
    before = _BEFORE.search(query)
    if before:
        left = query[: before.start()].strip(" ,.;")
        right = query[before.end() :].strip(" ,.;")
        if left and right:
            return (left, right), "before"
    after = _AFTER.search(query)
    if after:
        left = query[: after.start()].strip(" ,.;")
        right = query[after.end() :].strip(" ,.;")
        if left and right:
            return (right, left), "after"
    return (query.strip(" ,.;"),), "none"
