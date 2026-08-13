"""Deterministic, dependency-light query planning."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


SUPPORTED_PROFILES = ("auto", "kis", "avs", "qa", "temporal")
_TEMPORAL_SPLIT = re.compile(
    r"\b(?:then|after\s+that|next|followed\s+by|sau\s+đó|tiếp\s+theo|rồi)\b",
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
_ANSWER_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ocr",
        re.compile(
            r"\b(?:ghi|viết|đọc|hiển\s+thị)\s+(?:nội\s+dung\s+)?gì\b"
            r"|\b(?:what|which)\b.*\b(?:text|written|say|read|display)\b"
            r"|\bwhat\s+does\s+(?:the\s+)?(?:sign|label|menu|plate)\s+say\b",
            re.IGNORECASE,
        ),
    ),
    (
        "color",
        re.compile(
            r"\b(?:màu\s+gì|what\s+colou?r|which\s+colou?r)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "count",
        re.compile(
            r"\b(?:bao\s+nhiêu|mấy|how\s+many|what\s+number)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "location",
        re.compile(r"\b(?:ở\s+đâu|where)\b", re.IGNORECASE),
    ),
    (
        "identity",
        re.compile(r"\b(?:là\s+ai|người\s+nào|who)\b", re.IGNORECASE),
    ),
    (
        "action",
        re.compile(
            r"\b(?:làm\s+gì|đang\s+làm\s+gì|doing\s+what|what\s+is\s+.+\s+doing)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "object",
        re.compile(
            r"\b(?:cầm|giữ|mang)\b.*\b(?:gì|vật\s+gì)\b"
            r"|\bwhat\b.*\b(?:holding|hold|carrying|carry)\b"
            r"|\b(?:holding|hold|carrying|carry)\b.*\bwhat\b",
            re.IGNORECASE,
        ),
    ),
    (
        "yes_no",
        re.compile(
            r"^\s*(?:có|phải|liệu)\b.*\bkhông\b"
            r"|^\s*(?:is|are|was|were|do|does|did|can|could|has|have)\b",
            re.IGNORECASE,
        ),
    ),
)
_SUBJECT_TERMS = (
    "người phụ nữ",
    "người đàn ông",
    "cô gái",
    "chàng trai",
    "đứa trẻ",
    "trẻ em",
    "a woman",
    "the woman",
    "a man",
    "the man",
    "a child",
    "the child",
    "a person",
    "the person",
)
_COLOR_TERMS = (
    "đỏ",
    "xanh lá",
    "xanh dương",
    "vàng",
    "đen",
    "trắng",
    "cam",
    "hồng",
    "tím",
    "nâu",
    "red",
    "green",
    "blue",
    "yellow",
    "black",
    "white",
    "orange",
    "pink",
    "purple",
    "brown",
)
_ACTION_TERMS = (
    "cầm",
    "giữ",
    "mang",
    "ngồi",
    "đứng",
    "đi bộ",
    "chạy",
    "nấu",
    "ăn",
    "uống",
    "holding",
    "carrying",
    "sitting",
    "standing",
    "walking",
    "running",
    "cooking",
    "eating",
    "drinking",
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
    task_mode: str = ""
    answer_type: str = "unknown"
    retrieval_statement: str = ""
    known_constraints: tuple[tuple[str, tuple[str, ...]], ...] = ()
    needs_temporal: bool = False
    confidence: float = 0.0

    def query_for(self, modality: str) -> str:
        return dict(self.modality_queries).get(modality, self.retrieval_query)

    @property
    def constraints(self) -> dict[str, list[str]]:
        return {key: list(values) for key, values in self.known_constraints}

    @property
    def question(self) -> str:
        """Backward-compatible QA name for the original query."""
        return self.original_query

    @property
    def answer_target(self) -> str:
        """Backward-compatible answer slot used by the old evidence endpoint."""
        return "held_object" if self.answer_type == "object" else self.answer_type

    @property
    def retrieval_queries(self) -> tuple[str, ...]:
        """QA-owned queries; external expansions are deliberately not included."""
        return (self.retrieval_statement or self.retrieval_query,)

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
            "task_mode": self.task_mode or self.profile,
            "answer_type": self.answer_type,
            "retrieval_statement": self.retrieval_statement or self.retrieval_query,
            "known_constraints": self.constraints,
            "needs_temporal": self.needs_temporal,
            "confidence": self.confidence,
        }


def build_query_plan(
    query: str,
    profile: str = "auto",
    *,
    task_mode: str | None = None,
) -> QueryPlan:
    original = " ".join(str(query).split())
    if not original or re.search(r"\w", original, re.UNICODE) is None:
        raise ValueError("query must not be empty")
    requested = str(task_mode if task_mode is not None else profile or "auto")
    requested = requested.strip().casefold()
    if requested not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported retrieval profile {profile!r}; expected {SUPPORTED_PROFILES}"
        )

    normalized = _normalize_typos(original)
    quoted = tuple(value.strip() for value in _QUOTED.findall(normalized) if value.strip())
    events, relation = _temporal_events(normalized)
    classification = _QUOTED.sub(" ", normalized)
    answer_type = _answer_type(classification)
    constraints = _known_constraints(classification, quoted)
    retrieval_statement = _retrieval_statement(normalized, answer_type)
    hints: list[str] = []
    reasons: list[str] = []
    if _OCR.search(classification):
        hints.append("ocr")
        reasons.append("OCR/text cue")
    if _OBJECT.search(classification):
        hints.append("objects")
        reasons.append("object cue")
    answer_hints = {
        "ocr": ("ocr", "caption"),
        "object": ("visual", "objects", "caption"),
        "count": ("visual", "objects", "caption"),
        "color": ("visual", "caption"),
        "action": ("visual", "caption"),
        "location": ("visual", "caption"),
        "identity": ("visual", "caption"),
    }
    for hint in answer_hints.get(answer_type, ()):
        hints.append(hint)
    if answer_type != "unknown":
        reasons.append(f"answer slot: {answer_type}")

    if requested != "auto":
        resolved, source = requested, "explicit"
        reasons.append(f"explicit profile: {resolved}")
    elif len(events) > 1:
        resolved, source = "temporal", "inferred"
        reasons.append("ordered temporal event chain")
    elif _QUESTION.search(classification) or answer_type != "unknown":
        resolved, source = "qa", "inferred"
        reasons.append("question/evidence query")
    elif _AVS.search(classification):
        resolved, source = "avs", "inferred"
        reasons.append("broad multi-result query")
    else:
        resolved, source = "kis", "default"
        reasons.append("default exact-instance search")

    # QA owns no translation, synonym generation, or query expansion.  External
    # expanded queries are accepted later by the QA router.
    expansion_terms = [] if resolved == "qa" else _expand(normalized)
    base_retrieval_query = retrieval_statement if resolved == "qa" else normalized
    retrieval_query = " ".join(dict.fromkeys([base_retrieval_query, *expansion_terms]))
    quoted_query = " ".join(quoted)
    modality_queries = (
        ("visual", retrieval_query),
        ("caption", retrieval_query),
        ("objects", retrieval_query),
        ("ocr", quoted_query if quoted_query and "ocr" in hints else retrieval_query),
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
        expansions=tuple(expansion_terms),
        modality_queries=modality_queries,
        reasons=tuple(reasons),
        task_mode=resolved,
        answer_type=answer_type if resolved == "qa" else "unknown",
        retrieval_statement=retrieval_statement,
        known_constraints=constraints,
        needs_temporal=relation != "none" or len(events) > 1,
        confidence=_parser_confidence(
            resolved=resolved,
            answer_type=answer_type,
            constraints=constraints,
        ),
    )


def _answer_type(query: str) -> str:
    for answer_type, pattern in _ANSWER_TYPE_PATTERNS:
        if pattern.search(query):
            return answer_type
    return "unknown"


def _known_constraints(
    query: str,
    quoted_phrases: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    folded = query.casefold()
    subjects = tuple(term for term in _SUBJECT_TERMS if term in folded)
    colors = tuple(term for term in _COLOR_TERMS if _contains_phrase(folded, term))
    attributes: list[str] = []
    for color in colors:
        clothing = re.search(
            rf"\b(?:áo|quần|váy|mũ|shirt|jacket|dress|hat)\s+{re.escape(color)}\b"
            rf"|\b{re.escape(color)}\s+(?:shirt|jacket|dress|hat)\b",
            folded,
            re.IGNORECASE,
        )
        attributes.append(clothing.group(0) if clothing else color)
    actions = tuple(term for term in _ACTION_TERMS if _contains_phrase(folded, term))
    constraints: list[tuple[str, tuple[str, ...]]] = []
    if subjects:
        constraints.append(("subject", tuple(dict.fromkeys(subjects))))
    if attributes:
        constraints.append(("attributes", tuple(dict.fromkeys(attributes))))
    if actions:
        constraints.append(("actions", tuple(dict.fromkeys(actions))))
    if quoted_phrases:
        constraints.append(("ocr_terms", tuple(dict.fromkeys(quoted_phrases))))
    return tuple(constraints)


def _contains_phrase(query: str, phrase: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(phrase)}(?!\w)",
        query,
        re.IGNORECASE,
    ) is not None


def _retrieval_statement(query: str, answer_type: str) -> str:
    statement = query.strip(" ?.!;,")
    if answer_type == "object":
        statement = re.sub(
            r"\b(?:đang\s+)?(?:cầm|giữ|mang)\s+(?:cái\s+|vật\s+)?gì\b",
            lambda match: re.sub(
                r"\s+(?:cái\s+|vật\s+)?gì\b",
                " một vật",
                match.group(0),
                flags=re.IGNORECASE,
            ),
            statement,
            flags=re.IGNORECASE,
        )
        match = re.match(
            r"^\s*what\s+(?:is|was)\s+(.+?)\s+(holding|carrying)\s*$",
            statement,
            re.IGNORECASE,
        )
        if match:
            statement = f"{match.group(1).strip()} {match.group(2)} an object"
        match = re.match(
            r"^\s*what\s+(?:does|did)\s+(.+?)\s+(hold|carry)\s*$",
            statement,
            re.IGNORECASE,
        )
        if match:
            verb = "holding" if match.group(2).casefold() == "hold" else "carrying"
            statement = f"{match.group(1).strip()} {verb} an object"
    elif answer_type == "color":
        statement = re.sub(
            r"\b(?:có\s+)?màu\s+gì\b|\bwhat\s+colou?r\b|\bwhich\s+colou?r\b",
            "",
            statement,
            flags=re.IGNORECASE,
        )
    elif answer_type == "location":
        statement = re.sub(
            r"\b(?:đang\s+)?ở\s+đâu\b|^\s*where\s+(?:is|are|was|were)\s+",
            "",
            statement,
            flags=re.IGNORECASE,
        )
    elif answer_type == "identity":
        statement = re.sub(
            r"\b(?:đó\s+)?là\s+ai\b|^\s*who\s+(?:is|are|was|were)\s+",
            "",
            statement,
            flags=re.IGNORECASE,
        )
    elif answer_type == "action":
        statement = re.sub(
            r"\b(?:đang\s+)?làm\s+gì\b|\bdoing\s+what\b",
            "",
            statement,
            flags=re.IGNORECASE,
        )
    elif answer_type == "count":
        statement = re.sub(
            r"\b(?:có\s+)?bao\s+nhiêu\b|\bhow\s+many\b",
            "",
            statement,
            flags=re.IGNORECASE,
        )
    elif answer_type == "ocr":
        statement = re.sub(
            r"\b(?:ghi|viết|đọc|hiển\s+thị)\s+(?:nội\s+dung\s+)?gì\b",
            "có văn bản",
            statement,
            flags=re.IGNORECASE,
        )
        statement = re.sub(
            r"^\s*what\s+does\s+(.+?)\s+say\s*$",
            r"\1 text",
            statement,
            flags=re.IGNORECASE,
        )
    cleaned = " ".join(statement.split()).strip(" ?.!;,")
    return cleaned or query.strip(" ?.!;,")


def _parser_confidence(
    *,
    resolved: str,
    answer_type: str,
    constraints: tuple[tuple[str, tuple[str, ...]], ...],
) -> float:
    if resolved != "qa":
        return 0.98
    if answer_type == "unknown":
        return 0.55
    return 0.94 if constraints else 0.84


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
