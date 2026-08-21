"""Deterministic, dependency-light query planning."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any

from backend.app.services.agent.query_expansion import (
    QueryExpansionConfig,
    QueryExpansionPlan,
    QueryExpansionProvider,
    build_query_expansion_plan,
)


SUPPORTED_PROFILES = ("auto", "kis", "avs", "qa", "temporal")
QA_ANSWER_TYPES = (
    "object",
    "color",
    "ocr",
    "action",
    "count",
    "location",
    "yes_no",
    "identity",
)
CONSTRAINT_CATEGORIES = (
    "subject",
    "objects",
    "attributes",
    "actions",
    "locations",
    "ocr_terms",
)
MAX_TEMPORAL_EVENTS = 5
_TEMPORAL_SPLIT = re.compile(
    r"\b(?:then|after\s+that|next(?!\s+to\b)|followed\s+by|sau\s+đó|tiếp\s+theo|rồi)\b",
    re.IGNORECASE,
)
_BEFORE_OR_AFTER = re.compile(
    r"\b(?P<relation>before|after|trước\s+khi|sau\s+khi)\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r'''["'“”‘’]([^"'“”‘’]+)["'“”‘’]''')
_OCR = re.compile(
    r"\b(?:text|written|printed|read|subtitle|sign|signboard|plate|menu|logo|qr|bib)\b"
    r"|\b(?:chữ|văn\s+bản|ghi\s+(?:chữ\s+)?gì|viết\s+(?:chữ\s+)?gì|phụ\s+đề|biển\s+hiệu|biển\s+báo|biển\s+số|thực\s+đơn|mã\s+qr)\b",
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
            r"\b(?:ghi|viết|đọc|hiển\s+thị)\s+(?:(?:nội\s+dung|chữ)\s+)?gì\b"
            r"|\b(?:what|which)\b.*\b(?:text|written|printed|say|read|display)\b"
            r"|\bwhat\s+(?:number|word|name)\b.*\b(?:printed|written|displayed|shown)\b"
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
            r"\b(?:bao\s+nhiêu|mấy|how\s+many)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "location",
        re.compile(
            r"\b(?:ở\s+đâu|where|địa\s+chỉ|address)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "identity",
        re.compile(
            r"\b(?:là\s+ai|người\s+nào|who)\b"
            r"|\b(?:vai\s+trò|nghề)\s+gì\b|\bwhat\s+(?:role|job|occupation)\b"
            r"|\b(?:tên|name)\b.*\b(?:là\s+gì|what)\b"
            r"|\bwhat\s+(?:is|was)\b.*(?:'s|\bthe\b).*\bname\b",
            re.IGNORECASE,
        ),
    ),
    (
        "action",
        re.compile(
            r"\b(?:làm\s+gì|đang\s+làm\s+gì|doing\s+what|what\s+(?:is|are|was|were)\s+.+\s+doing)\b"
            r"|\bwhat\s+(?:did|does|do)\s+.+\s+do\b",
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
            r"(?:^|\s)(?:có|phải|liệu)\b.*\bkhông\b"
            r"|\b(?:có|phải)\s+không\b"
            r"|^\s*(?:is|are|was|were|do|does|did|can|could|has|have|"
            r"will|would|should|may|might|must)\b",
            re.IGNORECASE,
        ),
    ),
)
_SUBJECT_PHRASES = (
    ("người đi xe đạp", ("người đi xe đạp",)),
    ("người phụ nữ", ("người phụ nữ",)),
    ("người đàn ông", ("người đàn ông",)),
    ("cô gái", ("cô gái",)),
    ("chàng trai", ("chàng trai",)),
    ("đứa trẻ", ("đứa trẻ", "trẻ em")),
    ("woman", ("the woman", "a woman", "woman")),
    ("man", ("the man", "a man", "man")),
    ("child", ("the child", "a child", "child")),
    ("person", ("the person", "a person", "person")),
    ("runner", ("the runner", "a runner", "runner")),
    ("cyclist", ("the cyclist", "a cyclist", "cyclist")),
    ("người", ("người",)),
)
_OBJECT_PHRASES = (
    ("mũ bảo hiểm", ("mũ bảo hiểm",)),
    ("đèn giao thông", ("đèn giao thông",)),
    ("traffic light", ("the traffic light", "a traffic light", "traffic light")),
    ("chiếc ô", ("chiếc ô", "cái ô")),
    ("biển hiệu", ("biển hiệu",)),
    ("chiếc cốc", ("chiếc cốc", "cái cốc")),
    ("con mèo", ("con mèo",)),
    ("con chó", ("con chó",)),
    ("bicycles", ("the bicycles", "bicycles")),
    ("bicycle", ("a bicycle", "the bicycle", "bicycle")),
    ("suitcase", ("the suitcase", "a suitcase", "suitcase")),
    ("car", ("the car", "a car", "car")),
    ("bib", ("the runner's bib", "runner's bib", "the bib", "bib")),
    ("phone", ("a phone", "the phone", "phone")),
    ("book", ("a book", "the book", "book")),
    ("ball", ("a ball", "the ball", "ball")),
    ("sign", ("a sign", "the sign", "sign")),
    ("jacket", ("a jacket", "the jacket", "jacket")),
    ("bag", ("a bag", "the bag", "bag")),
    ("túi", ("cái túi", "chiếc túi", "túi")),
    ("cup", ("a cup", "the cup", "cup")),
    ("helmet", ("a helmet", "the helmet", "helmet")),
)
_ATTRIBUTE_PHRASES = (
    ("đồng phục phản quang", ("đồng phục phản quang",)),
    ("áo đỏ", ("áo đỏ",)),
    ("blue shirt", ("the blue shirt", "a blue shirt", "blue shirt")),
    ("xanh lá", ("xanh lá",)),
    ("xanh dương", ("xanh dương",)),
    ("đỏ", ("đỏ",)),
    ("vàng", ("vàng",)),
    ("đen", ("đen",)),
    ("trắng", ("trắng",)),
    ("green", ("green",)),
    ("blue", ("blue",)),
    ("red", ("red",)),
    ("yellow", ("yellow",)),
    ("black", ("black",)),
    ("white", ("white",)),
)
_ACTION_PHRASES = (
    ("directing traffic", ("directing traffic",)),
    ("đi bộ", ("đi bộ",)),
    ("cầm", ("cầm",)),
    ("giữ", ("giữ",)),
    ("mang", ("mang",)),
    ("nằm", ("nằm",)),
    ("ngồi", ("ngồi",)),
    ("đứng", ("đứng",)),
    ("chạy", ("chạy",)),
    ("nấu", ("nấu",)),
    ("ăn", ("ăn",)),
    ("uống", ("uống",)),
    ("holding", ("holding",)),
    ("hold", ("hold",)),
    ("carrying", ("carrying",)),
    ("carry", ("carry",)),
    ("wearing", ("wearing",)),
    ("wear", ("wear",)),
    ("sitting", ("sitting",)),
    ("standing", ("standing",)),
    ("walking", ("walking",)),
    ("running", ("running",)),
    ("cooking", ("cooking",)),
    ("eating", ("eating",)),
    ("drinking", ("drinking",)),
)
_LOCATION_PHRASES = (
    ("phía trên cửa", ("phía trên cửa",)),
    ("cạnh cửa", ("cạnh cửa",)),
    ("bên trái", ("bên trái",)),
    ("bên phải", ("bên phải",)),
    ("trên bàn", ("trên bàn",)),
    ("by the tree", ("by the tree",)),
    ("next to the window", ("next to the window",)),
    ("beside the wall", ("beside the wall",)),
    ("at the intersection", ("at the intersection",)),
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
    task_mode: str = ""
    answer_type: str = "unknown"
    retrieval_statement: str = ""
    known_constraints: tuple[tuple[str, tuple[str, ...]], ...] = ()
    constraint_roles: tuple[
        tuple[str, tuple[tuple[str, str], ...]], ...
    ] = ()
    needs_temporal: bool = False
    answer_event_index: int | None = None
    confidence: float = 0.0

    def query_for(self, modality: str) -> str:
        return dict(self.modality_queries).get(modality, self.retrieval_query)

    @property
    def constraints(self) -> dict[str, list[str]]:
        return {key: list(values) for key, values in self.known_constraints}

    @property
    def roles(self) -> dict[str, dict[str, str]]:
        return {
            category: {phrase: role for phrase, role in values}
            for category, values in self.constraint_roles
        }

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
            "expansion_plan": self.expansion_plan.to_dict(),
            "modality_queries": dict(self.modality_queries),
            "reasons": list(self.reasons),
            "task_mode": self.task_mode or self.profile,
            "answer_type": self.answer_type,
            "retrieval_statement": self.retrieval_statement or self.retrieval_query,
            "known_constraints": self.constraints,
            "constraint_roles": self.roles,
            "needs_temporal": self.needs_temporal,
            "answer_event_index": self.answer_event_index,
            "confidence": self.confidence,
        }


def build_query_plan(
    query: str,
    profile: str = "auto",
    *,
    task_mode: str | None = None,
    expansion_provider: QueryExpansionProvider | None = None,
    expansion_config: QueryExpansionConfig | None = None,
    expansion_plan: QueryExpansionPlan | None = None,
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
    classification = _QUOTED.sub(" ", normalized)
    answer_type = _answer_type(classification)
    raw_events, relation = _ordered_temporal_clauses(normalized)
    events = (
        tuple(_clean_event_clause(event) for event in raw_events)
        if relation != "none" or len(raw_events) > 1
        else raw_events
    )
    constraints = _known_constraints(
        classification,
        quoted,
        answer_type=answer_type,
    )
    constraint_roles = _constraint_roles(
        normalized,
        answer_type=answer_type,
        constraints=constraints,
    )
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
        "yes_no": ("visual", "caption", "objects"),
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

    # Explicit KIS/AVS plans do not silently become temporal executions. QA
    # retains parsed events because its router owns temporal evidence chains.
    if resolved not in {"temporal", "qa"}:
        events = (normalized,)
        relation = "none"

    # Generic query expansion belongs only to advanced KIS/AVS retrieval.
    # QA, temporal, and TRAKE own their task-specific query parsing downstream;
    # calling the generic provider for those routes only adds latency and the
    # resulting variants are intentionally unused.
    effective_expansion_config = expansion_config
    effective_expansion_provider = expansion_provider
    expansion_skip_reason = ""
    if resolved not in {"kis", "avs"} and expansion_plan is None:
        effective_expansion_config = QueryExpansionConfig(enabled=False)
        effective_expansion_provider = None
        expansion_skip_reason = f"{resolved}_route"
    resolved_expansion = expansion_plan or build_query_expansion_plan(
        original,
        provider=effective_expansion_provider,
        config=effective_expansion_config,
    )
    if expansion_skip_reason:
        resolved_expansion = replace(
            resolved_expansion,
            fallback_reason=expansion_skip_reason,
        )
    accepted_paraphrases = tuple(
        value.text
        for value in resolved_expansion.accepted_variants
        if value.type == "paraphrase"
    )
    retrieval_query = retrieval_statement if resolved == "qa" else normalized
    if resolved == "qa":
        quoted_query = " ".join(quoted)
        modality_queries = (
            ("visual", retrieval_query),
            ("caption", retrieval_query),
            ("objects", retrieval_query),
            ("ocr", quoted_query if quoted_query and "ocr" in hints else retrieval_query),
        )
    else:
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
        task_mode=resolved,
        # Answer semantics belong to the question, not to the execution profile.
        # In particular a temporal QA route must not erase the answer slot.
        answer_type=answer_type,
        retrieval_statement=retrieval_statement,
        known_constraints=constraints,
        constraint_roles=constraint_roles,
        needs_temporal=relation != "none" or len(events) > 1,
        answer_event_index=_answer_event_index(
            answer_type=answer_type,
            events=raw_events,
            relation=relation,
        ),
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
    *,
    answer_type: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    folded = query.casefold()
    locations, location_spans = _extract_phrases(folded, _LOCATION_PHRASES)
    subjects, subject_spans = _extract_phrases(folded, _SUBJECT_PHRASES)
    objects, _ = _extract_phrases(
        folded,
        _OBJECT_PHRASES,
        excluded_spans=(*location_spans, *subject_spans),
    )
    attributes, _ = _extract_phrases(folded, _ATTRIBUTE_PHRASES)
    actions, _ = _extract_phrases(folded, _ACTION_PHRASES)
    if not locations:
        locations = _generic_locations(folded)
    if not subjects:
        subjects = _generic_subjects(folded, answer_type=answer_type)
    if not objects:
        objects = _generic_objects(folded, answer_type=answer_type)

    values_by_category = {
        "subject": subjects,
        "objects": objects,
        "attributes": attributes,
        "actions": actions,
        "locations": locations,
        "ocr_terms": tuple(dict.fromkeys(quoted_phrases)),
    }
    return tuple(
        (category, values_by_category[category])
        for category in CONSTRAINT_CATEGORIES
        if values_by_category[category]
    )


def _generic_subjects(query: str, *, answer_type: str) -> tuple[str, ...]:
    patterns: tuple[str, ...] = ()
    if answer_type == "object":
        patterns = (
            r"^\s*what\s+(?:is|are|was|were)\s+(.+?)\s+(?:holding|carrying)\b",
            r"^\s*what\s+(?:does|do|did)\s+(.+?)\s+(?:hold|carry)\b",
            r"^\s*(.+?)\s+(?:đang\s+)?(?:cầm|giữ|mang)\b",
        )
    elif answer_type == "action":
        patterns = (
            r"^\s*what\s+(?:is|are|was|were)\s+(.+?)(?:\s+(?:next\s+to|beside|by|at|near)\b.*?)?\s+doing\b",
            r"^\s*what\s+(?:does|do|did)\s+(.+?)\s+do\b",
            r"^\s*(.+?)\s+(?:đang\s+)?làm\s+gì\b",
        )
    elif answer_type == "yes_no":
        predicate = r"(?:holding|carrying|wearing|hold|carry|wear|cầm|giữ|mang|đeo|đội|mặc)"
        patterns = (
            rf"^\s*(?:is|are|was|were|do|does|did|can|could|has|have|will|would|should|may|might|must)\s+(?!there\b)(.+?)\s+{predicate}\b",
            rf"^\s*(?:có\s+phải|phải\s+chăng|phải|liệu)\s+(.+?)\s+(?:có\s+)?(?:đang\s+)?{predicate}\b",
            rf"^\s*(?!có\s+phải\b)(?:liệu\s+)?(.+?)\s+(?:có|phải)\s+(?:đang\s+)?{predicate}\b",
        )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            value = _canonical_entity(match.group(1))
            if value:
                return (value,)
    return ()


def _generic_objects(query: str, *, answer_type: str) -> tuple[str, ...]:
    patterns_by_type = {
        "count": (
            r"\bhow\s+many\s+(.+?)(?=\s+(?:are|is|were|was|beside|by|at|on|under|near)\b|[?!.]|$)",
            r"\b(?:bao\s+nhiêu|mấy)\s+(.+?)(?=\s+(?:ở|trên|dưới|cạnh|gần|đang|có)\b|[?!.]|$)",
        ),
        "color": (
            r"\bwhat\s+colou?r\s+(?:is|are|was|were)\s+(.+?)(?=\s+(?:parked|placed|standing|sitting|beside|by|at|near|under|above|next)\b|[?!.]|$)",
            r"^\s*(.+?)(?=\s+(?:có\s+)?màu\s+gì\b)",
        ),
        "location": (
            r"^\s*where\s+(?:is|are|was|were)\s+(.+?)(?=\s+(?:placed|parked|located|standing|sitting)\b|[?!.]|$)",
            r"^\s*(.+?)(?=\s+(?:đang\s+)?(?:nằm\s+)?ở\s+đâu\b)",
        ),
        "yes_no": (
            r"\b(?:holding|carrying|wearing|hold|carry|wear)\s+(?:an|a|the)?\s*(.+?)(?=\s+(?:while|when|before|after)\b|[?!.]|$)",
            r"\b(?:cầm|giữ|mang|đeo|đội|mặc)\s+(?:cái|chiếc|con)?\s*(.+?)(?=\s+không\b|[?!.]|$)",
        ),
        "identity": (
            r"\bwhat\s+(?:is|was)\s+(?:the\s+)?(.+?)(?:'s|’s)\s+name\b",
            r"\btên\s+của\s+(.+?)(?=\s+(?:trong\b.*\s+)?là\s+gì\b|[?!.]|$)",
        ),
        "ocr": (
            r"\b(?:on|from)\s+(?:a|an|the)?\s*(.+?)(?=[?!.]|$)",
            r"\b(?:trên|ở)\s+(.+?)(?=\s+(?:ghi|viết|hiển\s+thị|đọc)\b|[?!.]|$)",
        ),
    }
    values: list[str] = []
    for pattern in patterns_by_type.get(answer_type, ()):
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            value = _canonical_entity(match.group(1))
            if value:
                values.append(value)
    return tuple(dict.fromkeys(values))


def _generic_locations(query: str) -> tuple[str, ...]:
    patterns = (
        r"\b(?:next\s+to|beside|by|at|under|above|near|behind|in\s+front\s+of)\s+(?:the\s+)?[\w'-]+(?:\s+[\w'-]+){0,2}",
        r"\b(?:phía\s+trên|phía\s+dưới|bên\s+trái|bên\s+phải|cạnh|trên|dưới|gần|phía\s+sau)\s+[\wÀ-ỹ-]+(?:\s+[\wÀ-ỹ-]+){0,2}",
    )
    stop_words = re.compile(
        r"\s+\b(?:doing|holding|carrying|is|are|was|were|đang|có|ghi|viết|làm|là)\b.*$",
        flags=re.IGNORECASE,
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            value = stop_words.sub("", match.group(0)).strip(" ,.;?!")
            if value:
                return (value,)
    return ()


def _canonical_entity(value: str) -> str:
    cleaned = " ".join(value.strip(" ,.;?!").split())
    cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:có|what|which)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_phrases(
    query: str,
    phrase_groups: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    excluded_spans: tuple[tuple[int, int], ...] = (),
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Extract canonical phrases, preferring longest non-overlapping matches."""

    candidates: list[tuple[int, int, str]] = []
    for canonical, variants in phrase_groups:
        for variant in variants:
            for match in re.finditer(
                rf"(?<!\w){re.escape(variant)}(?!\w)",
                query,
                flags=re.IGNORECASE,
            ):
                span = match.span()
                if any(_spans_overlap(span, excluded) for excluded in excluded_spans):
                    continue
                candidates.append((span[0], span[1], canonical))

    # Longest match wins at an overlap ("the child" -> "child", never an
    # additional generic "person").  Final values still follow query order.
    selected: list[tuple[int, int, str]] = []
    for start, end, canonical in sorted(
        candidates,
        key=lambda item: (-(item[1] - item[0]), item[0], item[2]),
    ):
        if any(_spans_overlap((start, end), (left, right)) for left, right, _ in selected):
            continue
        selected.append((start, end, canonical))
    selected.sort(key=lambda item: (item[0], item[1]))
    values = tuple(dict.fromkeys(item[2] for item in selected))
    spans = tuple((item[0], item[1]) for item in selected)
    return values, spans


def _spans_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _constraint_roles(
    query: str,
    *,
    answer_type: str,
    constraints: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Label each constraint as retrieval context or a yes/no hypothesis.

    Hypothesis values describe the proposition being tested and therefore must
    not receive a relevance bonus.  The representation mirrors
    ``known_constraints`` while assigning a role per canonical phrase.
    """

    if answer_type != "yes_no":
        return tuple(
            (category, tuple((phrase, "context") for phrase in values))
            for category, values in constraints
        )

    folded = query.casefold()
    hypothesis_start = _yes_no_hypothesis_start(folded, constraints)
    role_groups: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for category, values in constraints:
        roles: list[tuple[str, str]] = []
        for phrase in values:
            role = "context"
            occurrences = _phrase_occurrences(folded, phrase.casefold())
            if hypothesis_start is not None and any(
                start >= hypothesis_start for start, _end in occurrences
            ):
                role = "hypothesis"
            elif hypothesis_start is None and _phrase_is_hypothesis(folded, phrase):
                role = "hypothesis"
            roles.append((phrase, role))
        role_groups.append((category, tuple(roles)))
    return tuple(role_groups)


def _yes_no_hypothesis_start(
    query: str,
    constraints: tuple[tuple[str, tuple[str, ...]], ...],
) -> int | None:
    """Locate the proposition boundary without leaking identifying context.

    The boundary is deliberately positional.  For example, ``blue shirt`` is
    context in "Is the child in the blue shirt holding a ball?", while
    ``holding`` and ``ball`` are the proposition being tested.
    """

    # Vietnamese subject-first questions expose an explicit predicate marker.
    # Ignore a leading ``có phải``/``liệu`` wrapper and use a later marker when
    # present ("Liệu bác sĩ có cầm ... không?").
    leading_wrapper = re.match(
        r"^\s*(?:có\s+phải|phải\s+chăng|phải|liệu)\b",
        query,
        re.IGNORECASE,
    )
    wrapper_end = leading_wrapper.end() if leading_wrapper else 0
    terminal = re.search(r"\bkhông\b", query, re.IGNORECASE)
    markers = [
        match
        for match in re.finditer(r"\b(?:có|phải)\b", query, re.IGNORECASE)
        if match.start() >= wrapper_end and (terminal is None or match.end() <= terminal.start())
    ]
    if markers:
        return markers[-1].end()

    occurrences: dict[str, list[tuple[int, int, str]]] = {}
    for category, values in constraints:
        occurrences[category] = [
            (start, end, phrase)
            for phrase in values
            for start, end in _phrase_occurrences(query, phrase.casefold())
        ]

    english_aux = re.match(
        r"^\s*(?:is|are|was|were|do|does|did|can|could|has|have|"
        r"will|would|should|may|might|must)\b",
        query,
        re.IGNORECASE,
    )
    search_start = english_aux.end() if english_aux else wrapper_end
    if english_aux and re.match(r"\s+there\b", query[search_start:], re.IGNORECASE):
        there = re.search(r"\bthere\b", query[search_start:], re.IGNORECASE)
        return search_start + there.end() if there else search_start

    # A semantic subject is the strongest anchor.  For inanimate yes/no
    # questions the first object is normally the grammatical subject.
    anchors = occurrences.get("subject", []) or occurrences.get("objects", [])
    anchor = min(
        (item for item in anchors if item[0] >= search_start),
        key=lambda item: item[0],
        default=None,
    )
    anchor_end = anchor[1] if anchor else search_start

    # Prefer an explicit action/verb after the subject.  This keeps modifiers
    # such as "running" in "the running man" or "blue shirt" in the subject
    # context, while marking the main predicate as hypothesis.
    action_starts = [
        start
        for start, _end, _phrase in occurrences.get("actions", [])
        if start >= anchor_end
    ]
    if action_starts:
        return min(action_starts)
    predicate = re.search(
        r"\b(?:hold|holding|carry|carrying|wear|wearing|say|says|show|shows|"
        r"display|displays|read|reads|contain|contains|cầm|giữ|mang|đeo|đội|mặc)\b",
        query[anchor_end:],
        re.IGNORECASE,
    )
    if predicate:
        return anchor_end + predicate.start()

    # Copular propositions have no lexical action ("Is the light green?").
    # The first non-subject constraint after the anchor is the tested value.
    candidates = [
        start
        for category in ("attributes", "ocr_terms", "locations", "objects")
        for start, _end, _phrase in occurrences.get(category, [])
        if start >= anchor_end
        and not (anchor is not None and start == anchor[0])
    ]
    return min(candidates) if candidates else None


def _phrase_occurrences(query: str, phrase: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        match.span()
        for match in re.finditer(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            query,
            re.IGNORECASE,
        )
    )


def _phrase_is_hypothesis(
    query: str,
    phrase: str,
) -> bool:
    match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", query, re.IGNORECASE)
    if match is None:
        # Canonical English phrases may omit an article present in the query.
        match = re.search(
            rf"(?<!\w)(?:a|an|the)\s+{re.escape(phrase)}(?!\w)",
            query,
            re.IGNORECASE,
        )
    if match is None:
        return False
    prefix = query[: match.start()]
    return bool(
        re.search(
            r"\b(?:there|hold|holding|carry|carrying|have|has|wear|wearing|có|đội|cầm|giữ|mang|đeo)\s+(?:a|an|the)?\s*$",
            prefix,
            re.IGNORECASE,
        )
    )


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
            r"^\s*what\s+(?:is|are|was|were)\s+(.+?)\s+(holding|carrying)\s*$",
            statement,
            re.IGNORECASE,
        )
        if match:
            statement = f"{match.group(1).strip()} {match.group(2)} an object"
        match = re.match(
            r"^\s*what\s+(?:does|do|did)\s+(.+?)\s+(hold|carry)\s*$",
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
            r"\b(?:đang\s+)?ở\s+đâu\b"
            r"|^\s*where\s+(?:is|are|was|were)\s+"
            r"|\b(?:home\s+)?address\b|\bđịa\s+chỉ(?:\s+nhà)?\b",
            "",
            statement,
            flags=re.IGNORECASE,
        )
    elif answer_type == "identity":
        statement = re.sub(
            r"\b(?:đó\s+)?là\s+ai\b"
            r"|^\s*who\s+(?:is|are|was|were)\s+"
            r"|\b(?:tên|name)\b.*?\b(?:là\s+gì|what)\b",
            "",
            statement,
            flags=re.IGNORECASE,
        )
    elif answer_type == "action":
        match = re.match(
            r"^\s*what\s+(?:is|are|was|were)\s+(.+?)\s+doing\s*$",
            statement,
            re.IGNORECASE,
        )
        if match:
            statement = match.group(1)
        else:
            match = re.match(
                r"^\s*what\s+(?:does|do|did)\s+(.+?)\s+do\s*$",
                statement,
                re.IGNORECASE,
            )
            if match:
                statement = match.group(1)
            else:
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
            r"\b(?:ghi|viết|đọc|hiển\s+thị)\s+(?:(?:nội\s+dung|chữ)\s+)?gì\b",
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
    elif answer_type == "yes_no":
        statement = re.sub(
            r"^\s*(?:is|are|was|were|do|does|did|can|could|has|have|"
            r"will|would|should|may|might|must)\s+",
            "",
            statement,
            flags=re.IGNORECASE,
        )
        statement = re.sub(
            r"^\s*(?:có\s+phải|phải\s+chăng|phải|liệu)\s+",
            "",
            statement,
            flags=re.IGNORECASE,
        )
        statement = re.sub(
            r"\b(?:có|phải)\s+(?=(?:đang\s+)?(?:cầm|giữ|mang|đeo|đội|mặc|ngồi|đứng|chạy))",
            "",
            statement,
            flags=re.IGNORECASE,
        )
        statement = re.sub(r"\bkhông\s*$", "", statement, flags=re.IGNORECASE)
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
    """Backward-compatible wrapper returning clean chronological clauses."""

    raw_events, relation = _ordered_temporal_clauses(query)
    if relation == "none":
        return raw_events, relation
    return tuple(_clean_event_clause(event) for event in raw_events), relation


def _ordered_temporal_clauses(query: str) -> tuple[tuple[str, ...], str]:
    """Parse before/after/then expressions into chronological event clauses.

    The parser is intentionally small but recursive.  Recursion matters for
    mixed expressions such as ``A after B, then C`` and for repeated relations
    such as ``A after B after C``.  Leading subordinate forms (``Before B, A``)
    are handled before sequential splitting so their comma-delimited scope is
    preserved.
    """

    _preflight_temporal_complexity(query)
    events, relations = _parse_temporal_expression(query)
    events = [_normalize_temporal_leaf(event) for event in events]
    events = [event for event in events if event]
    if not events:
        events = [_normalize_temporal_leaf(query)]
    _validate_temporal_event_count(events)
    if len(events) <= 1 or not relations:
        return tuple(events), "none"
    unique_relations = tuple(dict.fromkeys(relations))
    relation = unique_relations[0] if len(unique_relations) == 1 else "mixed"
    return tuple(events), relation


def _parse_temporal_expression(value: str) -> tuple[list[str], list[str]]:
    text = value.strip(" ,.;?!")
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
                relation = _canonical_temporal_relation(leading.group("relation"))
                if relation == "before":
                    return (
                        [*main_events, *subordinate_events],
                        [*main_relations, *subordinate_relations, relation],
                    )
                return (
                    [*subordinate_events, *main_events],
                    [*subordinate_relations, *main_relations, relation],
                )

    sequential_parts = _TEMPORAL_SPLIT.split(text)
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
        if events:
            return events, relations

    comparison = _BEFORE_OR_AFTER.search(text)
    if comparison is not None:
        left = text[: comparison.start()]
        right = text[comparison.end() :]
        left_events, left_relations = _parse_temporal_expression(left)
        right_events, right_relations = _parse_temporal_expression(right)
        if left_events and right_events:
            relation = _canonical_temporal_relation(comparison.group("relation"))
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


def _canonical_temporal_relation(value: str) -> str:
    folded = value.casefold()
    return "before" if folded == "before" or folded.startswith("trước") else "after"


def _preflight_temporal_complexity(query: str) -> None:
    sequential_spans = [match.span() for match in _TEMPORAL_SPLIT.finditer(query)]
    comparison_count = sum(
        not any(_spans_overlap(match.span(), span) for span in sequential_spans)
        for match in _BEFORE_OR_AFTER.finditer(query)
    )
    if len(sequential_spans) + comparison_count >= MAX_TEMPORAL_EVENTS:
        raise ValueError(
            "temporal_query_too_complex: ordered temporal queries support "
            f"at most {MAX_TEMPORAL_EVENTS} events"
        )


def _clean_event_clause(value: str) -> str:
    clause = _normalize_temporal_leaf(value)
    if not clause:
        return ""
    answer_type = _answer_type(_QUOTED.sub(" ", clause))
    if answer_type != "unknown":
        clause = _retrieval_statement(clause, answer_type)
    return " ".join(clause.strip(" ,.;?!").split())


def _normalize_temporal_leaf(value: str) -> str:
    clause = " ".join(value.strip(" ,.;?!").split())
    clause = re.sub(
        r"^\s*(?:show|find|retrieve|tìm\s+cảnh|cho\s+tôi\s+xem|tìm)\s+",
        "",
        clause,
        flags=re.IGNORECASE,
    )
    clause = re.sub(
        r"^\s*(?:and|và)\s+",
        "",
        clause,
        flags=re.IGNORECASE,
    )
    return " ".join(clause.strip(" ,.;?!").split())


def _validate_temporal_event_count(events: list[str]) -> None:
    if len(events) > MAX_TEMPORAL_EVENTS:
        raise ValueError(
            "temporal_query_too_complex: ordered temporal queries support "
            f"at most {MAX_TEMPORAL_EVENTS} events"
        )


def _answer_event_index(
    *,
    answer_type: str,
    events: tuple[str, ...],
    relation: str,
) -> int | None:
    if relation == "none" or len(events) <= 1:
        return None
    if answer_type == "yes_no":
        # A yes/no proposition applies to the complete ordered chain.
        return None
    cue_indexes = [
        index
        for index, event in enumerate(events)
        if _answer_type(_QUOTED.sub(" ", event)) == answer_type
    ]
    if cue_indexes:
        # A well-formed question has one cue.  Last-cue wins is deterministic
        # for malformed multi-question input and avoids returning a context
        # clause merely because a before/after relation reordered the events.
        return cue_indexes[-1]
    return 0 if relation == "before" else len(events) - 1
