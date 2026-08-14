"""Safe TKIS query expansion with a reusable local production provider."""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from backend.app.services.agent.prompts import (
    QUERY_EXPANSION_PROMPT_REVISION,
    build_query_expansion_prompt,
)


DEFAULT_QUERY_EXPANSION_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_QUERY_EXPANSION_MODEL_REVISION = "c202236"
_PROVIDER_FIELDS = (
    "paraphrases",
    "objects",
    "attributes",
    "actions",
    "relations",
    "ocr_literals",
    "scene_terms",
)
_QUOTED = re.compile(r'''["'“”‘’]([^"'“”‘’]+)["'“”‘’]''')
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_CODE = re.compile(r"(?<!\w)(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{2,}(?!\w)")
_ALL_CAPS = re.compile(r"(?<!\w)[A-ZÀ-Ỹ][A-ZÀ-Ỹ0-9&.-]{1,}(?!\w)")
_OCR_CUE = re.compile(
    r"\b(?:text|word|written|reads?|sign|brand|logo|plate|code|subtitle)\s+(?:is\s+|says?\s+|numbered\s+|with\s+)?([A-ZÀ-Ỹ0-9][\wÀ-ỹ&.-]*)"
    r"|\b(?:chữ|từ|ghi|biển|nhãn|logo|mã|số)\s+([A-ZÀ-Ỹ0-9][\wÀ-ỹ&.-]*)",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "một": "1", "hai": "2", "ba": "3", "bốn": "4",
    "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9",
    "mười": "10",
}
_COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "gray", "grey", "brown", "đỏ", "xanh", "vàng", "cam", "tím",
    "hồng", "đen", "trắng", "xám", "nâu",
}
_NEGATIONS = {"no", "not", "without", "never", "không", "chẳng", "chưa"}
_RELATION_GROUPS = {
    "next_to": ("next to", "beside", "adjacent to", "cạnh", "bên cạnh", "kế bên"),
    "in_front_of": ("in front of", "phía trước", "trước mặt"),
    "behind": ("behind", "phía sau", "đằng sau"),
    "on": ("on top of", "on", "trên"),
    "under": ("under", "below", "beneath", "dưới"),
    "inside": ("inside", "in", "bên trong", "trong"),
    "outside": ("outside", "bên ngoài", "ngoài"),
}
_SUBJECT_GROUPS = {
    "man": ("man", "male", "gentleman", "ông", "đàn ông", "nam"),
    "woman": ("woman", "female", "lady", "bà", "phụ nữ", "nữ"),
    "person": ("person", "human", "someone", "người"),
    "child": ("child", "kid", "boy", "girl", "trẻ em", "đứa trẻ"),
}
_OBJECT_GROUPS = {
    "person": ("person", "people", "human", "someone", "man", "woman", "người"),
    "bus": ("bus", "coach", "xe buýt"),
    "car": ("car", "automobile", "xe hơi", "ô tô"),
    "bicycle": ("bicycle", "bike", "xe đạp"),
    "motorcycle": ("motorcycle", "motorbike", "scooter", "xe máy"),
    "truck": ("truck", "lorry", "xe tải"),
    "phone": ("phone", "smartphone", "mobile", "điện thoại"),
    "computer": ("computer", "laptop", "máy tính"),
    "shirt": ("shirt", "t-shirt", "áo"),
    "door": ("door", "cửa"),
    "book": ("book", "sách"),
    "bottle": ("bottle", "chai"),
}
_ACTION_GROUPS = {
    "stand": ("stand", "standing", "stands", "stood", "đứng"),
    "sit": ("sit", "sitting", "sits", "sat", "ngồi"),
    "walk": ("walk", "walking", "walks", "walked", "đi bộ"),
    "run": ("run", "running", "runs", "ran", "chạy"),
    "hold": ("hold", "holding", "holds", "held", "cầm", "giữ"),
    "wear": ("wear", "wearing", "wears", "wore", "mặc", "đeo"),
    "open": ("open", "opening", "opens", "opened", "mở"),
    "enter": ("enter", "entering", "enters", "entered", "vào", "lên xe"),
    "drive": ("drive", "driving", "drives", "drove", "lái"),
    "play": ("play", "playing", "plays", "played", "chơi"),
}


@dataclass(frozen=True)
class QueryExpansionConfig:
    enabled: bool = True
    max_paraphrases: int = 2
    original_weight: float = 1.0
    paraphrase_weight: float = 0.6
    max_expansion_contribution: float = 1.0
    max_query_chars: int = 512
    hard_paraphrase_limit: int = 8
    model_name: str = DEFAULT_QUERY_EXPANSION_MODEL
    model_revision: str = DEFAULT_QUERY_EXPANSION_MODEL_REVISION
    timeout_seconds: float = 120.0
    max_new_tokens: int = 384
    dtype: str = "auto"
    quantization: str = "4bit"

    def __post_init__(self) -> None:
        if not 0 <= int(self.max_paraphrases) <= 2:
            raise ValueError("max_paraphrases must be within [0, 2]")
        if self.enabled and int(self.max_paraphrases) == 0:
            raise ValueError("enabled query expansion requires at least one paraphrase slot")
        if self.original_weight <= 0 or self.paraphrase_weight < 0:
            raise ValueError("original_weight must be positive and paraphrase_weight non-negative")
        if self.enabled and self.paraphrase_weight <= 0:
            raise ValueError("enabled query expansion requires a positive paraphrase_weight")
        if self.original_weight < self.paraphrase_weight:
            raise ValueError("original_weight must be >= paraphrase_weight")
        if not 0 < self.max_expansion_contribution <= 1:
            raise ValueError("max_expansion_contribution must be within (0, 1]")
        if self.max_query_chars <= 0 or self.hard_paraphrase_limit <= 0:
            raise ValueError("query expansion limits must be positive")
        if self.timeout_seconds <= 0 or self.max_new_tokens <= 0:
            raise ValueError("query expansion runtime limits must be positive")
        if not self.model_name.strip() or not self.model_revision.strip():
            raise ValueError("query expansion model name/revision must be pinned")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("unsupported query expansion dtype")
        if self.quantization not in {"none", "8bit", "4bit"}:
            raise ValueError("unsupported query expansion quantization")


@dataclass(frozen=True)
class ProtectedLiterals:
    quoted: tuple[str, ...] = ()
    ocr_literals: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    counts: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()
    proper_names: tuple[str, ...] = ()
    negations: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class QueryVariant:
    text: str
    type: str
    weight: float
    accepted: bool = True
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "type": self.type,
            "weight": self.weight,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class QueryDecomposition:
    objects: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    ocr_literals: tuple[str, ...] = ()
    scene_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class QueryExpansionPlan:
    original: str
    variants: tuple[QueryVariant, ...]
    decomposition: QueryDecomposition
    protected_literals: ProtectedLiterals
    status: str
    fallback_reason: str = ""
    provider_name: str = ""
    model_name: str = ""
    model_revision: str = ""
    prompt_revision: str = QUERY_EXPANSION_PROMPT_REVISION
    cache_hit: bool = False
    provider_paraphrases: tuple[str, ...] = ()
    decomposition_rejections: tuple[str, ...] = ()

    @property
    def accepted_variants(self) -> tuple[QueryVariant, ...]:
        return tuple(value for value in self.variants if value.accepted)

    def to_dict(self) -> dict[str, object]:
        return {
            "original": self.original,
            "variants": [value.to_dict() for value in self.variants],
            "decomposition": self.decomposition.to_dict(),
            "protected_literals": self.protected_literals.to_dict(),
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "prompt_revision": self.prompt_revision,
            "cache_hit": self.cache_hit,
            "provider_paraphrases": list(self.provider_paraphrases),
            "decomposition_rejections": list(self.decomposition_rejections),
        }


@dataclass(frozen=True)
class ProviderResponse:
    payload: Mapping[str, object]
    cache_hit: bool = False


class QueryExpansionProvider(Protocol):
    provider_name: str
    model_name: str
    model_revision: str

    def expand(self, query: str, protected: ProtectedLiterals) -> ProviderResponse: ...
    def close(self) -> None: ...


class QwenQueryExpansionProvider:
    """Production local generative provider backed by the cached caption Qwen."""

    provider_name = "qwen_local_transformers"

    def __init__(
        self,
        *,
        config: QueryExpansionConfig | None = None,
        device: str = "cpu",
        cache_dir: Path = Path("data/model_cache/query_expansion"),
        model_cache_dir: Path | None = None,
        local_files_only: bool = False,
        runner: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config or QueryExpansionConfig()
        self.model_name = self.config.model_name
        self.model_revision = self.config.model_revision
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir is not None else self.cache_dir / "model"
        self.local_files_only = bool(local_files_only)
        self.runner = runner
        self._model: Any | None = None
        self._processor: Any | None = None
        self._load_error: Exception | None = None

    def expand(self, query: str, protected: ProtectedLiterals) -> ProviderResponse:
        prompt = build_query_expansion_prompt(query, protected.to_dict())
        cache_path = self.cache_dir / "responses" / f"{self._cache_key(query)}.json"
        if cache_path.is_file():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return ProviderResponse(_strict_payload(payload), cache_hit=True)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                # A partial/stale cache entry must not permanently force the
                # production path into original-only fallback. The successful
                # generation below atomically replaces it.
                pass
        # Model download/load is a one-time setup cost and can legitimately take
        # several minutes on a fresh Colab runtime. The deadline applies to the
        # provider inference itself, not that cold-start preparation.
        if self.runner is None:
            self._load()
        started = time.perf_counter()
        raw = self.runner(prompt) if self.runner is not None else self._run_local(prompt)
        elapsed = time.perf_counter() - started
        if elapsed > self.config.timeout_seconds:
            raise TimeoutError(
                f"query expansion exceeded {self.config.timeout_seconds}s"
            )
        payload = _strict_payload(_strict_json_object(raw))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        return ProviderResponse(payload, cache_hit=False)

    def _cache_key(self, query: str) -> str:
        value = {
            "query": query,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "prompt_revision": QUERY_EXPANSION_PROMPT_REVISION,
            "max_paraphrases": self.config.max_paraphrases,
            "max_new_tokens": self.config.max_new_tokens,
            "dtype": self.config.dtype,
            "quantization": self.config.quantization,
        }
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(f"query expansion model unavailable: {self._load_error}")
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            kwargs: dict[str, object] = {
                "revision": self.model_revision,
                "cache_dir": str(self.model_cache_dir),
                "local_files_only": self.local_files_only,
            }
            self._processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
            dtype = torch.float32
            if self.device.startswith("cuda"):
                if self.config.dtype == "float16":
                    dtype = torch.float16
                elif self.config.dtype == "float32":
                    dtype = torch.float32
                else:
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs = dict(kwargs)
            model_kwargs["torch_dtype"] = dtype
            quantization = self.config.quantization if self.device.startswith("cuda") else "none"
            if quantization != "none":
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quantization == "4bit",
                    load_in_8bit=quantization == "8bit",
                    bnb_4bit_compute_dtype=dtype,
                )
                model_kwargs["device_map"] = "auto"
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            if quantization == "none":
                self._model = self._model.to(self.device)
            self._model.eval()
        except Exception as exc:
            self._load_error = exc
            raise

    def _run_local(self, prompt: str) -> str:
        self._load()
        import torch

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = getattr(self._model, "device", torch.device(self.device))
        inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                max_time=self.config.timeout_seconds,
                do_sample=False,
            )
        return self._processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
        )[0].strip()

    def close(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def build_production_query_expansion_provider(
    *,
    config: QueryExpansionConfig,
    device: str,
    cache_dir: Path,
    model_cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> QueryExpansionProvider:
    """Build the default real generative provider without loading it eagerly."""
    return QwenQueryExpansionProvider(
        config=config,
        device=device,
        cache_dir=cache_dir,
        model_cache_dir=model_cache_dir,
        local_files_only=local_files_only,
    )


def protect_literals(query: str) -> ProtectedLiterals:
    quoted = _unique(value.strip() for value in _QUOTED.findall(query) if value.strip())
    ocr = list(quoted)
    for match in _OCR_CUE.finditer(query):
        value = next((group for group in match.groups() if group), "").strip(" ,.;:")
        if value:
            ocr.append(value)
    tokens = [value.casefold() for value in _WORD.findall(query)]
    numbers = _unique(_NUMBER.findall(query))
    counts = _unique([
        *numbers,
        *(_NUMBER_WORDS[value] for value in tokens if value in _NUMBER_WORDS),
    ])
    colors = _unique(value for value in tokens if value in _COLORS)
    codes = _unique(_CODE.findall(query))
    proper = _unique(
        value for value in _ALL_CAPS.findall(query)
        if value not in codes and value not in numbers
    )
    negations = _unique(value for value in tokens if value in _NEGATIONS)
    relations = tuple(sorted(_extract_groups(query, _RELATION_GROUPS)))
    return ProtectedLiterals(
        quoted=tuple(quoted),
        ocr_literals=tuple(_unique(ocr)),
        numbers=tuple(numbers),
        counts=tuple(counts),
        colors=tuple(colors),
        codes=tuple(codes),
        proper_names=tuple(proper),
        negations=tuple(negations),
        relations=relations,
    )


def build_query_expansion_plan(
    query: str,
    *,
    provider: QueryExpansionProvider | None,
    config: QueryExpansionConfig | None = None,
) -> QueryExpansionPlan:
    config = config or QueryExpansionConfig()
    original = " ".join(str(query).split())
    protected = protect_literals(original)
    fallback_decomposition = _deterministic_decomposition(original, protected)
    original_variant = QueryVariant(original, "original", config.original_weight)
    provider_name = getattr(provider, "provider_name", "") if provider is not None else ""
    model_name = getattr(provider, "model_name", "") if provider is not None else ""
    model_revision = getattr(provider, "model_revision", "") if provider is not None else ""
    if not original:
        return QueryExpansionPlan(
            original,
            (original_variant,),
            fallback_decomposition,
            protected,
            "fallback",
            "empty_query",
            provider_name,
            model_name,
            model_revision,
        )
    if len(original) > config.max_query_chars:
        return QueryExpansionPlan(
            original,
            (original_variant,),
            fallback_decomposition,
            protected,
            "fallback",
            "query_too_long_for_expansion",
            provider_name,
            model_name,
            model_revision,
        )
    if not config.enabled:
        return QueryExpansionPlan(
            original,
            (original_variant,),
            fallback_decomposition,
            protected,
            "disabled",
            "explicit_ablation",
        )
    if provider is None:
        return QueryExpansionPlan(
            original,
            (original_variant,),
            fallback_decomposition,
            protected,
            "fallback",
            "production_provider_unavailable",
        )
    try:
        response = provider.expand(original, protected)
        payload = _strict_payload(response.payload)
    except Exception as exc:
        return QueryExpansionPlan(
            original,
            (original_variant,),
            fallback_decomposition,
            protected,
            "fallback",
            f"{type(exc).__name__}: {exc}",
            provider_name,
            model_name,
            model_revision,
        )

    raw_paraphrases = tuple(str(value).strip() for value in payload["paraphrases"])
    accepted = 0
    seen = {_normalize(original)}
    variants: list[QueryVariant] = [original_variant]
    for index, value in enumerate(raw_paraphrases[: config.hard_paraphrase_limit]):
        reasons = _paraphrase_rejection_reasons(
            original,
            value,
            protected,
            max_query_chars=config.max_query_chars,
        )
        normalized = _normalize(value)
        if normalized in seen:
            reasons.append("duplicate_paraphrase")
        if not reasons and accepted >= config.max_paraphrases:
            reasons.append("max_paraphrases_exceeded")
        is_accepted = not reasons
        if is_accepted:
            accepted += 1
            seen.add(normalized)
        variants.append(
            QueryVariant(
                value,
                "paraphrase",
                config.paraphrase_weight,
                is_accepted,
                ",".join(dict.fromkeys(reasons)),
            )
        )
    if len(raw_paraphrases) > config.hard_paraphrase_limit:
        variants.append(
            QueryVariant(
                f"<{len(raw_paraphrases) - config.hard_paraphrase_limit} omitted>",
                "paraphrase",
                config.paraphrase_weight,
                False,
                "provider_payload_hard_limit_exceeded",
            )
        )

    decomposition, decomposition_rejections = _validated_decomposition(
        original,
        payload,
        protected,
        fallback_decomposition,
    )
    valid_empty = not raw_paraphrases
    status = "passed" if accepted or valid_empty else "fallback"
    fallback_reason = "" if status == "passed" else "no_valid_paraphrase"
    return QueryExpansionPlan(
        original=original,
        variants=tuple(variants),
        decomposition=decomposition,
        protected_literals=protected,
        status=status,
        fallback_reason=fallback_reason,
        provider_name=provider_name,
        model_name=model_name,
        model_revision=getattr(provider, "model_revision", model_revision),
        cache_hit=response.cache_hit,
        provider_paraphrases=raw_paraphrases,
        decomposition_rejections=tuple(decomposition_rejections),
    )


def normalize_object_labels(values: Sequence[str]) -> tuple[str, ...]:
    resolved: list[str] = []
    for value in values:
        matches = _extract_groups(value, _OBJECT_GROUPS)
        resolved.extend(sorted(matches))
    return tuple(_unique(resolved))


def _strict_json_object(raw: str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(raw, Mapping):
        return raw
    value = json.loads(str(raw).strip())
    if not isinstance(value, dict):
        raise ValueError("query expansion output must be a JSON object")
    return value


def _strict_payload(payload: Mapping[str, object]) -> dict[str, list[str]]:
    if set(payload) != set(_PROVIDER_FIELDS):
        raise ValueError(f"query expansion schema must contain exactly {_PROVIDER_FIELDS}")
    validated: dict[str, list[str]] = {}
    for field_name in _PROVIDER_FIELDS:
        value = payload[field_name]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"query expansion field {field_name} must be a list of strings")
        stripped = [item.strip() for item in value]
        # Preserve blank paraphrases so they are rejected and cannot masquerade
        # as the provider's legitimate `paraphrases: []` response.
        validated[field_name] = (
            stripped if field_name == "paraphrases" else [item for item in stripped if item]
        )
    return validated


def _paraphrase_rejection_reasons(
    original: str,
    paraphrase: str,
    protected: ProtectedLiterals,
    *,
    max_query_chars: int = 512,
) -> list[str]:
    reasons: list[str] = []
    if not paraphrase.strip():
        return ["empty_paraphrase"]
    if len(paraphrase) > max_query_chars:
        reasons.append("paraphrase_too_long")
    folded = _normalize(paraphrase)
    for value in (*protected.quoted, *protected.ocr_literals):
        if _normalize(value) not in folded:
            reasons.append("literal_missing")
    if set(_normalized_counts(original)) != set(_normalized_counts(paraphrase)):
        reasons.append("number_changed")
    if set(protected.codes) != set(_CODE.findall(paraphrase)):
        reasons.append("code_changed")
    if set(protected.colors) != set(_token_subset(paraphrase, _COLORS)):
        reasons.append("color_changed")
    if set(protected.counts) != set(_normalized_counts(paraphrase)):
        reasons.append("count_changed")
    if bool(protected.negations) != bool(_token_subset(paraphrase, _NEGATIONS)):
        reasons.append("negation_changed")
    if set(protected.relations) != _extract_groups(paraphrase, _RELATION_GROUPS):
        reasons.append("relation_changed")
    for value in protected.proper_names:
        if _normalize(value) not in folded:
            reasons.append("proper_name_changed")
    _compare_semantic_groups(original, paraphrase, _SUBJECT_GROUPS, "subject_changed", reasons)
    _compare_semantic_groups(original, paraphrase, _OBJECT_GROUPS, "object_changed", reasons)
    _compare_semantic_groups(original, paraphrase, _ACTION_GROUPS, "action_changed", reasons)
    return list(dict.fromkeys(reasons))


def _compare_semantic_groups(
    original: str,
    paraphrase: str,
    groups: Mapping[str, Sequence[str]],
    reason: str,
    reasons: list[str],
) -> None:
    original_groups = _extract_groups(original, groups)
    paraphrase_groups = _extract_groups(paraphrase, groups)
    if original_groups != paraphrase_groups:
        # Generic person is a safe relaxation of man/woman, but adding gender is not.
        if (
            groups is _SUBJECT_GROUPS
            and original_groups <= {"man", "woman"}
            and paraphrase_groups == {"person"}
        ):
            return
        reasons.append(reason)


def _validated_decomposition(
    original: str,
    payload: Mapping[str, list[str]],
    protected: ProtectedLiterals,
    fallback: QueryDecomposition,
) -> tuple[QueryDecomposition, list[str]]:
    rejections: list[str] = []
    objects = normalize_object_labels(payload["objects"])
    if not objects:
        objects = fallback.objects
    scene_terms: list[str] = []
    original_tokens = set(_WORD.findall(_normalize(original)))
    for value in payload["scene_terms"]:
        tokens = set(_WORD.findall(_normalize(value)))
        if tokens and tokens <= original_tokens:
            scene_terms.append(value)
        else:
            rejections.append(f"scene_term_not_grounded:{value}")
    ocr = _unique([*protected.ocr_literals, *payload["ocr_literals"]])
    ocr = [value for value in ocr if _normalize(value) in _normalize(original)]
    return QueryDecomposition(
        objects=tuple(objects),
        attributes=tuple(_grounded_terms(original, payload["attributes"])),
        actions=tuple(_grounded_or_canonical(original, payload["actions"], _ACTION_GROUPS)),
        relations=tuple(_grounded_or_canonical(original, payload["relations"], _RELATION_GROUPS)),
        ocr_literals=tuple(ocr),
        scene_terms=tuple(_unique(scene_terms)),
    ), rejections


def _deterministic_decomposition(
    query: str,
    protected: ProtectedLiterals,
) -> QueryDecomposition:
    return QueryDecomposition(
        objects=normalize_object_labels([query]),
        attributes=tuple(protected.colors),
        actions=tuple(sorted(_extract_groups(query, _ACTION_GROUPS))),
        relations=tuple(protected.relations),
        ocr_literals=tuple(protected.ocr_literals),
    )


def _grounded_terms(original: str, values: Sequence[str]) -> list[str]:
    folded = _normalize(original)
    return _unique(value for value in values if _normalize(value) in folded)


def _grounded_or_canonical(
    original: str,
    values: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> list[str]:
    allowed = _extract_groups(original, groups)
    resolved: list[str] = []
    for value in values:
        matches = _extract_groups(value, groups)
        resolved.extend(sorted(matches & allowed))
    return _unique(resolved or sorted(allowed))


def _extract_groups(text: str, groups: Mapping[str, Sequence[str]]) -> set[str]:
    folded = _normalize(text)
    found: set[str] = set()
    for canonical, aliases in groups.items():
        if any(_contains_phrase(folded, _normalize(alias)) for alias in aliases):
            found.add(canonical)
    return found


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _token_subset(text: str, allowed: set[str]) -> list[str]:
    return _unique(value.casefold() for value in _WORD.findall(text) if value.casefold() in allowed)


def _normalized_counts(text: str) -> list[str]:
    return _unique([
        *_NUMBER.findall(text),
        *(
            _NUMBER_WORDS[value.casefold()]
            for value in _WORD.findall(text)
            if value.casefold() in _NUMBER_WORDS
        ),
    ])


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def _unique(values: Sequence[str] | Any) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


__all__ = [
    "DEFAULT_QUERY_EXPANSION_MODEL",
    "DEFAULT_QUERY_EXPANSION_MODEL_REVISION",
    "ProtectedLiterals",
    "ProviderResponse",
    "QueryDecomposition",
    "QueryExpansionConfig",
    "QueryExpansionPlan",
    "QueryExpansionProvider",
    "QueryVariant",
    "QwenQueryExpansionProvider",
    "build_query_expansion_plan",
    "build_production_query_expansion_provider",
    "normalize_object_labels",
    "protect_literals",
]
