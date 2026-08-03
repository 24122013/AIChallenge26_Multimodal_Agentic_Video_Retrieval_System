"""Config loader for Retrieval Phase 2/3 settings."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode

from backend.app.services.retrieval.hybrid_search import HybridSearchConfig
from backend.app.services.retrieval.rerank import RerankConfig, RerankWeights


DEFAULT_RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")

_CONFIG_SCHEMA = {
    "hybrid": {
        "stage1_top_k": "positive integer",
        "text_stage1_top_k": "positive integer",
        "rerank_pool_size": "positive integer",
        "default_top_k": "positive integer",
        "max_top_k": "positive integer",
        "max_gap_seconds": "positive number",
    },
    "weights": {
        "visual": "non-negative number",
        "caption": "non-negative number",
        "ocr": "non-negative number",
        "asr": "non-negative number",
        "objects": "non-negative number",
        "temporal": "non-negative number",
    },
    "dedupe": {
        "same_shot": "boolean",
    },
    "text_index": {
        "path": "non-empty string",
        "default_top_k": "positive integer",
        "max_top_k": "positive integer",
    },
}


class RetrievalConfigError(ValueError):
    """Raised for invalid YAML syntax or an invalid Retrieval config schema."""


@dataclass(frozen=True)
class TextIndexConfig:
    path: Path = Path("data/indexes/retrieval_text_index.json")
    default_top_k: int = 20
    max_top_k: int = 200


@dataclass(frozen=True)
class RetrievalRuntimeConfig:
    hybrid: HybridSearchConfig = HybridSearchConfig()
    rerank: RerankConfig = RerankConfig()
    text_index: TextIndexConfig = TextIndexConfig()


def load_retrieval_runtime_config(
    config_path: str | Path | None = None,
) -> RetrievalRuntimeConfig:
    path = Path(
        config_path
        or os.getenv("RETRIEVAL_CONFIG_PATH")
        or DEFAULT_RETRIEVAL_CONFIG_PATH
    )
    raw = _load_yaml_config(path) if path.exists() else {}
    hybrid_raw = _section(raw, "hybrid")
    weights_raw = _section(raw, "weights")
    dedupe_raw = _section(raw, "dedupe")
    text_raw = _section(raw, "text_index")

    hybrid = HybridSearchConfig(
        stage1_top_k=_int_env(
            "RETRIEVAL_HYBRID_STAGE1_TOP_K",
            hybrid_raw.get("stage1_top_k"),
            HybridSearchConfig.stage1_top_k,
        ),
        text_stage1_top_k=_int_env(
            "RETRIEVAL_HYBRID_TEXT_STAGE1_TOP_K",
            hybrid_raw.get("text_stage1_top_k"),
            HybridSearchConfig.text_stage1_top_k,
        ),
        rerank_pool_size=_int_env(
            "RETRIEVAL_HYBRID_RERANK_POOL_SIZE",
            hybrid_raw.get("rerank_pool_size"),
            HybridSearchConfig.rerank_pool_size,
        ),
        default_top_k=_int_env(
            "RETRIEVAL_DEFAULT_TOP_K",
            hybrid_raw.get("default_top_k"),
            HybridSearchConfig.default_top_k,
        ),
        max_top_k=_int_env(
            "RETRIEVAL_MAX_TOP_K",
            hybrid_raw.get("max_top_k"),
            HybridSearchConfig.max_top_k,
        ),
        max_gap_seconds=_float_env(
            "RETRIEVAL_TEMPORAL_MAX_GAP_SECONDS",
            hybrid_raw.get("max_gap_seconds"),
            HybridSearchConfig.max_gap_seconds,
        ),
    )
    weights = RerankWeights(
        visual=_float_env(
            "RETRIEVAL_WEIGHT_VISUAL",
            weights_raw.get("visual"),
            RerankWeights.visual,
        ),
        caption=_float_env(
            "RETRIEVAL_WEIGHT_CAPTION",
            weights_raw.get("caption"),
            RerankWeights.caption,
        ),
        ocr=_float_env(
            "RETRIEVAL_WEIGHT_OCR",
            weights_raw.get("ocr"),
            RerankWeights.ocr,
        ),
        asr=_float_env(
            "RETRIEVAL_WEIGHT_ASR",
            weights_raw.get("asr"),
            RerankWeights.asr,
        ),
        objects=_float_env(
            "RETRIEVAL_WEIGHT_OBJECTS",
            weights_raw.get("objects"),
            RerankWeights.objects,
        ),
        temporal=_float_env(
            "RETRIEVAL_WEIGHT_TEMPORAL",
            weights_raw.get("temporal"),
            RerankWeights.temporal,
        ),
    )
    text_index = TextIndexConfig(
        path=Path(
            os.getenv("RETRIEVAL_TEXT_INDEX_PATH")
            or text_raw.get("path")
            or TextIndexConfig.path
        ),
        default_top_k=_int_env(
            "RETRIEVAL_TEXT_DEFAULT_TOP_K",
            text_raw.get("default_top_k"),
            TextIndexConfig.default_top_k,
        ),
        max_top_k=_int_env(
            "RETRIEVAL_TEXT_MAX_TOP_K",
            text_raw.get("max_top_k"),
            TextIndexConfig.max_top_k,
        ),
    )
    return RetrievalRuntimeConfig(
        hybrid=hybrid,
        rerank=RerankConfig(
            weights=weights,
            dedupe_same_shot=_bool_env(
                "RETRIEVAL_DEDUPE_SAME_SHOT",
                dedupe_raw.get("same_shot"),
                RerankConfig.dedupe_same_shot,
            ),
        ),
        text_index=text_index,
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RetrievalConfigError(
            f"Retrieval config section {name!r} must be a mapping"
        )
    return value


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load standard YAML with PyYAML, then validate the Retrieval schema."""
    text = path.read_text(encoding="utf-8")
    try:
        root_node = yaml.compose(text, Loader=yaml.SafeLoader)
        locations = _collect_config_locations(root_node, path=path)
        loaded = yaml.safe_load(text)
    except RetrievalConfigError:
        raise
    except yaml.YAMLError as exc:
        raise _yaml_parser_error(path, exc) from exc

    if loaded is None:
        root: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        root = loaded
    else:
        line_number = root_node.start_mark.line + 1 if root_node is not None else 1
        raise _yaml_error(
            path,
            line_number,
            f"document root must be a mapping; got {type(loaded).__name__}",
        )

    _validate_config_schema(root, locations=locations, path=path)
    return root


def _collect_config_locations(
    root_node: Node | None,
    *,
    path: Path,
) -> dict[tuple[str, ...], int]:
    """Collect setting lines and reject duplicate mapping keys before loading."""
    locations: dict[tuple[str, ...], int] = {}
    if not isinstance(root_node, MappingNode):
        return locations

    section_lines: dict[str, int] = {}
    for section_key_node, section_node in root_node.value:
        section_name = _config_key_name(section_key_node, path=path)
        if section_name is None:
            continue
        section_line = section_key_node.start_mark.line + 1
        if section_name in section_lines:
            raise _yaml_error(
                path,
                section_line,
                f"duplicate section {section_name!r}; first declared on line "
                f"{section_lines[section_name]}",
            )
        section_lines[section_name] = section_line
        locations[(section_name,)] = section_line

        if not isinstance(section_node, MappingNode):
            continue
        setting_lines: dict[str, int] = {}
        for setting_key_node, _ in section_node.value:
            setting_name = _config_key_name(setting_key_node, path=path)
            if setting_name is None:
                continue
            setting_line = setting_key_node.start_mark.line + 1
            if setting_name in setting_lines:
                raise _yaml_error(
                    path,
                    setting_line,
                    f"duplicate setting {section_name}.{setting_name}; first "
                    f"declared on line {setting_lines[setting_name]}",
                )
            setting_lines[setting_name] = setting_line
            locations[(section_name, setting_name)] = setting_line
    return locations


def _config_key_name(node: Node, *, path: Path) -> str | None:
    if not isinstance(node, ScalarNode):
        raise _yaml_error(
            path,
            node.start_mark.line + 1,
            "config mapping keys must be scalar strings",
        )
    if node.tag == "tag:yaml.org,2002:merge":
        return None
    if node.tag != "tag:yaml.org,2002:str":
        raise _yaml_error(
            path,
            node.start_mark.line + 1,
            f"config mapping keys must be strings; got {node.tag.rsplit(':', 1)[-1]}",
        )
    return node.value


def _validate_config_schema(
    root: dict[str, Any],
    *,
    locations: dict[tuple[str, ...], int],
    path: Path,
) -> None:
    for section_name, section in root.items():
        if not isinstance(section_name, str):
            raise _yaml_error(
                path,
                1,
                f"section names must be strings; got {section_name!r}",
            )
        section_line = locations.get((section_name,), 1)
        section_schema = _CONFIG_SCHEMA.get(section_name)
        if section_schema is None:
            supported = ", ".join(sorted(_CONFIG_SCHEMA))
            raise _yaml_error(
                path,
                section_line,
                f"unknown section {section_name!r}; supported sections: {supported}",
            )
        if not isinstance(section, dict):
            raise _yaml_error(
                path,
                section_line,
                f"section {section_name!r} must be a mapping; got "
                f"{type(section).__name__}",
            )

        for key, value in section.items():
            if not isinstance(key, str):
                raise _yaml_error(
                    path,
                    section_line,
                    f"setting names in {section_name!r} must be strings; got {key!r}",
                )
            expected = section_schema.get(key)
            line_number = locations.get((section_name, key), section_line)
            setting = f"{section_name}.{key}"
            if expected is None:
                supported = ", ".join(sorted(section_schema))
                raise _yaml_error(
                    path,
                    line_number,
                    f"unknown setting {setting}; supported settings in "
                    f"{section_name!r}: {supported}",
                )
            if not _matches_expected_type(value, expected):
                raise _yaml_error(
                    path,
                    line_number,
                    f"{setting} must be a {expected}; got {value!r}",
                )


def _matches_expected_type(value: Any, expected: str) -> bool:
    is_number = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )
    if expected == "positive integer":
        return not isinstance(value, bool) and isinstance(value, int) and value > 0
    if expected == "positive number":
        return is_number and value > 0
    if expected == "non-negative number":
        return is_number and value >= 0
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "non-empty string":
        return isinstance(value, str) and bool(value.strip())
    raise AssertionError(f"Unknown retrieval config schema rule: {expected}")


def _yaml_parser_error(path: Path, error: yaml.YAMLError) -> RetrievalConfigError:
    mark = getattr(error, "problem_mark", None)
    problem = getattr(error, "problem", None)
    detail = problem or str(error).splitlines()[0] or type(error).__name__
    if mark is None:
        return RetrievalConfigError(f"Invalid retrieval YAML at {path}: {detail}")
    return RetrievalConfigError(
        f"Invalid retrieval YAML at {path}:{mark.line + 1}:{mark.column + 1}: "
        f"{detail}"
    )


def _yaml_error(path: Path, line_number: int, message: str) -> RetrievalConfigError:
    return RetrievalConfigError(
        f"Invalid retrieval YAML at {path}:{line_number}: {message}"
    )


def _int_env(name: str, value: Any, default: int) -> int:
    raw = os.getenv(name)
    result = int(raw if raw is not None else value if value is not None else default)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _float_env(name: str, value: Any, default: float) -> float:
    raw = os.getenv(name)
    return float(raw if raw is not None else value if value is not None else default)


def _bool_env(name: str, value: Any, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(value) if value is not None else default
    normalized = raw.casefold().strip()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")
