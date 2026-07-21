"""scheme_validator — kiểm tra record metadata có đúng schema v1.1 không.

Team P3 dùng module này như "cổng gác": mọi sidecar (caption/ocr/asr/objects)
và keyframe metadata phải pass validate trước khi coi là hợp lệ. Không chứa
search logic — chỉ kiểm tra cấu trúc dữ liệu.

Usage:
    from backend.app.services.ingestion.scheme_validator import validate_keyframe
    result = validate_keyframe(record)
    if not result.valid:
        print(result.errors)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.models.metadata import TimestampSource

VALID_TIMESTAMP_SOURCES = {s.value for s in TimestampSource}


@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge_prefixed(self, other: "ValidationResult", prefix: str) -> None:
        if not other.valid:
            self.valid = False
        self.errors.extend(f"{prefix}: {e}" for e in other.errors)
        self.warnings.extend(f"{prefix}: {w}" for w in other.warnings)

    def to_dict(self) -> dict:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


# ---------------------------------------------------------------------------
# Helper kiểm tra field
# ---------------------------------------------------------------------------

def _require(res: ValidationResult, data: dict, key: str, types: tuple) -> None:
    if key not in data or data[key] in (None, ""):
        res.error(f"thiếu field bắt buộc '{key}'")
        return
    if not isinstance(data[key], types):
        res.error(
            f"field '{key}' sai kiểu: cần {'/'.join(t.__name__ for t in types)}, "
            f"got {type(data[key]).__name__}"
        )


def _check_id_prefix(res: ValidationResult, data: dict, key: str, prefix: str) -> None:
    value = data.get(key)
    if isinstance(value, str) and value and not value.startswith(prefix):
        res.warn(f"'{key}'={value!r} không bắt đầu bằng '{prefix}'")


# ---------------------------------------------------------------------------
# Validators theo từng loại record
# ---------------------------------------------------------------------------

def validate_video(data: dict) -> ValidationResult:
    res = ValidationResult()
    _require(res, data, "video_id", (str,))
    for numeric in ("duration", "fps"):
        if numeric in data and not isinstance(data[numeric], (int, float)):
            res.error(f"'{numeric}' phải là số")
    return res


def validate_segment(data: dict) -> ValidationResult:
    res = ValidationResult()
    _require(res, data, "segment_id", (str,))
    _require(res, data, "video_id", (str,))
    _require(res, data, "start_time", (int, float))
    _require(res, data, "end_time", (int, float))
    _check_id_prefix(res, data, "segment_id", "SEG_")
    if isinstance(data.get("start_time"), (int, float)) and isinstance(
        data.get("end_time"), (int, float)
    ):
        if data["end_time"] < data["start_time"]:
            res.error("end_time < start_time")
    return res


def validate_keyframe(data: dict) -> ValidationResult:
    """Kiểm tra keyframe record (nguồn cho frame_map)."""
    res = ValidationResult()
    _require(res, data, "frame_id", (str,))
    _require(res, data, "video_id", (str,))
    _require(res, data, "shot_id", (str,))
    _require(res, data, "segment_id", (str,))
    _require(res, data, "timestamp", (int, float))
    _require(res, data, "keyframe_path", (str,))

    _check_id_prefix(res, data, "frame_id", "FRAME_")
    _check_id_prefix(res, data, "shot_id", "SHOT_")

    source = data.get("timestamp_source")
    if source is not None and source not in VALID_TIMESTAMP_SOURCES:
        res.warn(
            f"timestamp_source={source!r} không thuộc "
            f"{sorted(VALID_TIMESTAMP_SOURCES)}"
        )
    conf = data.get("timestamp_confidence")
    if conf is not None and isinstance(conf, (int, float)) and not (0.0 <= conf <= 1.0):
        res.warn(f"timestamp_confidence={conf} nằm ngoài [0,1]")
    return res


def validate_caption(data: dict) -> ValidationResult:
    res = ValidationResult()
    _require(res, data, "frame_id", (str,))
    if "caption" not in data:
        res.error("thiếu field 'caption'")
    elif not isinstance(data["caption"], str):
        res.error("'caption' phải là str")
    elif not data["caption"].strip():
        res.warn("caption rỗng")
    return res


def validate_ocr(data: dict) -> ValidationResult:
    res = ValidationResult()
    _require(res, data, "frame_id", (str,))
    if "ocr_text" not in data:
        res.error("thiếu field 'ocr_text'")
    elif not isinstance(data["ocr_text"], str):
        res.error("'ocr_text' phải là str")
    conf = data.get("ocr_confidence")
    if conf is not None and isinstance(conf, (int, float)) and not (0.0 <= conf <= 1.0):
        res.warn(f"ocr_confidence={conf} ngoài [0,1]")
    return res


def validate_asr(data: dict) -> ValidationResult:
    res = ValidationResult()
    _require(res, data, "segment_id", (str,))
    if "transcript" not in data:
        res.error("thiếu field 'transcript'")
    elif not isinstance(data["transcript"], str):
        res.error("'transcript' phải là str")
    _check_id_prefix(res, data, "segment_id", "SEG_")
    return res


def validate_objects(data: dict) -> ValidationResult:
    res = ValidationResult()
    _require(res, data, "frame_id", (str,))
    objects = data.get("objects")
    if objects is None:
        res.error("thiếu field 'objects'")
    elif not isinstance(objects, list):
        res.error("'objects' phải là list")
    else:
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                res.error(f"objects[{i}] phải là dict")
                continue
            if "label" not in obj or not obj.get("label"):
                res.error(f"objects[{i}] thiếu 'label'")
            conf = obj.get("confidence")
            if conf is not None and isinstance(conf, (int, float)) and not (0.0 <= conf <= 1.0):
                res.warn(f"objects[{i}].confidence={conf} ngoài [0,1]")
    return res


def validate_embedding(data: dict) -> ValidationResult:
    res = ValidationResult()
    for key in ("embedding_id", "frame_id", "video_id", "model_name"):
        _require(res, data, key, (str,))
    _require(res, data, "vector_dim", (int,))
    _require(res, data, "embedding_index", (int,))
    return res


# ---------------------------------------------------------------------------
# Batch validate
# ---------------------------------------------------------------------------

_VALIDATORS = {
    "video": validate_video,
    "segment": validate_segment,
    "keyframe": validate_keyframe,
    "caption": validate_caption,
    "ocr": validate_ocr,
    "asr": validate_asr,
    "objects": validate_objects,
    "embedding": validate_embedding,
}


def validate_record(kind: str, data: dict) -> ValidationResult:
    """Validate một record theo `kind` (video/segment/keyframe/caption/...)."""
    validator = _VALIDATORS.get(kind)
    if validator is None:
        res = ValidationResult()
        res.error(f"kind không hỗ trợ: {kind!r}")
        return res
    return validator(data)


def validate_records(kind: str, rows: list[dict]) -> dict:
    """Validate cả list, trả summary (count valid/invalid + lỗi mẫu)."""
    total = len(rows)
    invalid: list[dict] = []
    warning_count = 0
    for i, row in enumerate(rows):
        result = validate_record(kind, row)
        warning_count += len(result.warnings)
        if not result.valid:
            invalid.append({"index": i, "errors": result.errors})

    return {
        "kind": kind,
        "total": total,
        "valid_count": total - len(invalid),
        "invalid_count": len(invalid),
        "warning_count": warning_count,
        "invalid_samples": invalid[:20],
    }


__all__ = [
    "ValidationResult",
    "validate_video",
    "validate_segment",
    "validate_keyframe",
    "validate_caption",
    "validate_ocr",
    "validate_asr",
    "validate_objects",
    "validate_embedding",
    "validate_record",
    "validate_records",
]
