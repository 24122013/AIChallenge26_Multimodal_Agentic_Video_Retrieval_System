"""Metadata data models — Single Source of Truth cho toàn hệ thống.

Các dataclass ở đây phản ánh chính xác `docs/metadata_schema.md` (v1.1).
Team P3 (Metadata) sở hữu file này. Mọi module sinh/đọc metadata phải dùng
các model này thay vì tự định nghĩa dict rời rạc, để tránh lệch schema.

Nguyên tắc:
- Chỉ mô tả dữ liệu (data holders) + serialize/deserialize + validate cơ bản.
- KHÔNG chứa search logic, rerank, hay agent planning (theo service_boundaries).
- Không phụ thuộc thư viện nặng (torch/pydantic); chỉ dùng stdlib dataclasses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums / hằng số nguồn timestamp
# ---------------------------------------------------------------------------

class TimestampSource(str, Enum):
    """Cách timestamp của một keyframe được xác định (xem metadata_schema.md)."""

    VIDEO_FPS = "video_fps"       # frame_index / fps — chính xác nhất
    MATCHED_FRAME = "matched_frame"  # khớp visual với video gốc
    INTERVAL = "interval"         # ước tính theo khoảng cố định
    UNKNOWN = "unknown"           # không xác định được nguồn


# Độ tin cậy mặc định gắn với từng nguồn timestamp.
TIMESTAMP_CONFIDENCE_BY_SOURCE: dict[str, float] = {
    TimestampSource.VIDEO_FPS.value: 1.0,
    TimestampSource.MATCHED_FRAME.value: 0.9,
    TimestampSource.INTERVAL.value: 0.5,
    TimestampSource.UNKNOWN.value: 0.0,
}


def _clean_str(value: object) -> str:
    return "" if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


# ---------------------------------------------------------------------------
# ID helpers — chuẩn hoá cách tạo ID để mọi pipeline sinh giống nhau
# ---------------------------------------------------------------------------

def make_frame_id(video_id: str, n: int) -> str:
    return f"FRAME_{video_id}_{n:06d}"


def make_shot_id(video_id: str, n: int) -> str:
    return f"SHOT_{video_id}_{n:06d}"


def make_segment_id(video_id: str, n: int) -> str:
    return f"SEG_{video_id}_{n:06d}"


def make_embedding_id(video_id: str, n: int) -> str:
    return f"EMB_{video_id}_{n:06d}"


# ---------------------------------------------------------------------------
# Video / Segment
# ---------------------------------------------------------------------------

@dataclass
class Video:
    """Metadata cấp video."""

    video_id: str
    video_path: str = ""
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Video":
        return cls(
            video_id=_clean_str(data.get("video_id")),
            video_path=_clean_str(data.get("video_path") or data.get("source_video_path")),
            duration=float(data.get("duration") or 0.0),
            fps=float(data.get("fps") or 0.0),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
        )


@dataclass
class Segment:
    """Một đoạn thời gian liên tục của video (dùng cho ASR / temporal)."""

    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    duration: float = 0.0

    def __post_init__(self) -> None:
        if not self.duration:
            self.duration = round(max(0.0, self.end_time - self.start_time), 3)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        start = float(data.get("start_time") or 0.0)
        end = float(data.get("end_time") or 0.0)
        return cls(
            segment_id=_clean_str(data.get("segment_id")),
            video_id=_clean_str(data.get("video_id")),
            start_time=start,
            end_time=end,
            duration=float(data.get("duration") or max(0.0, end - start)),
        )


# ---------------------------------------------------------------------------
# Keyframe — record trung tâm
# ---------------------------------------------------------------------------

@dataclass
class Keyframe:
    """Keyframe record theo schema v1.1.

    Đây là nguồn dữ liệu chính; `frame_map.json` được build từ tập keyframe này.
    """

    # Required
    frame_id: str
    video_id: str
    shot_id: str
    segment_id: str
    timestamp: float
    keyframe_path: str

    # Alias path (backward compat)
    frame_path: str = ""
    thumbnail_path: str = ""

    # Optional nhưng khuyến nghị
    timestamp_source: str = TimestampSource.UNKNOWN.value
    timestamp_confidence: float = 1.0
    frame_index: int | None = None
    embedding_id: str = ""
    embedding_index: int | None = None
    faiss_index: int | None = None

    # Provenance shot (giữ để builder / neighbor dùng)
    shot_start: float | None = None
    shot_end: float | None = None
    shot_index: int | None = None

    def __post_init__(self) -> None:
        if not self.frame_path:
            self.frame_path = self.keyframe_path
        if not self.thumbnail_path:
            self.thumbnail_path = self.keyframe_path

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Keyframe":
        keyframe_path = _clean_str(data.get("keyframe_path") or data.get("frame_path"))
        return cls(
            frame_id=_clean_str(data.get("frame_id")),
            video_id=_clean_str(data.get("video_id")),
            shot_id=_clean_str(data.get("shot_id")),
            segment_id=_clean_str(data.get("segment_id") or data.get("shot_id")),
            timestamp=float(data.get("timestamp") or 0.0),
            keyframe_path=keyframe_path,
            frame_path=_clean_str(data.get("frame_path") or keyframe_path),
            thumbnail_path=_clean_str(data.get("thumbnail_path") or keyframe_path),
            timestamp_source=_clean_str(
                data.get("timestamp_source") or TimestampSource.UNKNOWN.value
            ),
            timestamp_confidence=float(
                data.get("timestamp_confidence")
                if data.get("timestamp_confidence") is not None
                else 1.0
            ),
            frame_index=_optional_int(data.get("frame_index")),
            embedding_id=_clean_str(data.get("embedding_id")),
            embedding_index=_optional_int(data.get("embedding_index")),
            faiss_index=_optional_int(data.get("faiss_index")),
            shot_start=_optional_float(data.get("shot_start")),
            shot_end=_optional_float(data.get("shot_end")),
            shot_index=_optional_int(data.get("shot_index")),
        )


# ---------------------------------------------------------------------------
# Các sidecar metadata: Caption / OCR / ASR / Objects
# ---------------------------------------------------------------------------

@dataclass
class Caption:
    """Caption sinh cho một keyframe."""

    frame_id: str
    caption: str = ""
    caption_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Caption":
        return cls(
            frame_id=_clean_str(data.get("frame_id")),
            caption=_clean_str(data.get("caption")),
            caption_model=_clean_str(data.get("caption_model")),
        )


@dataclass
class OCR:
    """Văn bản OCR trên một keyframe."""

    frame_id: str
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    ocr_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OCR":
        return cls(
            frame_id=_clean_str(data.get("frame_id")),
            ocr_text=_clean_str(data.get("ocr_text")),
            ocr_confidence=float(data.get("ocr_confidence") or 0.0),
            ocr_model=_clean_str(data.get("ocr_model")),
        )


@dataclass
class ASR:
    """Lời thoại (transcript) gắn với một segment thời gian."""

    segment_id: str
    transcript: str = ""
    language: str = ""
    start_time: float | None = None
    end_time: float | None = None
    asr_confidence: float = 0.0
    asr_model: str = ""
    video_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ASR":
        return cls(
            segment_id=_clean_str(data.get("segment_id")),
            transcript=_clean_str(data.get("transcript")),
            language=_clean_str(data.get("language")),
            start_time=_optional_float(data.get("start_time")),
            end_time=_optional_float(data.get("end_time")),
            asr_confidence=float(data.get("asr_confidence") or 0.0),
            asr_model=_clean_str(data.get("asr_model")),
            video_id=_clean_str(data.get("video_id")),
        )


@dataclass
class DetectedObject:
    """Một object được phát hiện trong keyframe."""

    label: str
    confidence: float = 0.0
    bbox: list[float] = field(default_factory=list)  # [x1, y1, x2, y2] optional

    def to_dict(self) -> dict:
        data = {"label": self.label, "confidence": self.confidence}
        if self.bbox:
            data["bbox"] = self.bbox
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DetectedObject":
        return cls(
            label=_clean_str(data.get("label")),
            confidence=float(data.get("confidence") or 0.0),
            bbox=[float(x) for x in (data.get("bbox") or [])],
        )


@dataclass
class ObjectAnnotation:
    """Tập object phát hiện được trên một keyframe."""

    frame_id: str
    objects: list[DetectedObject] = field(default_factory=list)
    object_model: str = ""

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "objects": [obj.to_dict() for obj in self.objects],
            "object_model": self.object_model,
        }

    @property
    def labels(self) -> list[str]:
        """Danh sách nhãn duy nhất, giữ thứ tự xuất hiện."""
        seen: dict[str, None] = {}
        for obj in self.objects:
            if obj.label and obj.label not in seen:
                seen[obj.label] = None
        return list(seen.keys())

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectAnnotation":
        return cls(
            frame_id=_clean_str(data.get("frame_id")),
            objects=[DetectedObject.from_dict(o) for o in (data.get("objects") or [])],
            object_model=_clean_str(data.get("object_model")),
        )


# ---------------------------------------------------------------------------
# Embedding metadata
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingMetadata:
    """Mô tả một embedding vector (khớp với file .npy của indexing team)."""

    embedding_id: str
    frame_id: str
    video_id: str
    model_name: str
    vector_dim: int
    embedding_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingMetadata":
        return cls(
            embedding_id=_clean_str(data.get("embedding_id")),
            frame_id=_clean_str(data.get("frame_id")),
            video_id=_clean_str(data.get("video_id")),
            model_name=_clean_str(data.get("model_name")),
            vector_dim=int(data.get("vector_dim") or 0),
            embedding_index=int(data.get("embedding_index") or 0),
        )


# ---------------------------------------------------------------------------
# Unified metadata record — kết quả gộp của metadata_builder
# ---------------------------------------------------------------------------

@dataclass
class UnifiedMetadataRecord:
    """Record gộp mọi sidecar về một keyframe.

    Đây là "unified retrieval record" mô tả trong metadata_schema.md — retrieval
    và UI đọc từ đây thay vì tự join nhiều file.
    """

    video_id: str
    frame_id: str
    timestamp: float
    segment_id: str = ""
    shot_id: str = ""
    timestamp_source: str = TimestampSource.UNKNOWN.value
    timestamp_confidence: float = 1.0
    frame_index: int | None = None
    faiss_index: int | None = None
    keyframe_path: str = ""
    thumbnail_path: str = ""

    caption: str = ""
    ocr_text: str = ""
    objects: list[str] = field(default_factory=list)
    transcript: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedMetadataRecord":
        return cls(
            video_id=_clean_str(data.get("video_id")),
            frame_id=_clean_str(data.get("frame_id")),
            timestamp=float(data.get("timestamp") or 0.0),
            segment_id=_clean_str(data.get("segment_id")),
            shot_id=_clean_str(data.get("shot_id")),
            timestamp_source=_clean_str(
                data.get("timestamp_source") or TimestampSource.UNKNOWN.value
            ),
            timestamp_confidence=float(
                data.get("timestamp_confidence")
                if data.get("timestamp_confidence") is not None
                else 1.0
            ),
            frame_index=_optional_int(data.get("frame_index")),
            faiss_index=_optional_int(data.get("faiss_index")),
            keyframe_path=_clean_str(data.get("keyframe_path")),
            thumbnail_path=_clean_str(data.get("thumbnail_path") or data.get("keyframe_path")),
            caption=_clean_str(data.get("caption")),
            ocr_text=_clean_str(data.get("ocr_text")),
            objects=[_clean_str(o) for o in (data.get("objects") or [])],
            transcript=_clean_str(data.get("transcript")),
        )


__all__ = [
    "TimestampSource",
    "TIMESTAMP_CONFIDENCE_BY_SOURCE",
    "make_frame_id",
    "make_shot_id",
    "make_segment_id",
    "make_embedding_id",
    "Video",
    "Segment",
    "Keyframe",
    "Caption",
    "OCR",
    "ASR",
    "DetectedObject",
    "ObjectAnnotation",
    "EmbeddingMetadata",
    "UnifiedMetadataRecord",
]
