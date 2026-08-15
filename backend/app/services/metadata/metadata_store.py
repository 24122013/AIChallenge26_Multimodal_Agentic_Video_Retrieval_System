"""MetadataStore — Single entry point cho toàn bộ metadata hệ thống.

Dùng để:
- Load frame_map.json (metadata chính cho retrieval)
- Lookup frame_id / video_id / timestamp / path từ faiss_index
- Cung cấp iterator và query API dùng chung cho retrieval, UI, eval

Usage:
    store = MetadataStore.from_frame_map("data/metadata/openclip_vit_b16_frame_map.json")
    record = store.get_by_faiss_index(42)
    frames = store.get_by_video_id("L01_V001")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """Một keyframe record chuẩn hóa từ frame_map.json."""

    faiss_index: int
    frame_id: str
    video_id: str
    timestamp: float
    keyframe_path: str

    # Optional fields — có thể None nếu metadata chưa đầy đủ
    shot_id: str = ""
    segment_id: str = ""
    frame_index: int | None = None
    thumbnail_path: str = ""
    embedding_id: str = ""
    embedding_index: int | None = None
    shot_start: float | None = None
    shot_end: float | None = None
    shot_index: int | None = None
    model_name: str = ""
    model_revision: str = ""
    vector_dim: int | None = None

    # Timestamp provenance — bổ sung v1.1
    timestamp_source: str = "unknown"
    timestamp_confidence: float = 1.0

    # Optional multimodal metadata. Production frame maps may omit these fields;
    # text indexes can still provide them as independent candidate sources.
    caption: str = ""
    ocr_text: str = ""
    objects: list[str] = field(default_factory=list)

    # Phase 4 keyframe-selection provenance propagated from the FAISS frame map.
    candidate_id: str = ""
    candidate_index: int | None = None
    candidate_reasons: list[str] = field(default_factory=list)
    keyframe_strategy: str = ""
    selection_phase: str = ""
    selection_rank: int | None = None
    selection_reasons: list[str] = field(default_factory=list)
    covered_event_ids: list[str] = field(default_factory=list)
    selection_score: float | None = None
    protected: bool = False
    coverage_added: bool = False
    importance_score: float | None = None
    semantic_novelty: float | None = None
    component_scores: dict[str, float] = field(default_factory=dict)
    available_modalities: list[str] = field(default_factory=list)
    protected_event_ids: list[str] = field(default_factory=list)
    selection_provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.thumbnail_path:
            self.thumbnail_path = self.keyframe_path

    @classmethod
    def from_dict(cls, faiss_index: int, data: dict) -> "FrameRecord":
        """Tạo FrameRecord từ một entry trong frame_map.json."""
        return cls(
            faiss_index=faiss_index,
            frame_id=data.get("frame_id") or "",
            video_id=data.get("video_id") or "",
            timestamp=float(data.get("timestamp") or 0.0),
            keyframe_path=data.get("keyframe_path") or data.get("frame_path") or "",
            shot_id=data.get("shot_id") or "",
            segment_id=data.get("segment_id") or "",
            frame_index=data.get("frame_index"),
            thumbnail_path=data.get("thumbnail_path") or data.get("keyframe_path") or "",
            embedding_id=data.get("embedding_id") or "",
            embedding_index=data.get("embedding_index"),
            shot_start=_optional_float(data.get("shot_start")),
            shot_end=_optional_float(data.get("shot_end")),
            shot_index=data.get("shot_index"),
            model_name=data.get("model_name") or "",
            model_revision=data.get("model_revision") or "",
            vector_dim=_optional_int(data.get("vector_dim")),
            timestamp_source=data.get("timestamp_source") or _infer_timestamp_source(data),
            timestamp_confidence=float(
                data.get("timestamp_confidence") if data.get("timestamp_confidence") is not None
                else _infer_timestamp_confidence(data)
            ),
            caption=str(data.get("caption") or data.get("segment_caption") or ""),
            ocr_text=str(data.get("ocr_text") or _joined_text(data.get("ocr"))),
            objects=_object_labels(data.get("objects") or data.get("object_classes")),
            candidate_id=str(data.get("candidate_id") or ""),
            candidate_index=_optional_int(data.get("candidate_index")),
            candidate_reasons=_string_list(data.get("candidate_reasons")),
            keyframe_strategy=str(data.get("keyframe_strategy") or ""),
            selection_phase=str(data.get("selection_phase") or ""),
            selection_rank=_optional_int(data.get("selection_rank")),
            selection_reasons=_string_list(data.get("selection_reasons")),
            covered_event_ids=_string_list(data.get("covered_event_ids")),
            selection_score=_optional_float(data.get("selection_score")),
            protected=_bool_value(data.get("protected")),
            coverage_added=_bool_value(data.get("coverage_added")),
            importance_score=_optional_float(data.get("importance_score")),
            semantic_novelty=_optional_float(data.get("semantic_novelty")),
            component_scores=_float_mapping(data.get("component_scores")),
            available_modalities=_string_list(data.get("available_modalities")),
            protected_event_ids=_string_list(data.get("protected_event_ids")),
            selection_provenance=_object_mapping(data.get("selection_provenance")),
        )

    def to_dict(self) -> dict:
        return {
            "faiss_index": self.faiss_index,
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "segment_id": self.segment_id,
            "timestamp": self.timestamp,
            "timestamp_source": self.timestamp_source,
            "timestamp_confidence": self.timestamp_confidence,
            "frame_index": self.frame_index,
            "keyframe_path": self.keyframe_path,
            "thumbnail_path": self.thumbnail_path,
            "embedding_id": self.embedding_id,
            "embedding_index": self.embedding_index,
            "shot_start": self.shot_start,
            "shot_end": self.shot_end,
            "shot_index": self.shot_index,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "vector_dim": self.vector_dim,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "objects": list(self.objects),
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "candidate_reasons": list(self.candidate_reasons),
            "keyframe_strategy": self.keyframe_strategy,
            "selection_phase": self.selection_phase,
            "selection_rank": self.selection_rank,
            "selection_reasons": list(self.selection_reasons),
            "covered_event_ids": list(self.covered_event_ids),
            "selection_score": self.selection_score,
            "protected": self.protected,
            "coverage_added": self.coverage_added,
            "importance_score": self.importance_score,
            "semantic_novelty": self.semantic_novelty,
            "component_scores": dict(self.component_scores),
            "available_modalities": list(self.available_modalities),
            "protected_event_ids": list(self.protected_event_ids),
            "selection_provenance": dict(self.selection_provenance),
        }


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _string_list(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        return [str(value)]
    return [str(item) for item in value if item is not None and str(item)]


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw_score in value.items():
        try:
            result[str(key)] = float(raw_score)
        except (TypeError, ValueError):
            continue
    return result


def _object_mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _bool_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def _joined_text(value: object) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value)
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = item.get("text") or item.get("ocr_text")
            if text:
                parts.append(str(text))
        elif item:
            parts.append(str(item))
    return " ".join(parts)


def _object_labels(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, list):
        return [str(value)]
    labels: list[str] = []
    for item in value:
        if isinstance(item, dict):
            label = item.get("label") or item.get("class_name") or item.get("class")
            if label:
                labels.append(str(label))
        elif item:
            labels.append(str(item))
    return list(dict.fromkeys(labels))


def _infer_timestamp_source(data: dict) -> str:
    """Tự động suy ra timestamp_source nếu field bị thiếu."""
    if data.get("frame_index") is not None:
        return "video_fps"
    if data.get("timestamp") is not None:
        return "interval"
    return "unknown"


def _infer_timestamp_confidence(data: dict) -> float:
    """Tự động suy ra timestamp_confidence nếu field bị thiếu."""
    source = _infer_timestamp_source(data)
    return {
        "video_fps": 1.0,
        "matched_frame": 0.9,
        "interval": 0.5,
        "unknown": 0.0,
    }.get(source, 0.5)


# ---------------------------------------------------------------------------
# MetadataStore
# ---------------------------------------------------------------------------

@dataclass
class MetadataStore:
    """Store trung tâm cho metadata keyframe.

    Cung cấp O(1) lookup theo faiss_index, frame_id, và video_id.
    """

    _by_faiss_index: dict[int, FrameRecord] = field(default_factory=dict, repr=False)
    _by_frame_id: dict[str, FrameRecord] = field(default_factory=dict, repr=False)
    _by_video_id: dict[str, list[FrameRecord]] = field(default_factory=dict, repr=False)
    _source_path: str = ""

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_frame_map(cls, frame_map_path: str | Path) -> "MetadataStore":
        """Load từ frame_map.json (metadata chính).

        frame_map.json format:
            { "<faiss_index>": { "frame_id": ..., "video_id": ..., ... }, ... }

        Args:
            frame_map_path: Đường dẫn tới frame_map.json

        Returns:
            MetadataStore đã load đầy đủ

        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu JSON không đúng format
        """
        path = Path(frame_map_path)
        if not path.exists():
            raise FileNotFoundError(f"frame_map.json not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw: dict = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"frame_map.json phải là dict, got {type(raw).__name__}")

        store = cls(_source_path=str(path))
        missing_fields_count = 0

        for key, data in raw.items():
            try:
                faiss_index = int(key)
            except (ValueError, TypeError):
                logger.warning("Bỏ qua key không hợp lệ trong frame_map: %r", key)
                continue

            if not isinstance(data, dict):
                logger.warning("Bỏ qua entry không phải dict tại faiss_index=%s", key)
                continue

            # Kiểm tra và bổ sung timestamp_source / timestamp_confidence nếu thiếu
            if "timestamp_source" not in data or not data["timestamp_source"]:
                data["timestamp_source"] = _infer_timestamp_source(data)
                missing_fields_count += 1
            if "timestamp_confidence" not in data or data["timestamp_confidence"] is None:
                data["timestamp_confidence"] = _infer_timestamp_confidence(data)

            record = FrameRecord.from_dict(faiss_index, data)
            store._by_faiss_index[faiss_index] = record

            if record.frame_id:
                store._by_frame_id[record.frame_id] = record

            if record.video_id:
                store._by_video_id.setdefault(record.video_id, []).append(record)

        if missing_fields_count:
            logger.info(
                "Đã tự động bổ sung timestamp_source cho %d records (thiếu field)",
                missing_fields_count,
            )

        # Sắp xếp theo timestamp trong mỗi video
        for records in store._by_video_id.values():
            records.sort(key=lambda r: r.timestamp)

        logger.info(
            "MetadataStore loaded: %d frames, %d videos từ %s",
            len(store._by_faiss_index),
            len(store._by_video_id),
            path,
        )
        return store

    @classmethod
    def from_jsonl(cls, jsonl_path: str | Path) -> "MetadataStore":
        """Load từ JSONL metadata file (backup / indexing metadata).

        Mỗi dòng là một keyframe record có chứa `faiss_index`.

        Args:
            jsonl_path: Đường dẫn tới file .jsonl

        Returns:
            MetadataStore đã load đầy đủ
        """
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL metadata not found: {path}")

        store = cls(_source_path=str(path))
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data: dict = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("JSON parse error at line %d: %s", line_no, exc)
                    continue

                faiss_index = data.get("faiss_index")
                if faiss_index is None:
                    logger.debug("Dòng %d thiếu faiss_index, bỏ qua", line_no)
                    continue

                record = FrameRecord.from_dict(int(faiss_index), data)
                store._by_faiss_index[int(faiss_index)] = record
                if record.frame_id:
                    store._by_frame_id[record.frame_id] = record
                if record.video_id:
                    store._by_video_id.setdefault(record.video_id, []).append(record)

        for records in store._by_video_id.values():
            records.sort(key=lambda r: r.timestamp)

        logger.info(
            "MetadataStore loaded from JSONL: %d frames, %d videos",
            len(store._by_faiss_index),
            len(store._by_video_id),
        )
        return store

    # ------------------------------------------------------------------
    # Primary retrieval API — gọi sau khi có kết quả từ faiss_index
    # ------------------------------------------------------------------

    def get_by_faiss_index(self, faiss_index: int) -> FrameRecord | None:
        """Lookup bởi FAISS index — dùng sau khi search FAISS.

        Args:
            faiss_index: Vị trí trong FAISS index

        Returns:
            FrameRecord hoặc None nếu không tìm thấy
        """
        return self._by_faiss_index.get(faiss_index)

    def get_many_by_faiss_index(self, faiss_indices: list[int]) -> list[FrameRecord | None]:
        """Batch lookup cho nhiều FAISS indices cùng lúc.

        Args:
            faiss_indices: Danh sách FAISS indices (thường từ top-k search)

        Returns:
            List FrameRecord theo cùng thứ tự, None nếu index không tìm thấy
        """
        return [self._by_faiss_index.get(i) for i in faiss_indices]

    def get_by_frame_id(self, frame_id: str) -> FrameRecord | None:
        """Lookup bởi frame_id."""
        return self._by_frame_id.get(frame_id)

    def get_by_video_id(self, video_id: str) -> list[FrameRecord]:
        """Lấy tất cả frames của một video, sorted theo timestamp.

        Args:
            video_id: ID video

        Returns:
            List FrameRecord sorted theo timestamp, [] nếu không có
        """
        return self._by_video_id.get(video_id, [])

    def get_neighbor_frames(
        self,
        faiss_index: int,
        window: int = 5,
    ) -> list[FrameRecord]:
        """Lấy các frames lân cận (trước/sau) theo faiss_index.

        Dùng cho UI hiển thị neighbor frames quanh một kết quả.

        Args:
            faiss_index: FAISS index trung tâm
            window: Số frames lấy về mỗi phía

        Returns:
            List FrameRecord lân cận (không bao gồm frame trung tâm)
        """
        center = self._by_faiss_index.get(faiss_index)
        if center is None:
            return []

        video_frames = self.get_by_video_id(center.video_id)
        result = []
        for frame in video_frames:
            if frame.faiss_index == faiss_index:
                continue
            if abs(frame.timestamp - center.timestamp) <= window * 2.0:
                result.append(frame)
        return sorted(result, key=lambda r: r.timestamp)

    def get_same_shot_neighbors(
        self,
        faiss_index: int,
        max_neighbors: int = 4,
    ) -> list[FrameRecord]:
        """Return nearby keyframes from the same shot as the retrieval hit.

        New extraction metadata contains stable shot fields. Older metadata can
        still use the timestamp-window fallback, which keeps existing indexes
        usable while the team rebuilds artifacts.
        """
        center = self._by_faiss_index.get(faiss_index)
        if center is None:
            return []

        video_frames = self.get_by_video_id(center.video_id)
        if center.shot_id:
            candidates = [
                frame
                for frame in video_frames
                if frame.faiss_index != center.faiss_index
                and frame.shot_id == center.shot_id
            ]
        elif center.shot_start is not None and center.shot_end is not None:
            candidates = [
                frame
                for frame in video_frames
                if frame.faiss_index != center.faiss_index
                and center.shot_start <= frame.timestamp <= center.shot_end
            ]
        else:
            candidates = self.get_neighbor_frames(faiss_index=faiss_index, window=2)

        candidates = sorted(candidates, key=lambda frame: abs(frame.timestamp - center.timestamp))
        candidates = candidates[:max(0, max_neighbors)]
        return sorted(candidates, key=lambda frame: frame.timestamp)

    # ------------------------------------------------------------------
    # Stats & validation helpers
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        return len(self._by_faiss_index)

    @property
    def total_videos(self) -> int:
        return len(self._by_video_id)

    @property
    def video_ids(self) -> list[str]:
        return sorted(self._by_video_id.keys())

    @property
    def faiss_indices(self) -> list[int]:
        return sorted(self._by_faiss_index.keys())

    def has_faiss_index(self, faiss_index: int) -> bool:
        return faiss_index in self._by_faiss_index

    def iter_records(self) -> Iterator[FrameRecord]:
        """Iterator qua tất cả records, sorted theo faiss_index."""
        for idx in sorted(self._by_faiss_index):
            yield self._by_faiss_index[idx]

    def validate_against_faiss(self, faiss_ntotal: int) -> dict:
        """Kiểm tra frame_map có khớp với FAISS index không.

        Args:
            faiss_ntotal: Số vectors trong FAISS index (index.ntotal)

        Returns:
            dict với keys: valid (bool), errors (list), warnings (list), stats (dict)
        """
        errors = []
        warnings = []
        stored_count = self.total_frames

        # Kiểm tra số lượng khớp
        if stored_count != faiss_ntotal:
            errors.append(
                f"Số lượng không khớp: frame_map có {stored_count} records "
                f"nhưng FAISS index có {faiss_ntotal} vectors"
            )

        # Kiểm tra faiss_indices liên tục từ 0..N-1
        expected_indices = set(range(faiss_ntotal))
        actual_indices = set(self._by_faiss_index.keys())
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices

        if missing:
            sample = sorted(missing)[:10]
            errors.append(f"Thiếu {len(missing)} faiss_indices trong frame_map: {sample}...")

        if extra:
            sample = sorted(extra)[:10]
            warnings.append(f"frame_map có {len(extra)} faiss_indices ngoài range FAISS: {sample}...")

        # Kiểm tra required fields
        missing_fields_records = []
        for idx, record in self._by_faiss_index.items():
            if not record.frame_id:
                missing_fields_records.append((idx, "frame_id"))
            if not record.video_id:
                missing_fields_records.append((idx, "video_id"))
            if not record.keyframe_path:
                missing_fields_records.append((idx, "keyframe_path"))
            if record.timestamp is None:
                missing_fields_records.append((idx, "timestamp"))

        if missing_fields_records:
            errors.append(
                f"{len(missing_fields_records)} records thiếu required fields: "
                f"{missing_fields_records[:5]}"
            )

        # Timestamp source summary
        source_counts: dict[str, int] = {}
        for r in self._by_faiss_index.values():
            source_counts[r.timestamp_source] = source_counts.get(r.timestamp_source, 0) + 1

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "frame_map_count": stored_count,
                "faiss_ntotal": faiss_ntotal,
                "missing_indices": len(missing),
                "extra_indices": len(extra),
                "timestamp_source_counts": source_counts,
            },
        }

    def summary(self) -> dict:
        """Summary stats cho logging / debugging."""
        source_counts: dict[str, int] = {}
        for r in self._by_faiss_index.values():
            source_counts[r.timestamp_source] = source_counts.get(r.timestamp_source, 0) + 1

        return {
            "source_path": self._source_path,
            "total_frames": self.total_frames,
            "total_videos": self.total_videos,
            "timestamp_source_counts": source_counts,
        }

    def __repr__(self) -> str:
        return (
            f"MetadataStore(frames={self.total_frames}, "
            f"videos={self.total_videos}, "
            f"source={self._source_path!r})"
        )
