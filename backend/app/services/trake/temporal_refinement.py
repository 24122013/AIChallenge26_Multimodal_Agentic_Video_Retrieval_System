"""Optional dense local refinement around coarse TRAKE frame lineage.

No corpus-wide dense index is created.  Frames are decoded only inside bounded
windows for the highest-ranked coarse paths.  Without an injected semantic
scorer the module returns the canonical coarse ``frame_index`` with an explicit
warning; it never derives a frame id from a filename or timestamp.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from backend.app.services.retrieval.retrieval_config import TrakeConfig
from backend.app.services.trake.models import BoundaryType, TemporalEvent, TemporalEventPlan, TemporalPath


@dataclass(frozen=True)
class DecodedFrame:
    frame_index: int
    image: Any = field(repr=False, compare=False)


class LocalFrameDecoder(Protocol):
    def decode(
        self,
        video_path: Path,
        *,
        start_frame: int,
        end_frame: int,
        stride: int,
    ) -> Sequence[DecodedFrame]:
        ...


class LocalFrameScorer(Protocol):
    def score(
        self,
        event: TemporalEvent,
        frames: Sequence[DecodedFrame],
    ) -> Sequence[float]:
        ...


class Siglip2EmbeddingEncoder(Protocol):
    """Shared text/image surface exposed by the canonical SigLIP2 encoder."""

    def encode(self, query: str) -> np.ndarray: ...

    def encode_images(
        self,
        images: Sequence[Any],
        *,
        batch_size: int = 16,
    ) -> np.ndarray: ...


class Siglip2LocalFrameScorer:
    """Cheap semantic scorer over bounded, decoded TRAKE frame windows.

    The scorer owns no model.  Production injects the encoder already held by
    the canonical visual search engine, so both text and local image features
    use one lazy SigLIP2 checkpoint and one manifest-compatible vector space.
    """

    def __init__(
        self,
        encoder: Siglip2EmbeddingEncoder,
        *,
        batch_size: int = 16,
    ) -> None:
        if not callable(getattr(encoder, "encode", None)):
            raise TypeError("TRAKE local scorer requires SigLIP2 text encoding")
        if not callable(getattr(encoder, "encode_images", None)):
            raise TypeError("TRAKE local scorer requires SigLIP2 image encoding")
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("TRAKE local scorer batch_size must be positive")
        self.encoder = encoder
        self.batch_size = int(batch_size)
        self._text_cache: dict[str, np.ndarray] = {}
        self._image_cache: dict[int, np.ndarray] = {}
        self._stats: dict[str, int] = {}

    def begin_request(self) -> None:
        self._text_cache.clear()
        self._image_cache.clear()
        self._stats = {
            "text_encode_calls": 0,
            "image_encode_calls": 0,
            "frames_encoded": 0,
            "frame_embedding_cache_hits": 0,
        }

    @property
    def request_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def score(
        self,
        event: TemporalEvent,
        frames: Sequence[DecodedFrame],
    ) -> Sequence[float]:
        semantic, _ = self.score_with_motion(event, frames)
        return semantic

    def score_with_motion(
        self,
        event: TemporalEvent,
        frames: Sequence[DecodedFrame],
    ) -> tuple[list[float], list[float]]:
        if not frames:
            return [], []
        query = " ".join(
            str(event.refinement_query or event.retrieval_query).split()
        )
        if not query:
            raise ValueError("TRAKE local refinement query must not be empty")

        cached_query = self._text_cache.get(query)
        if cached_query is None:
            cached_query = np.asarray(self.encoder.encode(query), dtype=np.float32)
            self._text_cache[query] = cached_query
            self._stats["text_encode_calls"] = self._stats.get("text_encode_calls", 0) + 1
        query_vector = np.asarray(cached_query, dtype=np.float32)
        if query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_vector = query_vector[0]
        if query_vector.ndim != 1 or not np.isfinite(query_vector).all():
            raise ValueError("TRAKE local query embedding must be one finite vector")
        query_norm = float(np.linalg.norm(query_vector))
        if not math.isfinite(query_norm) or query_norm <= 0:
            raise ValueError("TRAKE local query embedding must not be zero")
        query_vector = query_vector / query_norm

        keys = [id(frame.image) for frame in frames]
        missing = [
            (key, _to_rgb_image(frame.image))
            for key, frame in zip(keys, frames)
            if key not in self._image_cache
        ]
        if missing:
            encoded = np.asarray(
                self.encoder.encode_images(
                    [image for _, image in missing],
                    batch_size=self.batch_size,
                ),
                dtype=np.float32,
            )
            if encoded.ndim != 2 or encoded.shape[0] != len(missing):
                raise ValueError("TRAKE local image embeddings must align with frames")
            for (key, _), vector in zip(missing, encoded):
                self._image_cache[key] = vector
            self._stats["image_encode_calls"] = self._stats.get("image_encode_calls", 0) + 1
            self._stats["frames_encoded"] = self._stats.get("frames_encoded", 0) + len(missing)
        self._stats["frame_embedding_cache_hits"] = self._stats.get(
            "frame_embedding_cache_hits", 0
        ) + len(frames) - len(missing)
        image_vectors = np.asarray([self._image_cache[key] for key in keys], dtype=np.float32)
        if image_vectors.ndim != 2 or image_vectors.shape[0] != len(frames):
            raise ValueError("TRAKE local image embeddings must align with frames")
        if image_vectors.shape[1] != query_vector.shape[0]:
            raise ValueError("TRAKE local text/image embedding dimensions differ")
        if not np.isfinite(image_vectors).all():
            raise ValueError("TRAKE local image embeddings must be finite")
        image_norms = np.linalg.norm(image_vectors, axis=1, keepdims=True)
        if not np.isfinite(image_norms).all() or np.any(image_norms <= 0):
            raise ValueError("TRAKE local image embeddings must not be zero")
        image_vectors = image_vectors / image_norms

        # Expose bounded [0, 1] semantic similarities while preserving cosine
        # ordering for the boundary-aware selector.
        scores = np.clip((image_vectors @ query_vector + 1.0) / 2.0, 0.0, 1.0)
        if scores.shape != (len(frames),) or not np.isfinite(scores).all():
            raise ValueError("TRAKE local scorer produced invalid scores")
        deltas = np.zeros(len(frames), dtype=np.float32)
        if len(frames) > 1:
            deltas[1:] = np.clip(
                1.0 - np.sum(image_vectors[1:] * image_vectors[:-1], axis=1),
                0.0,
                1.0,
            )
        return [float(value) for value in scores], [float(value) for value in deltas]


def _to_rgb_image(image: Any) -> Any:
    """Convert default OpenCV BGR/BGRA frames to processor-ready RGB."""

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return np.ascontiguousarray(np.repeat(image[..., None], 3, axis=2))
        if image.ndim != 3 or image.shape[2] not in {3, 4}:
            raise ValueError("TRAKE decoded frame must have 1, 3, or 4 channels")
        return np.ascontiguousarray(image[..., [2, 1, 0]])
    converter = getattr(image, "convert", None)
    if callable(converter):
        return converter("RGB")
    raise TypeError("TRAKE decoded frame must be an OpenCV array or PIL image")


@dataclass(frozen=True)
class LocalFrameHypothesis:
    frame_index: int
    score: float | None
    strategy: str
    confidence: float
    source: str
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "score": self.score,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "source": self.source,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class RefinementVariant:
    coarse_path: TemporalPath
    frame_indices: tuple[int, ...]
    score: float
    event_refinements: tuple[LocalFrameHypothesis, ...]
    warnings: tuple[str, ...] = ()

    @property
    def video_id(self) -> str:
        return self.coarse_path.video_id

    @property
    def path_id(self) -> str:
        return self.coarse_path.path_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_ids": list(self.frame_indices),
            "score": self.score,
            "path_id": self.path_id,
            "event_refinements": [item.to_dict() for item in self.event_refinements],
            "warnings": list(self.warnings),
        }


class OpenCVLocalFrameDecoder:
    """Lazy OpenCV decoder that seeks only to the requested local window."""

    def decode(
        self,
        video_path: Path,
        *,
        start_frame: int,
        end_frame: int,
        stride: int,
    ) -> Sequence[DecodedFrame]:
        if stride <= 0:
            raise ValueError("local decode stride must be positive")
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError("invalid local decode frame range")
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("opencv_unavailable") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("video_decoder_open_failed")
        output: list[DecodedFrame] = []
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_index = start_frame
            while frame_index <= end_frame:
                ok, image = capture.read()
                if not ok:
                    break
                if (frame_index - start_frame) % stride == 0:
                    output.append(DecodedFrame(frame_index=frame_index, image=image))
                frame_index += 1
        finally:
            capture.release()
        return output


class TemporalRefiner:
    """Boundary-aware refiner with dependency-injected decoder and scorer."""

    def __init__(
        self,
        *,
        config: TrakeConfig | None = None,
        video_root: str | Path | None = None,
        decoder: LocalFrameDecoder | None = None,
        scorer: LocalFrameScorer | None = None,
    ) -> None:
        self.config = config or TrakeConfig()
        self.video_root = Path(
            video_root
            or os.getenv("RETRIEVAL_TRAKE_VIDEO_ROOT")
            or "data/raw/video"
        )
        self.decoder = decoder or OpenCVLocalFrameDecoder()
        self.scorer = scorer
        self._frame_cache: dict[tuple[str, int], DecodedFrame] = {}
        self._stats: dict[str, float | int] = {}

    def begin_request(self) -> None:
        self._frame_cache.clear()
        self._stats = {
            "decode_calls": 0,
            "frames_decoded": 0,
            "decoded_frame_cache_hits": 0,
            "frame_decode_ms": 0.0,
            "frame_embedding_ms": 0.0,
        }
        begin = getattr(self.scorer, "begin_request", None)
        if callable(begin):
            begin()

    def end_request(self) -> None:
        return None

    @property
    def request_stats(self) -> dict[str, Any]:
        output = dict(self._stats)
        scorer_stats = getattr(self.scorer, "request_stats", {})
        if isinstance(scorer_stats, dict):
            output.update(scorer_stats)
        return output

    def _decode_cached(
        self,
        video_path: Path,
        video_id: str,
        *,
        start_frame: int,
        end_frame: int,
        stride: int,
    ) -> list[DecodedFrame]:
        requested = list(range(start_frame, end_frame + 1, stride))
        missing = [index for index in requested if (video_id, index) not in self._frame_cache]
        self._stats["decoded_frame_cache_hits"] = int(
            self._stats.get("decoded_frame_cache_hits", 0)
        ) + len(requested) - len(missing)
        ranges: list[tuple[int, int]] = []
        for index in missing:
            if ranges and index == ranges[-1][1] + stride:
                ranges[-1] = (ranges[-1][0], index)
            else:
                ranges.append((index, index))
        for range_start, range_end in ranges:
            started = time.perf_counter()
            decoded = list(
                self.decoder.decode(
                    video_path,
                    start_frame=range_start,
                    end_frame=range_end,
                    stride=stride,
                )
            )
            self._stats["frame_decode_ms"] = round(
                float(self._stats.get("frame_decode_ms", 0.0))
                + (time.perf_counter() - started) * 1000.0,
                3,
            )
            self._stats["decode_calls"] = int(self._stats.get("decode_calls", 0)) + 1
            self._stats["frames_decoded"] = int(self._stats.get("frames_decoded", 0)) + len(decoded)
            for frame in decoded:
                self._frame_cache[(video_id, frame.frame_index)] = frame
        return [
            self._frame_cache[(video_id, index)]
            for index in requested
            if (video_id, index) in self._frame_cache
        ]

    def refine(
        self,
        path: TemporalPath,
        plan: TemporalEventPlan,
    ) -> list[RefinementVariant]:
        if any(
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
            for frame_index in path.frame_ids
        ):
            # There is no trustworthy original-frame identity to preserve.
            # Returning no variant lets ranking drop this path fail-closed.
            return []
        if len(path.event_candidates) != len(plan.events):
            return [_coarse_variant(path, "refinement_event_count_mismatch")]
        if self.scorer is None:
            return [_coarse_variant(path, "local_refinement_scorer_unavailable")]
        try:
            video_path = resolve_video_path(self.video_root, path.video_id)
        except (FileNotFoundError, ValueError):
            return [_coarse_variant(path, "local_refinement_video_unavailable")]

        choices: list[list[LocalFrameHypothesis]] = []
        warnings: list[str] = []
        for event, candidate in zip(plan.events, path.event_candidates):
            coarse = candidate.result.frame_index
            if not isinstance(coarse, int) or isinstance(coarse, bool) or coarse < 0:
                return [_coarse_variant(path, "local_refinement_missing_lineage")]
            start = max(0, coarse - self.config.window_before_frames)
            end = coarse + self.config.window_after_frames
            try:
                frames = self._decode_cached(
                    video_path,
                    path.video_id,
                    start_frame=start,
                    end_frame=end,
                    stride=self.config.dense_stride_frames,
                )
                embedding_started = time.perf_counter()
                score_with_motion = getattr(self.scorer, "score_with_motion", None)
                if callable(score_with_motion):
                    scores, motion_scores = score_with_motion(event, frames)
                    scores = list(scores)
                    motion_scores = list(motion_scores)
                else:
                    scores = list(self.scorer.score(event, frames))
                    motion_scores = None
                self._stats["frame_embedding_ms"] = round(
                    float(self._stats.get("frame_embedding_ms", 0.0))
                    + (time.perf_counter() - embedding_started) * 1000.0,
                    3,
                )
                if not frames or len(scores) != len(frames):
                    raise ValueError("local_score_shape_mismatch")
                if any(not math.isfinite(float(score)) for score in scores):
                    raise ValueError("local_score_not_finite")
                selected = select_local_hypotheses(
                    event,
                    frames,
                    scores,
                    limit=self.config.local_hypotheses_per_event,
                    coarse_frame_index=coarse,
                    motion_scores=motion_scores,
                )
                if not selected:
                    raise ValueError("local_hypothesis_empty")
                choices.append(selected)
            except Exception:
                warning = f"event_{event.index}_local_refinement_fallback"
                warnings.append(warning)
                choices.append([_coarse_local_hypothesis(coarse, warning)])

        variants = _combine_local_choices(path, choices, warnings=tuple(warnings))
        return variants or [_coarse_variant(path, "local_refinement_order_fallback")]


def refine_temporal_path(
    path: TemporalPath,
    plan: TemporalEventPlan,
    *,
    config: TrakeConfig | None = None,
    video_root: str | Path | None = None,
    decoder: LocalFrameDecoder | None = None,
    scorer: LocalFrameScorer | None = None,
) -> list[RefinementVariant]:
    return TemporalRefiner(
        config=config,
        video_root=video_root,
        decoder=decoder,
        scorer=scorer,
    ).refine(path, plan)


def select_local_hypotheses(
    event: TemporalEvent,
    frames: Sequence[DecodedFrame],
    scores: Sequence[float],
    *,
    limit: int,
    coarse_frame_index: int | None = None,
    motion_scores: Sequence[float] | None = None,
    confirmation_frames: int = 2,
) -> list[LocalFrameHypothesis]:
    """Select boundary-aware local frame alternatives from fake or real scores."""

    if limit <= 0:
        raise ValueError("local hypothesis limit must be positive")
    if len(frames) != len(scores):
        raise ValueError("local frames and scores must align")
    if not frames:
        return []
    if motion_scores is not None and len(motion_scores) != len(frames):
        raise ValueError("local motion scores and frames must align")
    if confirmation_frames <= 0:
        raise ValueError("confirmation_frames must be positive")
    pairs = sorted(
        ((frame.frame_index, float(score)) for frame, score in zip(frames, scores)),
        key=lambda item: item[0],
    )
    boundary = (
        event.boundary_type.value
        if isinstance(event.boundary_type, BoundaryType)
        else str(event.boundary_type)
    )
    boundary = boundary.casefold().strip()
    center = coarse_frame_index if coarse_frame_index is not None else pairs[len(pairs) // 2][0]
    low_score = min(score for _, score in pairs)
    high_score = max(score for _, score in pairs)
    span = high_score - low_score
    if span <= 1e-12:
        # A flat scorer provides no evidence for a semantic transition or peak.
        # Preserve the coarse location (or the nearest actually decoded frame)
        # and do not manufacture confident alternatives at a window boundary.
        frame_index, score = min(
            pairs,
            key=lambda item: (abs(item[0] - center), item[0]),
        )
        return [
            LocalFrameHypothesis(
                frame_index=int(frame_index),
                score=round(score, 6),
                strategy="flat_local_score_fallback",
                confidence=0.0,
                source=(
                    "canonical_metadata"
                    if coarse_frame_index is not None and frame_index == coarse_frame_index
                    else "local_refinement"
                ),
                warning="flat_local_score_fallback",
            )
        ]
    strategy = "semantic_top_score"
    ordered: list[tuple[int, float]]

    if boundary == BoundaryType.PEAK.value:
        strategy = "local_peak"
        peaks: list[tuple[int, float]] = []
        for index, pair in enumerate(pairs):
            left = pairs[index - 1][1] if index > 0 else float("-inf")
            right = pairs[index + 1][1] if index + 1 < len(pairs) else float("-inf")
            if pair[1] >= left and pair[1] >= right:
                peaks.append(pair)
        ordered = sorted(peaks or pairs, key=lambda item: (-item[1], abs(item[0] - center), item[0]))
    elif boundary in {
        BoundaryType.FIRST_CONTACT.value,
        BoundaryType.FIRST_TRANSITION.value,
        BoundaryType.FIRST_LEAVE.value,
    }:
        strategy = (
            "first_fully_left_state"
            if boundary == BoundaryType.FIRST_LEAVE.value
            else "first_positive_transition"
        )
        values = [score for _, score in pairs]
        low, high = min(values), max(values)
        threshold = low + 0.55 * (high - low)
        motion_by_frame = {
            frame.frame_index: float(value)
            for frame, value in zip(frames, motion_scores or [0.0] * len(frames))
        }
        max_motion = max(motion_by_frame.values(), default=0.0)
        motion_threshold = 0.35 * max_motion
        crossings = []
        for index, pair in enumerate(pairs):
            confirmed = all(
                pairs[future][1] >= threshold
                for future in range(index, min(len(pairs), index + confirmation_frames))
            ) and min(len(pairs), index + confirmation_frames) - index >= confirmation_frames
            motion_ok = motion_scores is None or (
                max_motion > 1e-6 and motion_by_frame[pair[0]] >= motion_threshold
            )
            if (
                pair[1] >= threshold
                and (index == 0 or pairs[index - 1][1] < threshold)
                and confirmed
                and motion_ok
            ):
                crossings.append(pair)
        if not crossings and motion_scores is not None:
            fallback_frame, fallback_score = min(
                pairs,
                key=lambda item: (abs(item[0] - center), item[0]),
            )
            return [
                LocalFrameHypothesis(
                    frame_index=int(fallback_frame),
                    score=round(fallback_score, 6),
                    strategy="insufficient_temporal_signal_fallback",
                    confidence=0.0,
                    source="canonical_metadata",
                    warning="insufficient_temporal_boundary_signal",
                )
            ]
        primary = min(crossings, key=lambda item: item[0]) if crossings else max(
            pairs,
            key=lambda item: (item[1], -item[0]),
        )
        remaining = [pair for pair in pairs if pair[0] != primary[0]]
        remaining.sort(key=lambda item: (abs(item[0] - primary[0]), -item[1], item[0]))
        ordered = [primary, *remaining]
    else:
        ordered = sorted(pairs, key=lambda item: (-item[1], abs(item[0] - center), item[0]))

    output: list[LocalFrameHypothesis] = []
    seen: set[int] = set()
    for frame_index, score in ordered:
        if frame_index in seen:
            continue
        seen.add(frame_index)
        confidence = (score - low_score) / span
        output.append(
            LocalFrameHypothesis(
                frame_index=int(frame_index),
                score=round(score, 6),
                strategy=strategy,
                confidence=round(max(0.0, min(1.0, confidence)), 6),
                source="local_refinement",
            )
        )
        if len(output) >= limit:
            break
    return output


def resolve_video_path(video_root: str | Path, video_id: str) -> Path:
    """Resolve ``<root>/<video_id>.mp4`` without accepting path traversal."""

    root = Path(video_root).resolve()
    raw_id = str(video_id).strip()
    if (
        not raw_id
        or raw_id in {".", ".."}
        or Path(raw_id).name != raw_id
        or "/" in raw_id
        or "\\" in raw_id
        or Path(raw_id).suffix
    ):
        raise ValueError("unsafe_video_id")
    candidate = (root / f"{raw_id}.mp4").resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("unsafe_video_id") from exc
    if not candidate.is_file():
        raise FileNotFoundError("canonical_video_not_found")
    return candidate


def _combine_local_choices(
    path: TemporalPath,
    choices: Sequence[Sequence[LocalFrameHypothesis]],
    *,
    warnings: tuple[str, ...],
) -> list[RefinementVariant]:
    # Bounded Cartesian beam.  Keeping a small number of partial variants avoids
    # exponential growth while still exposing independent local alternatives.
    width = max(1, min(100, len(choices) * max(len(items) for items in choices)))
    states: list[tuple[tuple[LocalFrameHypothesis, ...], float]] = [((), 0.0)]
    for event_choices in choices:
        expanded: list[tuple[tuple[LocalFrameHypothesis, ...], float]] = []
        for selected, confidence_sum in states:
            for choice in event_choices:
                if selected and choice.frame_index <= selected[-1].frame_index:
                    continue
                expanded.append(((*selected, choice), confidence_sum + choice.confidence))
        expanded.sort(
            key=lambda item: (
                -(item[1] / len(item[0])),
                tuple(choice.frame_index for choice in item[0]),
            )
        )
        states = expanded[:width]
        if not states:
            return []

    output: list[RefinementVariant] = []
    for selected, confidence_sum in states:
        has_local_score = any(item.score is not None for item in selected)
        local_confidence = confidence_sum / len(selected)
        score = (
            0.85 * path.score + 0.15 * local_confidence
            if has_local_score
            else path.score
        )
        output.append(
            RefinementVariant(
                coarse_path=path,
                frame_indices=tuple(item.frame_index for item in selected),
                score=round(float(score), 6),
                event_refinements=selected,
                warnings=tuple(
                    dict.fromkeys(
                        (*path.warnings, *warnings, *(item.warning for item in selected if item.warning))
                    )
                ),
            )
        )
    output.sort(key=lambda item: (-item.score, item.frame_indices, item.video_id))
    return output


def _coarse_local_hypothesis(frame_index: int, warning: str) -> LocalFrameHypothesis:
    return LocalFrameHypothesis(
        frame_index=int(frame_index),
        score=None,
        strategy="coarse_frame_fallback",
        confidence=0.0,
        source="canonical_metadata",
        warning=warning,
    )


def _coarse_variant(path: TemporalPath, warning: str) -> RefinementVariant:
    if any(
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
        for frame_index in path.frame_ids
    ):
        raise ValueError("coarse TRAKE path has no original-frame lineage")
    frame_indices = tuple(int(frame_index) for frame_index in path.frame_ids)
    return RefinementVariant(
        coarse_path=path,
        frame_indices=frame_indices,
        score=float(path.score),
        event_refinements=tuple(
            _coarse_local_hypothesis(frame_index, warning)
            for frame_index in frame_indices
        ),
        warnings=tuple(dict.fromkeys((*path.warnings, warning))),
    )


__all__ = [
    "DecodedFrame",
    "LocalFrameDecoder",
    "LocalFrameHypothesis",
    "LocalFrameScorer",
    "OpenCVLocalFrameDecoder",
    "RefinementVariant",
    "Siglip2EmbeddingEncoder",
    "Siglip2LocalFrameScorer",
    "TemporalRefiner",
    "refine_temporal_path",
    "resolve_video_path",
    "select_local_hypotheses",
]
