"""Canonical orchestration for online KIS, AVS, temporal, and QA queries."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.app.services.agent.query_expansion import QueryExpansionProvider
from backend.app.services.retrieval.hybrid_search import HybridSearchEngine
from backend.app.services.retrieval.online_context import OnlineContextIndex
from backend.app.services.retrieval.planned_hybrid import planned_hybrid_search
from backend.app.services.retrieval.qa_evidence import QaEvidenceSearchEngine
from backend.app.services.retrieval.qa_pipeline import QaSearchPipeline
from backend.app.services.retrieval.query_plan import QueryPlan, build_query_plan
from backend.app.services.retrieval.retrieval_config import RetrievalRuntimeConfig


ONLINE_SCHEMA_VERSION = "1.0"
SUPPORTED_ONLINE_TASKS = ("auto", "kis", "avs", "temporal", "qa")


@dataclass(frozen=True)
class OnlinePipelineConfig:
    """Online-only feature switches.

    Canonical neighbor/segment context stays opt-in until retrieval ablations
    demonstrate that it should be promoted to the default critical path.
    """

    include_neighbors: bool = False
    include_segments: bool = False
    max_top_k: int = 200


@dataclass(frozen=True)
class Candidate:
    """Stable candidate schema shared by all online task routes."""

    video_id: str
    keyframe_id: str
    timestamp: float
    shot_id: str = ""
    segment_id: str = ""
    visual_score: float | None = None
    caption_score: float | None = None
    ocr_score: float | None = None
    object_score: float | None = None
    bge_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float = 0.0
    neighbors: tuple[dict[str, Any], ...] = ()
    segment_context: dict[str, Any] | None = None
    frame_index: int | None = None
    faiss_index: int | None = None
    keyframe_path: str = ""
    thumbnail_path: str = ""
    caption: str = ""
    ocr_text: str = ""
    objects: tuple[str, ...] = ()
    modality_scores: dict[str, float] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    context_sources: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        context_index: OnlineContextIndex | None = None,
        include_neighbors: bool = False,
        include_segments: bool = False,
    ) -> "Candidate":
        modality_scores = _float_mapping(value.get("modality_scores"))
        video_id = str(value.get("video_id") or "")
        keyframe_id = str(
            value.get("keyframe_id")
            or value.get("frame_id")
            or value.get("start_keyframe")
            or ""
        )
        timestamp = _float(value.get("timestamp"), default=0.0)
        segment_id = str(value.get("segment_id") or "")
        neighbors = tuple(
            dict(item)
            for item in (value.get("neighbors") or ())
            if isinstance(item, Mapping)
        )
        segment_context = None
        context_sources: tuple[str, ...] = ()
        if context_index is not None and (include_neighbors or include_segments):
            context = context_index.lookup(
                video_id=video_id,
                frame_id=keyframe_id,
                timestamp=timestamp,
                segment_id=segment_id,
                existing_neighbors=neighbors,
            )
            if include_neighbors:
                neighbors = context.neighbors
            if include_segments:
                segment_id = context.segment_id
                segment_context = context.segment
            allowed_sources = {
                "neighbors_all" if include_neighbors else "",
                "segments_all" if include_segments else "",
            }
            context_sources = tuple(
                source for source in context.sources if source in allowed_sources
            )

        temporal = {
            key: value[key]
            for key in (
                "temporal_event_index",
                "temporal_match_rank",
                "temporal_match_mode",
                "temporal_chain_id",
                "temporal_event_query",
                "temporal_event_role",
                "temporal_chain_score",
            )
            if key in value and value[key] is not None
        }
        fusion_value = value.get("fusion_score")
        if fusion_value is None:
            fusion_value = modality_scores.get(
                "rrf",
                modality_scores.get("fusion"),
            )
        rerank_value = value.get("rerank_score")
        if rerank_value is None:
            rerank_value = value.get("score", value.get("retrieval_score", 0.0))
        return cls(
            video_id=video_id,
            keyframe_id=keyframe_id,
            timestamp=timestamp,
            shot_id=str(value.get("shot_id") or ""),
            segment_id=segment_id,
            visual_score=_modality_score(modality_scores, "visual"),
            caption_score=_modality_score(modality_scores, "caption"),
            ocr_score=_modality_score(modality_scores, "ocr"),
            object_score=_modality_score(modality_scores, "objects", "object"),
            bge_score=_modality_score(modality_scores, "dense_text", "bge"),
            fusion_score=_optional_float(fusion_value),
            rerank_score=_float(rerank_value, default=0.0),
            neighbors=neighbors,
            segment_context=segment_context,
            frame_index=_optional_int(value.get("frame_index")),
            faiss_index=_optional_int(value.get("faiss_index")),
            keyframe_path=str(value.get("keyframe_path") or ""),
            thumbnail_path=str(value.get("thumbnail_path") or ""),
            caption=str(value.get("caption") or ""),
            ocr_text=str(value.get("ocr_text") or ""),
            objects=tuple(str(item) for item in (value.get("objects") or ()) if item),
            modality_scores=modality_scores,
            temporal=temporal,
            context_sources=context_sources,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return explicit scores plus compatibility aliases used by the UI."""

        return {
            "video_id": self.video_id,
            "keyframe_id": self.keyframe_id,
            "frame_id": self.keyframe_id,
            "timestamp": self.timestamp,
            "shot_id": self.shot_id,
            "segment_id": self.segment_id,
            "visual_score": self.visual_score,
            "caption_score": self.caption_score,
            "ocr_score": self.ocr_score,
            "object_score": self.object_score,
            "bge_score": self.bge_score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "score": self.rerank_score,
            "neighbors": [dict(item) for item in self.neighbors],
            "segment_context": (
                dict(self.segment_context) if self.segment_context is not None else None
            ),
            "frame_index": self.frame_index,
            "faiss_index": self.faiss_index,
            "keyframe_path": self.keyframe_path,
            "thumbnail_path": self.thumbnail_path,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "objects": list(self.objects),
            "modality_scores": dict(self.modality_scores),
            "temporal": dict(self.temporal),
            "context_sources": list(self.context_sources),
        }


class OnlinePipeline:
    """One public entrypoint over the existing online retrieval services."""

    def __init__(
        self,
        *,
        hybrid_engine: HybridSearchEngine,
        runtime_config: RetrievalRuntimeConfig | None = None,
        query_expansion_provider: QueryExpansionProvider | None = None,
        qa_pipeline: QaSearchPipeline | None = None,
        qa_evidence_engine: QaEvidenceSearchEngine | None = None,
        context_index: OnlineContextIndex | None = None,
        config: OnlinePipelineConfig | None = None,
    ) -> None:
        self.hybrid_engine = hybrid_engine
        self.runtime_config = runtime_config or RetrievalRuntimeConfig()
        self.query_expansion_provider = query_expansion_provider
        self.qa_pipeline = qa_pipeline
        self.qa_evidence_engine = qa_evidence_engine
        self.context_index = context_index
        self.config = config or OnlinePipelineConfig()
        if int(self.config.max_top_k) <= 0:
            raise ValueError("online max_top_k must be positive")

    def run(
        self,
        query: str,
        task: str = "kis",
        top_k: int | None = None,
        *,
        expanded_queries: Sequence[str] = (),
        include_context: bool | None = None,
    ) -> dict[str, Any]:
        """Plan once, route by task, then normalize all result candidates."""

        started_at = time.perf_counter()
        original_query = " ".join(str(query).split())
        if not original_query:
            raise ValueError("query must not be empty")
        requested_task = str(task or "kis").casefold().strip()
        if requested_task not in SUPPORTED_ONLINE_TASKS:
            raise ValueError(
                "Unsupported online task; expected "
                + ", ".join(SUPPORTED_ONLINE_TASKS)
            )
        requested_top_k = self._top_k(top_k, task=requested_task)

        plan: QueryPlan | None = None
        resolved_task = requested_task
        if requested_task in {"auto", "kis", "avs"}:
            plan = self._plan(original_query, requested_task)
            resolved_task = plan.profile if requested_task == "auto" else requested_task
            if requested_task == "auto" and top_k is None and resolved_task == "qa":
                requested_top_k = 5

        if resolved_task in {"kis", "avs"}:
            assert plan is not None
            raw = planned_hybrid_search(
                self.hybrid_engine,
                plan,
                top_k=requested_top_k,
                max_expansion_contribution=(
                    self.runtime_config.query_expansion.max_expansion_contribution
                ),
            ).to_dict()
        elif resolved_task == "qa":
            if self.qa_pipeline is None:
                raise RuntimeError("QA pipeline is unavailable in this online runtime")
            raw = self.qa_pipeline.search(
                original_query,
                top_k=min(requested_top_k, 5),
                task_mode="qa",
                expanded_queries=tuple(expanded_queries),
            )
        elif resolved_task == "temporal":
            if self.qa_evidence_engine is None:
                raise RuntimeError(
                    "Temporal evidence engine is unavailable in this online runtime"
                )
            raw = self.qa_evidence_engine.search(
                original_query,
                top_k=requested_top_k,
                task_mode="temporal",
                # Event decomposition owns temporal queries; whole-question
                # expansions must not alter event order or meaning.
                expanded_queries=(),
            )
        else:
            raise RuntimeError(f"Online task routing failed: {resolved_task}")

        query_plan = _query_plan_payload(raw, plan)
        _validate_original_anchor(original_query, query_plan)
        include_neighbors, include_segments = self._context_flags(include_context)
        if (include_neighbors or include_segments) and self.context_index is None:
            raise FileNotFoundError(
                "Online context was requested but canonical neighbor/segment "
                "artifacts are not loaded"
            )
        raw_candidates = raw.get("results") or ()
        candidates = [
            Candidate.from_mapping(
                item,
                context_index=self.context_index,
                include_neighbors=include_neighbors,
                include_segments=include_segments,
            )
            for item in raw_candidates
            if isinstance(item, Mapping)
        ]

        response: dict[str, Any] = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "query": original_query,
            "requested_task": requested_task,
            "task": resolved_task,
            "top_k": requested_top_k,
            "query_plan": query_plan,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "context": {
                "enabled": include_neighbors or include_segments,
                "neighbors_enabled": include_neighbors,
                "segments_enabled": include_segments,
                "index": (
                    self.context_index.summary()
                    if self.context_index is not None
                    else None
                ),
            },
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
        }
        trace = raw.get("routing_trace") or raw.get("trace")
        if isinstance(trace, Mapping):
            response["routing_trace"] = dict(trace)
        for field_name in (
            "answer",
            "answer_report",
            "answer_eligible",
            "preflight_block_reason",
            "evidence",
            "temporal_matches",
            "experiment_id",
        ):
            if field_name in raw:
                response[field_name] = raw[field_name]
        return response

    def _plan(self, query: str, task: str) -> QueryPlan:
        return build_query_plan(
            query,
            profile=task,
            expansion_provider=(
                self.query_expansion_provider
                if self.runtime_config.query_expansion.enabled
                else None
            ),
            expansion_config=self.runtime_config.query_expansion,
        )

    def _top_k(self, value: int | None, *, task: str) -> int:
        default = 5 if task == "qa" else self.runtime_config.hybrid.default_top_k
        requested = default if value is None else int(value)
        return max(1, min(requested, int(self.config.max_top_k)))

    def _context_flags(self, override: bool | None) -> tuple[bool, bool]:
        if override is None:
            return (self.config.include_neighbors, self.config.include_segments)
        enabled = bool(override)
        return (enabled, enabled)


def _query_plan_payload(
    raw: Mapping[str, Any],
    plan: QueryPlan | None,
) -> dict[str, Any]:
    value = raw.get("query_plan")
    if isinstance(value, Mapping):
        return dict(value)
    trace = raw.get("trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("query_plan"), Mapping):
        return dict(trace["query_plan"])
    if plan is not None:
        return plan.to_dict()
    raise RuntimeError("Online route did not return its query plan")


def _validate_original_anchor(query: str, plan: Mapping[str, Any]) -> None:
    original = " ".join(str(plan.get("original_query") or "").split())
    if original != query:
        raise RuntimeError(
            "Online query plan did not preserve the original query anchor"
        )


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, float] = {}
    for key, raw in value.items():
        try:
            output[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return output


def _modality_score(
    values: Mapping[str, float],
    *names: str,
) -> float | None:
    for name in names:
        if name in values:
            return float(values[name])
    return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: object, *, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _optional_int(value: object) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one query through the canonical online pipeline."
    )
    parser.add_argument(
        "--task",
        choices=SUPPORTED_ONLINE_TASKS,
        default="kis",
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--expanded-query",
        action="append",
        default=[],
        help="Caller-owned QA expansion; repeat for multiple values.",
    )
    parser.add_argument(
        "--with-context",
        action="store_const",
        const=True,
        default=None,
        help="Require canonical neighbor and segment context for this query.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; stdout is always printed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Lazy import avoids a module cycle: retrieval_manager owns the cached
    # production OnlinePipeline instance and imports this class definition.
    from backend.app.services.retrieval.retrieval_manager import search_online

    args = build_parser().parse_args(argv)
    response = search_online(
        query=args.query,
        task=args.task,
        top_k=args.top_k,
        expanded_queries=args.expanded_query,
        include_context=args.with_context,
    )
    serialized = json.dumps(response, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


__all__ = [
    "Candidate",
    "ONLINE_SCHEMA_VERSION",
    "OnlinePipeline",
    "OnlinePipelineConfig",
    "SUPPORTED_ONLINE_TASKS",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
