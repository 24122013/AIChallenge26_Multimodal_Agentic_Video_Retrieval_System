"""BGE cross-encoder reranking with deterministic Phase-5 fallback."""
from __future__ import annotations

import math
import threading
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from backend.app.models.retrieval import RetrievalResult
from backend.app.services.retrieval.bge_dense import field_tagged_document


DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BGE_RERANKER_REVISION = "main"
DEFAULT_CANDIDATE_LIMIT = 100
DEFAULT_OUTPUT_K = 20
DEFAULT_RETRIEVAL_ALPHA = 0.5


_SHARED_RUNNER_LOCK = threading.RLock()
_SHARED_RUNNERS: weakref.WeakValueDictionary[
    tuple[object, ...], "LazyBgeReranker"
] = weakref.WeakValueDictionary()


class PairScorer(Protocol):
    def __call__(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


@dataclass(frozen=True)
class BgeRerankReport:
    status: str
    model_name: str
    model_revision: str
    candidate_count: int
    scored_count: int
    output_count: int
    retrieval_alpha: float
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "candidate_count": self.candidate_count,
            "scored_count": self.scored_count,
            "output_count": self.output_count,
            "retrieval_alpha": self.retrieval_alpha,
            "fallback_reason": self.fallback_reason,
        }


def rerank_with_bge(
    candidates: Sequence[RetrievalResult],
    *,
    query: str,
    runner: PairScorer | None = None,
    model_name: str = DEFAULT_BGE_RERANKER_MODEL,
    model_revision: str = DEFAULT_BGE_RERANKER_REVISION,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    output_k: int = DEFAULT_OUTPUT_K,
    retrieval_alpha: float = DEFAULT_RETRIEVAL_ALPHA,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[list[RetrievalResult], BgeRerankReport]:
    """Rerank a caller-bounded candidate pool and return a bounded head.

    Candidates without caption/OCR/object text remain in the pool with their
    original retrieval score.  Any model/scoring error falls back to the input
    Phase-5 order instead of dropping results.  The public defaults remain
    100 input candidates and 20 outputs; task adapters may choose stricter or
    wider validated bounds explicitly.
    """

    normalized_query = " ".join(str(query).split())
    if not normalized_query:
        raise ValueError("BGE reranker query must not be empty")
    if candidate_limit <= 0 or output_k <= 0:
        raise ValueError("BGE reranker candidate_limit and output_k must be positive")
    if not 0.0 <= float(retrieval_alpha) <= 1.0:
        raise ValueError("BGE reranker retrieval_alpha must be within [0, 1]")

    limited = list(candidates[: min(int(candidate_limit), len(candidates))])
    fallback = limited[: min(int(output_k), len(limited))]
    if not limited:
        return [], BgeRerankReport(
            status="passed",
            model_name=model_name,
            model_revision=model_revision,
            candidate_count=0,
            scored_count=0,
            output_count=0,
            retrieval_alpha=float(retrieval_alpha),
        )

    positions: list[int] = []
    pairs: list[tuple[str, str]] = []
    for position, candidate in enumerate(limited):
        document = reranker_document(candidate)
        if _document_has_content(document):
            positions.append(position)
            pairs.append((normalized_query, document))

    try:
        scores: Sequence[float]
        if pairs:
            scorer = runner or LazyBgeReranker(
                model_name=model_name,
                model_revision=model_revision,
                batch_size=batch_size,
                device=device,
                cache_dir=Path(cache_dir) if cache_dir else None,
                local_files_only=local_files_only,
            )
            scores = scorer(pairs)
            if len(scores) != len(pairs):
                raise ValueError(
                    "BGE reranker score count mismatch: "
                    f"expected {len(pairs)}, got {len(scores)}"
                )
        else:
            scores = []

        score_by_position: dict[int, float] = {}
        for position, raw_score in zip(positions, scores):
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("BGE reranker scores must be finite values in [0, 1]")
            score_by_position[position] = score

        rescored: list[RetrievalResult] = []
        for position, candidate in enumerate(limited):
            reranker_score = score_by_position.get(position)
            if reranker_score is None:
                rescored.append(candidate)
                continue
            blended = (
                float(retrieval_alpha) * float(candidate.score)
                + (1.0 - float(retrieval_alpha)) * reranker_score
            )
            modalities = dict(candidate.modality_scores)
            modalities["bge_reranker"] = reranker_score
            rescored.append(
                replace(
                    candidate,
                    score=round(blended, 8),
                    modality_scores=modalities,
                )
            )
        rescored.sort(key=_ranking_key)
        output = rescored[: min(int(output_k), len(rescored))]
        return output, BgeRerankReport(
            status="passed",
            model_name=model_name,
            model_revision=model_revision,
            candidate_count=len(limited),
            scored_count=len(pairs),
            output_count=len(output),
            retrieval_alpha=float(retrieval_alpha),
        )
    except Exception as exc:
        return fallback, BgeRerankReport(
            status="fallback",
            model_name=model_name,
            model_revision=model_revision,
            candidate_count=len(limited),
            scored_count=0,
            output_count=len(fallback),
            retrieval_alpha=float(retrieval_alpha),
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def reranker_document(candidate: RetrievalResult | Mapping[str, object]) -> str:
    """Return only [CAPTION], [OCR], and [OBJECTS] fields (never ASR)."""

    if isinstance(candidate, RetrievalResult):
        record: Mapping[str, object] = {
            "caption": candidate.caption,
            "ocr_text": candidate.ocr_text,
            "objects": candidate.objects,
        }
    else:
        record = candidate
    return field_tagged_document(record)


class LazyBgeReranker:
    """Lazy sequence-classification scorer for bge-reranker-v2-m3."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_BGE_RERANKER_MODEL,
        model_revision: str = DEFAULT_BGE_RERANKER_REVISION,
        batch_size: int = 16,
        max_length: int = 1024,
        device: str = "auto",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("BGE reranker batch size and max length must be positive")
        if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
            raise ValueError("BGE reranker device must be auto, cpu, cuda, or cuda:<index>")
        self.model_name = model_name
        self.model_revision = model_revision
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.requested_device = device
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        # A cached runner can serve QA and TRAKE concurrently.  Serialize model
        # loading and inference on that instance so two requests cannot load a
        # duplicate checkpoint or race shared tokenizer/model state.
        self._call_lock = threading.RLock()

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        if not pairs:
            return []
        with self._call_lock:
            return self._score_pairs(pairs)

    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        self._load()
        assert self._torch is not None and self._model is not None and self._tokenizer is not None
        output_scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            queries = [query for query, _ in batch]
            documents = [document for _, document in batch]
            inputs = self._tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                raw = self._model(**inputs)
            logits = getattr(raw, "logits", None)
            if logits is None:
                raise RuntimeError("BGE reranker did not return logits")
            if logits.ndim == 1 or logits.shape[-1] == 1:
                probabilities = self._torch.sigmoid(logits.reshape(-1))
            elif logits.shape[-1] == 2:
                probabilities = self._torch.softmax(logits.float(), dim=-1)[:, 1]
            else:
                raise RuntimeError(f"Unexpected BGE reranker logits shape: {tuple(logits.shape)}")
            output_scores.extend(float(value) for value in probabilities.float().cpu().tolist())
        return output_scores

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("PyTorch and Transformers are required for BGE reranking") from exc
        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.requested_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for BGE reranking but is unavailable")
        common: dict[str, object] = {
            "revision": self.model_revision,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            common["cache_dir"] = self.cache_dir.as_posix()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **common)
        model_kwargs = dict(common)
        if device.startswith("cuda"):
            model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        model.to(device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device


def get_shared_bge_reranker_runner(
    *,
    model_name: str = DEFAULT_BGE_RERANKER_MODEL,
    model_revision: str = DEFAULT_BGE_RERANKER_REVISION,
    batch_size: int = 16,
    max_length: int = 1024,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> LazyBgeReranker:
    """Return one thread-safe lazy runner for an identical model contract."""

    normalized_cache = str(Path(cache_dir)) if cache_dir else ""
    key: tuple[object, ...] = (
        str(model_name),
        str(model_revision),
        int(batch_size),
        int(max_length),
        str(device),
        normalized_cache,
        bool(local_files_only),
    )
    with _SHARED_RUNNER_LOCK:
        runner = _SHARED_RUNNERS.get(key)
        if runner is None:
            runner = LazyBgeReranker(
                model_name=model_name,
                model_revision=model_revision,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                cache_dir=Path(cache_dir) if cache_dir else None,
                local_files_only=local_files_only,
            )
            _SHARED_RUNNERS[key] = runner
        return runner


def clear_shared_bge_reranker_runners() -> None:
    """Drop cached lazy runners after retrieval runtime caches are cleared."""

    with _SHARED_RUNNER_LOCK:
        _SHARED_RUNNERS.clear()


def _document_has_content(document: str) -> bool:
    return any(
        line.partition("]")[2].strip()
        for line in document.splitlines()
        if "]" in line
    )


def _ranking_key(candidate: RetrievalResult) -> tuple[object, ...]:
    return (
        -float(candidate.score),
        str(candidate.video_id),
        float(candidate.timestamp),
        str(candidate.frame_id),
    )
