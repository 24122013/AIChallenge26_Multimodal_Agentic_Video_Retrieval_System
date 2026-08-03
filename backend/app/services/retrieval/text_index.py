"""Dependency-light lexical indexes for caption, OCR, ASR, and object search."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse
from backend.app.services.retrieval.query_terms import (
    content_phrase_match,
    content_tokens,
    stem_token,
    weighted_term_coverage,
)


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
MODALITIES = ("caption", "ocr", "asr", "objects")
INDEX_VERSION = 2


class TextIndexSearcher:
    """BM25-style search over text metadata built by build_text_index.py."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._payload: dict[str, Any] | None = None
        self._stem_postings: dict[str, dict[str, dict[str, int]]] = {}

    def _load(self) -> dict[str, Any]:
        if self._payload is not None:
            return self._payload
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Retrieval text index not found: {self.index_path}. "
                "Metadata role must provide caption/OCR/ASR/object artifacts, then "
                "run backend/app/services/indexing/build_text_index.py."
            )
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION:
            raise ValueError(
                f"Unsupported retrieval text index in {self.index_path}; "
                f"expected version {INDEX_VERSION}"
            )
        modalities = payload.get("modalities")
        if not isinstance(modalities, dict):
            raise ValueError(f"Text index is missing modalities: {self.index_path}")
        self._payload = payload
        return payload

    def search(
        self,
        query: str,
        modality: str,
        top_k: int = 20,
    ) -> VisualSearchResponse:
        started_at = time.perf_counter()
        bounded_top_k = max(1, int(top_k))
        results = self.search_results(
            query=query,
            modality=modality,
            top_k=bounded_top_k,
        )
        return VisualSearchResponse(
            query=query,
            top_k=bounded_top_k,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
            results=results,
        )

    def search_results(
        self,
        query: str,
        modality: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        modality = normalize_modality(modality)
        query_terms = list(
            dict.fromkeys(content_tokens(query, fallback_to_all=True))
        )
        if not query_terms:
            return []

        payload = self._load()
        modality_payload = payload["modalities"].get(modality, {})
        documents = modality_payload.get("documents", {})
        postings = modality_payload.get("postings", {})
        stats = modality_payload.get("stats", {})
        if not isinstance(documents, dict) or not isinstance(postings, dict):
            raise ValueError(
                f"Invalid {modality} index payload in {self.index_path}"
            )

        doc_count = max(1, int(stats.get("doc_count") or len(documents) or 1))
        avg_len = max(1.0, float(stats.get("avg_doc_len") or 1.0))
        scores: dict[str, float] = {}
        matched_terms: dict[str, set[str]] = {}
        postings_by_stem = self._stem_postings.get(modality)
        if postings_by_stem is None:
            postings_by_stem = _merge_postings_by_stem(postings)
            self._stem_postings[modality] = postings_by_stem

        for term in query_terms:
            token_postings = postings_by_stem.get(term, {})
            if not isinstance(token_postings, dict):
                continue
            doc_freq = len(token_postings)
            if doc_freq == 0:
                continue
            idf = math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            for doc_id, term_freq in token_postings.items():
                document = documents.get(doc_id)
                if not isinstance(document, dict):
                    continue
                doc_len = max(1, int(document.get("doc_len") or 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + _bm25(
                    term_freq=float(term_freq),
                    doc_len=doc_len,
                    avg_len=avg_len,
                    idf=idf,
                )
                matched_terms.setdefault(doc_id, set()).add(term)

        quality_scores = {
            doc_id: _lexical_quality_score(
                query=query,
                query_terms=query_terms,
                matched_terms=matched_terms.get(doc_id, set()),
                bm25_score=score,
                document=documents.get(doc_id, {}),
            )
            for doc_id, score in scores.items()
        }
        ranked_ids = sorted(
            quality_scores,
            key=lambda doc_id: (
                quality_scores[doc_id],
                scores.get(doc_id, 0.0),
                doc_id,
            ),
            reverse=True,
        )[: max(0, int(top_k))]
        return [
            _document_to_result(
                documents[doc_id],
                modality=modality,
                score=quality_scores[doc_id],
            )
            for doc_id in ranked_ids
        ]


def build_text_index(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build deterministic BM25 postings for all supported metadata schemas."""
    documents_by_modality: dict[str, dict[str, dict[str, Any]]] = {
        modality: {} for modality in MODALITIES
    }
    postings_by_modality: dict[str, dict[str, dict[str, int]]] = {
        modality: {} for modality in MODALITIES
    }

    for ordinal, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for modality in MODALITIES:
            text = text_for_modality(record, modality)
            tokens = tokenize(text)
            if not tokens:
                continue
            doc_id = _document_id(record, modality=modality, ordinal=ordinal)
            document = _document_from_record(
                record,
                modality=modality,
                text=text,
                doc_len=len(tokens),
            )
            documents_by_modality[modality][doc_id] = document
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            for token, count in term_counts.items():
                postings_by_modality[modality].setdefault(token, {})[
                    doc_id
                ] = count

    return {
        "version": INDEX_VERSION,
        "modalities": {
            modality: {
                "documents": documents_by_modality[modality],
                "postings": postings_by_modality[modality],
                "stats": _stats(documents_by_modality[modality].values()),
            }
            for modality in MODALITIES
        },
    }


def tokenize(text: str) -> list[str]:
    """Tokenize English and Vietnamese text while preserving Unicode letters."""
    return _TOKEN_RE.findall(str(text).casefold())


def normalize_modality(modality: str) -> str:
    normalized = modality.casefold().strip()
    aliases = {
        "ocr_text": "ocr",
        "object": "objects",
        "object_classes": "objects",
        "caption_text": "caption",
        "transcript": "asr",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODALITIES:
        raise ValueError(f"Unsupported text modality: {modality}")
    return normalized


def text_for_modality(record: dict[str, Any], modality: str) -> str:
    modality = normalize_modality(modality)
    if modality == "caption":
        return _first_text(
            record.get("caption"),
            record.get("segment_caption"),
            record.get("captions_aggregated"),
        )
    if modality == "ocr":
        return _first_text(
            record.get("ocr_text"),
            _join_nested_text(record.get("ocr")),
            _join_nested_text(record.get("text_regions")),
        )
    if modality == "asr":
        return _first_text(
            record.get("asr_text"),
            record.get("transcript"),
            record.get("transcript_text"),
            _join_nested_text(record.get("asr")),
            _join_nested_text(record.get("transcript_segments")),
            record.get("text") if record.get("pipeline") == "asr" else None,
        )
    return " ".join(_object_labels(record))


def _document_id(record: dict[str, Any], modality: str, ordinal: int) -> str:
    identity = (
        record.get("frame_id")
        or record.get("segment_id")
        or record.get("faiss_index")
        or record.get("transcript_segment_id")
        or ordinal
    )
    video_id = record.get("video_id") or "unknown"
    return f"{modality}:{video_id}:{identity}"


def _document_from_record(
    record: dict[str, Any],
    modality: str,
    text: str,
    doc_len: int,
) -> dict[str, Any]:
    frame_id = record.get("frame_id") or record.get("start_keyframe") or ""
    timestamp = record.get("timestamp")
    if timestamp is None:
        timestamp = record.get(
            "start_time",
            record.get("segment_start", record.get("start", 0.0)),
        )
    return {
        "video_id": str(record.get("video_id") or ""),
        "frame_id": str(frame_id),
        "segment_id": str(record.get("segment_id") or record.get("shot_id") or ""),
        "shot_id": str(record.get("shot_id") or record.get("segment_id") or ""),
        "timestamp": float(timestamp or 0.0),
        "timestamp_confidence": float(record.get("timestamp_confidence") or 0.0),
        "faiss_index": record.get("faiss_index"),
        "frame_index": record.get("frame_index", record.get("start_frame")),
        "keyframe_path": str(
            record.get("keyframe_path") or record.get("frame_path") or ""
        ),
        "thumbnail_path": str(
            record.get("thumbnail_path")
            or record.get("keyframe_path")
            or record.get("frame_path")
            or ""
        ),
        "caption": text_for_modality(record, "caption"),
        "ocr_text": text_for_modality(record, "ocr"),
        "asr_text": text_for_modality(record, "asr"),
        "objects": _object_labels(record),
        "modality": modality,
        "text": text,
        "doc_len": doc_len,
    }


def _document_to_result(
    document: dict[str, Any],
    modality: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        video_id=str(document.get("video_id") or ""),
        frame_id=str(document.get("frame_id") or ""),
        segment_id=str(document.get("segment_id") or ""),
        shot_id=str(document.get("shot_id") or ""),
        timestamp=float(document.get("timestamp") or 0.0),
        timestamp_confidence=float(document.get("timestamp_confidence") or 0.0),
        frame_index=document.get("frame_index"),
        faiss_index=document.get("faiss_index"),
        score=score,
        keyframe_path=str(document.get("keyframe_path") or ""),
        thumbnail_path=str(document.get("thumbnail_path") or ""),
        caption=str(document.get("caption") or ""),
        ocr_text=str(document.get("ocr_text") or ""),
        asr_text=str(document.get("asr_text") or ""),
        objects=[str(value) for value in document.get("objects") or []],
        modality_scores={modality: score},
    )


def _first_text(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _join_nested_text(value: object) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return str(value).strip()
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("ocr_text")
                or item.get("transcript_text")
            )
            if text:
                parts.append(str(text).strip())
        elif item:
            parts.append(str(item).strip())
    return " ".join(part for part in parts if part)


def _object_labels(record: dict[str, Any]) -> list[str]:
    raw = record.get("object_classes")
    if not raw:
        raw = record.get("objects")
    if not raw:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
        return [value for value in values if value]
    if not isinstance(raw, list):
        return [str(raw)]
    labels: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            label = item.get("label") or item.get("class_name") or item.get("class")
            if label:
                labels.append(str(label))
        elif item:
            labels.append(str(item))
    return list(dict.fromkeys(labels))


def _bm25(term_freq: float, doc_len: int, avg_len: float, idf: float) -> float:
    k1 = 1.5
    b = 0.75
    denominator = term_freq + k1 * (1.0 - b + b * doc_len / max(avg_len, 1.0))
    return idf * (term_freq * (k1 + 1.0)) / denominator


def _normalize_score(score: float) -> float:
    return round(score / (score + 1.0), 6)


def _merge_postings_by_stem(
    postings: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Merge raw-token postings so old indexes gain query-time stemming."""
    merged: dict[str, dict[str, int]] = {}
    for raw_token, raw_postings in postings.items():
        if not isinstance(raw_postings, dict):
            continue
        stem = stem_token(raw_token)
        if not stem:
            continue
        stem_postings = merged.setdefault(stem, {})
        for doc_id, term_freq in raw_postings.items():
            try:
                count = int(term_freq)
            except (TypeError, ValueError):
                continue
            stem_postings[str(doc_id)] = (
                stem_postings.get(str(doc_id), 0) + max(0, count)
            )
    return merged


def _lexical_quality_score(
    *,
    query: str,
    query_terms: list[str],
    matched_terms: set[str],
    bm25_score: float,
    document: Any,
) -> float:
    """Blend BM25 with query coverage without hard-filtering partial matches."""
    document_text = (
        str(document.get("text") or "") if isinstance(document, dict) else ""
    )
    coverage = weighted_term_coverage(query_terms, matched_terms)
    phrase = content_phrase_match(query, document_text)
    score = (
        0.50 * _normalize_score(bm25_score)
        + 0.40 * coverage
        + 0.10 * phrase
    )
    return round(max(0.0, min(1.0, score)), 6)


def _stats(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    docs = list(documents)
    total_len = sum(int(document.get("doc_len") or 0) for document in docs)
    return {
        "doc_count": len(docs),
        "avg_doc_len": total_len / len(docs) if docs else 0.0,
    }
