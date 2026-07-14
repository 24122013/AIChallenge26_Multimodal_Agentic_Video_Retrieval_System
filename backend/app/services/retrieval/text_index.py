"""Lightweight lexical indexes for caption, OCR, ASR, and object search."""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse


_TOKEN_RE = re.compile(r"[a-z0-9]+")
MODALITIES = ("caption", "ocr", "asr", "objects")


@dataclass(frozen=True)
class TextDocument:
    doc_id: str
    video_id: str
    frame_id: str
    timestamp: float
    text: str
    modality: str
    segment_id: str = ""
    shot_id: str = ""
    faiss_index: int | None = None
    frame_index: int | None = None
    keyframe_path: str = ""
    thumbnail_path: str = ""
    caption: str = ""
    ocr_text: str = ""
    asr_text: str = ""
    objects: list[str] | None = None


class TextIndexSearcher:
    """BM25-style search over text metadata built by build_text_index.py."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._payload: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._payload is not None:
            return self._payload
        if not self.index_path.exists():
            raise FileNotFoundError(f"text index not found: {self.index_path}")
        self._payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        return self._payload

    def search(
        self,
        query: str,
        modality: str,
        top_k: int = 20,
    ) -> VisualSearchResponse:
        started_at = time.perf_counter()
        results = self.search_results(query=query, modality=modality, top_k=top_k)
        return VisualSearchResponse(
            query=query,
            top_k=max(1, int(top_k)),
            latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
            results=results,
        )

    def search_results(
        self,
        query: str,
        modality: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        modality = _normalize_modality(modality)
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        payload = self._load()
        modality_payload = payload.get("modalities", {}).get(modality, {})
        documents = modality_payload.get("documents", {})
        postings = modality_payload.get("postings", {})
        stats = modality_payload.get("stats", {})
        doc_count = max(1, int(stats.get("doc_count") or len(documents) or 1))
        avg_len = float(stats.get("avg_doc_len") or 1.0)

        scores: dict[str, float] = {}
        for token in query_tokens:
            token_postings = postings.get(token, {})
            doc_freq = len(token_postings)
            if doc_freq == 0:
                continue
            idf = math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            for doc_id, term_freq in token_postings.items():
                doc_len = max(1, int(documents[doc_id].get("doc_len") or 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + _bm25(
                    term_freq=float(term_freq),
                    doc_len=doc_len,
                    avg_len=avg_len,
                    idf=idf,
                )

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[: max(0, int(top_k))]
        return [
            _document_to_result(documents[doc_id], score=_normalize_score(scores[doc_id]))
            for doc_id in ranked_ids
        ]


def build_text_index(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    documents_by_modality: dict[str, dict[str, dict[str, Any]]] = {
        modality: {} for modality in MODALITIES
    }
    postings_by_modality: dict[str, dict[str, dict[str, int]]] = {
        modality: {} for modality in MODALITIES
    }

    for record in records:
        for modality in MODALITIES:
            text = _text_for_modality(record, modality)
            tokens = tokenize(text)
            if not tokens:
                continue
            doc_id = f"{modality}:{record.get('frame_id') or record.get('faiss_index')}"
            doc = _document_from_record(record, modality=modality, text=text, doc_len=len(tokens))
            documents_by_modality[modality][doc_id] = doc
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            for token, count in term_counts.items():
                postings_by_modality[modality].setdefault(token, {})[doc_id] = count

    return {
        "version": 1,
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
    return _TOKEN_RE.findall(text.lower())


def _bm25(term_freq: float, doc_len: int, avg_len: float, idf: float) -> float:
    k1 = 1.5
    b = 0.75
    denominator = term_freq + k1 * (1.0 - b + b * doc_len / max(avg_len, 1.0))
    return idf * (term_freq * (k1 + 1.0)) / denominator


def _normalize_score(score: float) -> float:
    return round(score / (score + 1.0), 6)


def _normalize_modality(modality: str) -> str:
    normalized = modality.lower().strip()
    aliases = {"ocr_text": "ocr", "object": "objects", "caption_text": "caption"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODALITIES:
        raise ValueError(f"unsupported text modality: {modality}")
    return normalized


def _text_for_modality(record: dict[str, Any], modality: str) -> str:
    if modality == "caption":
        return str(record.get("caption") or "")
    if modality == "ocr":
        return str(record.get("ocr_text") or record.get("ocr") or "")
    if modality == "asr":
        return str(record.get("asr_text") or record.get("asr") or record.get("transcript") or "")
    objects = record.get("objects") or []
    if isinstance(objects, list):
        return " ".join(str(item) for item in objects)
    return str(objects)


def _document_from_record(
    record: dict[str, Any],
    modality: str,
    text: str,
    doc_len: int,
) -> dict[str, Any]:
    return {
        "video_id": record.get("video_id") or "",
        "frame_id": record.get("frame_id") or "",
        "segment_id": record.get("segment_id") or "",
        "shot_id": record.get("shot_id") or "",
        "timestamp": float(record.get("timestamp") or 0.0),
        "timestamp_confidence": float(record.get("timestamp_confidence") or 0.0),
        "faiss_index": record.get("faiss_index"),
        "frame_index": record.get("frame_index"),
        "keyframe_path": record.get("keyframe_path") or record.get("frame_path") or "",
        "thumbnail_path": record.get("thumbnail_path") or record.get("keyframe_path") or "",
        "caption": record.get("caption") or "",
        "ocr_text": record.get("ocr_text") or record.get("ocr") or "",
        "asr_text": record.get("asr_text") or record.get("asr") or record.get("transcript") or "",
        "objects": _objects(record.get("objects")),
        "modality": modality,
        "text": text,
        "doc_len": doc_len,
    }


def _document_to_result(document: dict[str, Any], score: float) -> RetrievalResult:
    return RetrievalResult(
        video_id=document.get("video_id") or "",
        frame_id=document.get("frame_id") or "",
        segment_id=document.get("segment_id") or "",
        shot_id=document.get("shot_id") or "",
        timestamp=float(document.get("timestamp") or 0.0),
        timestamp_confidence=float(document.get("timestamp_confidence") or 0.0),
        frame_index=document.get("frame_index"),
        faiss_index=document.get("faiss_index"),
        score=score,
        keyframe_path=document.get("keyframe_path") or "",
        thumbnail_path=document.get("thumbnail_path") or "",
        caption=document.get("caption") or "",
        ocr_text=document.get("ocr_text") or "",
        objects=_objects(document.get("objects")),
    )


def _objects(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _stats(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    docs = list(documents)
    total_len = sum(int(doc.get("doc_len") or 0) for doc in docs)
    return {
        "doc_count": len(docs),
        "avg_doc_len": total_len / len(docs) if docs else 0.0,
    }
