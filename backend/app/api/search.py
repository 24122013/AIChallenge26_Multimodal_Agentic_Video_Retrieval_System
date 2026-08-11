"""Unified search API wrappers for Retrieval Phase 1-3."""
from __future__ import annotations

import time
import traceback
from backend.app.models.search import TextSearchPayload
from backend.app.models.retrieval import VisualSearchResponse, APIResponse
from backend.app.services.retrieval.retrieval_manager import (
    search_asr,
    search_caption,
    search_hybrid,
    search_object,
    search_ocr,
    search_qa_evidence,
    search_temporal,
    search_visual,
)

try:  # pragma: no cover - depends on optional API runtime.
    from fastapi import APIRouter, HTTPException
except ImportError:  # pragma: no cover
    APIRouter = None
    HTTPException = None
    BaseModel = object
    Field = None


if APIRouter is not None:
    search_router = APIRouter(prefix="/api/search", tags=["search"])

    @search_router.post("/kist")
    def search_kist_endpoint(body: TextSearchPayload) -> APIResponse[VisualSearchResponse]:
        try:
            results = _dispatch_search(body.query, body.top_k, body.mode)
            return APIResponse(
                success=True,
                data=results,
                message=None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - convert service errors to API response.
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc
else:
    search_router = None


def search(query: str, top_k: int = 20, mode: str = "visual") -> VisualSearchResponse:
    return _dispatch_search(query, top_k, mode)


def _dispatch_search(query: str, top_k: int, mode: str) -> VisualSearchResponse:
    normalized = mode.casefold().strip()
    if normalized in {"visual", "image", "baseline"}:
        return search_visual(query=query, top_k=top_k)
    if normalized == "hybrid":
        return search_hybrid(query=query, top_k=top_k)
    if normalized == "caption":
        return search_caption(query=query, top_k=top_k)
    if normalized in {"ocr", "ocr_text"}:
        return search_ocr(query=query, top_k=top_k)
    if normalized in {"asr", "transcript"}:
        return search_asr(query=query, top_k=top_k)
    if normalized in {"object", "objects"}:
        return search_object(query=query, top_k=top_k)
    if normalized in {"qa", "qa_evidence", "question", "question_answering"}:
        return search_qa_evidence(question=query, top_k=top_k)
    if normalized == "temporal":
        started_at = time.perf_counter()
        matches = search_temporal(query=query, top_k=top_k)
        return VisualSearchResponse(
            query=query,
            top_k=max(1, int(top_k)),
            latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
            results=[match.to_dict() for match in matches],
        )
    raise ValueError(
        "Unsupported search mode. Expected visual, hybrid, caption, OCR, "
        "ASR, object, QA evidence, or temporal."
    )
