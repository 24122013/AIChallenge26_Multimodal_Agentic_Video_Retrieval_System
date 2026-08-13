"""Unified search API wrappers for Retrieval Phase 1-3."""
from __future__ import annotations

from backend.app.services.retrieval.retrieval_manager import (
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
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None
    HTTPException = None
    BaseModel = object
    Field = None


if APIRouter is not None:
    router = APIRouter(prefix="/search", tags=["search"])

    class SearchBody(BaseModel):
        query: str
        mode: str = "visual"
        top_k: int = Field(default=20, ge=1, le=200)

    @router.post("")
    def search_endpoint(body: SearchBody) -> dict:
        try:
            return {
                "success": True,
                "data": _dispatch_search(body.query, body.top_k, body.mode),
                "message": None,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - convert service errors to API response.
            raise HTTPException(status_code=500, detail=str(exc)) from exc
else:
    router = None


def search(query: str, top_k: int = 20, mode: str = "visual") -> dict:
    return _dispatch_search(query, top_k, mode)


def _dispatch_search(query: str, top_k: int, mode: str) -> dict:
    normalized = mode.casefold().strip()
    if normalized in {"visual", "image", "baseline"}:
        return search_visual(query=query, top_k=top_k).to_dict()
    if normalized == "hybrid":
        return search_hybrid(query=query, top_k=top_k).to_dict()
    if normalized == "caption":
        return search_caption(query=query, top_k=top_k).to_dict()
    if normalized in {"ocr", "ocr_text"}:
        return search_ocr(query=query, top_k=top_k).to_dict()
    if normalized in {"object", "objects"}:
        return search_object(query=query, top_k=top_k).to_dict()
    if normalized in {"qa", "qa_evidence", "question", "question_answering"}:
        return search_qa_evidence(question=query, top_k=top_k)
    if normalized == "temporal":
        matches = search_temporal(query=query, top_k=top_k)
        return {
            "query": query,
            "top_k": max(1, int(top_k)),
            "results": [match.to_dict() for match in matches],
        }
    raise ValueError(
        "Unsupported search mode. Expected visual, hybrid, caption, OCR, "
        "object, QA evidence, or temporal."
    )
