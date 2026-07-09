"""Unified search API wrappers for visual and text retrieval."""
from __future__ import annotations

from backend.app.services.retrieval.retrieval_manager import (
    search_caption,
    search_hybrid,
    search_object,
    search_ocr,
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
        if body.mode not in {"visual", "image", "baseline", "hybrid", "caption", "ocr", "object"}:
            raise HTTPException(
                status_code=400,
                detail="Unsupported search mode.",
            )
        try:
            return {
                "success": True,
                "data": _dispatch_search(body.query, body.top_k, body.mode),
                "message": None,
            }
        except Exception as exc:  # noqa: BLE001 - convert service errors to API response.
            raise HTTPException(status_code=500, detail=str(exc)) from exc
else:
    router = None


def search(query: str, top_k: int = 20, mode: str = "visual") -> dict:
    if mode not in {"visual", "image", "baseline", "hybrid", "caption", "ocr", "object"}:
        raise ValueError("Unsupported search mode.")
    return _dispatch_search(query, top_k, mode)


def _dispatch_search(query: str, top_k: int, mode: str) -> dict:
    if mode in {"visual", "image", "baseline"}:
        return search_visual(query=query, top_k=top_k).to_dict()
    if mode == "hybrid":
        return search_hybrid(query=query, top_k=top_k).to_dict()
    if mode == "caption":
        return search_caption(query=query, top_k=top_k).to_dict()
    if mode == "ocr":
        return search_ocr(query=query, top_k=top_k).to_dict()
    return search_object(query=query, top_k=top_k).to_dict()
