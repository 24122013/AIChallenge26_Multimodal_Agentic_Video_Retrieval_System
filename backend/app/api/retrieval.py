"""Retrieval API endpoints.

FastAPI is optional in the current repo, so service functions remain importable
even when the web dependency has not been installed yet.
"""
from __future__ import annotations

from backend.app.services.retrieval.retrieval_manager import (
    search_hybrid,
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
    router = APIRouter(prefix="/retrieval", tags=["retrieval"])

    class VisualSearchBody(BaseModel):
        query: str
        top_k: int = Field(default=20, ge=1, le=200)

    @router.post("/visual")
    def visual_search_endpoint(body: VisualSearchBody) -> dict:
        try:
            return {
                "success": True,
                "data": search_visual(body.query, body.top_k).to_dict(),
                "message": None,
            }
        except Exception as exc:  # noqa: BLE001 - convert service errors to API response.
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/hybrid")
    def hybrid_search_endpoint(body: VisualSearchBody) -> dict:
        try:
            return {
                "success": True,
                "data": search_hybrid(body.query, body.top_k).to_dict(),
                "message": None,
            }
        except Exception as exc:  # noqa: BLE001 - convert service errors to API response.
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/temporal")
    def temporal_search_endpoint(body: VisualSearchBody) -> dict:
        try:
            return {
                "success": True,
                "data": [match.to_dict() for match in search_temporal(body.query, body.top_k)],
                "message": None,
            }
        except Exception as exc:  # noqa: BLE001 - convert service errors to API response.
            raise HTTPException(status_code=500, detail=str(exc)) from exc
else:
    router = None


def visual_search(query: str, top_k: int = 20) -> dict:
    """Plain Python wrapper useful before the FastAPI app is wired."""
    return search_visual(query=query, top_k=top_k).to_dict()


def hybrid_search(query: str, top_k: int = 20) -> dict:
    """Plain Python wrapper for Phase 3 hybrid reranking."""
    return search_hybrid(query=query, top_k=top_k).to_dict()


def temporal_search(query: str, top_k: int = 20) -> list[dict]:
    """Plain Python wrapper for Phase 3 temporal search."""
    return [match.to_dict() for match in search_temporal(query=query, top_k=top_k)]
