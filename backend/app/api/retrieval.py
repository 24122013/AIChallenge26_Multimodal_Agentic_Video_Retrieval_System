"""Retrieval API endpoints.

FastAPI is optional in the current repo, so service functions remain importable
even when the web dependency has not been installed yet.
"""
from __future__ import annotations

from backend.app.services.retrieval.retrieval_manager import (
    search_asr,
    search_caption,
    search_hybrid,
    search_object,
    search_ocr,
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
        return _response(lambda: search_visual(body.query, body.top_k).to_dict())

    @router.post("/hybrid")
    def hybrid_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(lambda: search_hybrid(body.query, body.top_k).to_dict())

    @router.post("/caption")
    def caption_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(lambda: search_caption(body.query, body.top_k).to_dict())

    @router.post("/ocr")
    def ocr_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(lambda: search_ocr(body.query, body.top_k).to_dict())

    @router.post("/asr")
    def asr_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(lambda: search_asr(body.query, body.top_k).to_dict())

    @router.post("/object")
    def object_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(lambda: search_object(body.query, body.top_k).to_dict())

    @router.post("/temporal")
    def temporal_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(
            lambda: {
                "query": body.query,
                "top_k": body.top_k,
                "results": [
                    match.to_dict()
                    for match in search_temporal(body.query, body.top_k)
                ],
            }
        )

    def _response(callable_) -> dict:
        try:
            return {
                "success": True,
                "data": callable_(),
                "message": None,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
else:
    router = None


def visual_search(query: str, top_k: int = 20) -> dict:
    """Plain Python wrapper useful before the FastAPI app is wired."""
    return search_visual(query=query, top_k=top_k).to_dict()


def hybrid_search(query: str, top_k: int = 20) -> dict:
    return search_hybrid(query=query, top_k=top_k).to_dict()


def caption_search(query: str, top_k: int = 20) -> dict:
    return search_caption(query=query, top_k=top_k).to_dict()


def ocr_search(query: str, top_k: int = 20) -> dict:
    return search_ocr(query=query, top_k=top_k).to_dict()


def asr_search(query: str, top_k: int = 20) -> dict:
    return search_asr(query=query, top_k=top_k).to_dict()


def object_search(query: str, top_k: int = 20) -> dict:
    return search_object(query=query, top_k=top_k).to_dict()


def temporal_search(query: str, top_k: int = 20) -> list[dict]:
    return [
        match.to_dict()
        for match in search_temporal(query=query, top_k=top_k)
    ]
