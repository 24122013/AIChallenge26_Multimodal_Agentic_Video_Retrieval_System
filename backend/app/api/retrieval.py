"""Retrieval API endpoints.

FastAPI is optional in the current repo, so service functions remain importable
even when the web dependency has not been installed yet.
"""
from __future__ import annotations

from backend.app.services.retrieval.retrieval_manager import (
    search_caption,
    search_object,
    search_ocr,
    search_online,
    search_trake,
    search_visual,
)
from backend.app.services.retrieval.qa_pipeline import RequiredQaPipelineError
from backend.app.services.trake import RequiredTrakePipelineError, TrakeStageDeadlineExceeded

try:  # pragma: no cover - depends on optional API runtime.
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None
    HTTPException = None
    JSONResponse = None
    BaseModel = object
    Field = None


def _online_request_options(
    *,
    include_context: bool | None,
    debug: bool | None,
) -> dict[str, bool]:
    """Return only explicit request overrides, preserving runtime defaults."""

    options: dict[str, bool] = {}
    if include_context is not None:
        options["include_context"] = include_context
    if debug is not None:
        options["debug"] = debug
    return options


if APIRouter is not None:
    router = APIRouter(prefix="/retrieval", tags=["retrieval"])

    class VisualSearchBody(BaseModel):
        query: str
        top_k: int = Field(default=20, ge=1, le=200)

    class QaSearchBody(BaseModel):
        query: str
        top_k: int = Field(default=20, ge=1, le=100)
        task_mode: str = "qa"
        expanded_queries: list[str] = Field(default_factory=list, max_length=20)

    class OnlineSearchBody(BaseModel):
        query: str
        task: str = "auto"
        top_k: int = Field(default=20, ge=1, le=200)
        expanded_queries: list[str] = Field(default_factory=list, max_length=20)
        include_context: bool | None = None
        debug: bool | None = None

    class TrakeSearchBody(BaseModel):
        query: str
        top_k: int = Field(default=100, ge=1, le=100)

    @router.post("/online")
    def online_search_endpoint(body: OnlineSearchBody) -> dict:
        return _response(
            lambda: search_online(
                body.query,
                task=body.task,
                top_k=body.top_k,
                expanded_queries=body.expanded_queries,
                **_online_request_options(
                    include_context=body.include_context,
                    debug=body.debug,
                ),
            )
        )

    @router.post("/visual")
    def visual_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(lambda: search_visual(body.query, body.top_k).to_dict())

    @router.post("/hybrid")
    def hybrid_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(
            lambda: search_online(body.query, task="kis", top_k=body.top_k)
        )

    @router.post("/caption")
    def caption_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(lambda: search_caption(body.query, body.top_k).to_dict())

    @router.post("/ocr")
    def ocr_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(lambda: search_ocr(body.query, body.top_k).to_dict())

    @router.post("/object")
    def object_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(lambda: search_object(body.query, body.top_k).to_dict())

    @router.post("/temporal")
    def temporal_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(
            lambda: search_online(body.query, task="temporal", top_k=body.top_k)
        )

    @router.post("/kis-visual")
    def kis_visual_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(
            lambda: search_online(body.query, task="kis_visual", top_k=body.top_k)
        )

    @router.post("/kis-temporal")
    def kis_temporal_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(
            lambda: search_online(
                body.query,
                task="kis_temporal",
                top_k=body.top_k,
            )
        )

    @router.post("/trake")
    def trake_search_endpoint(body: TrakeSearchBody) -> dict:
        return _response(lambda: search_trake(body.query, top_k=100))

    @router.post("/qa-evidence")
    def qa_evidence_search_endpoint(body: VisualSearchBody) -> dict:
        return _response(
            lambda: search_online(body.query, task="qa", top_k=body.top_k)
        )

    @router.post("/qa")
    def qa_search_endpoint(body: QaSearchBody) -> dict:
        return _response(
            lambda: search_online(
                body.query,
                task="qa",
                top_k=body.top_k,
                expanded_queries=body.expanded_queries,
            )
        )

    def _response(callable_) -> dict:
        try:
            return {
                "success": True,
                "data": callable_(),
                "message": None,
            }
        except RequiredQaPipelineError as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "data": exc.response,
                    "message": str(exc),
                },
            )
        except RequiredTrakePipelineError as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "data": exc.response,
                    "message": str(exc),
                },
            )
        except TrakeStageDeadlineExceeded as exc:
            return JSONResponse(
                status_code=504,
                content={
                    "success": False,
                    "data": exc.response,
                    "message": str(exc),
                },
            )
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
    return search_online(query=query, task="kis", top_k=top_k)


def kis_visual_search(query: str, top_k: int = 20) -> dict:
    return search_online(query=query, task="kis_visual", top_k=top_k)


def caption_search(query: str, top_k: int = 20) -> dict:
    return search_caption(query=query, top_k=top_k).to_dict()


def ocr_search(query: str, top_k: int = 20) -> dict:
    return search_ocr(query=query, top_k=top_k).to_dict()


def object_search(query: str, top_k: int = 20) -> dict:
    return search_object(query=query, top_k=top_k).to_dict()


def temporal_search(query: str, top_k: int = 20) -> list[dict]:
    return search_online(query=query, task="temporal", top_k=top_k)["candidates"]


def kis_temporal_search(query: str, top_k: int = 20) -> dict:
    return search_online(query=query, task="kis_temporal", top_k=top_k)


def trake_search(query: str, top_k: int = 100) -> dict:
    return search_trake(query=query, top_k=100)


def qa_evidence_search(question: str, top_k: int = 10) -> dict:
    return search_online(query=question, task="qa", top_k=top_k)


def qa_search(
    query: str,
    top_k: int = 20,
    *,
    task_mode: str = "qa",
    expanded_queries: list[str] | None = None,
) -> dict:
    return search_online(
        query=query,
        task="qa",
        top_k=top_k,
        expanded_queries=expanded_queries or [],
    )
