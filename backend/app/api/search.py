"""Unified search API wrappers for Retrieval Phase 1-3."""
from __future__ import annotations

from backend.app.services.retrieval.retrieval_manager import (
    search_caption,
    search_object,
    search_ocr,
    search_online,
    search_visual,
)
from backend.app.services.retrieval.qa_pipeline import RequiredQaPipelineError
from backend.app.services.trake import RequiredTrakePipelineError
from backend.app.services.submission.csv_export import (
    SubmissionExportError,
    export_query_csv,
)

try:  # pragma: no cover - depends on optional API runtime.
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse, Response
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
    router = APIRouter(prefix="/api/search", tags=["search"])

    class SearchBody(BaseModel):
        query: str
        mode: str = "online"
        top_k: int = Field(default=20, ge=1, le=200)
        task_mode: str = "auto"
        expanded_queries: list[str] = Field(default_factory=list, max_length=20)
        include_context: bool | None = None
        debug: bool | None = None

    class ExportBody(BaseModel):
        query: str
        task: str
        top_k: int = Field(default=100, ge=1, le=100)

    @router.post("")
    def search_endpoint(body: SearchBody) -> dict:
        try:
            return {
                "success": True,
                "data": _dispatch_search(
                    body.query,
                    body.top_k,
                    body.mode,
                    task_mode=body.task_mode,
                    expanded_queries=body.expanded_queries,
                    **_online_request_options(
                        include_context=body.include_context,
                        debug=body.debug,
                    ),
                ),
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - convert service errors to API response.
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/export")
    def export_search_endpoint(body: ExportBody) -> Response:
        try:
            exported = export_query_csv(body.query, body.task, body.top_k)
            return Response(
                content=exported.content.encode("utf-8"),
                media_type="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{exported.filename}"'
                    )
                },
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except RequiredQaPipelineError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RequiredTrakePipelineError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except SubmissionExportError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
else:
    router = None


def search(
    query: str,
    top_k: int = 20,
    mode: str = "online",
    *,
    task_mode: str = "auto",
    expanded_queries: list[str] | None = None,
    include_context: bool | None = None,
    debug: bool | None = None,
) -> dict:
    return _dispatch_search(
        query,
        top_k,
        mode,
        task_mode=task_mode,
        expanded_queries=expanded_queries,
        **_online_request_options(
            include_context=include_context,
            debug=debug,
        ),
    )


def _dispatch_search(
    query: str,
    top_k: int,
    mode: str,
    *,
    task_mode: str = "auto",
    expanded_queries: list[str] | None = None,
    include_context: bool | None = None,
    debug: bool | None = None,
) -> dict:
    normalized = mode.casefold().strip()
    online_options = _online_request_options(
        include_context=include_context,
        debug=debug,
    )
    if normalized in {"online", "auto"}:
        return search_online(
            query=query,
            task=task_mode,
            top_k=top_k,
            expanded_queries=expanded_queries or [],
            **online_options,
        )
    if normalized in {"kis", "hybrid"}:
        return search_online(
            query=query,
            task="kis",
            top_k=top_k,
            **online_options,
        )
    if normalized == "kis_visual":
        return search_online(
            query=query,
            task="kis_visual",
            top_k=top_k,
            **online_options,
        )
    if normalized == "kis_temporal":
        return search_online(
            query=query,
            task="kis_temporal",
            top_k=top_k,
            **online_options,
        )
    if normalized == "avs":
        return search_online(
            query=query,
            task="avs",
            top_k=top_k,
            **online_options,
        )
    if normalized in {
        "qa",
        "qa_evidence",
        "question",
        "question_answering",
        "qa_answer",
        "grounded_qa",
    }:
        return search_online(
            query=query,
            task="qa",
            top_k=min(5, int(top_k)),
            expanded_queries=expanded_queries or [],
            **online_options,
        )
    if normalized == "temporal":
        return search_online(
            query=query,
            task="temporal",
            top_k=top_k,
            **online_options,
        )
    if normalized == "trake":
        return search_online(
            query=query,
            task="trake",
            top_k=min(100, int(top_k)),
            **online_options,
        )

    # Modality-only modes are retained as diagnostics. User-facing task routes
    # above all pass through the canonical OnlinePipeline.
    if normalized in {"visual", "image", "baseline"}:
        return search_visual(query=query, top_k=top_k).to_dict()
    if normalized == "caption":
        return search_caption(query=query, top_k=top_k).to_dict()
    if normalized in {"ocr", "ocr_text"}:
        return search_ocr(query=query, top_k=top_k).to_dict()
    if normalized in {"object", "objects"}:
        return search_object(query=query, top_k=top_k).to_dict()
    raise ValueError(
        "Unsupported search mode. Expected online, KIS, kis_visual, kis_temporal, AVS, "
        "temporal, TRAKE, QA, "
        "or a modality diagnostic (visual, caption, OCR, object)."
    )
