"""Telemetry API for logging user's interaction and submission."""
from __future__ import annotations

try:  # pragma: no cover - depends on optional API runtime.
    import json
    import logging
    import time
    from typing import Dict, Any
    from pydantic import BaseModel, Field
    from fastapi import APIRouter, Request

    # Base directory for video sources
    telemetry_router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])
    logger = logging.getLogger(__name__)

    # POST /api/video/telemetry/log
    class TelemetryPayload(BaseModel):
        event_type: str
        timestamp: float = Field(default_factory=time.time)
        latency_ms: int = 0
        details: Dict[str, Any] = Field(default_factory=dict)

    @telemetry_router.post("/log")
    async def log_interaction(payload: TelemetryPayload, request: Request):
        """Ingests granular frontend user engagement metrics and submission."""
        user_id = getattr(request.state, "user", None) 
        
        log_entry = {
            "user_id": user_id,
            "event_type": payload.event_type,
            "timestamp": payload.timestamp,
            "latency_ms": payload.latency_ms,
            "details": payload.details,
        }
        
        logger.info(f"TELEMETRY_LOG: {json.dumps(log_entry)}")
        
        return {"status": "logged"}

except ImportError:  # pragma: no cover
    telemetry_router = None