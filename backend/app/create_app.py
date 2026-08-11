import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_application():
    """Factory function to initialize and configure the FastAPI app."""
    application = FastAPI(
        title="Multi-modal Search & Media API",
        description="Phase 1 Visual CLIP Search, Video Streaming, and Telemetry Engine",
        version="1.0.0",
    )

    # Configure Middleware (Add CORS, Trusted Hosts, etc. here)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===========================
    # Include Routers safely
    # ===========================
    # Search router
    from backend.app.api.search import search_router
    if search_router is not None:
        application.include_router(search_router)
        print("[INFO] Search router has been loaded.\n")
    else:
        print("[!Warning] Search router is not working properly!\n")

    # Video streaming router
    from backend.app.api.video import video_router
    if video_router is not None:
        application.include_router(video_router)
        print("[INFO] Video router has been loaded.\n")
    else:
        print("[!Warning] Video router is not working properly!\n")

    # Telemetry log (user log, submission) router
    from backend.app.api.telemetry import telemetry_router
    if telemetry_router is not None:
        application.include_router(telemetry_router)
        print("[INFO] Telemetry router has been loaded.\n")
    else:
        print("[!Warning] Telemetry router is not working properly!\n")

    return application
