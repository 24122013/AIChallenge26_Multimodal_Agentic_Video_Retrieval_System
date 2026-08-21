from backend.app.core.environment import load_project_env

# Load the canonical repository environment before importing application
# modules whose settings are evaluated at import time.
load_project_env()

# Create the app instance that Uvicorn/Gunicorn will point to
from backend.app.create_app import create_application
app = create_application()

@app.get("/health", tags=["health"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
