from backend.app.create_app import create_application

# Create the app instance that Uvicorn/Gunicorn will point to
app = create_application()

@app.get("/health", tags=["health"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}