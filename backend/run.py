from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Create the app instance that Uvicorn/Gunicorn will point to
from backend.app.create_app import create_application
app = create_application()

@app.get("/health", tags=["health"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}