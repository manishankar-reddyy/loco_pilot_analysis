"""ASGI entry point: `uvicorn server:app` (or `python server.py`)."""
import os

from loco_monitor.api import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
