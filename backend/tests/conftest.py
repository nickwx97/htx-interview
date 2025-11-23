import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure the backend package root is on sys.path so tests can import `main` reliably
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    # Prefer absolute import when running from the backend folder
    from main import app as fastapi_app
except Exception:
    # Fallback: try relative import path (for other invocation contexts)
    from ..main import app as fastapi_app


@pytest.fixture(scope="session")
def client():
    """Test client for the FastAPI app."""
    with TestClient(fastapi_app) as c:
        yield c
