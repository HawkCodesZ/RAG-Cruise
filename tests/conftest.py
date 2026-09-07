import os
import pytest
from unittest.mock import AsyncMock, MagicMock

# Ensure Settings() can construct even if the real .env isn't loaded in
# this test run — no real API calls happen, so the values just need to
# be present, not valid.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.invalid/v1")

from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_collection():
    """A MagicMock standing in for a chromadb Collection. Configure
    .get()/.query() return values per-test as needed."""
    return MagicMock()


@pytest.fixture
def mock_embed(monkeypatch):
    async def _fake_embed(texts):
        return [[0.0] * 5 for _ in texts]
    mock = AsyncMock(side_effect=_fake_embed)
    monkeypatch.setattr("app.vectorstore.embed_texts", mock)  # was "app.llm.embed_texts"
    return mock