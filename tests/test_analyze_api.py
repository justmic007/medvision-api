"""Tests for the /analyze endpoint.

Mocks the orchestrator so the test exercises the HTTP layer (upload handling,
validation, response shape) without loading models or hitting the network.
"""
import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.orchestrator import AnalysisResult
from app.services.orchestrator import FindingAnalysis
from app.services.literature import Article
import app.api.analyze as analyze_module
from app.core.deps import get_current_user
from app.models import Role, User

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_orchestrator(monkeypatch):
    """Replace the endpoint's orchestrator with a fake one."""
    fake_result = AnalysisResult(
        model_name="test-model",
        findings=[
            FindingAnalysis(
                name="Nodule",
                probability=0.9,
                threshold=0.1,
                heatmap_base64="FAKEB64",
                articles=[Article(pmid="1", title="t", journal="j", year="2020")],
            )
        ],
        disclaimer="NOT a diagnostic tool.",
    )
    fake = MagicMock()
    fake.analyze.return_value = fake_result
    monkeypatch.setattr(analyze_module, "get_orchestrator", lambda: fake)

    # Override auth so the protected endpoint has an authenticated user
    # without needing a real token.
    fake_user = User(
        id="u1", email="doc@medvision.dev", hashed_password="x", role=Role.clinician
    )
    app.dependency_overrides[get_current_user] = lambda: fake_user
    yield
    app.dependency_overrides.clear()


def test_analyze_accepts_image_and_returns_analysis():
    img_bytes = io.BytesIO(b"fake image content")
    resp = client.post(
        "/analyze",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "test-model"
    assert body["num_present"] == 1
    assert body["findings"][0]["name"] == "Nodule"
    assert body["findings"][0]["heatmap_base64"] == "FAKEB64"
    assert "diagnostic" in body["disclaimer"].lower()


def test_analyze_rejects_unsupported_file_type():
    resp = client.post(
        "/analyze",
        files={"file": ("notes.txt", io.BytesIO(b"text"), "text/plain")},
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()
