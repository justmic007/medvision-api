"""Tests for patient endpoints + analyze->case wiring (Phase 6 capstone)."""
import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core.deps import get_current_user
from app.db import SessionLocal, engine, get_db
from app.main import app
from app.models import Case, Patient, Role, User
from app.services.user_service import create_user


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable")


@pytest.fixture
def ctx():
    connection = engine.connect()
    trans = connection.begin()
    session = SessionLocal(bind=connection)
    app.dependency_overrides[get_db] = lambda: session
    try:
        yield TestClient(app), session
    finally:
        app.dependency_overrides.clear()
        session.close()
        trans.rollback()
        connection.close()


def _as(user):
    app.dependency_overrides[get_current_user] = lambda: user


def test_create_and_list_patient_scoped_to_clinician(ctx):
    c, db = ctx
    doc = create_user(db, "doc1@medvision.dev", "pw", role=Role.clinician)
    other = create_user(db, "doc2@medvision.dev", "pw", role=Role.clinician)
    db.flush()

    _as(doc)
    r = c.post("/patients", json={"mrn": "P-1", "sex": "female", "age": 40})
    assert r.status_code == 201
    pid = r.json()["id"]

    # doc sees their patient
    r = c.get("/patients")
    assert r.status_code == 200
    assert pid in [p["id"] for p in r.json()]

    # other clinician does NOT see doc's patient (ownership isolation)
    _as(other)
    r = c.get("/patients")
    assert pid not in [p["id"] for p in r.json()]


def test_patient_endpoint_requires_auth(ctx):
    c, _ = ctx
    app.dependency_overrides.pop(get_current_user, None)
    assert c.get("/patients").status_code == 401


def test_analyze_persists_case_for_patient(ctx, monkeypatch):
    import app.api.analyze as analyze_module
    from app.services.orchestrator import AnalysisResult, FindingAnalysis

    c, db = ctx
    doc = create_user(db, "doc3@medvision.dev", "pw", role=Role.clinician)
    db.flush()
    patient = Patient(mrn="P-9", sex=__import__("app.models", fromlist=["Sex"]).Sex.male,
                      age=50, clinician_id=doc.id)
    db.add(patient); db.flush()

    # Stub the orchestrator so no model runs.
    fake = type("F", (), {})()
    fake.analyze = lambda *a, **k: AnalysisResult(
        model_name="test-model",
        findings=[FindingAnalysis(name="Nodule", probability=0.9, threshold=0.1)],
        disclaimer="NOT diagnostic.",
    )
    monkeypatch.setattr(analyze_module, "get_orchestrator", lambda: fake)

    _as(doc)
    r = c.post(
        "/analyze",
        files={"file": ("x.jpg", io.BytesIO(b"data"), "image/jpeg")},
        data={"patient_id": patient.id},
    )
    assert r.status_code == 200
    # a case now exists for this patient
    case = db.query(Case).filter_by(patient_id=patient.id).first()
    assert case is not None
    assert case.clinician_id == doc.id
    assert case.model_name == "test-model"
