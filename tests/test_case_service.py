"""Tests for case persistence (Phase 6, increment 2).

Each test runs in a transaction rolled back afterward, so tests never leave
rows behind. create_case only flushes (caller owns the commit), which keeps
this isolation clean. Skipped if Postgres isn't reachable.
"""
import pytest
from sqlalchemy import text

from app.db import SessionLocal, engine
from app.models import Patient, Role, Sex, User
from app.services.case_service import _results_json, create_case
from app.services.literature import Article
from app.services.orchestrator import AnalysisResult, FindingAnalysis


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable")


@pytest.fixture
def db():
    connection = engine.connect()
    trans = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()


def _fake_result() -> AnalysisResult:
    return AnalysisResult(
        model_name="test-model",
        findings=[
            FindingAnalysis(
                name="Nodule",
                probability=0.9,
                threshold=0.1,
                heatmap_base64="FAKE",
                articles=[Article(pmid="1", title="t", journal="j", year="2020")],
            )
        ],
        disclaimer="NOT diagnostic.",
    )


def test_results_json_excludes_base64():
    js = _results_json(_fake_result())
    assert js["findings"][0]["has_heatmap"] is True
    assert "heatmap_base64" not in js["findings"][0]
    assert js["findings"][0]["articles"][0]["pmid"] == "1"


def test_create_case_persists_and_links(db):
    clinician = User(email="t@x.local", hashed_password="h", role=Role.clinician)
    db.add(clinician)
    db.flush()
    patient = Patient(mrn="T-1", sex=Sex.male, age=40, clinician_id=clinician.id)
    db.add(patient)
    db.flush()

    case = create_case(
        db, _fake_result(), patient_id=patient.id, clinician_id=clinician.id
    )
    assert case.id is not None
    assert case.model_name == "test-model"
    assert len(case.results["findings"]) == 1
    assert case.patient.mrn == "T-1"
    assert case.clinician.email == "t@x.local"
