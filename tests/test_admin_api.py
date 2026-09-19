"""Tests for admin endpoints + auth enforcement (Phase 6, 3e)."""
import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core.deps import get_current_user
from app.db import SessionLocal, engine, get_db
from app.main import app
from app.models import Role, Status, User
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
    """Rolled-back DB session + a TestClient with get_db overridden to it."""
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


def _as_user(user: User):
    app.dependency_overrides[get_current_user] = lambda: user


def test_analyze_requires_auth():
    # No auth override here -> the protected endpoint must reject.
    app.dependency_overrides.pop(get_current_user, None)
    c = TestClient(app)
    resp = c.post(
        "/analyze",
        files={"file": ("x.jpg", io.BytesIO(b"data"), "image/jpeg")},
    )
    assert resp.status_code == 401


def test_admin_endpoint_rejects_clinician(ctx):
    c, db = ctx
    clinician = create_user(db, "c@medvision.dev", "pw", role=Role.clinician)
    db.flush()
    _as_user(clinician)
    resp = c.get("/admin/pending-clinicians")
    assert resp.status_code == 403


def test_admin_endpoint_rejects_no_token(ctx):
    c, _ = ctx
    app.dependency_overrides.pop(get_current_user, None)
    resp = c.get("/admin/pending-clinicians")
    assert resp.status_code == 401


def test_admin_can_list_and_approve(ctx):
    c, db = ctx
    admin = create_user(db, "a@medvision.dev", "pw", role=Role.admin)
    pending = create_user(db, "p@medvision.dev", "pw", role=Role.clinician)
    db.flush()
    _as_user(admin)

    # list includes the pending clinician
    r = c.get("/admin/pending-clinicians")
    assert r.status_code == 200
    ids = [u["id"] for u in r.json()]
    assert pending.id in ids

    # approve it
    r = c.post(f"/admin/clinicians/{pending.id}/approve")
    assert r.status_code == 200
    assert r.json()["status"] == "approved"

    # re-approving a non-pending clinician is a conflict
    r = c.post(f"/admin/clinicians/{pending.id}/approve")
    assert r.status_code == 409
