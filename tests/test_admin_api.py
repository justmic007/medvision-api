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


def _set_status(c, user_id, status):
    return c.post(f"/admin/clinicians/{user_id}/status", json={"status": status})


def test_admin_lists_all_clinicians(ctx):
    c, db = ctx
    admin = create_user(db, "a@medvision.dev", "pw", role=Role.admin)
    p1 = create_user(db, "p1@medvision.dev", "pw", role=Role.clinician)
    p2 = create_user(db, "p2@medvision.dev", "pw", role=Role.clinician)
    p2.status = Status.approved
    db.flush()
    _as_user(admin)

    r = c.get("/admin/clinicians")
    assert r.status_code == 200
    ids = {u["id"] for u in r.json()}
    assert {p1.id, p2.id} <= ids  # all clinicians, any status


def test_admin_approve_and_reject(ctx):
    c, db = ctx
    admin = create_user(db, "a@medvision.dev", "pw", role=Role.admin)
    a = create_user(db, "a1@medvision.dev", "pw", role=Role.clinician)
    r = create_user(db, "r1@medvision.dev", "pw", role=Role.clinician)
    db.flush()
    _as_user(admin)

    assert _set_status(c, a.id, "approved").json()["status"] == "approved"
    assert _set_status(c, r.id, "rejected").json()["status"] == "rejected"


def test_admin_full_lifecycle_suspend_reinstate(ctx):
    c, db = ctx
    admin = create_user(db, "a@medvision.dev", "pw", role=Role.admin)
    doc = create_user(db, "doc@medvision.dev", "pw", role=Role.clinician)
    db.flush()
    _as_user(admin)

    # pending -> approved
    assert _set_status(c, doc.id, "approved").json()["status"] == "approved"
    # approved -> suspended
    assert _set_status(c, doc.id, "suspended").json()["status"] == "suspended"
    # suspended -> approved (reinstate)
    assert _set_status(c, doc.id, "approved").json()["status"] == "approved"


def test_admin_rejects_invalid_transition(ctx):
    c, db = ctx
    admin = create_user(db, "a@medvision.dev", "pw", role=Role.admin)
    doc = create_user(db, "doc2@medvision.dev", "pw", role=Role.clinician)
    db.flush()
    _as_user(admin)

    # pending -> suspended is not allowed
    r = _set_status(c, doc.id, "suspended")
    assert r.status_code == 400


def test_status_change_requires_admin(ctx):
    c, db = ctx
    clinician = create_user(db, "cc@medvision.dev", "pw", role=Role.clinician)
    target = create_user(db, "tt@medvision.dev", "pw", role=Role.clinician)
    db.flush()
    _as_user(clinician)
    r = _set_status(c, target.id, "approved")
    assert r.status_code == 403
