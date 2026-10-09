"""Tests for the demo-account hardening.

Covers the HealthTrack pattern as applied to MedVision:
  * /auth/demo-login 404s when DEMO_LOGIN_ENABLED is off (the default).
  * A non-demo account sitting on the demo email is never reachable, and an
    unapproved demo account is rejected too -- the is_demo + approved gate
    means demo-login can only ever mint a token for a real demo account.
  * A demo clinician may read but not write (403 demo_read_only).
  * A demo admin may read (scoped to demo data) but not drive the RBAC
    lifecycle on anyone except the sacrificial demo clinician.
  * Normal /auth/login for real accounts is completely unaffected, and a real
    admin's writes are not blocked.

Same harness as the other API tests: TestClient with get_db overridden to a
rolled-back transaction; skipped if Postgres is unreachable. Because the demo
accounts are also present in the seeded dev DB, _make() purges any existing
row for an email (inside the rolled-back txn) before recreating it, so the
tests are independent of seed state and safe on a fresh CI database.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core.config import get_settings
from app.core.demo import DEMO_ACCOUNTS, DEMO_SACRIFICIAL_CLINICIAN
from app.core.deps import get_current_user
from app.db import SessionLocal, engine, get_db
from app.main import app
from app.models import Case, Patient, Role, Status, User
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


@pytest.fixture
def enable_demo(monkeypatch):
    """Turn DEMO_LOGIN_ENABLED on for the duration of a test (restored after)."""
    monkeypatch.setattr(get_settings(), "demo_login_enabled", True)
    yield


def _as_user(user: User) -> None:
    app.dependency_overrides[get_current_user] = lambda: user


def _purge(db, email: str) -> None:
    """Remove an existing user + its patients/cases, within the test txn."""
    existing = db.query(User).filter_by(email=email).first()
    if existing is None:
        return
    pats = db.query(Patient).filter_by(clinician_id=existing.id).all()
    for p in pats:
        db.query(Case).filter_by(patient_id=p.id).delete(synchronize_session=False)
        db.delete(p)
    db.query(Case).filter_by(clinician_id=existing.id).delete(synchronize_session=False)
    db.delete(existing)
    db.flush()


def _make(
    db,
    email,
    role=Role.clinician,
    *,
    is_demo=False,
    status=Status.approved,
    verified=True,
    password="secret123",
) -> User:
    """Create a user with forced gate state, purging any pre-existing row for
    the same email first (seeded demo accounts live at these addresses)."""
    email = email.strip().lower()
    _purge(db, email)
    u = create_user(db, email, password, role=role)
    u.is_demo = is_demo
    u.status = status
    u.email_verified = verified
    db.flush()
    return u


# --- demo-login endpoint -------------------------------------------------

def test_demo_login_404_when_flag_off(ctx):
    c, _ = ctx
    r = c.post("/auth/demo-login", json={"role": "clinician"})
    assert r.status_code == 404


def test_demo_login_rejects_unknown_role(ctx):
    c, _ = ctx
    r = c.post("/auth/demo-login", json={"role": "nurse"})
    assert r.status_code == 422


def test_demo_login_succeeds_for_seeded_demo_account(ctx, enable_demo):
    c, db = ctx
    _make(db, DEMO_ACCOUNTS["clinician"], Role.clinician, is_demo=True)
    r = c.post("/auth/demo-login", json={"role": "clinician"})
    assert r.status_code == 200, r.text
    assert "access_token" in r.json()
    assert "refresh_token" in r.cookies


def test_demo_login_unreachable_for_non_demo_account(ctx, enable_demo):
    # A real account (is_demo=False) parked on the demo email must NOT be
    # reachable, even with the flag on -- this is the core safety property.
    c, db = ctx
    _make(db, DEMO_ACCOUNTS["clinician"], Role.clinician, is_demo=False)
    r = c.post("/auth/demo-login", json={"role": "clinician"})
    assert r.status_code == 404


def test_demo_login_unreachable_when_not_approved(ctx, enable_demo):
    c, db = ctx
    _make(
        db,
        DEMO_ACCOUNTS["clinician"],
        Role.clinician,
        is_demo=True,
        status=Status.pending,
    )
    r = c.post("/auth/demo-login", json={"role": "clinician"})
    assert r.status_code == 404


# --- demo clinician: read but not write ----------------------------------

def test_demo_clinician_can_read_patients(ctx):
    c, db = ctx
    clinician = _make(db, DEMO_ACCOUNTS["clinician"], Role.clinician, is_demo=True)
    _as_user(clinician)
    r = c.get("/patients")
    assert r.status_code == 200


def test_demo_clinician_cannot_write_patients(ctx):
    c, db = ctx
    clinician = _make(db, DEMO_ACCOUNTS["clinician"], Role.clinician, is_demo=True)
    _as_user(clinician)
    r = c.post(
        "/patients",
        json={"first_name": "A", "last_name": "B", "sex": "female", "age": 40},
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "demo_read_only"


# --- demo admin: scoped reads, writes only on the sacrificial clinician ---

def test_demo_admin_read_is_scoped_to_demo(ctx):
    c, db = ctx
    admin = _make(db, DEMO_ACCOUNTS["admin"], Role.admin, is_demo=True)
    _make(db, DEMO_ACCOUNTS["clinician"], Role.clinician, is_demo=True)
    _make(db, "real-doc@medvision.dev", Role.clinician, is_demo=False)
    _as_user(admin)
    r = c.get("/admin/clinicians")
    assert r.status_code == 200
    emails = {row["email"] for row in r.json()}
    assert DEMO_ACCOUNTS["clinician"] in emails
    assert "real-doc@medvision.dev" not in emails


def test_demo_admin_cannot_change_status_of_regular_clinician(ctx):
    c, db = ctx
    admin = _make(db, DEMO_ACCOUNTS["admin"], Role.admin, is_demo=True)
    target = _make(db, DEMO_ACCOUNTS["clinician"], Role.clinician, is_demo=True)
    _as_user(admin)
    r = c.post(
        f"/admin/clinicians/{target.id}/status", json={"status": "suspended"}
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "demo_read_only"


def test_demo_admin_can_change_status_of_sacrificial(ctx):
    c, db = ctx
    admin = _make(db, DEMO_ACCOUNTS["admin"], Role.admin, is_demo=True)
    sac = _make(db, DEMO_SACRIFICIAL_CLINICIAN, Role.clinician, is_demo=True)
    _as_user(admin)
    r = c.post(
        f"/admin/clinicians/{sac.id}/status", json={"status": "suspended"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "suspended"


# --- real accounts are completely unaffected -----------------------------

def test_real_login_still_works(ctx):
    c, db = ctx
    _make(db, "realuser@medvision.dev", Role.clinician, is_demo=False)
    r = c.post(
        "/auth/login",
        json={"email": "realuser@medvision.dev", "password": "secret123"},
    )
    assert r.status_code == 200, r.text
    assert "access_token" in r.json()


def test_real_admin_write_not_blocked(ctx):
    c, db = ctx
    admin = _make(db, "realadmin@medvision.dev", Role.admin, is_demo=False)
    target = _make(db, "pending-doc@medvision.dev", Role.clinician,
                   is_demo=False, status=Status.pending, verified=False)
    _as_user(admin)
    r = c.post(
        f"/admin/clinicians/{target.id}/status", json={"status": "approved"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "approved"
