"""Tests for the auth endpoints (Phase 6, 3d).

Uses TestClient with the get_db dependency overridden to a rolled-back
transaction, so tests never leave users behind. Skipped if Postgres is down.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import SessionLocal, engine, get_db
from app.main import app
from app.models import Status, User


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable")


@pytest.fixture
def client():
    connection = engine.connect()
    trans = connection.begin()
    session = SessionLocal(bind=connection)

    def _override():
        yield session

    app.dependency_overrides[get_db] = _override
    try:
        yield TestClient(app), session
    finally:
        app.dependency_overrides.clear()
        session.close()
        trans.rollback()
        connection.close()


def _register(c, email="a@example.com", pw="secret123"):
    return c.post("/auth/register", json={"email": email, "password": pw})


def test_register_rejects_weak_passwords(client):
    c, _ = client
    for weak in ["short1", "abcdefgh", "12345678", ""]:
        r = c.post(
            "/auth/register",
            json={"email": "weak@example.com", "password": weak},
        )
        assert r.status_code == 422, f"weak password {weak!r} should be rejected"


def test_register_creates_pending_unverified(client):
    c, _ = client
    r = _register(c)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "pending"
    assert body["email_verified"] is False


def test_login_blocked_until_verified_and_approved(client):
    c, db = client
    _register(c, "b@example.com")
    # unverified -> 403
    r = c.post("/auth/login", json={"email": "b@example.com", "password": "secret123"})
    assert r.status_code == 403

    # verify + approve, then login succeeds
    user = db.query(User).filter_by(email="b@example.com").first()
    user.email_verified = True
    user.status = Status.approved
    db.flush()
    r = c.post("/auth/login", json={"email": "b@example.com", "password": "secret123"})
    assert r.status_code == 200
    assert "access_token" in r.json()
    # Refresh token is set as an httpOnly cookie, not returned in the body.
    assert "refresh_token" not in r.json()
    assert "refresh_token" in r.cookies


def test_wrong_password_generic_error(client):
    c, db = client
    _register(c, "c@example.com")
    user = db.query(User).filter_by(email="c@example.com").first()
    user.email_verified = True
    user.status = Status.approved
    db.flush()
    r = c.post("/auth/login", json={"email": "c@example.com", "password": "WRONG"})
    assert r.status_code == 401
    assert "Invalid email or password" in r.json()["detail"]


def test_refresh_and_logout_revokes(client):
    c, db = client
    _register(c, "d@example.com")
    user = db.query(User).filter_by(email="d@example.com").first()
    user.email_verified = True
    user.status = Status.approved
    db.flush()
    # Login sets the refresh cookie; TestClient persists it across requests.
    r = c.post(
        "/auth/login", json={"email": "d@example.com", "password": "secret123"}
    )
    assert r.status_code == 200

    # Refresh reads the cookie (no body); works while valid.
    assert c.post("/auth/refresh").status_code == 200
    # Logout revokes the token and clears the cookie.
    assert c.post("/auth/logout").status_code == 200
    # After logout the cookie is cleared, so refresh has no token -> 401.
    assert c.post("/auth/refresh").status_code == 401


def test_me_returns_current_user(client):
    c, db = client
    _register(c, "e@example.com")
    user = db.query(User).filter_by(email="e@example.com").first()
    user.email_verified = True
    user.status = Status.approved
    db.flush()
    access = c.post(
        "/auth/login", json={"email": "e@example.com", "password": "secret123"}
    ).json()["access_token"]

    r = c.get("/auth/me", headers={"Authorization": f"Bearer {access}"})
    assert r.status_code == 200
    assert r.json()["email"] == "e@example.com"

    # No token -> 401.
    c.cookies.clear()
    assert c.get("/auth/me").status_code == 401
