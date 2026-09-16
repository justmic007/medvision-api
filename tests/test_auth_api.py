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
    assert "refresh_token" in r.json()


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
    tokens = c.post(
        "/auth/login", json={"email": "d@example.com", "password": "secret123"}
    ).json()
    rt = tokens["refresh_token"]

    assert c.post("/auth/refresh", json={"refresh_token": rt}).status_code == 200
    assert c.post("/auth/logout", json={"refresh_token": rt}).status_code == 200
    # revoked -> refresh now fails
    assert c.post("/auth/refresh", json={"refresh_token": rt}).status_code == 401
