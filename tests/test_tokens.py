"""Tests for JWT access + refresh tokens (Phase 6, 3b)."""
import jwt as pyjwt
import pytest
from sqlalchemy import text

from app.core.tokens import (
    create_access_token,
    decode_token,
    issue_refresh_token,
    refresh_token_is_valid,
    revoke_refresh_token,
)
from app.db import SessionLocal, engine
from app.models import Role
from app.services.user_service import create_user


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def test_access_token_roundtrip():
    tok = create_access_token("user-123", "clinician")
    claims = decode_token(tok)
    assert claims["sub"] == "user-123"
    assert claims["role"] == "clinician"
    assert claims["type"] == "access"


def test_tampered_token_rejected():
    tok = create_access_token("user-123", "clinician")
    with pytest.raises(pyjwt.PyJWTError):
        decode_token(tok + "tamper")


@pytest.mark.skipif(not _db_available(), reason="Postgres not reachable")
class TestRefreshTokens:
    @pytest.fixture
    def db(self):
        conn = engine.connect()
        trans = conn.begin()
        session = SessionLocal(bind=conn)
        try:
            yield session
        finally:
            session.close()
            trans.rollback()
            conn.close()

    def test_issue_then_valid(self, db):
        user = create_user(db, "rt1@x.local", "pw", role=Role.clinician)
        raw = issue_refresh_token(db, user.id)
        assert refresh_token_is_valid(db, raw) is True

    def test_revoke_invalidates(self, db):
        user = create_user(db, "rt2@x.local", "pw", role=Role.clinician)
        raw = issue_refresh_token(db, user.id)
        assert revoke_refresh_token(db, raw) is True
        assert refresh_token_is_valid(db, raw) is False

    def test_access_token_not_valid_as_refresh(self, db):
        user = create_user(db, "rt3@x.local", "pw", role=Role.clinician)
        access = create_access_token(user.id, "clinician")
        assert refresh_token_is_valid(db, access) is False
