"""Tests for password hashing and user creation (Phase 6 auth foundation)."""
import pytest
from sqlalchemy import text

from app.core.security import hash_password, verify_password
from app.db import SessionLocal, engine
from app.models import Role, Status
from app.services.user_service import create_user


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def test_password_hash_and_verify():
    h = hash_password("s3cret-pass")
    assert h != "s3cret-pass"                    # not plaintext
    assert h.startswith("$2b$") or h.startswith("$2a$")   # bcrypt
    assert verify_password("s3cret-pass", h)     # correct verifies
    assert not verify_password("wrong", h)       # wrong rejected


@pytest.mark.skipif(not _db_available(), reason="Postgres not reachable")
class TestUserCreation:
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

    def test_clinician_starts_pending_unverified(self, db):
        u = create_user(db, "doc@x.local", "pw", role=Role.clinician)
        assert u.status == Status.pending
        assert u.email_verified is False
        assert verify_password("pw", u.hashed_password)

    def test_admin_starts_approved_verified(self, db):
        u = create_user(db, "admin@x.local", "pw", role=Role.admin)
        assert u.status == Status.approved
        assert u.email_verified is True

    def test_email_normalized_and_duplicate_rejected(self, db):
        create_user(db, "Dupe@X.Local", "pw")
        # same email, different case -> normalized -> duplicate
        with pytest.raises(ValueError):
            create_user(db, "dupe@x.local", "pw")
