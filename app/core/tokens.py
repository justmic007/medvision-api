"""JWT access + refresh tokens (Phase 6, 3b).

Access token  — short-lived (config), carries user id + role; sent with each
                request to prove identity.
Refresh token — longer-lived; used only to mint a new access token. The token
                is stored HASHED in the refresh_tokens table (like a password),
                so a DB leak never exposes usable tokens, and logout can revoke
                it. No rotation yet (a refresh token is reusable until expiry or
                revocation) — a deliberate later enhancement.

Tokens carry a 'type' claim ('access'/'refresh') so one can't be used as the
other. Expiry ('exp') is validated by pyjwt on decode.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import RefreshToken

_settings = get_settings()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: str, role: str) -> str:
    """Short-lived JWT proving identity + role."""
    expire = _now() + timedelta(minutes=_settings.access_token_expire_minutes)
    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "iat": _now(),
        "exp": expire,
    }
    return jwt.encode(payload, _settings.jwt_secret_key, algorithm=_settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    """Decode + validate a JWT (raises jwt exceptions on invalid/expired)."""
    return jwt.decode(
        token, _settings.jwt_secret_key, algorithms=[_settings.jwt_algorithm]
    )


def _hash_token(raw: str) -> str:
    """Hash a refresh token for storage (sha256 — these are high-entropy)."""
    return hashlib.sha256(raw.encode()).hexdigest()


def issue_refresh_token(db: Session, user_id: str) -> str:
    """Create a refresh token, store its hash, return the RAW token to the client.

    The raw token is a random secret embedded in a JWT so it also carries an exp
    the API can check without a DB hit; the DB row lets us revoke it. Caller owns
    the transaction (flush, not commit).
    """
    expire = _now() + timedelta(days=_settings.refresh_token_expire_days)
    jti = secrets.token_urlsafe(32)  # unique random id for this token
    payload = {
        "sub": user_id,
        "type": "refresh",
        "jti": jti,
        "iat": _now(),
        "exp": expire,
    }
    raw = jwt.encode(payload, _settings.jwt_secret_key, algorithm=_settings.jwt_algorithm)

    db.add(
        RefreshToken(
            user_id=user_id,
            token_hash=_hash_token(raw),
            expires_at=expire,
            revoked=False,
        )
    )
    db.flush()
    return raw


def refresh_token_is_valid(db: Session, raw: str) -> bool:
    """True if the refresh token decodes, isn't revoked, and exists in the DB."""
    try:
        payload = decode_token(raw)
    except jwt.PyJWTError:
        return False
    if payload.get("type") != "refresh":
        return False
    row = db.query(RefreshToken).filter_by(token_hash=_hash_token(raw)).first()
    return row is not None and not row.revoked


def revoke_refresh_token(db: Session, raw: str) -> bool:
    """Mark a refresh token revoked (logout). Returns True if one was revoked."""
    row = db.query(RefreshToken).filter_by(token_hash=_hash_token(raw)).first()
    if row is None:
        return False
    row.revoked = True
    db.flush()
    return True
