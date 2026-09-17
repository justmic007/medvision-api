"""Auth dependencies: current user + role checks.

get_current_user validates the Bearer access token and loads the user.
require_admin / require_clinician gate endpoints by role. These are the
enforcement points that make the roles mean something.
"""
from __future__ import annotations

import jwt as pyjwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.tokens import decode_token
from app.db import get_db
from app.models import Role, User

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Validate the access token and return the current user."""
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    try:
        claims = decode_token(creds.credentials)
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    if claims.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type.")

    user = db.query(User).filter_by(id=claims.get("sub")).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Allow only admins."""
    if user.role != Role.admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user
