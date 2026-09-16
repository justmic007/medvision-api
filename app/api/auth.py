"""Auth endpoints: register, login, refresh, logout.

Login enforces BOTH gates (email_verified AND status=approved) and returns a
generic error for bad credentials so it doesn't leak which emails exist. It uses
specific messages for the gate failures (verify email / pending approval) — a
deliberate UX-vs-info-leak tradeoff: a legitimate user needs to know why login
is blocked, and registration is open anyway so email existence isn't secret.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.core.tokens import (
    create_access_token,
    decode_token,
    issue_refresh_token,
    refresh_token_is_valid,
    revoke_refresh_token,
)
from app.db import get_db
from app.models import Role, Status, User
from app.schemas.auth import (
    AccessTokenResponse,
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
    UserResponse,
)
from app.services.user_service import create_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> UserResponse:
    try:
        user = create_user(db, body.email, body.password, role=Role.clinician)
    except ValueError:
        # Duplicate email -> 409, but keep message generic-ish.
        raise HTTPException(status_code=409, detail="Could not register with that email.")
    db.commit()
    return UserResponse(
        id=user.id, email=user.email, role=user.role.value,
        status=user.status.value, email_verified=user.email_verified,
    )


@router.post("/login", response_model=TokenPair)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenPair:
    user = db.query(User).filter_by(email=body.email.strip().lower()).first()

    # Generic failure for bad email OR bad password (no email enumeration).
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Please verify your email first.")

    if user.status != Status.approved:
        raise HTTPException(
            status_code=403, detail="Your account is pending admin approval."
        )

    access = create_access_token(user.id, user.role.value)
    refresh = issue_refresh_token(db, user.id)
    db.commit()
    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=AccessTokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)) -> AccessTokenResponse:
    if not refresh_token_is_valid(db, body.refresh_token):
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.")
    claims = decode_token(body.refresh_token)
    user = db.query(User).filter_by(id=claims["sub"]).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid refresh token.")
    access = create_access_token(user.id, user.role.value)
    return AccessTokenResponse(access_token=access)


@router.post("/logout", response_model=MessageResponse)
def logout(body: RefreshRequest, db: Session = Depends(get_db)) -> MessageResponse:
    revoke_refresh_token(db, body.refresh_token)
    db.commit()
    return MessageResponse(message="Logged out.")
