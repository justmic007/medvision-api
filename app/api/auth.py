"""Auth endpoints: register, login, refresh, logout.

Login enforces BOTH gates (email_verified AND status=approved) and returns a
generic error for bad credentials so it doesn't leak which emails exist. It uses
specific messages for the gate failures (verify email / pending approval) — a
deliberate UX-vs-info-leak tradeoff: a legitimate user needs to know why login
is blocked, and registration is open anyway so email existence isn't secret.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.core.config import get_settings
from app.core.tokens import (
    create_verification_token,
    create_access_token,
    decode_token,
    issue_refresh_token,
    refresh_token_is_valid,
    revoke_refresh_token,
)
from app.core.cookies import (
    REFRESH_COOKIE_NAME,
    clear_refresh_cookie,
    set_refresh_cookie,
)
from app.core.deps import get_current_user
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
from app.services.email import get_email_sender

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> UserResponse:
    try:
        user = create_user(db, body.email, body.password, role=Role.clinician)
    except ValueError:
        # Duplicate email -> 409, but keep message generic-ish.
        raise HTTPException(status_code=409, detail="Could not register with that email.")
    db.commit()

    # Send a verification email (console backend in dev prints the link).
    settings = get_settings()
    token = create_verification_token(user.id, user.email)
    link = f"{settings.email_verification_base_url}/auth/verify?token={token}"
    get_email_sender().send(
        to=user.email,
        subject="Verify your MedVision AI account",
        html=f'Please verify your email: <a href="{link}">{link}</a>',
    )
    return UserResponse(
        id=user.id, email=user.email, role=user.role.value,
        status=user.status.value, email_verified=user.email_verified,
    )


@router.post("/login", response_model=AccessTokenResponse)
def login(
    body: LoginRequest, response: Response, db: Session = Depends(get_db)
) -> AccessTokenResponse:
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
    # Refresh token goes in an httpOnly cookie (JS can't read it); the access
    # token is returned in the body for the client to hold in memory.
    set_refresh_cookie(response, refresh)
    return AccessTokenResponse(access_token=access)


@router.post("/refresh", response_model=AccessTokenResponse)
def refresh(request: Request, db: Session = Depends(get_db)) -> AccessTokenResponse:
    token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not token or not refresh_token_is_valid(db, token):
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.")
    claims = decode_token(token)
    user = db.query(User).filter_by(id=claims["sub"]).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid refresh token.")
    access = create_access_token(user.id, user.role.value)
    return AccessTokenResponse(access_token=access)


@router.post("/logout", response_model=MessageResponse)
def logout(
    request: Request, response: Response, db: Session = Depends(get_db)
) -> MessageResponse:
    token = request.cookies.get(REFRESH_COOKIE_NAME)
    if token:
        revoke_refresh_token(db, token)
        db.commit()
    clear_refresh_cookie(response)
    return MessageResponse(message="Logged out.")


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    """Return the currently authenticated user (from the access token)."""
    return UserResponse(
        id=user.id, email=user.email, role=user.role.value,
        status=user.status.value, email_verified=user.email_verified,
    )


@router.get("/verify", response_model=MessageResponse)
def verify_email(token: str, db: Session = Depends(get_db)) -> MessageResponse:
    """Verify a user's email from the token in the emailed link."""
    import jwt as pyjwt

    try:
        claims = decode_token(token)
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=400, detail="Invalid or expired verification link.")

    if claims.get("type") != "verify":
        raise HTTPException(status_code=400, detail="Invalid verification token.")

    user = db.query(User).filter_by(id=claims["sub"]).first()
    if user is None:
        raise HTTPException(status_code=400, detail="Invalid verification token.")

    if user.email_verified:
        return MessageResponse(message="Email already verified.")

    user.email_verified = True
    db.commit()
    return MessageResponse(message="Email verified. Awaiting admin approval.")
