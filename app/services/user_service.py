"""User creation (Phase 6, auth foundation).

Creates users with hashed passwords and the correct initial gate states:
  - clinician  -> status=pending, email_verified=False
                  (must verify email AND be admin-approved before login)
  - admin      -> status=approved, email_verified=True
                  (the bootstrap admin, created by seed/CLI — there's no admin
                  to approve the first admin, so it's created already-active)

create_user flushes (caller owns the transaction), consistent with case_service.
Duplicate emails raise ValueError (the caller maps this to an HTTP 409).
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models import Role, Status, User


def create_user(
    db: Session,
    email: str,
    password: str,
    role: Role = Role.clinician,
) -> User:
    """Create a user with a hashed password and role-appropriate gate state."""
    email = email.strip().lower()

    existing = db.query(User).filter_by(email=email).first()
    if existing is not None:
        raise ValueError(f"A user with email {email!r} already exists.")

    if role == Role.admin:
        status = Status.approved
        email_verified = True
    else:
        status = Status.pending
        email_verified = False

    user = User(
        email=email,
        hashed_password=hash_password(password),
        role=role,
        status=status,
        email_verified=email_verified,
    )
    db.add(user)
    db.flush()
    return user
