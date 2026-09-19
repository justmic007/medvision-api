"""Admin endpoints: manage clinician approval.

All protected by require_admin. Approve/reject actions are recorded in the
AuditLog (who did what to whom, when) — the provenance the clinical workflow
needs. Only clinicians in 'pending' can be approved/rejected.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import require_admin
from app.db import get_db
from app.models import AuditLog, Role, Status, User
from app.schemas.admin import ActionResponse, PendingClinician

router = APIRouter(prefix="/admin", tags=["admin"])


def _audit(db: Session, admin: User, action: str, target_id: str) -> None:
    db.add(
        AuditLog(
            user_id=admin.id,
            action=action,
            entity="user",
            entity_id=target_id,
        )
    )


@router.get("/pending-clinicians", response_model=list[PendingClinician])
def list_pending(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[PendingClinician]:
    users = (
        db.query(User)
        .filter_by(role=Role.clinician, status=Status.pending)
        .all()
    )
    return [
        PendingClinician(
            id=u.id, email=u.email,
            email_verified=u.email_verified, status=u.status.value,
        )
        for u in users
    ]


def _get_pending_clinician(db: Session, user_id: str) -> User:
    user = db.query(User).filter_by(id=user_id).first()
    if user is None or user.role != Role.clinician:
        raise HTTPException(status_code=404, detail="Clinician not found.")
    if user.status != Status.pending:
        raise HTTPException(
            status_code=409, detail=f"Clinician is already {user.status.value}."
        )
    return user


@router.post("/clinicians/{user_id}/approve", response_model=ActionResponse)
def approve(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> ActionResponse:
    user = _get_pending_clinician(db, user_id)
    user.status = Status.approved
    _audit(db, admin, "approve_clinician", user.id)
    db.commit()
    return ActionResponse(id=user.id, status=user.status.value, message="Clinician approved.")


@router.post("/clinicians/{user_id}/reject", response_model=ActionResponse)
def reject(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> ActionResponse:
    user = _get_pending_clinician(db, user_id)
    user.status = Status.rejected
    _audit(db, admin, "reject_clinician", user.id)
    db.commit()
    return ActionResponse(id=user.id, status=user.status.value, message="Clinician rejected.")
