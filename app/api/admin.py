"""Admin endpoints: manage the clinician access lifecycle.

All protected by require_admin. Status changes are recorded in the AuditLog
(who did what to whom, when) — the provenance the clinical workflow needs.

Status lifecycle (separation of duties: admins manage access, not clinical work):
    pending   -> approved | rejected
    approved  -> suspended            (revoke access)
    suspended -> approved             (reinstate)
    rejected  -> approved             (admit a previously-denied applicant)
Current status is the source of truth for access; full history lives in the
audit log (a reinstated clinician is simply 'approved' again).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import require_admin
from app.core.demo import DEMO_SACRIFICIAL_CLINICIAN
from app.db import get_db
from app.models import AuditLog, Role, Status, User
from app.schemas.admin import ClinicianSummary, StatusChangeRequest, ActionResponse

router = APIRouter(prefix="/admin", tags=["admin"])

# Allowed status transitions. Any pair not listed here is rejected (400).
_ALLOWED_TRANSITIONS: dict[Status, set[Status]] = {
    Status.pending: {Status.approved, Status.rejected},
    Status.approved: {Status.suspended},
    Status.suspended: {Status.approved},
    Status.rejected: {Status.approved},
}

# Human-readable audit action per target status.
_AUDIT_ACTION: dict[Status, str] = {
    Status.approved: "approve_clinician",
    Status.rejected: "reject_clinician",
    Status.suspended: "suspend_clinician",
}


def _audit(db: Session, admin: User, action: str, target_id: str) -> None:
    db.add(
        AuditLog(
            user_id=admin.id, action=action, entity="user", entity_id=target_id
        )
    )


def _summary(u: User) -> ClinicianSummary:
    return ClinicianSummary(
        id=u.id, email=u.email, email_verified=u.email_verified, status=u.status.value
    )


@router.get("/clinicians", response_model=list[ClinicianSummary])
def list_clinicians(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[ClinicianSummary]:
    """All clinicians, any status, newest first — the management view.

    A demo admin sees only demo clinicians, never real registrants' emails."""
    q = db.query(User).filter_by(role=Role.clinician)
    if _admin.is_demo:
        q = q.filter_by(is_demo=True)
    users = q.order_by(User.created_at.desc()).all()
    return [_summary(u) for u in users]


@router.get("/pending-clinicians", response_model=list[ClinicianSummary])
def list_pending(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[ClinicianSummary]:
    """Clinicians awaiting a first decision (kept for convenience)."""
    q = db.query(User).filter_by(role=Role.clinician, status=Status.pending)
    if _admin.is_demo:
        q = q.filter_by(is_demo=True)
    users = q.order_by(User.created_at.desc()).all()
    return [_summary(u) for u in users]


@router.post("/clinicians/{user_id}/status", response_model=ActionResponse)
def change_status(
    user_id: str,
    body: StatusChangeRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> ActionResponse:
    """Change a clinician's status, enforcing the allowed transitions."""
    user = db.query(User).filter_by(id=user_id).first()
    if user is None or user.role != Role.clinician:
        raise HTTPException(status_code=404, detail="Clinician not found.")

    # A demo admin may drive the lifecycle ONLY on the sacrificial demo
    # clinician — never on a real clinician or the primary demo clinician.
    if admin.is_demo and user.email != DEMO_SACRIFICIAL_CLINICIAN:
        raise HTTPException(status_code=403, detail="demo_read_only")

    target = Status(body.status)
    allowed = _ALLOWED_TRANSITIONS.get(user.status, set())
    if target not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot change status from {user.status.value} to {target.value}.",
        )

    user.status = target
    # Reinstating (-> approved from suspended/rejected) logs a distinct action.
    action = _AUDIT_ACTION.get(target, "reinstate_clinician")
    _audit(db, admin, action, user.id)
    db.commit()
    return ActionResponse(
        id=user.id, status=user.status.value, message=f"Clinician {target.value}."
    )
