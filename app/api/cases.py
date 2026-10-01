"""Case-viewing endpoints — list and view, scoped to the authenticated clinician.

Ownership is enforced on every read: a clinician only ever sees their own cases.
Listing returns lightweight summaries; the detail endpoint returns the full
results JSON. A patient's case history is available via /patients/{id}/cases.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.deps import require_clinician
from app.db import get_db
from app.models import Case, Patient, User
from app.schemas.case import CaseDetail, CaseSummary
from app.services.storage import download_scan

router = APIRouter(tags=["cases"])


def _summary(c: Case) -> CaseSummary:
    return CaseSummary(
        id=c.id,
        patient_id=c.patient_id,
        patient_mrn=c.patient.mrn,
        patient_name=f"{c.patient.first_name} {c.patient.last_name}",
        model_name=c.model_name,
        created_at=c.created_at.isoformat(),
    )


@router.get("/cases", response_model=list[CaseSummary])
def list_cases(
    db: Session = Depends(get_db),
    user: User = Depends(require_clinician),
) -> list[CaseSummary]:
    cases = (
        db.query(Case)
        .filter_by(clinician_id=user.id)
        .order_by(Case.created_at.desc())
        .all()
    )
    return [_summary(c) for c in cases]


@router.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(
    case_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_clinician),
) -> CaseDetail:
    case = db.query(Case).filter_by(id=case_id, clinician_id=user.id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    return CaseDetail(
        id=case.id, patient_id=case.patient_id, clinician_id=case.clinician_id,
        model_name=case.model_name, scan_key=case.scan_key,
        results=case.results, created_at=case.created_at.isoformat(),
    )


@router.get("/patients/{patient_id}/cases", response_model=list[CaseSummary])
def patient_case_history(
    patient_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_clinician),
) -> list[CaseSummary]:
    # Ownership: the patient must belong to this clinician.
    patient = db.query(Patient).filter_by(id=patient_id, clinician_id=user.id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found.")
    cases = (
        db.query(Case)
        .filter_by(patient_id=patient_id, clinician_id=user.id)
        .order_by(Case.created_at.desc())
        .all()
    )
    return [_summary(c) for c in cases]


@router.get("/cases/{case_id}/scan")
def get_case_scan(
    case_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_clinician),
) -> Response:
    """Download the original scan for a case (ownership-checked)."""
    case = db.query(Case).filter_by(id=case_id, clinician_id=user.id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    if case.scan_key is None:
        raise HTTPException(status_code=404, detail="No scan stored for this case.")
    data = download_scan(case.scan_key)
    # Content type from the key's suffix (jpg/png/dcm).
    suffix = case.scan_key.rsplit(".", 1)[-1].lower()
    media = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "dcm": "application/dicom",
    }.get(suffix, "application/octet-stream")
    return Response(content=data, media_type=media)
