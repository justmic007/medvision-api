"""Patient endpoints — create and list, scoped to the authenticated clinician.

Per-clinician ownership: a clinician only sees and creates their own patients.
(An admin-wide view could be added later; for now these are clinician-scoped.)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db import get_db
from app.models import Patient, Sex, User
from app.schemas.patient import PatientCreate, PatientResponse

router = APIRouter(prefix="/patients", tags=["patients"])


def _to_response(p: Patient) -> PatientResponse:
    return PatientResponse(
        id=p.id, mrn=p.mrn, sex=p.sex.value, age=p.age, clinician_id=p.clinician_id
    )


@router.post("", response_model=PatientResponse, status_code=201)
def create_patient(
    body: PatientCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PatientResponse:
    patient = Patient(
        mrn=body.mrn,
        sex=Sex(body.sex),
        age=body.age,
        clinician_id=user.id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return _to_response(patient)


@router.get("", response_model=list[PatientResponse])
def list_patients(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[PatientResponse]:
    patients = db.query(Patient).filter_by(clinician_id=user.id).all()
    return [_to_response(p) for p in patients]
