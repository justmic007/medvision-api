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
        id=p.id, mrn=p.mrn, first_name=p.first_name, last_name=p.last_name,
        sex=p.sex.value, age=p.age, clinician_id=p.clinician_id
    )


@router.post("", response_model=PatientResponse, status_code=201)
def create_patient(
    body: PatientCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PatientResponse:
    # Auto-generate a per-clinician MRN (MRN-0001, MRN-0002, ...). Using the
    # max existing sequence + 1 so it survives deletions. Production would use a
    # DB sequence to avoid the rare concurrent-create race.
    existing = db.query(Patient).filter_by(clinician_id=user.id).all()
    nums = [
        int(p.mrn.split("-")[1])
        for p in existing
        if p.mrn.startswith("MRN-") and p.mrn.split("-")[1].isdigit()
    ]
    mrn = f"MRN-{(max(nums) + 1) if nums else 1:04d}"

    patient = Patient(
        mrn=mrn,
        first_name=body.first_name,
        last_name=body.last_name,
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
