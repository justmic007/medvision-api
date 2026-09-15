"""Seed one dev clinician + patient so cases have something to link to.

Temporary scaffolding for increment 2 (persistence before auth, D-09). Once the
auth layer exists, real registered users replace this seed. Idempotent.

Usage:  python scripts/seed_dev_data.py
"""
from app.db import SessionLocal
from app.models import Patient, Role, Sex, User


def seed():
    db = SessionLocal()
    try:
        clinician = db.query(User).filter_by(
            email="dev.clinician@medvision.local"
        ).first()
        if clinician is None:
            clinician = User(
                email="dev.clinician@medvision.local",
                hashed_password="not-a-real-hash-seed-only",
                role=Role.clinician,
            )
            db.add(clinician)
            db.commit()
            db.refresh(clinician)

        patient = db.query(Patient).filter_by(
            mrn="DEV-0001", clinician_id=clinician.id
        ).first()
        if patient is None:
            patient = Patient(
                mrn="DEV-0001", sex=Sex.female, age=54, clinician_id=clinician.id
            )
            db.add(patient)
            db.commit()
            db.refresh(patient)

        print(f"clinician_id={clinician.id}")
        print(f"patient_id={patient.id}")
        return clinician.id, patient.id
    finally:
        db.close()


if __name__ == "__main__":
    seed()
