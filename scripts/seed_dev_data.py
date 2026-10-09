"""Seed development / demo data for MedVision.

Two distinct populations, keyed off users.is_demo:

  * A real admin (is_demo=False) you can log in with normally. Real accounts
    are never touched by the demo reset below.
  * Three demo accounts on the demo domain (is_demo=True), used only by the
    public role-only /auth/demo-login flow:
      - demo-clinician@demo.medvision.dev   primary clinician
      - demo-admin@demo.medvision.dev       admin (scoped to demo data)
      - demo-clinician-b@demo.medvision.dev sacrificial clinician for the
                                            RBAC lifecycle demo (admin may
                                            suspend / reinstate this one only)

Demo passwords are random and unknown on purpose: nobody signs in with them,
the demo-login endpoint looks accounts up by role. Re-running this script IS
the reset — every demo-domain user, patient and case is wiped and rebuilt.
The wipe is scoped to the demo domain only; nothing with is_demo=False is
read or written beyond ensuring the one real admin exists.
"""

import secrets

from app.core.demo import (
    DEMO_ACCOUNTS,
    DEMO_DOMAIN,
    DEMO_SACRIFICIAL_CLINICIAN,
    is_demo_email,
)
from app.core.deps import get_db
from app.models import Case, Patient, Role, Sex, Status, User
from app.services.demo_service import CANONICAL_DEMO_PATIENT_MRN
from app.services.user_service import create_user

REAL_ADMIN_EMAIL = "admin@medvision.dev"
REAL_ADMIN_PASSWORD = "SeedPass123!"


def _new_demo_password() -> str:
    return secrets.token_urlsafe(24)


def _make_demo_user(db, email: str, role: Role) -> User:
    """Create a demo user: is_demo, approved, verified, random password."""
    user = create_user(db, email=email, password=_new_demo_password(), role=role)
    user.is_demo = True
    user.status = Status.approved
    user.email_verified = True
    db.flush()
    return user


def _ensure_real_admin(db) -> None:
    existing = db.query(User).filter_by(email=REAL_ADMIN_EMAIL).first()
    if existing is not None:
        return
    admin = create_user(
        db, email=REAL_ADMIN_EMAIL, password=REAL_ADMIN_PASSWORD, role=Role.admin
    )
    admin.is_demo = False
    admin.status = Status.approved
    admin.email_verified = True
    db.flush()


def _wipe_demo_domain(db) -> None:
    """Delete every demo-domain user plus their patients and cases. Scoped
    strictly to is_demo accounts — real data is never removed."""
    demo_users = (
        db.query(User)
        .filter((User.is_demo.is_(True)) | (User.email.ilike(f"%@{DEMO_DOMAIN}")))
        .all()
    )
    demo_user_ids = [u.id for u in demo_users]
    if demo_user_ids:
        demo_patient_ids = [
            p.id
            for p in db.query(Patient)
            .filter(Patient.clinician_id.in_(demo_user_ids))
            .all()
        ]
        if demo_patient_ids:
            db.query(Case).filter(
                Case.patient_id.in_(demo_patient_ids)
            ).delete(synchronize_session=False)
        db.query(Case).filter(
            Case.clinician_id.in_(demo_user_ids)
        ).delete(synchronize_session=False)
        db.query(Patient).filter(
            Patient.clinician_id.in_(demo_user_ids)
        ).delete(synchronize_session=False)
        for u in demo_users:
            db.delete(u)
    db.flush()


def _seed_demo(db) -> None:
    clinician = _make_demo_user(db, DEMO_ACCOUNTS["clinician"], Role.clinician)
    _make_demo_user(db, DEMO_ACCOUNTS["admin"], Role.admin)
    _make_demo_user(db, DEMO_SACRIFICIAL_CLINICIAN, Role.clinician)

    patient = Patient(
        mrn=CANONICAL_DEMO_PATIENT_MRN,
        first_name="Demo",
        last_name="Patient",
        sex=Sex.female,
        age=57,
        clinician_id=clinician.id,
    )
    db.add(patient)
    db.flush()

    case = Case(
        patient_id=patient.id,
        clinician_id=clinician.id,
        scan_key=None,
        model_name="medvision-cxr-demo",
        results={
            "findings": [
                {"label": "Cardiomegaly", "probability": 0.78},
                {"label": "Pleural Effusion", "probability": 0.41},
                {"label": "No Finding", "probability": 0.12},
            ],
            "disclaimer": "Decision support only; not a diagnosis.",
        },
    )
    db.add(case)
    db.flush()


def main() -> None:
    gen = get_db()
    db = next(gen)
    try:
        _ensure_real_admin(db)
        _wipe_demo_domain(db)
        _seed_demo(db)
        db.commit()
        print("Seed complete.")
        print(f"  real admin: {REAL_ADMIN_EMAIL} / {REAL_ADMIN_PASSWORD} (is_demo=False)")
        print(f"  demo clinician:  {DEMO_ACCOUNTS['clinician']}")
        print(f"  demo admin:      {DEMO_ACCOUNTS['admin']}")
        print(f"  demo sacrificial:{DEMO_SACRIFICIAL_CLINICIAN}")
        print("  demo passwords are random/unknown; use /auth/demo-login by role.")
    except Exception:
        db.rollback()
        raise
    finally:
        gen.close()


if __name__ == "__main__":
    main()
