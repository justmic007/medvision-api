"""Demo state management: the per-login reset that keeps the public demo
in a known-good state.

Called at the start of every POST /auth/demo-login. Scoped entirely to
is_demo accounts and the demo domain, so it can never touch real data.

With the read-only rules in place a demo user can mutate almost nothing, so
the reset is intentionally small:
  - restore the sacrificial demo clinician to its canonical status
    (what a demo admin may have changed by approve/suspend/reinstate)
  - defensively clear any patients/cases owned by a demo clinician that are
    NOT the canonical seeded demo patient (should be none while writes are
    blocked, but this keeps the demo pristine regardless)

It never deletes the demo accounts themselves, the canonical demo patient, or
its seeded case; the seed script owns creating those.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.demo import (
    DEMO_SACRIFICIAL_CLINICIAN,
    DEMO_SACRIFICIAL_CANONICAL_STATUS,
    DEMO_ACCOUNTS,
)
from app.models import Case, Patient, Status, User

CANONICAL_DEMO_PATIENT_MRN = "DEMO-0001"


def reset_demo_state(db: Session) -> None:
    """Restore the demo to its canonical state. Safe to call on every login."""
    sac = (
        db.query(User)
        .filter_by(email=DEMO_SACRIFICIAL_CLINICIAN, is_demo=True)
        .first()
    )
    if sac is not None:
        canonical = Status(DEMO_SACRIFICIAL_CANONICAL_STATUS)
        if sac.status != canonical:
            sac.status = canonical

    demo_clinician = (
        db.query(User)
        .filter_by(email=DEMO_ACCOUNTS["clinician"], is_demo=True)
        .first()
    )
    if demo_clinician is not None:
        junk = (
            db.query(Patient)
            .filter(
                Patient.clinician_id == demo_clinician.id,
                Patient.mrn != CANONICAL_DEMO_PATIENT_MRN,
            )
            .all()
        )
        for p in junk:
            db.query(Case).filter(Case.patient_id == p.id).delete(
                synchronize_session=False
            )
            db.delete(p)

    db.commit()
