"""Seed the bootstrap admin plus public demo accounts.

The admin approves clinicians and can't be approved by another admin (there's
none yet), so it's created already active. The demo accounts back the one-click
"Explore as..." buttons on the public landing page: a demo clinician (for the
analysis + patient workflow) and a demo admin (for the access-control workflow),
both created already active (approved + verified) with synthetic data only.

Idempotent: re-running won't duplicate.

Seeded credentials (synthetic/demo only):
    Role       Email                        Password
    Admin      admin@medvision.dev          SeedPass123!
    Clinician  demo-clinician@medvision.dev DemoPass123!
    Admin      demo-admin@medvision.dev     DemoPass123!

Usage:  python -m scripts.seed_dev_data
"""
from app.db import SessionLocal
from app.models import Role, Status, User
from app.services.user_service import create_user

ACCOUNTS = [
    # (email, password, role, label)
    ("admin@medvision.dev", "SeedPass123!", Role.admin, "bootstrap admin"),
    ("demo-clinician@medvision.dev", "DemoPass123!", Role.clinician, "demo clinician"),
    ("demo-admin@medvision.dev", "DemoPass123!", Role.admin, "demo admin"),
]


def _ensure(db, email, password, role, label):
    existing = db.query(User).filter_by(email=email).first()
    if existing is not None:
        print(f"{label} already exists: {existing.email} (id={existing.id})")
        return existing

    user = create_user(db, email, password, role=role)
    # Demo/clinician accounts must be usable immediately: force both gates open.
    user.email_verified = True
    user.status = Status.approved
    db.commit()
    print(f"seeded {label}: {user.email} -> {user.status.value}, verified={user.email_verified}")
    return user


def seed():
    db = SessionLocal()
    try:
        for email, password, role, label in ACCOUNTS:
            _ensure(db, email, password, role, label)
        print("\nlogin:")
        print("  admin@medvision.dev / SeedPass123!")
        print("  demo-clinician@medvision.dev / DemoPass123!")
        print("  demo-admin@medvision.dev / DemoPass123!")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
