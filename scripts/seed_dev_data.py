"""Seed the bootstrap admin account.

The admin is the account that approves clinicians. It can't be approved by
another admin (there's none yet), so it's created here already active
(approved + verified) via create_user, which sets those for role=admin.

Idempotent: re-running won't duplicate.

Seeded credentials (dev only — CHANGE for any real deployment):
    Role   Email                      Password
    Admin  admin@medvision.dev   SeedPass123!

Usage:  python -m scripts.seed_dev_data
"""
from app.db import SessionLocal
from app.models import Role, User
from app.services.user_service import create_user

ADMIN_EMAIL = "admin@medvision.dev"
ADMIN_PASSWORD = "SeedPass123!"


def seed():
    db = SessionLocal()
    try:
        existing = db.query(User).filter_by(email=ADMIN_EMAIL).first()
        if existing is not None:
            print(f"admin already exists: {existing.email} (id={existing.id})")
            return existing.id

        admin = create_user(db, ADMIN_EMAIL, ADMIN_PASSWORD, role=Role.admin)
        db.commit()
        print(f"seeded admin: {admin.email} (id={admin.id})")
        print(f"  status={admin.status.value}, verified={admin.email_verified}")
        print(f"  login with: {ADMIN_EMAIL} / {ADMIN_PASSWORD}")
        return admin.id
    finally:
        db.close()


if __name__ == "__main__":
    seed()
