"""add suspended to status enum

Revision ID: a1b2c3d4e5f6
Revises: 6587dd68e8bf
Create Date: 2026-10-01 06:47:48

Adds 'suspended' to the status enum (for clinicians whose access was revoked
after approval, distinct from 'rejected' = never approved). Postgres 12+ allows
ALTER TYPE ... ADD VALUE inside a transaction; IF NOT EXISTS makes it idempotent.
"""
from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = "6587dd68e8bf"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE status ADD VALUE IF NOT EXISTS 'suspended'")


def downgrade() -> None:
    # Postgres has no direct DROP VALUE for enums; leaving 'suspended' in place
    # is harmless if unused. A full downgrade would require recreating the type.
    pass
