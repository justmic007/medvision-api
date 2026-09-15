"""ORM models — the Phase 6 clinical-workflow schema.

Entities:
  User          — admin or clinician (auth identity)
  RefreshToken  — server-side refresh tokens (revocable)
  Patient       — synthetic, de-identified; owned by a clinician
  Case          — one analysis: a scan + its results, for a patient
  AuditLog      — provenance: who did what, when

Ownership (per-clinician): patients and cases belong to a clinician; an admin
transcends this. Access checks live in the query/service layer, not here.

Scans and analysis results: a Case stores a reference to the stored scan
(object-storage key, D-10) and the structured analysis output as JSON — not the
raw pixels in the DB.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    admin = "admin"
    clinician = "clinician"


class Status(str, enum.Enum):
    """Account approval state (admin-gated for clinicians)."""
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class Sex(str, enum.Enum):
    male = "male"
    female = "female"
    other = "other"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.clinician)
    # Two independent gates before a clinician is fully active:
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[Status] = mapped_column(Enum(Status), default=Status.pending)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    patients: Mapped[list["Patient"]] = relationship(back_populates="clinician")
    cases: Mapped[list["Case"]] = relationship(back_populates="clinician")
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user"
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship(back_populates="refresh_tokens")


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mrn: Mapped[str] = mapped_column(String(64), index=True)  # pseudonymous label
    sex: Mapped[Sex] = mapped_column(Enum(Sex))
    age: Mapped[int] = mapped_column(Integer)
    clinician_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    clinician: Mapped["User"] = relationship(back_populates="patients")
    cases: Mapped[list["Case"]] = relationship(back_populates="patient")


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    clinician_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    # Object-storage key for the stored scan (D-10); pixels live in MinIO/R2.
    scan_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_name: Mapped[str] = mapped_column(String(128))
    # Structured analysis output (findings/probabilities/literature) as JSON.
    results: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    patient: Mapped["Patient"] = relationship(back_populates="cases")
    clinician: Mapped["User"] = relationship(back_populates="cases")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(64))       # e.g. "create_case"
    entity: Mapped[str] = mapped_column(String(64))       # e.g. "case"
    entity_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
