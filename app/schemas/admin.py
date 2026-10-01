"""Schemas for admin clinician-management endpoints."""
from pydantic import BaseModel, field_validator


class ClinicianSummary(BaseModel):
    id: str
    email: str
    email_verified: bool
    status: str


class StatusChangeRequest(BaseModel):
    status: str  # target status: approved | rejected | suspended

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str) -> str:
        allowed = {"approved", "rejected", "suspended"}
        if v not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return v


class ActionResponse(BaseModel):
    id: str
    status: str
    message: str
