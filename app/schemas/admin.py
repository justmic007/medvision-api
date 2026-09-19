"""Schemas for admin clinician-management endpoints."""
from pydantic import BaseModel


class PendingClinician(BaseModel):
    id: str
    email: str
    email_verified: bool
    status: str


class ActionResponse(BaseModel):
    id: str
    status: str
    message: str
