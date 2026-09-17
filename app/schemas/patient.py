"""Schemas for patient endpoints."""
from pydantic import BaseModel


class PatientCreate(BaseModel):
    mrn: str          # pseudonymous label / medical record number
    sex: str          # "male" | "female" | "other"
    age: int


class PatientResponse(BaseModel):
    id: str
    mrn: str
    sex: str
    age: int
    clinician_id: str
