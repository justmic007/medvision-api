"""Schemas for patient endpoints."""
from pydantic import BaseModel


class PatientCreate(BaseModel):
    mrn: str          # system identifier / medical record number
    first_name: str
    last_name: str
    sex: str          # "male" | "female" | "other"
    age: int


class PatientResponse(BaseModel):
    id: str
    mrn: str
    first_name: str
    last_name: str
    sex: str
    age: int
    clinician_id: str
