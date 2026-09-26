"""Schemas for patient endpoints."""
from pydantic import BaseModel


class PatientCreate(BaseModel):
    # mrn is system-generated (MRN-0001 per clinician), not supplied by the client.
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
