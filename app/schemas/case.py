"""Schemas for case-viewing endpoints."""
from typing import Any

from pydantic import BaseModel


class CaseSummary(BaseModel):
    """Lightweight case listing (no full results payload)."""
    id: str
    patient_id: str
    patient_mrn: str
    patient_name: str
    model_name: str
    created_at: str


class CaseDetail(BaseModel):
    """Full case, including the results JSON."""
    id: str
    patient_id: str
    clinician_id: str
    model_name: str
    scan_key: str | None
    results: dict[str, Any]
    created_at: str
