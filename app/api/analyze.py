"""POST /analyze — upload a chest X-ray, get the full structured analysis.

The Orchestrator (and its models) is expensive to construct, so it is built
once and reused across requests via a module-level lazy singleton — never
per-request, which would reload the model every call.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db import get_db
from app.services.case_service import create_case
from app.models import Patient, User
from app.schemas.analysis import AnalysisResponse, ArticleOut, FindingOut
from app.services.orchestrator import AnalysisResult, Orchestrator

router = APIRouter(tags=["analyze"])

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """Lazily construct the orchestrator once, then reuse it."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def _to_response(result: AnalysisResult) -> AnalysisResponse:
    return AnalysisResponse(
        model_name=result.model_name,
        num_present=result.num_present,
        findings=[
            FindingOut(
                name=f.name,
                probability=f.probability,
                threshold=f.threshold,
                heatmap_base64=f.heatmap_base64,
                articles=[
                    ArticleOut(
                        pmid=a.pmid,
                        title=a.title,
                        journal=a.journal,
                        year=a.year,
                        citation=a.citation,
                        url=a.url,
                    )
                    for a in f.articles
                ],
            )
            for f in result.findings
        ],
        disclaimer=result.disclaimer,
    )


ALLOWED = {".dcm", ".png", ".jpg", ".jpeg"}


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    patient_id: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED)}",
        )

    # Persist the upload to a temp file for the preprocessor to read, then remove.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = get_orchestrator().analyze(tmp_path, image_id=file.filename)

        # If a patient is specified and owned by this clinician, persist a case.
        if patient_id is not None:
            patient = (
                db.query(Patient)
                .filter_by(id=patient_id, clinician_id=current_user.id)
                .first()
            )
            if patient is None:
                raise HTTPException(
                    status_code=404,
                    detail="Patient not found or not owned by you.",
                )
            create_case(
                db, result, patient_id=patient.id, clinician_id=current_user.id
            )
            db.commit()

        return _to_response(result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
