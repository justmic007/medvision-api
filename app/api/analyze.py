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

from app.core.deps import require_clinician
from app.db import get_db
from app.services.case_service import create_case
from app.services.storage import upload_scan
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
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    patient_id: str | None = Form(None),
    current_user: User = Depends(require_clinician),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED)}",
        )

    # Read the upload once; reused for the preprocessor tempfile and scan storage.
    contents = await file.read()

    # Size guard: reject oversized uploads before doing any work.
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )

    # Content guard: a valid extension does not guarantee a valid image
    # (e.g. a renamed PDF). Verify it loads as an image -> clean 400, not a 500.
    # DICOM is not a standard image; skip the PIL check for .dcm.
    if suffix != ".dcm":
        import io
        from PIL import Image, UnidentifiedImageError
        try:
            Image.open(io.BytesIO(contents)).verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(
                status_code=400, detail="File is not a valid image."
            )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
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
            # Store the scan in object storage; the case keeps only the key (D-10).
            scan_key = upload_scan(contents, suffix)
            create_case(
                db, result, patient_id=patient.id,
                clinician_id=current_user.id, scan_key=scan_key,
            )
            db.commit()

        return _to_response(result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
