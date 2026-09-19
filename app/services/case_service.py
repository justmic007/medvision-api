"""Case persistence (Phase 6, increment 2).

Saves an analysis as a Case row: the structured findings + literature as JSON,
linked to a patient and clinician. Heatmaps and the raw scan are NOT stored here
— they go to object storage (MinIO/R2) by reference in a later increment (D-10),
keeping case rows lean.

No auth yet (D-09 ordering): the clinician_id is passed in; the auth layer that
supplies it from a logged-in user comes next.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Case
from app.services.orchestrator import AnalysisResult


def _results_json(result: AnalysisResult) -> dict:
    """Serialize the analysis into a lean JSON structure (no base64 heatmaps)."""
    return {
        "num_present": result.num_present,
        "disclaimer": result.disclaimer,
        "findings": [
            {
                "name": f.name,
                "probability": f.probability,
                "threshold": f.threshold,
                "has_heatmap": f.heatmap_base64 is not None,
                "articles": [
                    {
                        "pmid": a.pmid,
                        "title": a.title,
                        "journal": a.journal,
                        "year": a.year,
                        "url": a.url,
                    }
                    for a in f.articles
                ],
            }
            for f in result.findings
        ],
    }


def create_case(
    db: Session,
    result: AnalysisResult,
    patient_id: str,
    clinician_id: str,
    scan_key: str | None = None,
) -> Case:
    """Persist an analysis result as a Case and return it."""
    case = Case(
        patient_id=patient_id,
        clinician_id=clinician_id,
        scan_key=scan_key,
        model_name=result.model_name,
        results=_results_json(result),
    )
    db.add(case)
    db.flush()      # assign the id without committing; caller owns the transaction
    return case
