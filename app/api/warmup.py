"""Warm-up endpoint.

On Render's free tier the instance spins down after inactivity and cold-starts
(~50s) on the next request, and the ONNX model lazy-loads on the first analyze.
This public endpoint lets the frontend trigger both the moment it loads: the
request wakes the instance, and building the orchestrator loads the model into
memory — so by the time the user clicks Analyze the model is already warm.

It does NOT run inference (no image, no DB, no upload) — it only loads the
model. get_orchestrator() is a lazy singleton, so repeat calls return instantly.
Unauthenticated on purpose: it reveals nothing and must be callable before login.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.analyze import get_orchestrator

router = APIRouter(tags=["warmup"])


@router.get("/warmup")
def warmup() -> dict:
    """Load the model into memory if it isn't already. Returns quickly."""
    try:
        get_orchestrator()  # builds classifier + GradCAM -> loads the ONNX model
        return {"status": "warm"}
    except Exception:
        # Never surface a warm-up failure to the client; the real analyze path
        # will load the model on demand regardless.
        return {"status": "cold"}
