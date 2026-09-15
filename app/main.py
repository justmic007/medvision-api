"""MedVision AI — FastAPI application entrypoint.

Phase 0: app skeleton with a real /health endpoint and the project-wide
non-diagnostic disclaimer surfaced at the root. Imaging, explainability,
and literature layers are added in later phases as additional routers under
app/api/, backed by logic in app/services/.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analyze, health
from app.core.config import get_settings
from app.demo import build_demo
import gradio as gr

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Decision-support research prototype for chest X-ray analysis. "
        "Non-diagnostic; clinician-in-the-loop."
    ),
)

# CORS: the frontend is a separate app on a different origin (separate repo),
# so it must be allowed to call this API from the browser. Permissive in dev;
# Origins come from CORS_ALLOWED_ORIGINS (.env); set to the real frontend
# URL(s) in production. Never "*" with credentials — browsers reject it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze.router)


@app.get("/", tags=["root"])
def root() -> dict:
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "disclaimer": settings.disclaimer,
        "docs": "/docs",
    }


# Mount the self-contained Gradio demo UI at /demo.
app = gr.mount_gradio_app(app, build_demo(), path="/demo")
