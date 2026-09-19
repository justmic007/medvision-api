# MedVision AI

A decision-support research prototype for chest X-ray analysis.

MedVision AI is designed with low-resource settings in mind — contexts where
radiologist access is scarce and a single specialist may serve a large
population, delaying interpretation of routine chest films. It pairs a
multi-pathology chest X-ray classifier with visual explainability (showing
*where* a finding is grounded in the image) and cited literature retrieval
(linking each detected finding to peer-reviewed evidence).

The system is explicitly **non-diagnostic** and keeps a **clinician in the
loop**: it surfaces, the clinician decides. It is a research/educational
prototype, not a validated clinical tool.

> **Disclaimer.** MedVision AI is a research/educational prototype and is NOT a
> diagnostic tool. Outputs are not a substitute for evaluation by a qualified
> clinician.

## System

Frontal chest X-ray in (DICOM or PNG/JPG) to calibrated multi-pathology
predictions + per-finding GradCAM heatmap overlay + cited PubMed literature,
wrapped in a multi-user clinical workflow. One API, containerized. See
ARCHITECTURE.md for the full design and DECISIONS.md for the decision log.

## Status

The stateless analysis core (Phases 0-5) is complete: DICOM/PNG preprocessing,
TorchXRayVision classification, GradCAM explainability, and PubMed literature
grounding, orchestrated behind a single API, with a Gradio demo and MLflow
run tracking.

Phase 6 (the clinical workflow) is built: Postgres persistence, JWT auth
(access + refresh), a two-gate clinician activation flow (email verification +
admin approval), per-clinician patients and cases, and S3-compatible object
storage for scans (MinIO local / Cloudflare R2 prod).

## Sample data

Test chest X-rays are public images, not committed to the repo (D-03). Fetch
them into data/ with:

    bash scripts/fetch_sample_data.sh

Downloads a few public, de-identified chest X-rays for local dev and the demo.

## Running locally

Start the backing services (Postgres + MinIO):

    docker compose up db minio -d

Set up the app environment and run migrations:

    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    alembic upgrade head
    python -m scripts.seed_dev_data

Run the API (with auto-reload for development):

    uvicorn app.main:app --reload

## Inspecting the running system

| Surface | URL / access | How |
|---|---|---|
| API docs (Swagger) | http://localhost:8000/docs | uvicorn app.main:app |
| Demo UI | http://localhost:8000/demo | (same server) |
| Health check | http://localhost:8000/health | - |
| Database | localhost:5432 | any Postgres client (below) |
| Stored scans | http://localhost:9001 | MinIO console (below) |
| Experiment tracking | http://localhost:5000 | run mlflow ui |

**Database (TablePlus or any Postgres client):** Host 127.0.0.1, Port 5432,
User medvision, Password medvision_dev, Database medvision. Tables: users,
patients, cases, refresh_tokens, audit_logs.

**Stored scans:** original scan images live in MinIO object storage, not on the
filesystem. Browse them in the MinIO console, bucket medvision-scans, folder
scans/. Each case's scan_key references the object. In production these live in
Cloudflare R2 (same S3-compatible code, different endpoint).

**Experiment tracking:** MLflow logs each inference run to the local mlruns/
store. It is not auto-started - run mlflow ui and open http://localhost:5000.

## Dev credentials (local only - never used in production)

| Service | Access | Username | Password |
|---|---|---|---|
| Admin login (API) | POST /auth/login | admin@medvision.dev | SeedPass123! |
| MinIO console | http://localhost:9001 | minioadmin | minioadmin |
| Postgres | localhost:5432 | medvision | medvision_dev |

Clinicians self-register via POST /auth/register, then must verify their email
(in dev the link prints to the server console) and be approved by an admin
before they can log in.

## Tests

    pytest -q

DB-backed tests need the Postgres container running.

## Project structure

    medvision-ai/
    |- app/
    |  |- main.py       FastAPI entrypoint
    |  |- api/          routers: health, analyze, auth, admin, patients, cases
    |  |- core/         config, security, tokens, deps
    |  |- services/     preprocessing, classifier, gradcam, literature,
    |  |                orchestrator, case, user, storage, email, tracking
    |  |- schemas/      pydantic models
    |  |- db.py         SQLAlchemy engine + session
    |  |- models.py     ORM models
    |- alembic/         database migrations
    |- scripts/         seed + data-fetch scripts
    |- tests/           pytest suite
    |- data/            gitignored - public/synthetic only (D-03)
    |- mlruns/          gitignored - local MLflow store
    |- Dockerfile
    |- docker-compose.yml
    |- requirements.txt
    |- ARCHITECTURE.md
    |- DECISIONS.md
