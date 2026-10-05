# MedVision AI — backend image
# Python 3.11 pinned per DECISIONS.md D-02 (verified against torch/monai/torchxrayvision).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# System deps some wheels may need at runtime (kept minimal).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer caches unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + migrations + scripts.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY scripts ./scripts

EXPOSE 8000

# Run migrations then start the server, binding to the platform's $PORT
# (Render/most PaaS set $PORT; default 8000 for local `docker run`).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
