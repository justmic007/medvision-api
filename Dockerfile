# MedVision AI — backend image
# Python 3.11 pinned per DECISIONS.md D-02 (verified against torch/monai/torchxrayvision).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1

WORKDIR /code

# System deps some wheels may need at runtime (kept minimal).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer caches unless requirements change.
COPY requirements.txt .
# CPU-only torch/torchvision: Render has no GPU, so the CUDA build just
# wastes RAM and disk. Same operators, same weights, identical results --
# only lighter. requirements.txt then finds them satisfied, installs the rest.
RUN pip install --no-cache-dir \
        torch==2.2.2+cpu torchvision==0.17.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
 && pip install --no-cache-dir -r requirements.txt

# App code + migrations + scripts.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY scripts ./scripts

EXPOSE 8000

# Run migrations then start the server, binding to the platform's $PORT
# (Render/most PaaS set $PORT; default 8000 for local `docker run`).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
