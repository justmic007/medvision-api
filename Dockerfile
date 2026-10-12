# MedVision AI — backend image
# Python 3.11 pinned per DECISIONS.md D-02.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1

WORKDIR /code

# System deps some wheels may need at runtime (kept minimal).
# libgomp1: OpenMP runtime used by onnxruntime / scikit-image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer caches unless requirements change.
# No torch index needed any more: the model runs on onnxruntime (CPU), whose
# wheels are on PyPI. The exported model lives in app/ml_assets/ (copied below).
COPY requirements.txt .
# --timeout/--retries: ride out slow or flaky networks during the build so a
# single slow wheel download doesn't abort the whole image build.
RUN pip install --no-cache-dir --timeout=1000 --retries=20 -r requirements.txt

# App code + migrations + scripts. app/ml_assets/ carries the ONNX model.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY scripts ./scripts

EXPOSE 8000

# Run migrations then start the server, binding to the platform's $PORT.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
