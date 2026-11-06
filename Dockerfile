# Hosted demo image (Render). CPU-only; models are downloaded at build time from their official sources.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SAREE_HOSTED=1 TORCH_HOME=/app/models/torch_home \
    MIN_RAM_GB=1.5 MIN_CPU_CORES=1 MAX_SESSIONS=2 OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-deploy.txt

COPY saree_ai/ saree_ai/
COPY backend/ backend/
COPY frontend/ frontend/
COPY config/ config/
COPY scripts/ scripts/
# pose baseline (torchvision COCO), Vosk English + Telugu, Piper Telugu voice
RUN python scripts/download_models.py --pose --asr --asr_te --tts_te_config --tts_te \
    && rm -f models/asr/*.zip

EXPOSE 10000
CMD ["python", "-m", "backend.server", "--hosted"]
