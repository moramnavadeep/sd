# FREE online demo image (Render free plan: 512 MB RAM, 0.1 CPU) - "lite" mode.
# The server serves the app, runs the tutorial logic and recognises TELUGU voice commands (~320 MB RAM).
# English recognition runs in the visitor's browser (vosk-browser, WebAssembly); Telugu replies are
# pre-recorded at build time.
# Body tracking is OFF here (it needs ~2 GB RAM): use Dockerfile.full on a paid plan for that.

# ---- build stage: download models, pack them for the browser, pre-record Telugu sentences
FROM python:3.12-slim AS assets
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN pip install pyyaml numpy opencv-python-headless piper-tts
COPY saree_ai/ saree_ai/
COPY config/ config/
COPY scripts/ scripts/
COPY frontend/ frontend/
RUN python scripts/download_models.py --asr --asr_te --tts_te_config --tts_te \
    && python scripts/build_lite_assets.py

# ---- runtime stage: tiny
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SAREE_LITE=1 LITE_SERVER_ASR=te MAX_ASR_STREAMS=2 MIN_RAM_GB=0.25 MIN_CPU_CORES=1 MAX_SESSIONS=20
WORKDIR /app
COPY requirements-lite.txt .
RUN pip install -r requirements-lite.txt
COPY saree_ai/ saree_ai/
COPY backend/ backend/
COPY config/ config/
COPY --from=assets /app/frontend/ frontend/
COPY --from=assets /app/models/asr/vosk-model-small-te-0.42/ models/asr/vosk-model-small-te-0.42/
EXPOSE 10000
CMD ["python", "-m", "backend.server", "--lite"]
