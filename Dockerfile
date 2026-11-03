# syntax=docker/dockerfile:1
ARG RUNTIME=cpu
FROM python:3.10-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PROJECT_DIR=/workspace \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True USE_TF=0
RUN useradd --create-home --uid 1000 app
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libreoffice-writer libreoffice-calc fonts-dejavu-core fonts-liberation \
    libgl1 libglib2.0-0 libgomp1 libxext6 libsm6 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

FROM base AS cpu
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-cpu.txt \
    && pip install --no-deps torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
ENV DEVICE=cpu

FROM base AS gpu
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-gpu.txt \
    && pip install --no-deps torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
ENV DEVICE=auto

FROM ${RUNTIME} AS final
COPY app ./app
COPY scripts ./scripts
COPY notebooks ./notebooks
COPY tests ./tests
COPY models.lock.json README.md MODEL_NOTES.md VERIFICATION.md ./
RUN chmod +x scripts/*.sh && python -m scripts.check_dependencies
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "127.0.0.1", "--port", "8000"]
HEALTHCHECK --interval=20s --timeout=5s --start-period=20s CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1
