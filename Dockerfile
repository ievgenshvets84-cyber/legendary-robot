# ---------------------------------------------------------------------------
# LLM-Forge – Docker-Image
#
# Standard-Build (CPU):
#   docker build -t llm-forge .
#
# GPU-Build (CUDA): Basisimage austauschen und mit NVIDIA-Runtime starten:
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04 \
#                --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
#                -t llm-forge-gpu .
#   docker run --gpus all -p 8000:8000 llm-forge-gpu
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_FORGE_HOME=/app

WORKDIR /app

# Python bereitstellen, falls das Basisimage (z. B. nvidia/cuda) keines enthält
RUN if ! command -v python3 >/dev/null 2>&1; then \
        apt-get update && \
        apt-get install -y --no-install-recommends python3 python3-pip python3-venv && \
        rm -rf /var/lib/apt/lists/*; \
    fi && \
    ln -sf "$(command -v python3)" /usr/local/bin/python || true

# Abhängigkeiten zuerst installieren (bessere Layer-Caching-Eigenschaften)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install torch --index-url ${TORCH_INDEX} && \
    python -m pip install -r requirements.txt

# Projektcode kopieren
COPY . .

# Laufzeitverzeichnisse sicherstellen
RUN mkdir -p datasets/raw datasets/processed checkpoints exports logs config

EXPOSE 8000

# Healthcheck gegen die API
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/system/health')" || exit 1

CMD ["python", "-m", "backend.main"]
