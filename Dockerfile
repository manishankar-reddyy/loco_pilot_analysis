# ===================================
# Loco Pilot Crew Behavior Monitoring
# Requires NVIDIA GPU with 12GB+ VRAM
# ===================================

# CUDA 12.1 + cuDNN 8 on Ubuntu 22.04 — matches PyTorch 2.x CUDA builds
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.1 first (separate layer — large and slow)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY templates/ templates/

# Copy model weights
COPY brake_horn_detection.pt .
COPY face_landmarker.task .

# Pre-download YOLO base model into image so first run is fast
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" 2>/dev/null || true

# Runtime directories
RUN mkdir -p frames annotated detections face_crops profiles

# HuggingFace model cache will be mounted as a volume
ENV HF_HOME=/hf_cache

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV ENABLE_VLM=1
ENV VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

HEALTHCHECK --interval=60s --timeout=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
