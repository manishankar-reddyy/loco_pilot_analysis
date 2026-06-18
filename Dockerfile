# ===================================
# Loco Pilot Crew Behavior Monitoring
# Requires NVIDIA GPU with 12GB+ VRAM
# ===================================

# Official PyTorch image — has CUDA 12.1 + cuDNN + torch pre-installed
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Qwen VLM dependencies (torch is already in the base image)
RUN pip install --no-cache-dir \
    "transformers>=4.52.0" \
    "accelerate>=0.34.0" \
    "bitsandbytes>=0.44.0" \
    "qwen-vl-utils>=0.0.8"

# Install remaining app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY templates/ templates/

# Copy model weights
COPY brake_horn_detection.pt .
COPY face_landmarker.task .

# Pre-download YOLO base model into the image
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" 2>/dev/null || true

# Runtime directories
RUN mkdir -p frames annotated detections face_crops profiles

# HuggingFace model cache — mount as volume to persist across restarts
ENV HF_HOME=/hf_cache
ENV PYTHONUNBUFFERED=1
ENV ENABLE_VLM=1
ENV VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
