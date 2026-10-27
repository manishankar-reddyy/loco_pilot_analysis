# ===================================
# Loco Pilot Crew Behavior Monitoring
# Multi-stage Docker build
# ===================================

FROM python:3.11-slim AS base

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main3.py main.py
COPY templates/ templates/

# Copy YOLO model weights
COPY brake_horn_detection.pt .

# Pre-download YOLO base model so first run is faster
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" 2>/dev/null || true

# Create directories for runtime data
RUN mkdir -p frames annotated detections faces persons face_crops videos

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV GEMINI_API_KEY=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
