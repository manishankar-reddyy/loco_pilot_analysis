# Knowledge Transfer Document: Loco Pilot Crew Behavior Monitoring

## 1. Project Overview
The **Loco Pilot Crew Behavior Monitoring** project is an AI-powered safety monitoring system that analyzes CCTV footage from locomotive cabs. It detects 9 critical crew behaviors—ranging from napping and mobile phone usage to horn and brake operation—using a multi-tiered Artificial Intelligence pipeline (Local CV + LLM Validation).

## 2. Tech Stack & Architecture
- **Web Framework**: FastAPI, Uvicorn, Jinja2 templates (Frontend).
- **Computer Vision (Local Edge processing)**: 
  - OpenCV (Image processing & fallback DNN face detection).
  - MediaPipe Tasks API (`FaceLandmarker` for precise EAR, yawning, and head pose estimation).
  - Ultralytics YOLOv8 (for object detection like persons, phones, bags).
- **Custom Object Detection**: Custom-trained YOLOv8 model for Loco cabin specifics (Horn & Brake application).
- **Vision-Language Model (VLM)**: Google Gemini (`gemini-2.5-flash`) for multi-modal contextual cross-validation of behaviors.
- **Reporting**: Python `fpdf2` for dynamic PDF report generation.
- **Containerization**: Docker & Docker Compose (`python:3.11-slim`).

---

## 3. The 3-Tier AI Analysis Pipeline
The core of the system lies in `main.py` which triggers when a video is posted to the `/analyze` endpoint.

### Tier 1: Local Computer Vision Extraction (Dense Sampling)
- **Frame Extraction**: Extracts frames every 4 seconds, and every 1 second in the last 8 seconds.
- **YOLO Detection (`yolov8s.pt`)**: Detects standard COCO objects like `person`, `cell_phone`, `backpack`, `book`, etc.
- **Custom Brake/Horn YOLO (`brake_horn_detection.pt`)**: Detects `horn_active`, `brake_applied`, etc.
- **MediaPipe FaceLandmarker**: Crops YOLO person detections and passes them to MediaPipe to extract:
  - **EAR (Eye Aspect Ratio)** & **Blink Blendshapes**: For sleep/micro-sleep detection.
  - **MAR (Mouth Aspect Ratio)**: For yawn detection.
  - **SolvePnP Head Pose**: Yaw, Pitch, Roll to detect looking away (distraction) or drooping head.

### Tier 2: Local Rule-Based Engine
Calculates parameters locally, analyzing sequences over the frames:
- Validates the camera baselines (calculating median yaw/pitch to adjust for side-mounted CCTV cameras).
- Tracks person count drops across consecutive frames to flag "Leaving Place".
- Matches "hands-on horn bounding boxes" to flag Horn Operations.

### Tier 3: LLM Cross-Validation (Gemini)
To reduce false positives and capture complex context, frames are passed to `gemini-2.5-flash` every 7 seconds. 
- It uses a specialized highly descriptive prompt (see `GEMINI_PER_FRAME_PROMPT` in `main.py`).
- It parses structured LLM feedback (e.g., `PARAMETER: VALUE | reason`).
- Final statuses are a consensus of YOLO/MediaPipe alerts + Gemini context (especially for Phone usage, Alertness, and Distraction). 
- Custom Brake & Horn triggers rely purely on the specialized local model since VLMs struggle with specific loco mechanics.

---

## 4. Key Behaviors Monitored (The 9 Parameters)
The system actively scores:
1. **Crew Alertness**: Blended score of Eye Closure (PERCLOS), drooping head, and yawning.
2. **Distraction**: Sustained gaze away from the camera baseline.
3. **Writing**: Detection of books/papers combined with downward gaze.
4. **Nap/Micro-sleep**: Consecutive frames with eyes closed.
5. **Mobile Use**: YOLO cell phone detection + Gemini active usage verification.
6. **Packing belongings**: Detecting handling of bags/backpacks.
7. **Leaving Place**: Drops in person count (from 2 to 1 or 0). 
8. **Horn Lever Operation**: Custom YOLO detection of `horn_active`.
9. **A9/SA9 Brake Application**: Custom YOLO detection of `brake_applied`.

---

## 5. Model Training Pipeline
The repository includes dedicated scripts for retraining the Brake & Horn custom model on new cab layouts:

### `prepare_yolo_dataset.py`
- Converts standard **LabelMe JSON annotations** into **YOLO normalized bounding boxes**.
- Class mapping: `horn` -> `horn_active`, `no_brake` -> `brake_released`.
- Shuffles and automatically splits data (80% Train, 20% Val).
- Creates the necessary `data.yaml` automatically.

### `train_yolo.py`
- Training script utilizing Ultralytics YOLOv8 engine.
- Base Model: `yolov8s.pt` (Small variant offers a good speed/accuracy trade-off).
- Embedded augmentations geared toward Loco cabs (Hue mapping, minor 5-degree rotation, NO vertical flip since cab orientation is fixed).
- Model outputs save to `./runs/brake_horn_detection/weights/best.pt`.

---

## 6. Setup & Deployment Instructions

### Prerequisites
- Docker & Docker Compose
- Google Gemini API Key

### Running Locally (Docker)
1. Add your Gemini API key to a `.env` file (or pass it via environment):
   `GEMINI_API_KEY=your_key_here`
2. Start the container:
   `docker-compose up -d --build`
3. Access the web interface at: `http://localhost:8000`

### Important File Paths
- **Uploaded Videos**: Mounted to `./videos`
- **Output Detections / Images**: Mounted to `./detections`

---

## 7. Troubleshooting & Maintenance Tips
- **MediaPipe Failing to Load Faces**: The model relies on CCTV resolution. The pipeline automatically tries cropping standard YOLO person boundaries and upscaling them to 400x400px before feeding to MediaPipe to ensure accurate eye/mouth extraction.
- **High False Positive Mobile Usage**: Tweak `GEMINI_PER_FRAME_PROMPT` to restrict Gemini from flagging objects that look like phones (e.g., reflective handles).
- **Adding New Safety Parameters**: 
  1. Add UI Config in `PARAMETER_CONFIG` dict.
  2. Implement CV rule logic in `compute_parameters_locally`.
  3. Expand the `GEMINI_PER_FRAME_PROMPT` prompt format.
- **Reporting Issues**: Formatting problems in the PDF are usually due to non-Latin-1 characters breaking `fpdf2`. Keep the `strip_emoji` function up-to-date.
