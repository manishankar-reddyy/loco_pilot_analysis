"""Configuration: paths, environment flags, thresholds and class maps."""
from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

os.chdir(BASE_DIR)

AUTO_INSTALL_DEPS = os.getenv("AUTO_INSTALL_DEPS", "0").strip().lower() in {"1", "true", "yes", "on"}

ENABLE_VLM = os.getenv("ENABLE_VLM", "1").strip().lower() in {"1", "true", "yes", "on"}

FACE_LANDMARKER_PATH = str(BASE_DIR / "face_landmarker.task")

FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

BRAKE_HORN_MODEL_PATH = str(BASE_DIR / "brake_horn_detection.pt") if (BASE_DIR / "brake_horn_detection.pt").exists() else None

FRAME_DIR     = str(BASE_DIR / 'frames')

ANNOTATED_DIR = str(BASE_DIR / 'annotated')

DETECTION_DIR = str(BASE_DIR / 'detections')

CROP_DIR      = str(BASE_DIR / 'face_crops')

PROFILE_DIR   = str(BASE_DIR / 'profiles')

for d in [FRAME_DIR, ANNOTATED_DIR, DETECTION_DIR, CROP_DIR, PROFILE_DIR]:
    os.makedirs(d, exist_ok=True)

MAX_FRAMES        = 200

SAMPLE_INTERVAL   = 4

END_DENSE_SECONDS = 8

KEEP_FRAMES_IN_MEMORY = True

WRITE_EXTRACTED_FRAMES = False

YOLO_BATCH_SIZE = int(os.getenv('YOLO_BATCH_SIZE', '32'))

YOLO_BATCH_CANDIDATES = [16, 32, 64]

MEDIAPIPE_THREADS = int(os.getenv('MEDIAPIPE_THREADS', '4'))

MEDIAPIPE_MIN_PERSON_CONF = 0.4

FACE_ROI_MIN_SIZE = 400

VLM_MODEL_ID       = os.getenv('VLM_MODEL_ID', 'Qwen/Qwen2.5-VL-7B-Instruct')

VLM_MAX_EVENTS     = int(os.getenv('VLM_MAX_EVENTS', '8'))

VLM_MAX_TOKENS     = int(os.getenv('VLM_MAX_TOKENS', '140'))

VLM_IMAGE_SIZE     = int(os.getenv('VLM_IMAGE_SIZE', '448'))

VLM_CONTEXT_IMAGES = int(os.getenv('VLM_CONTEXT_IMAGES', '6'))

DENSE_EVENT_WINDOW_SECONDS = 6

DENSE_EVENT_FPS = 1

DENSE_EVENT_BEHAVIORS = {'writing', 'nap', 'alertness', 'distraction', 'attention_state'}

FUSION_THRESHOLDS = {
    'mobile': 0.62, 'writing': 0.48, 'packing': 0.62, 'leaving': 0.65,
    'nap': 0.62, 'distraction': 0.58, 'horn': 0.55, 'brake_a9': 0.55,
}

GROUND_TRUTH_JSON = None

YOLO_CLASSES = {
    0: 'person', 67: 'cell_phone', 24: 'backpack',
    26: 'handbag', 28: 'suitcase', 73: 'book', 63: 'laptop',
}

BRAKE_HORN_CLASSES = {
    0: 'horn_active', 1: 'horn_inactive', 2: 'brake_applied', 3: 'brake_released',
}

YOLO_COLORS = {
    'person': (0,255,0), 'cell_phone': (0,0,255), 'backpack': (255,0,255),
    'handbag': (255,0,255), 'suitcase': (255,0,255), 'book': (0,165,255),
    'laptop': (0,165,255), 'horn_active': (255,0,0), 'brake_applied': (0,0,255),
    'horn_inactive': (180,180,180), 'brake_released': (180,180,180),
}

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".ts", ".mpg", ".mpeg"}

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "4096"))
