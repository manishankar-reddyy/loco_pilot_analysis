import os
import cv2
import shutil
import time
import re
import json
import math
import numpy as np
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
from fpdf import FPDF

# =========================
# MEDIAPIPE TASKS API SETUP (v0.10.31+)
# =========================
MEDIAPIPE_AVAILABLE = False
face_landmarker = None

# Model file for MediaPipe FaceLandmarker
FACE_LANDMARKER_MODEL = "face_landmarker.task"
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


def download_model_if_needed(model_path, url):
    """Auto-download MediaPipe model file if not present."""
    if os.path.exists(model_path):
        return True
    try:
        print(f"Downloading {os.path.basename(model_path)} from Google...")
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded {os.path.basename(model_path)} ({os.path.getsize(model_path) / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


try:
    import mediapipe as mp
    if download_model_if_needed(FACE_LANDMARKER_MODEL, FACE_LANDMARKER_URL):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
            running_mode=VisionRunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=4,
            min_face_detection_confidence=0.25,
            min_face_presence_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        face_landmarker = FaceLandmarker.create_from_options(options)
        MEDIAPIPE_AVAILABLE = True
        print("MediaPipe FaceLandmarker (Tasks API) initialized with blendshapes + 478 landmarks.")
    else:
        print("MediaPipe model not available. Face analysis will use OpenCV fallback.")
except Exception as e:
    print(f"MediaPipe Tasks API not available: {e}")
    print("Face analysis will use OpenCV DNN fallback.")

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available.")

# =========================
# CONFIG
# =========================
FRAME_DIR = "frames"
PERSON_DIR = "persons"
DETECTION_DIR = "detections"
ANNOTATED_DIR = "annotated"

# Ensure directories exist
for folder in [FRAME_DIR, PERSON_DIR, DETECTION_DIR, ANNOTATED_DIR]:
    if os.path.exists(folder) and folder != DETECTION_DIR:
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# YOLO COCO classes of interest for railway cab monitoring
YOLO_CLASSES_OF_INTEREST = {
    0: "person",
    67: "cell_phone",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    73: "book",
    63: "laptop",
    41: "cup",
    39: "bottle",
}

# Color map for YOLO annotations (BGR format for OpenCV)
YOLO_COLORS = {
    "person": (0, 200, 0),        # Green
    "cell_phone": (0, 0, 255),    # Red — critical!
    "book": (255, 100, 0),        # Blue
    "backpack": (0, 180, 255),    # Orange
    "handbag": (0, 180, 255),     # Orange
    "suitcase": (0, 180, 255),    # Orange
    "laptop": (255, 0, 128),      # Purple
    "cup": (200, 200, 0),         # Cyan
    "bottle": (200, 200, 0),      # Cyan
    "chair": (128, 128, 128),     # Gray
    "tv": (255, 255, 0),          # Cyan-bright
    #'lever':(0,0,0), #white
    #'horn':(0,0,255), #blue
    #'push':(255,0,0), #red
}

# =========================
# FASTAPI SETUP
# =========================
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/detections", StaticFiles(directory=DETECTION_DIR), name="detections")

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# Global store for the last analysis (used by PDF download)
last_report = {
    "parameters": {},
    "overall_summary": "",
    "recommendations": "",
    "video_info": {},
    "result_text": "",
}

# =========================
# YOLO SETUP (auto-downloads yolov8s for better accuracy)
# =========================
yolo_model = None
if YOLO_AVAILABLE:
    try:
        yolo_model = YOLO("yolov8s.pt")  # Auto-downloads if not present (~22MB)
        print("YOLO v8s (small) model loaded — better accuracy for small objects.")
    except Exception:
        try:
            yolo_model = YOLO("yolov8n.pt")  # Fallback to nano
            print("YOLO v8n (nano) model loaded as fallback.")
        except Exception as e:
            print(f"Warning: Could not load any YOLO model: {e}")
            YOLO_AVAILABLE = False

# =========================
# CUSTOM BRAKE & HORN YOLO MODEL
# =========================
BRAKE_HORN_MODEL_PATH = "brake_horn_detection.pt"
brake_horn_model = None
BRAKE_HORN_AVAILABLE = False
BRAKE_HORN_CLASSES = {
    0: "horn_active",
    1: "horn_inactive",
    2: "brake_applied",
    3: "brake_released",
}
if YOLO_AVAILABLE and os.path.exists(BRAKE_HORN_MODEL_PATH):
    try:
        brake_horn_model = YOLO(BRAKE_HORN_MODEL_PATH)
        BRAKE_HORN_AVAILABLE = True
        print(f"Custom Brake & Horn YOLO model loaded from {BRAKE_HORN_MODEL_PATH}.")
    except Exception as e:
        print(f"Warning: Could not load Brake & Horn model: {e}")
else:
    if not os.path.exists(BRAKE_HORN_MODEL_PATH):
        print(f"Warning: Brake & Horn model not found at {BRAKE_HORN_MODEL_PATH}.")

# =========================
# OPENCV DNN FACE DETECTOR (fallback when MediaPipe unavailable)
# =========================
opencv_face_net = None
try:
    # Use OpenCV's built-in DNN face detector (more accurate than Haar cascade)
    opencv_face_net = cv2.FaceDetectorYN.create(
        "",  # Will use built-in model
        "",
        (320, 320),
        0.5,  # score threshold
        0.3,  # nms threshold
        5000  # top_k
    )
    print("OpenCV FaceDetectorYN loaded (DNN-based fallback).")
except Exception:
    # Fall back to Haar cascade if DNN not available
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        opencv_face_cascade = cv2.CascadeClassifier(cascade_path)
        if opencv_face_cascade.empty():
            opencv_face_cascade = None
        else:
            print("OpenCV Haar face detector loaded (fallback).")
    except Exception as e:
        print(f"OpenCV face detector not available: {e}")
        opencv_face_cascade = None

# =========================
# FRAME EXTRACTION (denser sampling)
# =========================
MAX_FRAMES = 200  # Higher cap since local analysis is fast
SAMPLE_INTERVAL = 4  # seconds between frames for local analysis


def extract_frames(video_path):
    """Extract frames every SAMPLE_INTERVAL seconds, capped at MAX_FRAMES.
    In the last 8 seconds, extracts every second for denser end-of-video analysis."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames // fps

    # Adaptive interval: if video is long, increase interval to stay within cap
    interval = SAMPLE_INTERVAL
    estimated_frames = duration // interval
    if estimated_frames > MAX_FRAMES:
        interval = max(1, duration // MAX_FRAMES)

    END_DENSE_SECONDS = 8  # Extract every second for the last 8 seconds
    dense_start = max(0, duration - END_DENSE_SECONDS)

    frames = []
    extracted_seconds = set()

    # Regular interval extraction (up to dense_start)
    for sec in range(0, duration, interval):
        if sec >= dense_start:
            break  # Will be handled by dense extraction
        frame_no = sec * fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            path = f"{FRAME_DIR}/frame_{sec}.jpg"
            cv2.imwrite(path, frame)
            frames.append((sec, path))
            extracted_seconds.add(sec)
        if len(frames) >= MAX_FRAMES - END_DENSE_SECONDS:
            break

    # Dense extraction for last 8 seconds (every 1 second)
    for sec in range(dense_start, duration):
        if sec in extracted_seconds:
            continue
        frame_no = sec * fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            path = f"{FRAME_DIR}/frame_{sec}.jpg"
            cv2.imwrite(path, frame)
            frames.append((sec, path))

    cap.release()
    print(f"  Extracted {len(frames)} frames (dense sampling in last {END_DENSE_SECONDS}s: {dense_start}s-{duration}s)")
    return frames, fps, duration


# =========================
# YOLO FULL OBJECT DETECTION
# =========================
def detect_objects_yolo(frame_path):
    """Run YOLO on a frame. Returns list of all relevant detected objects with class, confidence, bbox."""
    if not YOLO_AVAILABLE or yolo_model is None:
        return []
    try:
        results = yolo_model(frame_path, verbose=False, conf=0.2)  # Lower threshold to catch small phones
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in YOLO_CLASSES_OF_INTEREST:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "class_id": cls_id,
                        "class_name": YOLO_CLASSES_OF_INTEREST[cls_id],
                        "confidence": round(conf, 2),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    })
        return detections
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return []


# =========================
# BRAKE & HORN DETECTION (Custom YOLO)
# =========================
def detect_brake_horn(frame_path):
    """Run custom Brake & Horn YOLO model on a frame.
    Returns list of detections with class_name, confidence, bbox."""
    if not BRAKE_HORN_AVAILABLE or brake_horn_model is None:
        return []
    try:
        results = brake_horn_model(frame_path, verbose=False, conf=0.3)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in BRAKE_HORN_CLASSES:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "class_id": cls_id,
                        "class_name": BRAKE_HORN_CLASSES[cls_id],
                        "confidence": round(conf, 2),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    })
        return detections
    except Exception as e:
        print(f"Brake/Horn detection error: {e}")
        return []


# =========================
# ADVANCED FACE ANALYSIS
# (MediaPipe FaceLandmarker + Blendshapes + solvePnP + EAR + MAR)
# =========================

# Eye landmark indices for 478-point FaceLandmarker model
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]   # p1-p6
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # p1-p6

# Mouth landmark indices for MAR (Mouth Aspect Ratio) — yawn detection
MOUTH_IDX = [61, 291, 13, 14, 81, 178, 311, 402]  # outer corners + top/bottom

# Key blendshape names we monitor
BLENDSHAPE_KEYS = [
    "eyeBlinkLeft", "eyeBlinkRight",
    "jawOpen",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "mouthFunnel", "mouthPucker",
]

# 3D model points for solvePnP head pose estimation (canonical face)
# These correspond to: nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),     # Chin (landmark 152)
    (-225.0, 170.0, -135.0),  # Left eye outer (landmark 263)
    (225.0, 170.0, -135.0),   # Right eye outer (landmark 33)
    (-150.0, -150.0, -125.0), # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0),  # Right mouth corner (landmark 291)
], dtype=np.float64)

# Landmark indices matching the 3D model points above
SOLVEPNP_LANDMARK_IDX = [1, 152, 263, 33, 61, 291]


def compute_ear(landmarks, eye_indices):
    """Compute Eye Aspect Ratio from 6 eye landmarks (NormalizedLandmark)."""
    try:
        pts = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
        # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        vertical1 = math.dist(pts[1], pts[5])
        vertical2 = math.dist(pts[2], pts[4])
        horizontal = math.dist(pts[0], pts[3])
        if horizontal < 1e-6:
            return 0.3
        return round((vertical1 + vertical2) / (2.0 * horizontal), 3)
    except Exception:
        return None


def compute_mar(landmarks):
    """Compute Mouth Aspect Ratio for yawn detection."""
    try:
        # Mouth corners (horizontal)
        left_corner = (landmarks[61].x, landmarks[61].y)
        right_corner = (landmarks[291].x, landmarks[291].y)
        # Top and bottom lip (vertical)
        top_lip = (landmarks[13].x, landmarks[13].y)
        bottom_lip = (landmarks[14].x, landmarks[14].y)
        # Additional vertical points
        top2 = (landmarks[81].x, landmarks[81].y)
        bottom2 = (landmarks[178].x, landmarks[178].y)
        top3 = (landmarks[311].x, landmarks[311].y)
        bottom3 = (landmarks[402].x, landmarks[402].y)

        horizontal = math.dist(left_corner, right_corner)
        vertical1 = math.dist(top_lip, bottom_lip)
        vertical2 = math.dist(top2, bottom2)
        vertical3 = math.dist(top3, bottom3)

        if horizontal < 1e-6:
            return 0.0
        mar = (vertical1 + vertical2 + vertical3) / (3.0 * horizontal)
        return round(mar, 3)
    except Exception:
        return None


def compute_head_pose_solvepnp(landmarks, img_w, img_h):
    """Compute precise head pose (pitch, yaw, roll) using cv2.solvePnP."""
    try:
        # 2D image points from face landmarks
        image_points = np.array([
            (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
            for idx in SOLVEPNP_LANDMARK_IDX
        ], dtype=np.float64)

        # Camera matrix approximation
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "pitch_label": "unknown", "yaw_label": "unknown"}

        # Convert rotation vector to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = round(angles[0], 1)  # Positive = looking up, Negative = looking down
        yaw = round(angles[1], 1)    # Positive = turned right, Negative = turned left
        roll = round(angles[2], 1)   # Head tilt

        # Generate human-readable labels
        if pitch < -15:
            pitch_label = "drooping"
        elif pitch < -8:
            pitch_label = "slight_tilt_down"
        elif pitch > 15:
            pitch_label = "looking_up"
        else:
            pitch_label = "upright"

        if abs(yaw) > 25:
            yaw_label = "turned_right" if yaw > 0 else "turned_left"
        elif abs(yaw) > 12:
            yaw_label = "slightly_right" if yaw > 0 else "slightly_left"
        else:
            yaw_label = "forward"

        return {
            "pitch": pitch, "yaw": yaw, "roll": roll,
            "pitch_label": pitch_label, "yaw_label": yaw_label
        }
    except Exception:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "pitch_label": "unknown", "yaw_label": "unknown"}


def extract_blendshape_scores(blendshapes):
    """Extract relevant blendshape scores from MediaPipe FaceLandmarker result."""
    scores = {}
    if blendshapes:
        for bs in blendshapes:
            if bs.category_name in BLENDSHAPE_KEYS:
                scores[bs.category_name] = round(bs.score, 3)
    return scores


def analyze_face(frame_path):
    """Advanced face analysis using MediaPipe FaceLandmarker (Tasks API).
    Returns: EAR, MAR, blendshapes (eyeBlink, jawOpen, gaze), solvePnP head pose."""
    result = {"faces_detected": 0, "face_analyses": []}

    img = cv2.imread(frame_path)
    if img is None:
        return result

    h, w = img.shape[:2]

    # -------- PRIMARY: MediaPipe FaceLandmarker (Tasks API) --------
    if MEDIAPIPE_AVAILABLE and face_landmarker is not None:
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = face_landmarker.detect(mp_image)

            if detection_result.face_landmarks:
                result["faces_detected"] = len(detection_result.face_landmarks)

                for face_idx, face_lm in enumerate(detection_result.face_landmarks):
                    # --- EAR (Eye Aspect Ratio) ---
                    ear_left = compute_ear(face_lm, LEFT_EYE_IDX)
                    ear_right = compute_ear(face_lm, RIGHT_EYE_IDX)
                    avg_ear = None
                    eye_status = "unknown"
                    if ear_left is not None and ear_right is not None:
                        avg_ear = round((ear_left + ear_right) / 2, 3)
                        if avg_ear < 0.18:
                            eye_status = "CLOSED"
                        elif avg_ear < 0.23:
                            eye_status = "PARTIALLY_CLOSED"
                        else:
                            eye_status = "OPEN"

                    # --- MAR (Mouth Aspect Ratio) for yawn detection ---
                    mar = compute_mar(face_lm)
                    yawn_status = "NOT_YAWNING"
                    if mar is not None and mar > 0.6:
                        yawn_status = "YAWNING"
                    elif mar is not None and mar > 0.4:
                        yawn_status = "MOUTH_OPEN"

                    # --- solvePnP Head Pose (precise angles in degrees) ---
                    head_pose = compute_head_pose_solvepnp(face_lm, w, h)

                    # --- Blendshape scores (direct from MediaPipe model) ---
                    blendshapes = {}
                    if detection_result.face_blendshapes and face_idx < len(detection_result.face_blendshapes):
                        blendshapes = extract_blendshape_scores(
                            detection_result.face_blendshapes[face_idx]
                        )

                    # Cross-validate EAR with blendshape eyeBlink
                    blink_left = blendshapes.get("eyeBlinkLeft", 0)
                    blink_right = blendshapes.get("eyeBlinkRight", 0)
                    avg_blink = (blink_left + blink_right) / 2

                    # If blendshapes strongly say eyes closed, override EAR
                    if avg_blink > 0.6:
                        eye_status = "CLOSED"
                    elif avg_blink > 0.4 and eye_status == "OPEN":
                        eye_status = "PARTIALLY_CLOSED"

                    # Gaze direction from blendshapes
                    look_down = (blendshapes.get("eyeLookDownLeft", 0) + blendshapes.get("eyeLookDownRight", 0)) / 2
                    look_up = (blendshapes.get("eyeLookUpLeft", 0) + blendshapes.get("eyeLookUpRight", 0)) / 2
                    gaze = "down" if look_down > 0.4 else "up" if look_up > 0.4 else "forward"

                    result["face_analyses"].append({
                        "ear_left": ear_left,
                        "ear_right": ear_right,
                        "ear_avg": avg_ear,
                        "eye_status": eye_status,
                        "mar": mar,
                        "yawn_status": yawn_status,
                        "head_pitch": head_pose["pitch_label"],
                        "head_yaw": head_pose["yaw_label"],
                        "head_pitch_deg": head_pose["pitch"],
                        "head_yaw_deg": head_pose["yaw"],
                        "head_roll_deg": head_pose["roll"],
                        "gaze_direction": gaze,
                        "blink_score": round(avg_blink, 3),
                        "jaw_open_score": blendshapes.get("jawOpen", 0),
                        "blendshapes": blendshapes,
                    })
                return result
        except Exception as e:
            print(f"FaceLandmarker error: {e}")

    # -------- FALLBACK: OpenCV face detection --------
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = []

        # Try DNN detector first
        if opencv_face_net is not None:
            opencv_face_net.setInputSize((w, h))
            _, face_dets = opencv_face_net.detect(img)
            if face_dets is not None:
                for det in face_dets:
                    x, y, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                    faces.append((x, y, fw, fh))
        elif opencv_face_cascade is not None:
            face_rects = opencv_face_cascade.detectMultiScale(gray, 1.3, 5)
            faces = [(x, y, fw, fh) for (x, y, fw, fh) in face_rects]

        result["faces_detected"] = len(faces)
        for (x, y, fw, fh) in faces:
            face_center_y = (y + fh / 2) / h
            pitch_guess = "drooping" if face_center_y > 0.6 else "upright"
            result["face_analyses"].append({
                "ear_left": None, "ear_right": None, "ear_avg": None,
                "eye_status": "unknown (opencv fallback)",
                "mar": None, "yawn_status": "unknown",
                "head_pitch": pitch_guess, "head_yaw": "unknown",
                "head_pitch_deg": None, "head_yaw_deg": None, "head_roll_deg": None,
                "gaze_direction": "unknown",
                "blink_score": None, "jaw_open_score": None,
                "blendshapes": {},
            })
    except Exception as e:
        print(f"OpenCV face detection error: {e}")

    return result


# =========================
# CREATE ANNOTATED FRAME (YOLO boxes drawn on full frame)
# =========================
YOLO_COLORS = {
    "person": (0, 255, 0),
    "cell_phone": (0, 0, 255),
    "backpack": (255, 0, 255),
    "handbag": (255, 0, 255),
    "suitcase": (255, 0, 255),
    "book": (0, 165, 255),
    "laptop": (0, 165, 255),
    "cup": (128, 128, 0),
    "bottle": (128, 128, 0),
    # Custom Brake & Horn classes
    "horn_active": (255, 0, 0),       # Blue — horn engaged
    "horn_inactive": (180, 180, 180), # Gray — horn idle
    "brake_applied": (0, 0, 255),     # Red — brake applied
    "brake_released": (180, 180, 180),# Gray — brake released
}


def create_annotated_frame(frame_path, yolo_detections, timestamp):
    """Draw rich YOLO bounding boxes + labels on the full frame. Returns path to annotated image."""
    img = cv2.imread(frame_path)
    if img is None:
        return frame_path

    h, w = img.shape[:2]

    # Critical objects get thicker boxes and highlighted labels
    CRITICAL_CLASSES = {"cell_phone", "book", "laptop"}

    for det in yolo_detections:
        conf = det["confidence"]
        if conf < 0.5:  # Skip low-confidence detections
            continue

        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class_name"]
        color = YOLO_COLORS.get(cls_name, (255, 255, 255))
        is_critical = cls_name in CRITICAL_CLASSES

        # Draw bounding box (thicker for critical objects)
        thickness = 3 if is_critical else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw corner markers for critical objects
        if is_critical:
            corner_len = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
            # Top-left
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 4)
            # Top-right
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 4)
            # Bottom-left
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 4)
            # Bottom-right
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 4)

        # Label with background
        display_name = cls_name.replace("_", " ").upper() if is_critical else cls_name
        label = f"{display_name} {conf:.0%}"
        font_scale = 0.55 if is_critical else 0.45
        font_thick = 2
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        # Label background
        label_y = max(y1, label_size[1] + 10)
        cv2.rectangle(img, (x1, label_y - label_size[1] - 6), (x1 + label_size[0] + 6, label_y + 2), color, -1)
        cv2.putText(img, label, (x1 + 3, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

    # Add timestamp watermark (bottom-left)
    ts_label = f"T={timestamp}s"
    cv2.putText(img, ts_label, (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Add detection summary legend (top-right corner) — only ≥50% confidence
    high_conf_dets = [d for d in yolo_detections if d["confidence"] >= 0.5]
    if high_conf_dets:
        obj_counts = {}
        for det in high_conf_dets:
            name = det["class_name"]
            obj_counts[name] = obj_counts.get(name, 0) + 1
        
        legend_y = 15
        for name, count in obj_counts.items():
            color = YOLO_COLORS.get(name, (255, 255, 255))
            legend_text = f"{name}: {count}"
            text_size, _ = cv2.getTextSize(legend_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            x_pos = w - text_size[0] - 10
            # Background
            cv2.rectangle(img, (x_pos - 3, legend_y - 12), (w - 2, legend_y + 3), (0, 0, 0), -1)
            cv2.putText(img, legend_text, (x_pos, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            legend_y += 16

    annotated_path = f"{ANNOTATED_DIR}/annotated_{timestamp}.jpg"
    cv2.imwrite(annotated_path, img)
    return annotated_path


# =========================
# COMPREHENSIVE FRAME ANALYSIS PIPELINE
# (Crops person regions from YOLO boxes for better face detection)
# =========================
CROP_DIR = "face_crops"
os.makedirs(CROP_DIR, exist_ok=True)


def analyze_face_on_person_crops(frame_path, yolo_dets, timestamp):
    """Run face analysis on YOLO person crops (upscaled) for better detection on low-res CCTV.
    Falls back to full-frame analysis if no person crops yield results."""
    merged_result = {"faces_detected": 0, "face_analyses": []}

    img = cv2.imread(frame_path)
    if img is None:
        return merged_result

    h, w = img.shape[:2]

    # Extract person bounding boxes from YOLO detections
    person_boxes = [d for d in yolo_dets if d["class_name"] == "person" and d["confidence"] > 0.4]

    if not person_boxes:
        # No YOLO persons — try full frame as last resort
        return analyze_face(frame_path)

    for p_idx, person in enumerate(person_boxes):
        x1, y1, x2, y2 = person["bbox"]

        # Focus on upper 75% of person bbox (head + shoulders + upper body)
        person_h = y2 - y1
        head_y2 = y1 + int(person_h * 0.75)

        # Add generous padding around the head region (25% each side)
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((head_y2 - y1) * 0.25)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, head_y2 + pad_y)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue

        # Upscale small crops aggressively for face landmark detection
        crop_h, crop_w = crop.shape[:2]
        min_dim = 400  # Larger crops = better face detection on CCTV
        if crop_w < min_dim or crop_h < min_dim:
            scale = max(min_dim / crop_w, min_dim / crop_h)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Save crop temporarily
        crop_path = f"{CROP_DIR}/crop_{timestamp}_{p_idx}.jpg"
        cv2.imwrite(crop_path, crop)

        # Run face analysis on the cropped + upscaled person region
        crop_face_data = analyze_face(crop_path)

        if crop_face_data["faces_detected"] > 0:
            merged_result["faces_detected"] += crop_face_data["faces_detected"]
            merged_result["face_analyses"].extend(crop_face_data["face_analyses"])

        # Clean up crop
        try:
            os.remove(crop_path)
        except OSError:
            pass

    # If person crops didn't yield face detections, try full frame
    if merged_result["faces_detected"] == 0:
        return analyze_face(frame_path)

    return merged_result


def analyze_all_frames(frames):
    """Run YOLO object detection + brake/horn detection + face analysis on every frame. Returns structured per-frame data."""
    frame_analyses = []
    all_annotated = []

    for timestamp, frame_path in frames:
        print(f"  Analyzing frame at {timestamp}s...")

        # YOLO object detection (persons, phones, bags, books, etc.)
        yolo_dets = detect_objects_yolo(frame_path)

        # Custom Brake & Horn YOLO detection
        brake_horn_dets = detect_brake_horn(frame_path)

        # Face analysis — crop person regions from YOLO for better face detection on CCTV
        face_data = analyze_face_on_person_crops(frame_path, yolo_dets, timestamp)

        # Merge all detections for annotation drawing
        all_dets_for_annotation = yolo_dets + brake_horn_dets

        # Create annotated frame with all bounding boxes
        annotated_path = create_annotated_frame(frame_path, all_dets_for_annotation, timestamp)

        # Summarize object counts
        obj_counts = {}
        for det in yolo_dets:
            name = det["class_name"]
            obj_counts[name] = obj_counts.get(name, 0) + 1

        # Determine horn/brake states from custom model detections
        horn_active_detected = any(d["class_name"] == "horn_active" for d in brake_horn_dets)
        horn_inactive_detected = any(d["class_name"] == "horn_inactive" for d in brake_horn_dets)
        brake_applied_detected = any(d["class_name"] == "brake_applied" for d in brake_horn_dets)
        brake_released_detected = any(d["class_name"] == "brake_released" for d in brake_horn_dets)

        frame_info = {
            "timestamp": timestamp,
            "frame_path": frame_path,
            "annotated_path": annotated_path,
            "yolo_detections": yolo_dets,
            "brake_horn_detections": brake_horn_dets,
            "object_counts": obj_counts,
            "person_count": obj_counts.get("person", 0),
            "phone_detected": "cell_phone" in obj_counts,
            "bag_detected": any(k in obj_counts for k in ["backpack", "handbag", "suitcase"]),
            "book_detected": "book" in obj_counts,
            "horn_active_detected": horn_active_detected,
            "horn_inactive_detected": horn_inactive_detected,
            "brake_applied_detected": brake_applied_detected,
            "brake_released_detected": brake_released_detected,
            "face_analysis": face_data,
        }

        frame_analyses.append(frame_info)
        all_annotated.append((timestamp, annotated_path))

        # Console log key detections
        if yolo_dets:
            det_summary = ", ".join([f"{d['class_name']}({d['confidence']:.0%})" for d in yolo_dets])
            print(f"    YOLO: {det_summary}")
        if brake_horn_dets:
            bh_summary = ", ".join([f"{d['class_name']}({d['confidence']:.0%})" for d in brake_horn_dets])
            print(f"    Brake/Horn: {bh_summary}")
        if face_data["faces_detected"] > 0:
            for i, fa in enumerate(face_data["face_analyses"]):
                extras = ""
                if fa.get('head_pitch_deg') is not None:
                    extras += f", pitch={fa['head_pitch_deg']}deg, yaw={fa['head_yaw_deg']}deg"
                if fa.get('blink_score') is not None:
                    extras += f", blink={fa['blink_score']}"
                if fa.get('yawn_status') and fa['yawn_status'] != 'NOT_YAWNING':
                    extras += f", {fa['yawn_status']}(MAR={fa.get('mar')})"
                print(f"    Face {i+1}: eyes={fa['eye_status']}, EAR={fa['ear_avg']}, head={fa['head_pitch']}/{fa['head_yaw']}, gaze={fa.get('gaze_direction','?')}{extras}")

    return frame_analyses, all_annotated


# =========================
# LOCAL RULE-BASED PARAMETER ENGINE
# =========================
def find_max_consecutive(indices, total):
    """Find max number of consecutive indices (e.g., consecutive frames with closed eyes)."""
    if not indices:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def compute_parameters_locally(frame_analyses):
    """Determine all 7 safety parameters from per-frame CV data using rule-based thresholds.
    Returns: parameters dict, overall_summary, recommendations (all generated locally)."""
    total = len(frame_analyses)
    if total == 0:
        return {}, "No frames analyzed.", ""

    # --- Pre-compute per-frame metrics ---
    eyes_closed_indices = []  # Frame indices where any face has eyes closed
    eyes_partial_indices = []
    head_drooping_indices = []
    head_turned_indices = []
    gaze_down_indices = []
    yawning_indices = []
    phone_indices = []
    bag_indices = []
    book_indices = []
    horn_active_indices = []
    brake_applied_indices = []
    person_counts = []

    # Collect all yaw values to compute camera baseline
    all_yaw_values = []
    all_pitch_values = []

    for i, fa in enumerate(frame_analyses):
        person_counts.append(fa["person_count"])
        if fa["phone_detected"]:
            phone_indices.append(i)
        if fa["bag_detected"]:
            bag_indices.append(i)
        if fa["book_detected"]:
            book_indices.append(i)
        if fa.get("horn_active_detected"):
            horn_active_indices.append(i)
        if fa.get("brake_applied_detected"):
            brake_applied_indices.append(i)

        for face in fa["face_analysis"]["face_analyses"]:
            if face["eye_status"] == "CLOSED":
                eyes_closed_indices.append(i)
            elif face["eye_status"] == "PARTIALLY_CLOSED":
                eyes_partial_indices.append(i)

            pitch_deg = face.get("head_pitch_deg")
            yaw_deg = face.get("head_yaw_deg")

            if pitch_deg is not None:
                all_pitch_values.append(pitch_deg)
            if yaw_deg is not None:
                all_yaw_values.append(yaw_deg)

            # Drooping: only use significant negative pitch (< -25°)
            # Higher threshold because CCTV angle can cause baseline offset
            if pitch_deg is not None and pitch_deg < -25:
                head_drooping_indices.append(i)

            if face.get("gaze_direction") == "down":
                gaze_down_indices.append(i)

            if face.get("yawn_status") == "YAWNING":
                yawning_indices.append(i)

    # --- Camera baseline calibration ---
    # CCTV cameras are usually side-mounted, so the "normal" yaw is NOT 0°
    # Compute median yaw as the baseline and measure DEVIATION from it
    baseline_yaw = 0.0
    baseline_pitch = 0.0
    if all_yaw_values:
        sorted_yaws = sorted(all_yaw_values)
        baseline_yaw = sorted_yaws[len(sorted_yaws) // 2]  # Median
    if all_pitch_values:
        sorted_pitches = sorted(all_pitch_values)
        baseline_pitch = sorted_pitches[len(sorted_pitches) // 2]

    print(f"  Camera baseline: yaw={baseline_yaw:.1f}°, pitch={baseline_pitch:.1f}°")

    # Now recompute head_turned using DEVIATION from baseline
    for i, fa in enumerate(frame_analyses):
        for face in fa["face_analysis"]["face_analyses"]:
            yaw_deg = face.get("head_yaw_deg")
            if yaw_deg is not None:
                yaw_deviation = abs(yaw_deg - baseline_yaw)
                if yaw_deviation > 30:  # More than 30° away from normal position
                    head_turned_indices.append(i)

    # Deduplicate (a frame can appear multiple times if multiple faces)
    eyes_closed_indices = sorted(set(eyes_closed_indices))
    eyes_partial_indices = sorted(set(eyes_partial_indices))
    head_drooping_indices = sorted(set(head_drooping_indices))
    head_turned_indices = sorted(set(head_turned_indices))
    gaze_down_indices = sorted(set(gaze_down_indices))
    yawning_indices = sorted(set(yawning_indices))
    phone_indices = sorted(set(phone_indices))
    bag_indices = sorted(set(bag_indices))
    book_indices = sorted(set(book_indices))

    # --- PERCLOS (Percentage of Eye CLOSure) ---
    all_impaired = sorted(set(eyes_closed_indices + eyes_partial_indices))
    perclos = len(all_impaired) / total if total > 0 else 0
    consecutive_closed = find_max_consecutive(eyes_closed_indices, total)
    consecutive_phone = find_max_consecutive(phone_indices, total)

    # --- Helper to pick best evidence frame ---
    def best_evidence(indices):
        if not indices:
            return None
        # Prefer a frame in the middle of the longest run
        return indices[len(indices) // 2]

    def get_best_mobile_evidence(indices):
        if not indices:
            return None
        best_idx = indices[len(indices) // 2]
        min_y = float('inf')
        for idx in indices:
            for det in frame_analyses[idx].get("yolo_detections", []):
                if det["class_name"] == "cell phone":
                    y1 = det["bbox"][1]
                    if y1 < min_y:
                        min_y = y1
                        best_idx = idx
        return best_idx

    def best_brake_horn_evidence(indices, target_class):
        """Pick the frame with the highest confidence for the target brake/horn class."""
        if not indices:
            return None
        best_idx = indices[0]
        best_conf = 0.0
        for idx in indices:
            for det in frame_analyses[idx].get("brake_horn_detections", []):
                if det["class_name"] == target_class and det["confidence"] > best_conf:
                    best_conf = det["confidence"]
                    best_idx = idx
        return best_idx

    parameters = {}

    # ====== 1. ALERTNESS ======
    drooping_pct = len(head_drooping_indices) / total
    yawn_pct = len(yawning_indices) / total
    alertness_score = perclos * 0.5 + drooping_pct * 0.3 + yawn_pct * 0.2

    if alertness_score > 0.3 or perclos > 0.4:
        alertness_status = "LOW"
    elif alertness_score > 0.12 or perclos > 0.15:
        alertness_status = "MODERATE"
    else:
        alertness_status = "HIGH"

    alert_evidence = best_evidence(eyes_closed_indices or head_drooping_indices or yawning_indices)
    if alertness_status == "LOW":
        alert_desc = "Low. Crew members exhibited signs of fatigue and lack of focus. Extended periods of drooping head or closed eyes observed."
    elif alertness_status == "MODERATE":
        alert_desc = "Moderate. Crew showed occasional signs of reduced alertness, such as brief drooping or yawning, but maintained general awareness."
    else:
        alert_desc = "High. Crew members remained alert, maintaining upright posture and active engagement."

    parameters["alertness"] = {
        "key": "alertness", "label": "Crew Alertness", "icon": "🧠",
        "description": alert_desc,
        "status": alertness_status,
        "confidence": "HIGH" if total > 20 else "MEDIUM",
        "timestamp": "N/A",
        "image_url": None,
        "evidence_frame_idx": alert_evidence,
    }

    # ====== 2. DISTRACTION ======
    # Based ONLY on head significantly turned from camera baseline
    # gaze_down is NOT distraction — crew looks down at controls/instruments
    distracted_pct = len(head_turned_indices) / total if total > 0 else 0
    dist_evidence = best_evidence(head_turned_indices)

    if distracted_pct > 0.3:
        distraction_status = "HIGH"
    elif distracted_pct > 0.15:
        distraction_status = "MODERATE"
    elif distracted_pct > 0.08:
        distraction_status = "LOW"
    else:
        distraction_status = "NONE"

    if distraction_status == "HIGH":
        dist_desc = "High. A crew member was observed looking away from the track frequently, focusing on other directions for extended periods."
    elif distraction_status == "MODERATE":
        dist_desc = "Moderate. A crew member occasionally turned their head away from the forward view, indicating some level of divided attention."
    elif distraction_status == "LOW":
        dist_desc = "Low. Minor head turns observed, likely for checking necessary operational displays or surroundings."
    else:
        dist_desc = "None. Crew maintained forward focus without significant distraction."

    parameters["distraction"] = {
        "key": "distraction", "label": "Crew Distraction", "icon": "👀",
        "description": dist_desc,
        "status": distraction_status,
        "confidence": "HIGH" if total > 20 else "MEDIUM",
        "timestamp": "N/A",
        "image_url": None,
        "evidence_frame_idx": dist_evidence,
    }

    # ====== 3. WRITING ======
    writing_frames = sorted(set(book_indices) & set(gaze_down_indices))
    writing_only_book = book_indices  # Book alone is also evidence

    if len(writing_frames) >= 3:
        writing_status = "OBSERVED"
        writing_conf = "HIGH"
    if len(writing_frames) >= 3:
        writing_status = "OBSERVED"
        writing_conf = "HIGH"
        writing_desc = "Observed. A crew member was seen actively focused downward on paperwork or a document while the locomotive was in motion."
    elif len(writing_only_book) >= 3:
        writing_status = "OBSERVED"
        writing_conf = "MEDIUM"
        writing_desc = "Observed. Books or documents were present and crew attention appeared directed towards them."
    elif len(writing_only_book) >= 1:
        writing_status = "OBSERVED"
        writing_conf = "LOW"
        writing_desc = "Observed. Writing materials were visible, though active writing was brief or unclear."
    else:
        writing_status = "NOT_OBSERVED"
        writing_conf = "HIGH"
        writing_desc = "Not Observed. No writing materials were detected and no crew member was seen doing writing work."

    parameters["writing"] = {
        "key": "writing", "label": "Crew Doing Writing Work", "icon": "✍️",
        "description": writing_desc,
        "status": writing_status,
        "confidence": writing_conf,
        "timestamp": "N/A",
        "image_url": None,
        "evidence_frame_idx": best_evidence(writing_frames or writing_only_book),
    }

    # ====== 4. NAP / MICRO-SLEEP ======
    if consecutive_closed >= 5:
        nap_status = "OBSERVED"
        nap_conf = "HIGH"
    if consecutive_closed >= 5:
        nap_status = "OBSERVED"
        nap_conf = "HIGH"
        nap_desc = "Observed. A crew member appeared to be in a state of sleep, with prolonged periods of closed eyes."
    elif consecutive_closed >= 3:
        nap_status = "OBSERVED"
        nap_conf = "MEDIUM"
        nap_desc = "Observed. A crew member exhibited signs of micro-sleep or deep rest with eyes closed multiple times."
    elif perclos > 0.4:
        nap_status = "OBSERVED"
        nap_conf = "MEDIUM"
        nap_desc = "Observed. High frequency of prolonged eye closure indicates possible micro-sleep."
    else:
        nap_status = "NOT_OBSERVED"
        nap_conf = "HIGH"
        nap_desc = "Not Observed. No significant evidence of napping or prolonged eye closure was seen."

    parameters["nap"] = {
        "key": "nap", "label": "Crew Taking Nap / Micro-sleep", "icon": "😴",
        "description": nap_desc,
        "status": nap_status,
        "confidence": nap_conf,
        "timestamp": "N/A",
        "image_url": None,
        "evidence_frame_idx": best_evidence(eyes_closed_indices),
    }

    # ====== 5. MOBILE ======
    if len(phone_indices) >= 3 and consecutive_phone >= 2:
        mobile_status = "OBSERVED"
        mobile_conf = "HIGH"
        mobile_desc = "Observed. A crew member was seen actively using or holding a mobile phone during the monitored period."
    elif len(phone_indices) >= 2:
        mobile_status = "OBSERVED"
        mobile_conf = "MEDIUM"
        mobile_desc = "Observed. Evidence suggests brief use or checking of a mobile phone by a crew member."
    elif len(phone_indices) == 1:
        mobile_status = "NOT_OBSERVED"
        mobile_conf = "MEDIUM"
        mobile_desc = "Not Observed. Although a device was visible briefly, sustained usage was not confirmed."
    else:
        mobile_status = "NOT_OBSERVED"
        mobile_conf = "HIGH"
        mobile_desc = "Not Observed. Neither crew member was seen using a mobile phone during the recorded period."

    # Check for phone-like posture (gaze down + hand position) even without direct detection
    gaze_down_no_book = sorted(set(gaze_down_indices) - set(book_indices))
    if len(gaze_down_no_book) > total * 0.2 and mobile_status == "NOT_OBSERVED":
        mobile_desc += " Note: Prolonged downward gaze observed without visible documents, suggesting possible hidden phone use."

    parameters["mobile"] = {
        "key": "mobile", "label": "Crew Using Mobile", "icon": "📱",
        "description": mobile_desc,
        "status": mobile_status,
        "confidence": mobile_conf,
        "timestamp": "N/A",
        "image_url": None,
        "evidence_frame_idx": get_best_mobile_evidence(phone_indices),
    }

    # ====== 6. PACKING ======


    if len(bag_indices) > 0:
        packing_status = "NOT_OBSERVED"
        packing_conf = "LOW"
        packing_desc = "Not Observed. Bags or belongings were detected in the cab, but they appear to be stationary."
    else:
        packing_status = "NOT_OBSERVED"
        packing_conf = "HIGH"
        packing_desc = "Not Observed. No crew member was seen actively packing bags or gathering personal items."

    parameters["packing"] = {
        "key": "packing", "label": "Crew Packing Belongings", "icon": "🎒",
        "description": packing_desc,
        "status": packing_status,
        "confidence": packing_conf,
        "timestamp": "N/A",
        "image_url": None,
        "evidence_frame_idx": best_evidence(bag_indices),
    }

    # ====== 7. LEAVING PLACE ======
    # Track person count changes
    leaving_evidence = None
    if len(person_counts) >= 3:
        # Check if person count drops and stays low
        initial_count = max(person_counts[:min(5, len(person_counts))])  # Max in first 5 frames
        min_count = min(person_counts)
        drop_indices = [i for i, c in enumerate(person_counts) if c < initial_count]
        sustained_drop = find_max_consecutive(drop_indices, total)

        if initial_count >= 2 and min_count <= initial_count - 1 and sustained_drop >= 3:
            leaving_status = "OBSERVED"
            leaving_conf = "HIGH" if sustained_drop >= 5 else "MEDIUM"
            leaving_desc = "Observed. A crew member vacated their position and left the designated seat area."
            leaving_evidence = best_evidence(drop_indices)
        elif initial_count >= 2 and min_count < initial_count:
            leaving_status = "OBSERVED"
            leaving_conf = "LOW"
            leaving_desc = "Observed. Possible brief absence of a crew member from their designated monitoring position."
            leaving_evidence = best_evidence(drop_indices)
        else:
            leaving_status = "NOT_OBSERVED"
            leaving_conf = "HIGH"
            leaving_desc = "Not Observed. Both crew members remained in their designated seats or expected operational areas."
    else:
        leaving_status = "NOT_OBSERVED"
        leaving_conf = "LOW"
        leaving_desc = "Insufficient frames to track person count changes."

    parameters["leaving"] = {
        "key": "leaving", "label": "Crew Leaving Place", "icon": "🚪",
        "description": leaving_desc,
        "status": leaving_status,
        "confidence": leaving_conf,
        "timestamp": "N/A",
        "image_url": None,
        "evidence_frame_idx": leaving_evidence,
    }

    # --- Auto-flag leaving AND packing if person count drops to 0 in last 60 seconds ---
    if len(person_counts) >= 5:
        last_timestamp = frame_analyses[-1]["timestamp"]
        cutoff_time = max(0, last_timestamp - 60)  # Last 60 seconds
        final_zero_indices = [
            i for i in range(total)
            if person_counts[i] == 0 and frame_analyses[i]["timestamp"] >= cutoff_time
        ]

        if len(final_zero_indices) >= 2:
            # Empty cab at the end — crew has left (and likely packed)
            if parameters["leaving"]["status"] != "OBSERVED":
                parameters["leaving"]["status"] = "OBSERVED"
                parameters["leaving"]["confidence"] = "HIGH" if len(final_zero_indices) >= 3 else "MEDIUM"
                parameters["leaving"]["description"] = (
                    f"Cab completely empty (0 persons) in {len(final_zero_indices)} frames "
                    f"in the final portion of the video. Crew has left the cab."
                )
                parameters["leaving"]["evidence_frame_idx"] = final_zero_indices[-1]

            if parameters["packing"]["status"] != "OBSERVED":
                parameters["packing"]["status"] = "OBSERVED"
                parameters["packing"]["confidence"] = "MEDIUM"
                parameters["packing"]["description"] = (
                    f"Cab vacated with 0 persons detected in final {len(final_zero_indices)} frames. "
                    f"Crew likely packed belongings before leaving."
                )
                parameters["packing"]["evidence_frame_idx"] = final_zero_indices[0]

    # --- If crew leaving is OBSERVED, auto-flag packing too ---
    if parameters["leaving"]["status"] == "OBSERVED" and parameters["packing"]["status"] != "OBSERVED":
        parameters["packing"]["status"] = "OBSERVED"
        parameters["packing"]["confidence"] = "MEDIUM"
        parameters["packing"]["description"] = (
            "Crew leaving detected — crew likely packed belongings before departing. "
            + parameters["packing"]["description"]
        )
        if not parameters["packing"].get("evidence_frame_idx"):
            parameters["packing"]["evidence_frame_idx"] = parameters["leaving"].get("evidence_frame_idx")

    # ====== 8. HORN OPERATION (Custom YOLO) ======
    horn_active_indices = sorted(set(horn_active_indices))
    consecutive_horn = find_max_consecutive(horn_active_indices, total)

    if len(horn_active_indices) >= 3 and consecutive_horn >= 2:
        horn_status = "OBSERVED"
        horn_conf = "HIGH"
        horn_desc = f"Horn lever active detected by custom YOLO in {len(horn_active_indices)}/{total} frames ({consecutive_horn} consecutive). Strong evidence of horn operation."
    elif len(horn_active_indices) >= 2:
        horn_status = "OBSERVED"
        horn_conf = "MEDIUM"
        horn_desc = f"Horn lever active detected by custom YOLO in {len(horn_active_indices)} frames. Likely horn operation."
    elif len(horn_active_indices) == 1:
        horn_status = "OBSERVED"
        horn_conf = "LOW"
        horn_desc = f"Horn lever active detected in only 1 frame — brief horn usage."
    else:
        horn_status = "NOT_OBSERVED"
        horn_conf = "HIGH" if BRAKE_HORN_AVAILABLE else "LOW"
        horn_desc = "No horn lever activation detected by custom YOLO in any frame." if BRAKE_HORN_AVAILABLE else "Brake/Horn YOLO model not available."

    parameters["horn"] = {
        "key": "horn", "label": "Crew Operating Horn Lever/Switch", "icon": "📯",
        "description": horn_desc,
        "status": horn_status,
        "confidence": horn_conf,
        "timestamp": "N/A",
        "image_url": None,
        "evidence_frame_idx": best_brake_horn_evidence(horn_active_indices, "horn_active"),
    }

    # ====== 9. BRAKE A9/SA9 APPLICATION (Custom YOLO) ======
    brake_applied_indices = sorted(set(brake_applied_indices))
    consecutive_brake = find_max_consecutive(brake_applied_indices, total)

    if len(brake_applied_indices) >= 3 and consecutive_brake >= 2:
        brake_status = "OBSERVED"
        brake_conf = "HIGH"
        brake_desc = f"A9/SA9 brake application detected by custom YOLO in {len(brake_applied_indices)}/{total} frames ({consecutive_brake} consecutive). Strong evidence of brake application."
    elif len(brake_applied_indices) >= 2:
        brake_status = "OBSERVED"
        brake_conf = "MEDIUM"
        brake_desc = f"A9/SA9 brake application detected by custom YOLO in {len(brake_applied_indices)} frames. Likely brake application."
    elif len(brake_applied_indices) == 1:
        brake_status = "OBSERVED"
        brake_conf = "LOW"
        brake_desc = f"Brake application detected in only 1 frame — brief or transitional."
    else:
        brake_status = "NOT_OBSERVED"
        brake_conf = "HIGH" if BRAKE_HORN_AVAILABLE else "LOW"
        brake_desc = "No A9/SA9 brake application detected by custom YOLO in any frame." if BRAKE_HORN_AVAILABLE else "Brake/Horn YOLO model not available."

    parameters["brake_a9"] = {
        "key": "brake_a9", "label": "LP Applying A9/SA9 After Halt", "icon": "🛑",
        "description": brake_desc,
        "status": brake_status,
        "confidence": brake_conf,
        "timestamp": "N/A",
        "image_url": None,
        "evidence_frame_idx": best_brake_horn_evidence(brake_applied_indices, "brake_applied"),
    }

    # --- Save evidence images ---
    for key, param in parameters.items():
        idx = param.get("evidence_frame_idx")
        status = param.get("status", "")
        # Only save evidence if flagged
        if status in ["NOT_OBSERVED", "NONE"] or (key == "alertness" and status == "HIGH"):
            param["evidence_frame_idx"] = None
            continue

        if idx is not None and 0 <= idx < total:
            source_path = frame_analyses[idx].get("annotated_path", frame_analyses[idx]["frame_path"])
            config = PARAMETER_CONFIG.get(next((k for k, v in PARAMETER_CONFIG.items() if v["key"] == key), ""), {})
            if config:
                param["image_url"] = save_detection_image(
                    source_path, None,
                    config.get("bbox_label", key.upper()),
                    config.get("bbox_color", (255, 255, 255)),
                    config.get("prefix", key)
                )

    # --- Overall summary ---
    issues = []
    if alertness_status == "LOW":
        issues.append("low crew alertness")
    if nap_status == "OBSERVED":
        issues.append("napping/micro-sleep detected")
    if mobile_status == "OBSERVED":
        issues.append("mobile phone usage")
    if distraction_status in ["HIGH", "MODERATE"]:
        issues.append(f"{distraction_status.lower()} distraction")
    if leaving_status == "OBSERVED":
        issues.append("crew member leaving position")
    if writing_status == "OBSERVED":
        issues.append("writing work during duty")
    if packing_status == "OBSERVED":
        issues.append("packing belongings")
    if horn_status == "OBSERVED":
        issues.append("horn lever operation detected")
    if brake_status == "OBSERVED":
        issues.append("A9/SA9 brake application detected")

    if not issues:
        overall_summary = f"Analysis of {total} frames shows generally safe crew behavior. Alertness is {alertness_status}, no critical safety violations detected."
    else:
        overall_summary = f"Analysis of {total} frames flagged {len(issues)} concern(s): {', '.join(issues)}. Immediate review recommended."

    # --- Recommendations ---
    recs = []
    if alertness_status != "HIGH":
        recs.append(f"- Monitor crew alertness levels (PERCLOS={perclos:.0%}). Consider alertness management interventions.")
    if nap_status == "OBSERVED":
        recs.append("- CRITICAL: Napping/micro-sleep detected. Immediate intervention required for safety.")
    if mobile_status == "OBSERVED":
        recs.append("- Enforce strict no-phone policy during locomotive operation.")
    if distraction_status in ["HIGH", "MODERATE"]:
        recs.append("- Address crew distraction. Remind crew to maintain focus on track and controls.")
    if leaving_status == "OBSERVED":
        recs.append("- Investigate why crew member left their position. Ensure minimum staffing requirements.")
    if horn_status == "OBSERVED":
        recs.append("- Horn lever operation detected. Verify appropriate horn usage per operating rules.")
    if brake_status == "OBSERVED":
        recs.append("- A9/SA9 brake application after halt detected. Confirm proper braking procedure followed.")
    if not recs:
        recs.append("- Continue monitoring. Overall crew behavior appears compliant with safety standards.")
    recommendations = "\n".join(recs)

    return parameters, overall_summary, recommendations


# =========================
# GEMINI DENSE ANALYSIS (every 7 seconds)
# =========================
GEMINI_FRAME_INTERVAL = 4  # Analyze one frame every 4 seconds
GEMINI_BATCH_SIZE = 8      # Send up to 8 selected frames in one Gemini request
GEMINI_BATCH_WORKERS = 3   # Parallel batch requests; reduce if Gemini rate-limits

GEMINI_PER_FRAME_PROMPT = """Analyze this locomotive cab CCTV frame for crew safety monitoring. 
For EACH of the 9 parameters below, provide your assessment.
Reply in EXACTLY this format (one line per parameter):

ALERTNESS: HIGH/MODERATE/LOW | <brief reason>
DISTRACTION: YES/NO | <brief reason>
WRITING: YES/NO | <brief reason>
NAPPING: YES/NO | <brief reason>
MOBILE_PHONE: YES/NO | <brief reason>
PACKING: YES/NO | <brief reason>
LEAVING: YES/NO | <brief reason>
HORN: YES/NO | <brief reason>
BRAKE_A9: YES/NO | <brief reason>

Rules:
- ALERTNESS: Are crew members alert, eyes open, posture upright, engaged with controls?
- DISTRACTION: Is anyone looking away from track/controls, turned around, not focused on duties?
- WRITING: Say YES ONLY if you can clearly see a person actively writing on paper with a pen/pencil, or reading a physical document/book. Do NOT confuse instruments, control panels, gauges, or dashboards as writing materials. Operating controls is NOT writing.
- NAPPING: Are eyes closed, head drooping, slumped posture? Signs of sleep?
- MOBILE_PHONE: Is ANYONE (including crew members standing in the background or partially obscured) actively holding, looking at, or using a mobile phone? You MUST scan ALL visible people in the entire cab. STRICTLY report NO if the phone is merely resting in a pocket, bag, or on a desk without being actively used.
- PACKING: Is anyone handling bags, picking up personal items, organizing belongings, or preparing to leave? Look for bags being moved, items being gathered, crew standing up to collect things. Also say YES if seats appear recently vacated with belongings missing compared to normal.
- LEAVING: Is any crew seat EMPTY that was previously occupied? Is someone standing, walking to the door, or at the exit? Is the cab partially or fully vacated? Fewer crew visible than expected (normally 2)? If you see an EMPTY CAB with no crew, this is definitely YES.
- HORN: Is any crew member reaching for, pressing, or operating the horn lever/switch? Look for hand on or near the horn lever (usually located on the upper portion of the control panel/console).
- BRAKE_A9: Is the Loco Pilot (LP) applying the A9 or SA9 brake handle? Look for the LP's hand on the brake handle, the brake handle in the applied/pulled position.

Be accurate. If unsure, lean toward NO for safety-critical items except ALERTNESS."""


def parse_gemini_per_frame(text):
    """Parse Gemini's structured per-frame response into a dict."""
    result = {}
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue
        # Parse "PARAMETER: VALUE | reason"
        parts = line.split(":", 1)
        if len(parts) < 2:
            continue
        key = parts[0].strip().upper()
        rest = parts[1].strip()
        
        # Extract value and reason
        if "|" in rest:
            value, reason = rest.split("|", 1)
            value = value.strip().upper()
            reason = reason.strip()
        else:
            value = rest.strip().upper()
            reason = ""

        key_map = {
            "ALERTNESS": "alertness",
            "DISTRACTION": "distraction",
            "WRITING": "writing",
            "NAPPING": "nap",
            "MOBILE_PHONE": "mobile",
            "MOBILE PHONE": "mobile",
            "PACKING": "packing",
            "LEAVING": "leaving",
            "HORN": "horn",
            "BRAKE_A9": "brake_a9",
            "BRAKE A9": "brake_a9",
        }
        mapped_key = key_map.get(key)
        if mapped_key:
            result[mapped_key] = {"value": value, "reason": reason}
    return result


def chunked(items, size):
    """Yield fixed-size chunks from a list."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def build_gemini_batch_prompt(batch_indices, frame_analyses):
    """Build a prompt that asks Gemini to analyze multiple images in order."""
    frame_map = "\n".join(
        f"- Image {pos}: FRAME_INDEX {idx}, timestamp {frame_analyses[idx]['timestamp']}s"
        for pos, idx in enumerate(batch_indices, start=1)
    )
    return f"""Analyze these locomotive cab CCTV frames for crew safety monitoring.

Frame mapping:
{frame_map}

For EACH image, provide your assessment for the 9 parameters below.
Reply in EXACTLY this format for every image:

FRAME_INDEX: <frame_index from the mapping>
ALERTNESS: HIGH/MODERATE/LOW | <brief reason>
DISTRACTION: YES/NO | <brief reason>
WRITING: YES/NO | <brief reason>
NAPPING: YES/NO | <brief reason>
MOBILE_PHONE: YES/NO | <brief reason>
PACKING: YES/NO | <brief reason>
LEAVING: YES/NO | <brief reason>
HORN: YES/NO | <brief reason>
BRAKE_A9: YES/NO | <brief reason>
END_FRAME

Rules:
- ALERTNESS: Are crew members alert, eyes open, posture upright, engaged with controls?
- DISTRACTION: Is anyone looking away from track/controls, turned around, not focused on duties?
- WRITING: Say YES ONLY if you can clearly see a person actively writing on paper with a pen/pencil, or reading a physical document/book. Do NOT confuse instruments, control panels, gauges, or dashboards as writing materials. Operating controls is NOT writing.
- NAPPING: Are eyes closed, head drooping, slumped posture? Signs of sleep?
- MOBILE_PHONE: Is ANYONE (including crew members standing in the background or partially obscured) actively holding, looking at, or using a mobile phone? You MUST scan ALL visible people in the entire cab. STRICTLY report NO if the phone is merely resting in a pocket, bag, or on a desk without being actively used.
- PACKING: Is anyone handling bags, picking up personal items, organizing belongings, or preparing to leave? Look for bags being moved, items being gathered, crew standing up to collect things. Also say YES if seats appear recently vacated with belongings missing compared to normal.
- LEAVING: Is any crew seat EMPTY that was previously occupied? Is someone standing, walking to the door, or at the exit? Is the cab partially or fully vacated? Fewer crew visible than expected (normally 2)? If you see an EMPTY CAB with no crew, this is definitely YES.
- HORN: Is any crew member reaching for, pressing, or operating the horn lever/switch? Look for hand on or near the horn lever (usually located on the upper portion of the control panel/console).
- BRAKE_A9: Is the Loco Pilot (LP) applying the A9 or SA9 brake handle? Look for the LP's hand on the brake handle, the brake handle in the applied/pulled position.

Be accurate. If unsure, lean toward NO for safety-critical items except ALERTNESS."""


def parse_gemini_batch_response(text, expected_indices):
    """Parse Gemini's multi-frame response into {frame_idx: parsed_result}."""
    results = {}
    current_idx = None
    current_lines = []
    expected_set = set(expected_indices)

    def flush_current():
        nonlocal current_idx, current_lines
        if current_idx in expected_set and current_lines:
            results[current_idx] = parse_gemini_per_frame("\n".join(current_lines))
        current_idx = None
        current_lines = []

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith("FRAME_INDEX") or upper.startswith("FRAME "):
            flush_current()
            match = re.search(r"\d+", line)
            if match:
                current_idx = int(match.group(0))
            continue

        if upper.startswith("END_FRAME"):
            flush_current()
            continue

        if current_idx is not None:
            current_lines.append(line)

    flush_current()
    return results


def analyze_frame_batch_with_gemini(batch_indices, frame_analyses):
    """Analyze up to GEMINI_BATCH_SIZE selected frames in one Gemini request."""
    images = []
    try:
        for idx in batch_indices:
            images.append(Image.open(frame_analyses[idx]["frame_path"]))

        response = model.generate_content([
            build_gemini_batch_prompt(batch_indices, frame_analyses),
            *images,
        ])
        return parse_gemini_batch_response(response.text.strip(), batch_indices)
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass


def analyze_with_gemini_dense(parameters, frame_analyses):
    """Send one frame every 7 seconds to Gemini for comprehensive per-frame analysis.
    Aggregates Gemini votes across all frames to determine final parameter statuses.
    Merges with local CV data for maximum accuracy."""
    total = len(frame_analyses)
    if total == 0:
        return parameters, "No frames to analyze."

    # Select frames strictly every GEMINI_FRAME_INTERVAL seconds, PLUS dense at the end
    gemini_indices = []
    next_target_time = 0
    last_timestamp = frame_analyses[-1]["timestamp"] if frame_analyses else 0
    dense_start_time = max(0, last_timestamp - 8)

    for i, fa in enumerate(frame_analyses):
        ts = fa["timestamp"]
        if ts >= dense_start_time:
            gemini_indices.append(i)
        elif ts >= next_target_time:
            gemini_indices.append(i)
            next_target_time = ts + GEMINI_FRAME_INTERVAL

    gemini_indices = sorted(set(gemini_indices))
    print(f"  Sending {len(gemini_indices)} frames to Gemini (every {GEMINI_FRAME_INTERVAL}s + dense end)...")

    # Per-frame Gemini results
    gemini_votes = {
        "alertness": [],   # list of "HIGH"/"MODERATE"/"LOW"
        "distraction": [],  # list of "YES"/"NO"
        "writing": [],
        "nap": [],
        "mobile": [],
        "packing": [],
        "leaving": [],
        "horn": [],
        "brake_a9": [],
    }
    gemini_reasons = {k: [] for k in gemini_votes}
    gemini_evidence = {k: [] for k in gemini_votes}  # List of all YES indices

    result_text_parts = []
    gemini_results = {}

    batches = list(chunked(gemini_indices, GEMINI_BATCH_SIZE))
    worker_count = min(GEMINI_BATCH_WORKERS, len(batches))
    print(
        f"  Sending Gemini requests in batches of {GEMINI_BATCH_SIZE} "
        f"({len(batches)} API calls, {worker_count} parallel workers)..."
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_batch = {
            executor.submit(analyze_frame_batch_with_gemini, batch_indices, frame_analyses): (batch_num, batch_indices)
            for batch_num, batch_indices in enumerate(batches, start=1)
        }

        for future in as_completed(future_to_batch):
            batch_num, batch_indices = future_to_batch[future]
            try:
                batch_results = future.result()
            except Exception as e:
                first_ts = frame_analyses[batch_indices[0]]["timestamp"]
                last_ts = frame_analyses[batch_indices[-1]]["timestamp"]
                print(f"  Batch {batch_num} ({first_ts}s-{last_ts}s): Gemini error: {e}")
                result_text_parts.append(f"Batch {batch_num}: error - {e}")
                continue

            gemini_results.update({
                idx: {
                    "idx": idx,
                    "timestamp": frame_analyses[idx]["timestamp"],
                    "parsed": parsed,
                    "error": None,
                }
                for idx, parsed in batch_results.items()
            })

            missing = [idx for idx in batch_indices if idx not in batch_results]
            if missing:
                missing_text = ", ".join(str(idx) for idx in missing)
                print(f"  Batch {batch_num}: Gemini response missing frame(s): {missing_text}")
                result_text_parts.append(f"Batch {batch_num}: missing frame(s) {missing_text}")
            else:
                print(f"  Batch {batch_num}/{len(batches)} complete ({len(batch_indices)} frames).")

    for idx in gemini_indices:
        result = gemini_results.get(idx)
        if not result:
            continue

        timestamp = result["timestamp"]
        parsed = result["parsed"]
        status_line = f"  Frame {idx} ({timestamp}s): "
        flags = []

        for param_key in gemini_votes:
            if param_key in parsed:
                value = parsed[param_key]["value"]
                reason = parsed[param_key]["reason"]
                gemini_votes[param_key].append(value)
                gemini_reasons[param_key].append(reason)

                # Track evidence frames for flagged items
                if param_key == "alertness" and value in ["LOW", "MODERATE"]:
                    gemini_evidence[param_key].append(idx)
                    flags.append(f"alert={value}")
                elif param_key != "alertness" and value == "YES":
                    gemini_evidence[param_key].append(idx)
                    flags.append(param_key)

        flag_str = ", ".join(flags) if flags else "OK"
        print(f"{status_line}{flag_str}")

    # --- Aggregate Gemini votes into final parameters ---
    gemini_total = len(gemini_indices)
    print(f"\n  Aggregating {gemini_total} Gemini frame analyses...")

    # 1. ALERTNESS — use Gemini vote distribution
    if gemini_votes["alertness"]:
        low_count = sum(1 for v in gemini_votes["alertness"] if v == "LOW")
        mod_count = sum(1 for v in gemini_votes["alertness"] if v == "MODERATE")
        high_count = sum(1 for v in gemini_votes["alertness"] if v == "HIGH")
        
        if low_count > gemini_total * 0.2:
            gemini_alertness = "LOW"
        elif (low_count + mod_count) > gemini_total * 0.3:
            gemini_alertness = "MODERATE"
        else:
            gemini_alertness = "HIGH"
        
        parameters["alertness"]["status"] = gemini_alertness
        # Description is already set by local CV, we'll keep it or let Gemini overwrite it later
        if gemini_evidence["alertness"]:
            parameters["alertness"]["evidence_frame_idx"] = gemini_evidence["alertness"][0]

    # For binary parameters (YES/NO), use Gemini vote counts
    binary_params = {
        "distraction": {"label": "Crew Distraction", "threshold": 0.2},
        "writing": {"label": "Crew Doing Writing Work", "threshold": 0.01},
        "nap": {"label": "Crew Taking Nap / Micro-sleep", "threshold": 0.02},
        "mobile": {"label": "Crew Using Mobile", "threshold": 0.05},
        "packing": {"label": "Crew Packing Belongings", "threshold": 0.02},
        "leaving": {"label": "Crew Leaving Place", "threshold": 0.02},
        "horn": {"label": "Crew Operating Horn Lever/Switch", "threshold": 0.1},
        "brake_a9": {"label": "LP Applying A9/SA9 After Halt", "threshold": 0.1},
    }

    for param_key, config in binary_params.items():
        votes = gemini_votes[param_key]
        if not votes:
            continue

        # For horn and brake_a9: if custom YOLO model already detected them as OBSERVED,
        # keep the YOLO results — they are more reliable than Gemini for these specific classes.
        # Only let Gemini override if YOLO did NOT detect them or model wasn't available.
        if param_key in ("horn", "brake_a9"):
            current_status = parameters.get(param_key, {}).get("status", "")
            if current_status == "OBSERVED" and BRAKE_HORN_AVAILABLE:
                # Custom YOLO already detected this — append Gemini info but don't override
                yes_count = sum(1 for v in votes if v == "YES")
                yes_pct = yes_count / len(votes) if votes else 0
                # Custom YOLO already detected this
                if yes_count > 0:
                    pass # We do not alter description, keep local human-readable text
                continue

        # For packing and leaving: if local CV already flagged as OBSERVED (e.g. zero-person rule),
        # don't let Gemini override back to NOT_OBSERVED.
        if param_key in ("packing", "leaving"):
            current_status = parameters.get(param_key, {}).get("status", "")
            if current_status == "OBSERVED":
                yes_count = sum(1 for v in votes if v == "YES")
                yes_pct = yes_count / len(votes) if votes else 0
                # Already observed locally, do not append model internal details
                continue

        yes_count = sum(1 for v in votes if v == "YES")
        yes_pct = yes_count / len(votes) if votes else 0
        
        if param_key == "mobile":
            if yes_count >= 2:
                gemini_status = "OBSERVED"
                conf = "HIGH"
            else:
                gemini_status = "NOT_OBSERVED"
                conf = "MEDIUM" if yes_count == 1 else "HIGH"
        elif yes_pct >= config["threshold"]:
            gemini_status = "OBSERVED"
            if yes_count >= 3:
                conf = "HIGH"
            elif yes_count >= 2:
                conf = "MEDIUM"
            else:
                conf = "LOW"
        else:
            gemini_status = "NOT_OBSERVED"
            conf = "HIGH" if yes_count == 0 else "MEDIUM"

        # Collect Gemini reasons for YES frames
        yes_reasons = [r for v, r in zip(votes, gemini_reasons[param_key]) if v == "YES" and r]
        reason_str = " | ".join(yes_reasons[:3]) if yes_reasons else "No specific evidence."

        # For horn/brake_a9 when YOLO didn't detect: let Gemini set the status
        parameters[param_key]["status"] = gemini_status
        parameters[param_key]["confidence"] = conf
        
        if gemini_status == "OBSERVED":
            yes_indices = gemini_evidence[param_key]
            if param_key == "mobile" and yes_indices:
                # Find the frame where the phone is highest up among the Gemini YES frames
                best_idx = yes_indices[len(yes_indices)//2]
                min_y = float('inf')
                for idx in yes_indices:
                    for det in frame_analyses[idx].get("yolo_detections", []):
                        if det["class_name"] == "cell phone":
                            y1 = det["bbox"][1]
                            if y1 < min_y:
                                min_y = y1
                                best_idx = idx
                parameters[param_key]["evidence_frame_idx"] = best_idx
            elif yes_indices:
                parameters[param_key]["evidence_frame_idx"] = yes_indices[0]
        
        # Use a human-readable description instead of the raw analysis metrics
        if param_key == "alertness":
            if parameters[param_key]["status"] == "LOW":
                desc = "Low. Crew members exhibited signs of fatigue and lack of focus. Extended periods of drooping head or closed eyes observed."
            elif parameters[param_key]["status"] == "MODERATE":
                desc = "Moderate. Crew showed occasional signs of reduced alertness, such as brief drooping or yawning, but maintained general awareness."
            else:
                desc = "High. Crew members remained alert, maintaining upright posture and active engagement."
        elif param_key == "distraction":
            if parameters[param_key]["status"] in ["HIGH", "OBSERVED"]:
                desc = "High. A crew member was observed looking away from the track frequently, focusing on other directions for extended periods."
            else:
                desc = "Not Observed. Crew maintained forward focus without significant distraction."
        elif param_key == "writing":
            if parameters[param_key]["status"] == "OBSERVED":
                desc = "Observed. A crew member was seen actively focused downward on paperwork or a document while the locomotive was in motion."
            else:
                desc = "Not Observed. No crew member was seen doing writing work."
        elif param_key == "nap":
            if parameters[param_key]["status"] == "OBSERVED":
                desc = "Observed. A crew member appeared to be in a state of sleep, with prolonged periods of closed eyes."
            else:
                desc = "Not Observed. No significant evidence of napping or prolonged eye closure was seen."
        elif param_key == "mobile":
            if parameters[param_key]["status"] == "OBSERVED":
                desc = "Observed. A crew member was seen actively using or holding a mobile phone during the monitored period."
            else:
                desc = "Not Observed. Neither crew member was seen using a mobile phone during the recorded period."
        elif param_key == "packing":
            if parameters[param_key]["status"] == "OBSERVED":
                desc = "Observed. Activity indicating handling of personal belongings or packing bags was noted."
            else:
                desc = "Not Observed. No crew member was seen actively packing bags or gathering personal items."
        elif param_key == "leaving":
            if parameters[param_key]["status"] == "OBSERVED":
                desc = "Observed. A crew member vacated their position and left the designated seat area."
            else:
                desc = "Not Observed. Both crew members remained in their designated seats or expected operational areas."
        elif param_key == "horn":
            if parameters[param_key]["status"] == "OBSERVED":
                desc = "Observed. A crew member was seen operating the horn lever/switch."
            else:
                desc = "Not Observed. No horn lever activation was detected."
        elif param_key == "brake_a9":
            if parameters[param_key]["status"] == "OBSERVED":
                desc = "Observed. The Loco Pilot was seen applying the A9/SA9 brake handle."
            else:
                desc = "Not Observed. No A9/SA9 brake application was detected."
        else:
            desc = parameters[param_key]["description"]
            
        parameters[param_key]["description"] = desc

    # --- Save evidence images ---
    for key, param in parameters.items():
        idx = param.get("evidence_frame_idx")
        status = param.get("status", "")
        # Only save evidence if flagged
        if status in ["NOT_OBSERVED", "NONE"] or (key == "alertness" and status == "HIGH"):
            param["evidence_frame_idx"] = None
            continue

        if idx is not None and 0 <= idx < total:
            source_path = frame_analyses[idx].get("annotated_path", frame_analyses[idx]["frame_path"])
            param["image_url"] = save_detection_image(
                source_path, None, key.upper(), (0, 0, 255), key
            )

    # --- Regenerate summary and recommendations ---
    issues = []
    if parameters.get("alertness", {}).get("status") == "LOW":
        issues.append("low crew alertness")
    if parameters.get("nap", {}).get("status") == "OBSERVED":
        issues.append("napping/micro-sleep detected")
    if parameters.get("mobile", {}).get("status") == "OBSERVED":
        issues.append("mobile phone usage")
    if parameters.get("distraction", {}).get("status") == "OBSERVED":
        issues.append("crew distraction")
    if parameters.get("leaving", {}).get("status") == "OBSERVED":
        issues.append("crew leaving position")
    if parameters.get("writing", {}).get("status") == "OBSERVED":
        issues.append("writing work during duty")
    if parameters.get("packing", {}).get("status") == "OBSERVED":
        issues.append("packing belongings")
    if parameters.get("horn", {}).get("status") == "OBSERVED":
        issues.append("horn lever operation detected")
    if parameters.get("brake_a9", {}).get("status") == "OBSERVED":
        issues.append("A9/SA9 brake application detected")

    if not issues:
        overall_summary = "Analysis complete. Generally safe crew behavior. No critical violations."
    else:
        overall_summary = f"Analysis flagged {len(issues)} concern(s): {', '.join(issues)}."

    recs = []
    if parameters.get("alertness", {}).get("status") != "HIGH":
        recs.append("- Monitor crew alertness levels. Consider breaks or interventions.")
    if parameters.get("nap", {}).get("status") == "OBSERVED":
        recs.append("- CRITICAL: Napping detected. Immediate intervention required.")
    if parameters.get("mobile", {}).get("status") == "OBSERVED":
        recs.append("- Enforce no-phone policy during locomotive operation.")
    if parameters.get("distraction", {}).get("status") == "OBSERVED":
        recs.append("- Address crew distraction. Remind crew to focus on track and controls.")
    if parameters.get("leaving", {}).get("status") == "OBSERVED":
        recs.append("- Investigate crew member leaving position. Ensure minimum staffing.")
    if parameters.get("horn", {}).get("status") == "OBSERVED":
        recs.append("- Horn lever operation detected. Verify appropriate horn usage per operating rules.")
    if parameters.get("brake_a9", {}).get("status") == "OBSERVED":
        recs.append("- A9/SA9 brake application after halt detected. Confirm proper braking procedure followed.")
    if not recs:
        recs.append("- Continue monitoring. Crew behavior appears compliant.")
    recommendations = "\n".join(recs)

    result_text = f"Analysis complete. {overall_summary}"
    return parameters, overall_summary, recommendations, result_text


# =========================
# BOUNDING BOX DRAWING (for Gemini response parsing)
# =========================
def draw_bounding_box(image_path, bbox_coords, label, color):
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w, _ = img.shape
    x_min = int(bbox_coords.get('x_min', 0) * w / 1000)
    y_min = int(bbox_coords.get('y_min', 0) * h / 1000)
    x_max = int(bbox_coords.get('x_max', 1000) * w / 1000)
    y_max = int(bbox_coords.get('y_max', 1000) * h / 1000)
    x_min = max(0, min(x_min, w))
    y_min = max(0, min(y_min, h))
    x_max = max(0, min(x_max, w))
    y_max = max(0, min(y_max, h))
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x_min, y_min - label_size[1] - 10),
                  (x_min + label_size[0] + 10, y_min), color, -1)
    cv2.putText(img, label, (x_min + 5, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


# =========================
# PARSE HELPERS
# =========================
def parse_bbox(bbox_text):
    if not bbox_text or bbox_text.strip().upper() == 'NONE':
        return None
    try:
        numbers = re.findall(r'\d+', bbox_text)
        if len(numbers) >= 4:
            return {
                'x_min': int(numbers[0]),
                'y_min': int(numbers[1]),
                'x_max': int(numbers[2]),
                'y_max': int(numbers[3])
            }
    except Exception as e:
        print(f"Error parsing bbox: {e}")
    return None


def save_detection_image(source_path, bbox_coords, label, color, prefix):
    """Save a detection image with optional bounding box. Returns the web-accessible path."""
    detection_filename = f"{prefix}_{int(time.time())}.jpg"
    detection_dest = os.path.join(DETECTION_DIR, detection_filename)
    if bbox_coords:
        img_with_bbox = draw_bounding_box(source_path, bbox_coords, label, color)
        if isinstance(img_with_bbox, str):
            shutil.copy(source_path, detection_dest)
        else:
            cv2.imwrite(detection_dest, img_with_bbox)
    else:
        shutil.copy(source_path, detection_dest)
    return f"/detections/{detection_filename}"


# =========================
# BUILD ENHANCED GEMINI PROMPT (with CV evidence)
# =========================
def build_analysis_prompt(num_images, frame_analyses):
    """Build a detailed Gemini prompt that includes structured CV evidence from YOLO + Face analysis."""

    # --- Per-frame CV evidence ---
    cv_evidence = ""
    total_phone_frames = sum(1 for f in frame_analyses if f["phone_detected"])
    total_bag_frames = sum(1 for f in frame_analyses if f["bag_detected"])
    total_book_frames = sum(1 for f in frame_analyses if f["book_detected"])
    person_counts = [f["person_count"] for f in frame_analyses]

    eyes_closed_frames = 0
    head_drooping_frames = 0

    for i, fa in enumerate(frame_analyses):
        cv_evidence += f"\nFrame {i+1} (at {fa['timestamp']}s):\n"

        # YOLO detections
        if fa['yolo_detections']:
            for det in fa['yolo_detections']:
                bbox = det['bbox']
                cv_evidence += f"  YOLO: {det['class_name']} (conf={det['confidence']:.0%}, pixel_bbox=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}])\n"
        else:
            cv_evidence += "  YOLO: No relevant objects detected\n"

        cv_evidence += f"  Persons in frame: {fa['person_count']}\n"

        # Face analysis (advanced: EAR + blendshapes + solvePnP + MAR)
        if fa['face_analysis']['faces_detected'] > 0:
            for j, face in enumerate(fa['face_analysis']['face_analyses']):
                cv_evidence += f"  Face {j+1}: eyes={face['eye_status']}"
                if face['ear_avg'] is not None:
                    cv_evidence += f" (EAR={face['ear_avg']})"
                if face.get('blink_score') is not None:
                    cv_evidence += f" [blink_score={face['blink_score']}]"
                cv_evidence += f", gaze={face.get('gaze_direction', '?')}"
                if face.get('head_pitch_deg') is not None:
                    cv_evidence += f", head_pose(pitch={face['head_pitch_deg']}deg, yaw={face['head_yaw_deg']}deg, roll={face.get('head_roll_deg', 0)}deg)"
                else:
                    cv_evidence += f", head_pitch={face['head_pitch']}, head_yaw={face['head_yaw']}"
                if face.get('yawn_status') and face['yawn_status'] != 'NOT_YAWNING':
                    cv_evidence += f", YAWNING(MAR={face.get('mar')})"
                if face.get('jaw_open_score') and face['jaw_open_score'] > 0.3:
                    cv_evidence += f", jaw_open={face['jaw_open_score']}"
                cv_evidence += "\n"

                if face['eye_status'] in ['CLOSED', 'PARTIALLY_CLOSED']:
                    eyes_closed_frames += 1
                if face.get('head_pitch') == 'drooping' or (face.get('head_pitch_deg') is not None and face['head_pitch_deg'] < -15):
                    head_drooping_frames += 1
        else:
            cv_evidence += "  Face: No faces detected by CV\n"

    # --- Temporal summary ---
    person_count_timeline = ", ".join([f"{fa['timestamp']}s:{fa['person_count']}" for fa in frame_analyses])

    temporal_summary = f"""
================================================================================
TEMPORAL / CROSS-FRAME SUMMARY (from computer vision)
================================================================================
- Phone detected by YOLO in {total_phone_frames}/{num_images} frames
- Bag/backpack detected in {total_bag_frames}/{num_images} frames
- Book/writing material detected in {total_book_frames}/{num_images} frames
- Eyes closed/partially closed in {eyes_closed_frames} face detections across all frames
- Head drooping in {head_drooping_frames} face detections across all frames
- Person count range: {min(person_counts) if person_counts else 0} to {max(person_counts) if person_counts else 0}
- Person count timeline: [{person_count_timeline}]
"""

    return f"""
You are an expert railway safety monitoring system. You are analyzing {num_images} SEQUENTIAL images from a locomotive cab (train driver cabin) CCTV camera.

These images show loco pilots (LP = Loco Pilot, ALP = Assistant Loco Pilot) during their duty.
Images are numbered 1 to {num_images} in chronological order.

IMPORTANT: Each image has YOLO object-detection bounding boxes DRAWN on it:
  - GREEN boxes = detected persons
  - RED boxes = detected cell phones
  - MAGENTA boxes = detected bags/backpacks
  - ORANGE boxes = detected books/laptops
Use these visual annotations as evidence alongside your own analysis.

================================================================================
COMPUTER VISION PRE-ANALYSIS DATA (YOLO + Face Mesh Analysis)
================================================================================
{cv_evidence}
{temporal_summary}

================================================================================
ANALYSIS INSTRUCTIONS
================================================================================
1. CROSS-VALIDATE: Use BOTH the CV pre-analysis data above AND your own visual analysis
   of the images. If YOLO detects a phone but you don't see one, note the discrepancy.
   If you see something YOLO missed, still report it.

2. TEMPORAL PATTERNS matter:
   - Phone in 1 frame = possibly false positive. In 3+ frames = confirmed usage.
   - Person count dropping across several frames = someone left.
   - Eyes closed in multiple consecutive frames = napping. Single frame = blinking.

3. Be STRICT: Only report OBSERVED if you have strong evidence.

4. For bounding boxes in your response, use 0-1000 normalized coordinates.

================================================================================
ANALYZE THESE 9 SAFETY PARAMETERS
================================================================================

1. CREW ALERTNESS
   - Use EAR data: EAR < 0.20 = eyes closed, 0.20-0.25 = drowsy, > 0.25 = alert
   - Use head_pitch: "drooping" = low alertness, "upright" = alert
   - Also visually check: posture, engagement with controls, active track scanning

2. CREW DISTRACTION
   - Use head_yaw: "turned_left/right" = not looking at track
   - Visually check: looking at non-operational items, turned away, animated gesturing

3. CREW DOING WRITING WORK
   - Check YOLO "book" detections
   - Visually check: pen in hand, paper/logbook on desk, writing posture, looking down at document

4. CREW TAKING NAP / MICRO-SLEEP
   - Use EAR data: consistent EAR < 0.20 across multiple frames = napping
   - Use head_pitch: "drooping" in multiple frames
   - Visually check: eyes fully closed, slumped, head resting on surface
   - NEED multi-frame evidence, not single eye blink

5. CREW USING MOBILE
   - Check YOLO "cell_phone" detections with confidence scores
   - Visually check: rectangular device in hand, screen glow, phone posture
   - CROSS-VALIDATE: YOLO + visual confirmation = high confidence

6. CREW PACKING BELONGINGS
   - Check YOLO "backpack"/"handbag"/"suitcase" detections
   - Visually check: actively handling bags, organizing items
   - Bags visible but stationary = normal. Actively handled = packing.

7. CREW LEAVING PLACE
   - Check person count changes across frames (timeline provided above)
   - Visually check: empty seats, person standing/moving toward exit

8. CREW OPERATING HORN LEVER/SWITCH
   - Visually check: Is any crew member reaching for, pressing, or operating the horn lever/switch?
   - The horn lever is typically located on the upper portion of the control panel/console
   - Look for hand on or near the horn lever (pull-down or push-type switch at top of cab console)
   - Note: YOLO cannot detect horn levers, rely entirely on visual observation

9. LP APPLYING A9/SA9 BRAKE AFTER HALT
   - Visually check: Is the Loco Pilot (LP) applying the A9 or SA9 brake handle?
   - Look for LP's hand on the brake handle, brake handle in applied/pulled position
   - The A9/SA9 brake valve is typically a large handle on the left side of the LP's control stand
   - Check if train appears halted (stationary) and brake is being applied
   - Note: YOLO cannot detect brake handles, rely entirely on visual observation

================================================================================
REQUIRED OUTPUT FORMAT — FOLLOW EXACTLY
================================================================================

---ANALYSIS START---

PARAMETER_1_ALERTNESS:
Description: [Detailed observations referencing specific frames, EAR values, head pose data]
Status: [HIGH / MODERATE / LOW]
Detected_Image: [Image number (1-{num_images}) showing most relevant evidence, or NONE]
BBox: [x_min, y_min, x_max, y_max] or NONE
Confidence: [HIGH / MEDIUM / LOW]
Timestamp_Approx: [N/A]

---

PARAMETER_2_DISTRACTION:
Description: [Reference head_yaw data and visual observations]
Status: [NONE / LOW / MODERATE / HIGH]
Detected_Image: [Image number or NONE]
BBox: [x_min, y_min, x_max, y_max] or NONE
Confidence: [HIGH / MEDIUM / LOW]
Timestamp_Approx: [N/A]

---

PARAMETER_3_WRITING:
Description: [Reference YOLO book detections and visual writing observations]
Status: [OBSERVED / NOT_OBSERVED]
Detected_Image: [Image number or NONE]
BBox: [x_min, y_min, x_max, y_max] or NONE
Confidence: [HIGH / MEDIUM / LOW]
Timestamp_Approx: [N/A]

---

PARAMETER_4_NAP_MICROSLEEP:
Description: [Reference EAR data across frames, head pose, multi-frame patterns]
Status: [OBSERVED / NOT_OBSERVED]
Detected_Image: [Image number or NONE]
BBox: [x_min, y_min, x_max, y_max] or NONE
Confidence: [HIGH / MEDIUM / LOW]
Timestamp_Approx: [N/A]

---

PARAMETER_5_MOBILE:
Description: [Reference YOLO cell_phone detections with confidence, cross-validate visually]
Status: [OBSERVED / NOT_OBSERVED]
Detected_Image: [Image number or NONE]
BBox: [x_min, y_min, x_max, y_max] or NONE
Confidence: [HIGH / MEDIUM / LOW]
Timestamp_Approx: [N/A]

---

PARAMETER_6_PACKING:
Description: [Reference YOLO bag detections and visual observations]
Status: [OBSERVED / NOT_OBSERVED]
Detected_Image: [Image number or NONE]
BBox: [x_min, y_min, x_max, y_max] or NONE
Confidence: [HIGH / MEDIUM / LOW]
Timestamp_Approx: [N/A]

---

PARAMETER_7_LEAVING:
Description: [Reference person count changes across frames]
Status: [OBSERVED / NOT_OBSERVED]
Detected_Image: [Image number or NONE]
BBox: [x_min, y_min, x_max, y_max] or NONE
Confidence: [HIGH / MEDIUM / LOW]
Timestamp_Approx: [N/A]

---

PARAMETER_8_HORN:
Description: [Visual observations of horn lever/switch operation]
Status: [OBSERVED / NOT_OBSERVED]
Detected_Image: [Image number or NONE]
BBox: [x_min, y_min, x_max, y_max] or NONE
Confidence: [HIGH / MEDIUM / LOW]
Timestamp_Approx: [N/A]

---

PARAMETER_9_BRAKE_A9:
Description: [Visual observations of A9/SA9 brake handle application after halt]
Status: [OBSERVED / NOT_OBSERVED]
Detected_Image: [Image number or NONE]
BBox: [x_min, y_min, x_max, y_max] or NONE
Confidence: [HIGH / MEDIUM / LOW]
Timestamp_Approx: [N/A]

---

OVERALL_SAFETY_SUMMARY:
[2-3 sentence overall safety assessment combining all 9 parameters]

RECOMMENDATIONS:
[Bullet points of safety recommendations based on findings]

---ANALYSIS END---
"""


# =========================
# PARSE GEMINI RESPONSE
# =========================
PARAMETER_CONFIG = {
    "1_ALERTNESS": {
        "key": "alertness",
        "label": "Crew Alertness",
        "icon": "🧠",
        "bbox_label": "ALERTNESS",
        "bbox_color": (0, 165, 255),  # Orange
        "prefix": "alertness",
    },
    "2_DISTRACTION": {
        "key": "distraction",
        "label": "Crew Distraction",
        "icon": "👀",
        "bbox_label": "DISTRACTION",
        "bbox_color": (0, 140, 255),  # Dark Orange
        "prefix": "distraction",
    },
    "3_WRITING": {
        "key": "writing",
        "label": "Crew Doing Writing Work",
        "icon": "✍️",
        "bbox_label": "WRITING",
        "bbox_color": (255, 0, 0),  # Blue
        "prefix": "writing",
    },
    "4_NAP_MICROSLEEP": {
        "key": "nap",
        "label": "Crew Taking Nap / Micro-sleep",
        "icon": "😴",
        "bbox_label": "NAP/MICROSLEEP",
        "bbox_color": (0, 0, 255),  # Red
        "prefix": "nap",
    },
    "5_MOBILE": {
        "key": "mobile",
        "label": "Crew Using Mobile",
        "icon": "📱",
        "bbox_label": "MOBILE PHONE",
        "bbox_color": (0, 0, 255),  # Red
        "prefix": "mobile",
    },
    "6_PACKING": {
        "key": "packing",
        "label": "Crew Packing Belongings",
        "icon": "🎒",
        "bbox_label": "PACKING",
        "bbox_color": (128, 0, 128),  # Purple
        "prefix": "packing",
    },
    "7_LEAVING": {
        "key": "leaving",
        "label": "Crew Leaving Place",
        "icon": "🚪",
        "bbox_label": "LEAVING",
        "bbox_color": (128, 0, 128),  # Purple
        "prefix": "leaving",
    },
    "8_HORN": {
        "key": "horn",
        "label": "Crew Operating Horn Lever/Switch",
        "icon": "📯",
        "bbox_label": "HORN OPERATION",
        "bbox_color": (0, 200, 200),  # Yellow-ish (Cyan in BGR)
        "prefix": "horn",
    },
    "9_BRAKE_A9": {
        "key": "brake_a9",
        "label": "LP Applying A9/SA9 After Halt",
        "icon": "🛑",
        "bbox_label": "BRAKE A9/SA9",
        "bbox_color": (0, 0, 180),  # Dark Red (BGR)
        "prefix": "brake_a9",
    },
}


def parse_gemini_response(result_text, person_images):
    """Parse the structured Gemini response into a dict of parameter results."""
    parameters = {}

    for param_id, config in PARAMETER_CONFIG.items():
        param_data = {
            "key": config["key"],
            "label": config["label"],
            "icon": config["icon"],
            "description": "",
            "status": "N/A",
            "confidence": "N/A",
            "timestamp": "N/A",
            "image_url": None,
        }

        # Find the block for this parameter
        block_pattern = rf'PARAMETER_{param_id}:\s*(.*?)(?=---|$)'
        block_match = re.search(block_pattern, result_text, re.DOTALL | re.IGNORECASE)

        if block_match:
            block_text = block_match.group(1)

            # Parse Description
            desc_match = re.search(r'Description:\s*(.+?)(?=\nStatus:)', block_text, re.DOTALL | re.IGNORECASE)
            if desc_match:
                param_data["description"] = desc_match.group(1).strip()

            # Parse Status
            status_match = re.search(r'Status:\s*(.+)', block_text, re.IGNORECASE)
            if status_match:
                param_data["status"] = status_match.group(1).strip()

            # Parse Confidence
            conf_match = re.search(r'Confidence:\s*(.+)', block_text, re.IGNORECASE)
            if conf_match:
                param_data["confidence"] = conf_match.group(1).strip()

            # Parse Timestamp
            ts_match = re.search(r'Timestamp_Approx:\s*(.+)', block_text, re.IGNORECASE)
            if ts_match:
                param_data["timestamp"] = ts_match.group(1).strip()

            # Parse Detected_Image and BBox → save detection image
            img_match = re.search(r'Detected_Image:\s*(\d+)', block_text, re.IGNORECASE)
            bbox_match = re.search(r'BBox:\s*(\[[\d,\s]+\]|NONE)', block_text, re.IGNORECASE)

            if img_match:
                detected_idx = int(img_match.group(1)) - 1
                if 0 <= detected_idx < len(person_images):
                    _, source_path = person_images[detected_idx]
                    bbox_coords = None
                    if bbox_match:
                        bbox_coords = parse_bbox(bbox_match.group(1))
                    param_data["image_url"] = save_detection_image(
                        source_path, bbox_coords,
                        config["bbox_label"], config["bbox_color"], config["prefix"]
                    )

        parameters[config["key"]] = param_data

    # Parse overall summary
    summary_match = re.search(r'OVERALL_SAFETY_SUMMARY:\s*(.+?)(?=RECOMMENDATIONS:|---ANALYSIS END---|$)',
                              result_text, re.DOTALL | re.IGNORECASE)
    overall_summary = summary_match.group(1).strip() if summary_match else ""

    rec_match = re.search(r'RECOMMENDATIONS:\s*(.+?)(?=---ANALYSIS END---|$)',
                          result_text, re.DOTALL | re.IGNORECASE)
    recommendations = rec_match.group(1).strip() if rec_match else ""

    return parameters, overall_summary, recommendations


# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_video(request: Request, video: UploadFile = File(...)):
    video_path = f"temp_{video.filename}"
    with open(video_path, "wb") as f:
        f.write(await video.read())

    result_text = ""
    parameters = {}
    overall_summary = ""
    recommendations = ""
    video_info = {
        "filename": video.filename,
        "duration": 0,
        "fps": 0,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frames_analyzed": 0,
    }

    try:
        # 1. Extract frames (every 2s, up to 200 for local analysis)
        frames, fps, duration = extract_frames(video_path)
        video_info["fps"] = fps
        video_info["duration"] = duration
        video_info["frames_analyzed"] = len(frames)
        print(f"\nExtracted {len(frames)} frames from {duration}s video at {fps} FPS.")

        if not frames:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "result": "No frames could be extracted from the video.",
                "video_info": video_info,
            })

        # 2. Run YOLO + MediaPipe on EVERY frame (local, fast)
        print("=" * 60)
        print("TIER 1: Running YOLO + MediaPipe Face Analysis on all frames...")
        print("=" * 60)
        frame_analyses, annotated_frames = analyze_all_frames(frames)
        print(f"\nCompleted local CV analysis of {len(frame_analyses)} frames.")

        # Print detection summary
        phone_frames = sum(1 for f in frame_analyses if f["phone_detected"])
        bag_frames = sum(1 for f in frame_analyses if f["bag_detected"])
        book_frames = sum(1 for f in frame_analyses if f["book_detected"])
        faces_detected = sum(f["face_analysis"]["faces_detected"] for f in frame_analyses)
        print(f"  Phone: {phone_frames} frames | Bag: {bag_frames} frames | Book: {book_frames} frames")
        print(f"  Total face detections: {faces_detected} across {len(frame_analyses)} frames")

        # 3. Compute all 7 parameters LOCALLY using rule-based engine
        print("\n" + "=" * 60)
        print("TIER 2: Computing parameters from local CV data (rule engine)...")
        print("=" * 60)
        parameters, overall_summary, recommendations = compute_parameters_locally(frame_analyses)

        for key, param in parameters.items():
            status_icon = "🔴" if param["status"] in ["LOW", "OBSERVED"] else "🟢" if param["status"] in ["HIGH", "NOT_OBSERVED", "NONE"] else "🟡"
            print(f"  {status_icon} {param['label']}: {param['status']} (conf={param['confidence']})")

        # 4. Dense Gemini analysis (every 7 seconds for high accuracy)
        print("\n" + "=" * 60)
        print("TIER 3: Dense Gemini analysis (one frame every 7 seconds)...")
        print("=" * 60)
        try:
            parameters, gemini_summary, gemini_recs, gemini_detail = analyze_with_gemini_dense(parameters, frame_analyses)
            overall_summary = gemini_summary
            recommendations = gemini_recs
            result_text = f"Analysis complete. {gemini_detail}"
            print("Gemini dense analysis complete.")
        except Exception as e:
            print(f"Gemini analysis skipped: {e}")
            result_text = "Local analysis complete. Gemini analysis was not available."

        # Final status summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS:")
        print("=" * 60)
        for key, param in parameters.items():
            print(f"  {param['label']}: {param['status']} ({param['confidence']})")
        print(f"\n{overall_summary}")

    except Exception as e:
        result_text = f"Error: {str(e)}"
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

    # Store results for PDF download
    last_report["parameters"] = parameters
    last_report["overall_summary"] = overall_summary
    last_report["recommendations"] = recommendations
    last_report["video_info"] = video_info
    last_report["result_text"] = result_text

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result_text,
        "parameters": parameters,
        "overall_summary": overall_summary,
        "recommendations": recommendations,
        "video_info": video_info,
    })


# =========================
# PDF REPORT GENERATION
# =========================
def strip_emoji(text):
    """Remove emoji and other non-latin-1 characters for PDF compatibility."""
    if not text:
        return text
    return text.encode('latin-1', 'ignore').decode('latin-1')


class CrewReportPDF(FPDF):
    """Custom PDF class for the crew monitoring report."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 64, 175)
        self.cell(0, 10, "Crew Behavior Monitoring Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        vi = last_report.get("video_info", {})
        self.cell(0, 5, f"Generated: {vi.get('analysis_time', 'N/A')}   |   File: {vi.get('filename', 'N/A')}   |   Duration: {vi.get('duration', 0)}s   |   Frames Analyzed: {vi.get('frames_analyzed', 0)}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Powered by Gemini AI + YOLO v8 + Face Analysis   |   Page {self.page_no()}/{{nb}}", align="C")


def generate_pdf_report():
    """Generate a PDF report from the last analysis results."""
    pdf = CrewReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    parameters = last_report.get("parameters", {})
    overall_summary = last_report.get("overall_summary", "")
    recommendations = last_report.get("recommendations", "")

    # Status color map
    def get_status_colors(key, status):
        s = status.upper()
        if s in ["LOW", "DROWSY", "DROWSY_DETECTED"] or ("NAP" in key.upper() and s == "OBSERVED") or ("MOBILE" in key.upper() and s == "OBSERVED"):
            return (220, 38, 38)  # Red
        elif s in ["NOT_OBSERVED", "NONE"] or ("ALERTNESS" in key.upper() and s == "HIGH"):
            return (34, 197, 94)  # Green
        elif s == "OBSERVED":
            return (59, 130, 246)  # Blue
        else:
            return (245, 158, 11)  # Orange/warning

    # Render each parameter
    param_num = 0
    for key, param in parameters.items():
        param_num += 1

        # Check if we need a new page (leave room for content + possible image)
        if pdf.get_y() > 220:
            pdf.add_page()

        # Parameter header bar
        pdf.set_fill_color(24, 30, 50)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, strip_emoji(f"  #{param_num}  {param.get('icon', '')}  {param.get('label', key)}"), fill=True, new_x="LMARGIN", new_y="NEXT")

        # Status badge
        status = param.get("status", "N/A")
        r, g, b = get_status_colors(key, status)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(r, g, b)
        pdf.cell(20, 7, "  Status: ", new_x="END")
        pdf.cell(0, 7, status, new_x="LMARGIN", new_y="NEXT")

        # Description
        desc = param.get("description", "")
        if desc:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5, strip_emoji(desc), new_x="LMARGIN", new_y="NEXT")

        # Confidence & Timestamp
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(130, 130, 130)
        meta_line = f"Confidence: {param.get('confidence', 'N/A')}"
        ts = param.get("timestamp", "N/A")
        if ts and ts != "N/A":
            meta_line += f"   |   Timestamp: {ts}"
        pdf.cell(0, 5, meta_line, new_x="LMARGIN", new_y="NEXT")

        # Evidence image (if available)
        image_url = param.get("image_url", None)
        if image_url:
            # Convert web path to filesystem path
            img_fs_path = image_url.lstrip("/")  # e.g. "detections/alertness_123.jpg"
            if os.path.exists(img_fs_path):
                try:
                    if pdf.get_y() > 210:
                        pdf.add_page()
                    pdf.image(img_fs_path, x=15, w=80, h=55)
                    pdf.ln(2)
                except Exception as e:
                    print(f"Error embedding image {img_fs_path}: {e}")

        # Spacing between parameters
        pdf.ln(4)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    # Overall Safety Summary
    if overall_summary:
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "  Overall Safety Assessment", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.ln(2)
        pdf.multi_cell(0, 5, strip_emoji(overall_summary), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # Recommendations
    if recommendations:
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "  Recommendations", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.ln(2)
        pdf.multi_cell(0, 5, strip_emoji(recommendations), new_x="LMARGIN", new_y="NEXT")

    # Save and return path
    pdf_path = os.path.join(DETECTION_DIR, "crew_monitoring_report.pdf")
    pdf.output(pdf_path)
    return pdf_path


@app.get("/download-report")
async def download_report():
    """Generate and download the PDF report for the last analysis."""
    if not last_report.get("parameters"):
        return HTMLResponse("<h3>No analysis available. Please analyze a video first.</h3>", status_code=400)
    pdf_path = generate_pdf_report()
    vi = last_report.get("video_info", {})
    filename = f"crew_report_{vi.get('filename', 'video').rsplit('.', 1)[0]}.pdf"
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
