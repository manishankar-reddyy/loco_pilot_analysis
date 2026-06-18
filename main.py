from __future__ import annotations

import os
import sys
import json
import math
import time
import shutil
import traceback
import urllib.request
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

AUTO_INSTALL_DEPS = os.getenv("AUTO_INSTALL_DEPS", "0").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_VLM = os.getenv("ENABLE_VLM", "1").strip().lower() in {"1", "true", "yes", "on"}

def _ensure_package(module_name: str, package_spec: str) -> None:
    try:
        __import__(module_name)
    except Exception:
        if not AUTO_INSTALL_DEPS:
            raise RuntimeError(
                f"Missing dependency '{module_name}'. Install '{package_spec}' or set AUTO_INSTALL_DEPS=1."
            )
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])

for _module, _package in [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("fpdf", "fpdf2==2.8.1"),
    ("ultralytics", "ultralytics==8.2.92"),
    ("mediapipe", "mediapipe==0.10.18"),
    ("fastapi", "fastapi"),
    ("jinja2", "jinja2"),
    ("uvicorn", "uvicorn[standard]"),
    ("multipart", "python-multipart"),
]:
    _ensure_package(_module, _package)

if ENABLE_VLM:
    for _module, _package in [
        ("torch", "torch"),
        ("transformers", "transformers>=4.52.0"),
        ("accelerate", "accelerate>=0.34.0"),
        ("bitsandbytes", "bitsandbytes>=0.44.0"),
        ("qwen_vl_utils", "qwen-vl-utils>=0.0.8"),
    ]:
        _ensure_package(_module, _package)

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

FACE_LANDMARKER_PATH = str(BASE_DIR / "face_landmarker.task")
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
BRAKE_HORN_MODEL_PATH = str(BASE_DIR / "brake_horn_detection.pt") if (BASE_DIR / "brake_horn_detection.pt").exists() else None

if not os.path.exists(FACE_LANDMARKER_PATH):
    print("Downloading face_landmarker.task (~3.7 MB)...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
    print(f"face_landmarker.task ready ({os.path.getsize(FACE_LANDMARKER_PATH)/1e6:.1f} MB)")

# CELL 5: Configuration + profiling utilities
import os, json, time, math, statistics, traceback
from dataclasses import dataclass, asdict, field
from collections import defaultdict

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
ENABLE_VLM         = ENABLE_VLM
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

class PipelineProfiler:
    def __init__(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.events = []
        self.stage_stack = []

    def cuda_sync(self):
        try:
            if 'torch' in globals() and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def start(self, name, sync_cuda=False):
        if sync_cuda:
            self.cuda_sync()
        self.stage_stack.append((name, time.time(), sync_cuda))

    def stop(self, extra=None):
        name, t0, sync_cuda = self.stage_stack.pop()
        if sync_cuda:
            self.cuda_sync()
        dt = time.time() - t0
        self.timings[name] += dt
        self.counts[name] += 1
        item = {'stage': name, 'seconds': round(dt, 4)}
        if extra:
            item.update(extra)
        self.events.append(item)
        return dt

    def add(self, name, seconds, extra=None):
        self.timings[name] += float(seconds)
        self.counts[name] += 1
        item = {'stage': name, 'seconds': round(float(seconds), 4)}
        if extra:
            item.update(extra)
        self.events.append(item)

    def snapshot_gpu(self):
        try:
            if 'torch' in globals() and torch.cuda.is_available():
                return {
                    'gpu_name': torch.cuda.get_device_name(0),
                    'vram_allocated_gb': round(torch.cuda.memory_allocated()/1e9, 3),
                    'vram_reserved_gb': round(torch.cuda.memory_reserved()/1e9, 3),
                    'vram_total_gb': round(torch.cuda.get_device_properties(0).total_memory/1e9, 3),
                }
        except Exception:
            pass
        return {}

    def report(self):
        rows = []
        for k, v in sorted(self.timings.items(), key=lambda kv: -kv[1]):
            rows.append({'stage': k, 'seconds': round(v, 3), 'calls': self.counts[k]})
        return {'timings': rows, 'events': self.events, 'gpu': self.snapshot_gpu()}

PROFILER = PipelineProfiler()

@dataclass
class EventCandidate:
    behavior: str
    start_ts: int
    end_ts: int
    frame_indices: list
    track_ids: list = field(default_factory=list)
    local_score: float = 0.0
    vlm_score: float = 0.0
    final_score: float = 0.0
    decision: str = 'UNRESOLVED'
    evidence_frame: int = None
    reason: str = ''
    needs_vlm: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        data = asdict(self)
        meta = dict(data.get('metadata') or {})
        if 'dense_frames' in meta:
            meta['dense_frames'] = [f.get('timestamp') for f in meta['dense_frames']]
        if 'evidence_frame_item' in meta:
            item = meta['evidence_frame_item']
            meta['evidence_frame_item'] = {
                'timestamp': item.get('timestamp'),
                'is_dense_event_frame': bool(item.get('is_dense_event_frame')),
            }
        data['metadata'] = meta
        return data

print('Configuration ready.')
print(f'  VLM        : {VLM_MODEL_ID}')
print(f'  YOLO batch : {YOLO_BATCH_SIZE}')
print(f'  MP threads : {MEDIAPIPE_THREADS}')
print(f'  Max VLM events: {VLM_MAX_EVENTS}')

# Local lazy Qwen loader. The Colab notebook loaded Qwen immediately; local UI loads it only during analysis.
vlm_model = None
vlm_processor = None
process_vision_info = None

def load_vlm_model():
    global vlm_model, vlm_processor, process_vision_info
    if not ENABLE_VLM:
        return None, None, None
    if vlm_model is not None and vlm_processor is not None:
        return vlm_model, vlm_processor, process_vision_info
    if not torch.cuda.is_available():
        raise RuntimeError("ENABLE_VLM=1 but CUDA is not available. Set ENABLE_VLM=0 or install CUDA PyTorch.")
    from transformers import AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info as _process_vision_info
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _VLMClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as _VLMClass

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f'Loading {VLM_MODEL_ID} (4-bit NF4)...')
    print('  First run will download ~15 GB. Subsequent runs use cache.\n')
    vlm_model = _VLMClass.from_pretrained(
        VLM_MODEL_ID,
        quantization_config=bnb,
        device_map='auto',
        torch_dtype=torch.float16,
    )
    vlm_model.eval()
    vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    process_vision_info = _process_vision_info
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VLM loaded. VRAM used: {used:.1f} GB / {total:.1f} GB')
    return vlm_model, vlm_processor, process_vision_info

# ── CELL 6: YOLO + MediaPipe initialisation ───────────────────────────────────
import cv2, numpy as np, math, threading
from ultralytics import YOLO
import mediapipe as mp

# YOLO models
print('Loading YOLO models...')
yolo_model = YOLO('yolov8s.pt')
print('  yolov8s.pt loaded')

brake_horn_model   = None
BRAKE_HORN_AVAILABLE = False
if BRAKE_HORN_MODEL_PATH and os.path.exists(BRAKE_HORN_MODEL_PATH):
    brake_horn_model   = YOLO(BRAKE_HORN_MODEL_PATH)
    BRAKE_HORN_AVAILABLE = True
    print('  brake_horn_detection.pt loaded')
else:
    print('  WARNING: brake_horn_detection.pt not found — horn/brake detection unavailable')

# MediaPipe FaceLandmarker (thread-local — one instance per worker thread)
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

def _make_landmarker_options():
    return FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=4,
        min_face_detection_confidence=0.25,
        min_face_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )

_thread_local = threading.local()

def get_thread_landmarker():
    """Return a thread-local FaceLandmarker (safe for concurrent use)."""
    if not hasattr(_thread_local, 'lm'):
        _thread_local.lm = FaceLandmarker.create_from_options(_make_landmarker_options())
    return _thread_local.lm

# Warm up main-thread instance
_main_lm = FaceLandmarker.create_from_options(_make_landmarker_options())
print(f'  MediaPipe FaceLandmarker ready ({MEDIAPIPE_THREADS} worker threads)')

# Load VLM at startup (matches notebook Cell 7 — eager load)
if ENABLE_VLM:
    load_vlm_model()

# CELL 8: Vision helper functions (EAR, MAR, head pose, face analysis)
from PIL import Image

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
SOLVEPNP_IDX  = [1, 152, 263, 33, 61, 291]
BLENDSHAPE_KEYS = [
    'eyeBlinkLeft','eyeBlinkRight','jawOpen',
    'eyeLookDownLeft','eyeLookDownRight','eyeLookUpLeft','eyeLookUpRight',
]

MODEL_3D = np.array([
    (0.,0.,0.), (0.,-330.,-65.), (-225.,170.,-135.),
    (225.,170.,-135.), (-150.,-150.,-125.), (150.,-150.,-125.),
], dtype=np.float64)

def compute_ear(lm, idx):
    try:
        p = [(lm[i].x, lm[i].y) for i in idx]
        h = math.dist(p[0], p[3])
        return round((math.dist(p[1],p[5]) + math.dist(p[2],p[4])) / (2*h), 3) if h>1e-6 else 0.3
    except Exception:
        return None

def compute_mar(lm):
    try:
        h = math.dist((lm[61].x,lm[61].y), (lm[291].x,lm[291].y))
        if h < 1e-6: return 0.0
        v1 = math.dist((lm[13].x,lm[13].y),  (lm[14].x,lm[14].y))
        v2 = math.dist((lm[81].x,lm[81].y),  (lm[178].x,lm[178].y))
        v3 = math.dist((lm[311].x,lm[311].y),(lm[402].x,lm[402].y))
        return round((v1+v2+v3)/(3*h), 3)
    except Exception:
        return None

def compute_head_pose(lm, w, h):
    try:
        pts2d = np.array([(lm[i].x*w, lm[i].y*h) for i in SOLVEPNP_IDX], dtype=np.float64)
        cam   = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts2d, cam, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return {'pitch':0.,'yaw':0.,'pitch_label':'unknown','yaw_label':'unknown'}
        angles,*_ = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])
        pitch, yaw = round(angles[0],1), round(angles[1],1)
        pl = 'drooping' if pitch<-15 else 'slight_down' if pitch<-8 else 'upright'
        yl = ('turned_right' if yaw>0 else 'turned_left') if abs(yaw)>25 else 'forward'
        return {'pitch':pitch,'yaw':yaw,'pitch_label':pl,'yaw_label':yl}
    except Exception:
        return {'pitch':0.,'yaw':0.,'pitch_label':'unknown','yaw_label':'unknown'}

def get_blendshapes(bs_list):
    return {b.category_name: round(b.score,3) for b in bs_list
            if b.category_name in BLENDSHAPE_KEYS} if bs_list else {}

def _run_landmarker_on_array(bgr_img, landmarker):
    result = {'faces_detected': 0, 'face_analyses': []}
    if bgr_img is None or bgr_img.size == 0:
        return result
    h, w = bgr_img.shape[:2]
    rgb  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    try:
        det = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception:
        return result
    if not det.face_landmarks:
        return result
    result['faces_detected'] = len(det.face_landmarks)
    for fi, face_lm in enumerate(det.face_landmarks):
        el = compute_ear(face_lm, LEFT_EYE_IDX)
        er = compute_ear(face_lm, RIGHT_EYE_IDX)
        ear = round((el+er)/2, 3) if el and er else None
        eye = 'CLOSED' if ear and ear<0.18 else 'PARTIALLY_CLOSED' if ear and ear<0.23 else 'OPEN'
        mar = compute_mar(face_lm)
        yawn= 'YAWNING' if mar and mar>0.6 else 'MOUTH_OPEN' if mar and mar>0.4 else 'NOT_YAWNING'
        pose= compute_head_pose(face_lm, w, h)
        bs  = get_blendshapes(det.face_blendshapes[fi] if det.face_blendshapes and fi<len(det.face_blendshapes) else [])
        avg_blink = (bs.get('eyeBlinkLeft',0)+bs.get('eyeBlinkRight',0))/2
        if avg_blink>0.6:  eye='CLOSED'
        elif avg_blink>0.4 and eye=='OPEN': eye='PARTIALLY_CLOSED'
        ld = (bs.get('eyeLookDownLeft',0)+bs.get('eyeLookDownRight',0))/2
        lu = (bs.get('eyeLookUpLeft',0)+bs.get('eyeLookUpRight',0))/2
        gaze = 'down' if ld>0.4 else 'up' if lu>0.4 else 'forward'
        result['face_analyses'].append({
            'ear_avg':ear, 'eye_status':eye, 'mar':mar, 'yawn_status':yawn,
            'head_pitch_deg':pose['pitch'], 'head_yaw_deg':pose['yaw'],
            'head_pitch':pose['pitch_label'], 'head_yaw':pose['yaw_label'],
            'gaze_direction':gaze, 'blink_score':round(avg_blink,3), 'blendshapes':bs,
            'face_quality': 'valid' if ear is not None else 'weak',
        })
    return result

def _read_frame(frame_item):
    if isinstance(frame_item, dict):
        if frame_item.get('image') is not None:
            return frame_item['image']
        if frame_item.get('frame_path'):
            return cv2.imread(frame_item['frame_path'])
    if isinstance(frame_item, str):
        return cv2.imread(frame_item)
    return None

def _run_landmarker_on_img(frame_item, landmarker):
    return _run_landmarker_on_array(_read_frame(frame_item), landmarker)

def analyze_face_on_crops(frame_item, yolo_dets, timestamp, landmarker):
    merged = {'faces_detected':0, 'face_analyses':[]}
    img = _read_frame(frame_item)
    if img is None:
        return merged
    H, W = img.shape[:2]
    persons = [d for d in yolo_dets if d['class_name']=='person' and d['confidence']>MEDIAPIPE_MIN_PERSON_CONF]
    if not persons:
        return _run_landmarker_on_array(img, landmarker)
    for pi, p in enumerate(persons):
        x1,y1,x2,y2 = p['bbox']
        hy2 = y1 + int((y2-y1)*0.75)
        px, py = int((x2-x1)*0.25), int((hy2-y1)*0.25)
        cx1,cy1,cx2,cy2 = max(0,x1-px),max(0,y1-py),min(W,x2+px),min(H,hy2+py)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size==0:
            continue
        ch, cw = crop.shape[:2]
        if cw<FACE_ROI_MIN_SIZE or ch<FACE_ROI_MIN_SIZE:
            scale = max(FACE_ROI_MIN_SIZE/cw, FACE_ROI_MIN_SIZE/ch)
            crop  = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        fd = _run_landmarker_on_array(crop, landmarker)
        if fd['faces_detected']>0:
            merged['faces_detected'] += fd['faces_detected']
            for face in fd['face_analyses']:
                face['person_bbox'] = p['bbox']
                face['person_confidence'] = p['confidence']
                merged['face_analyses'].append(face)
    if merged['faces_detected']==0:
        return _run_landmarker_on_array(img, landmarker)
    return merged

def smooth_face_metrics(frame_analyses, window=3):
    vals = {'ear_avg': [], 'head_pitch_deg': [], 'head_yaw_deg': []}
    for fa in frame_analyses:
        faces = fa.get('face_analysis', {}).get('face_analyses', [])
        best = faces[0] if faces else {}
        for k in vals:
            vals[k].append(best.get(k))
    half = window // 2
    for i, fa in enumerate(frame_analyses):
        faces = fa.get('face_analysis', {}).get('face_analyses', [])
        if not faces:
            continue
        for k in vals:
            nearby = [vals[k][j] for j in range(max(0, i-half), min(len(vals[k]), i+half+1)) if vals[k][j] is not None]
            if nearby:
                faces[0][f'{k}_smooth'] = round(float(statistics.median(nearby)), 3)
    return frame_analyses

print('Vision helpers defined.')


# CELL 9: Fast frame extraction with decord fallback + dense event decode
def _sample_seconds(duration):
    ivl = SAMPLE_INTERVAL
    if duration // ivl > MAX_FRAMES:
        ivl = max(1, duration // MAX_FRAMES)
    dense_start = max(0, duration - END_DENSE_SECONDS)
    seconds, seen = [], set()
    for s in range(0, duration, ivl):
        if s >= dense_start:
            break
        seconds.append(s); seen.add(s)
        if len(seconds) >= MAX_FRAMES - END_DENSE_SECONDS:
            break
    for s in range(dense_start, duration):
        if s not in seen:
            seconds.append(s)
    return seconds, ivl

def _extract_frames_decord(video_path):
    import decord
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = int(round(vr.get_avg_fps())) or 30
    total = len(vr)
    dur = total // fps
    sample_secs, ivl = _sample_seconds(dur)
    frame_ids = [min(total-1, int(s*fps)) for s in sample_secs]
    rgb_batch = vr.get_batch(frame_ids).asnumpy()
    frames = []
    for sec, rgb in zip(sample_secs, rgb_batch):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        p = f'{FRAME_DIR}/frame_{sec}.jpg'
        if WRITE_EXTRACTED_FRAMES:
            t = time.time(); cv2.imwrite(p, bgr); PROFILER.add('jpeg_write_extracted', time.time()-t)
            frame_path = p
        else:
            frame_path = None
        frames.append({'timestamp': sec, 'frame_path': frame_path, 'image': bgr if KEEP_FRAMES_IN_MEMORY else None})
    return frames, fps, dur, ivl

def _extract_frames_opencv(video_path):
    cap   = cv2.VideoCapture(video_path)
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur   = total // fps
    sample_secs, ivl = _sample_seconds(dur)
    frames = []
    frame_idx = 0
    for sec in sorted(sample_secs):
        target = int(sec * fps)
        while frame_idx < target:
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
        ret, frm = cap.read()
        frame_idx += 1
        if not ret:
            break
        p = f'{FRAME_DIR}/frame_{sec}.jpg'
        if WRITE_EXTRACTED_FRAMES:
            t = time.time(); cv2.imwrite(p, frm); PROFILER.add('jpeg_write_extracted', time.time()-t)
            frame_path = p
        else:
            frame_path = None
        frames.append({'timestamp': sec, 'frame_path': frame_path, 'image': frm if KEEP_FRAMES_IN_MEMORY else None})
    cap.release()
    return frames, fps, dur, ivl


def extract_frames(video_path):
    PROFILER.start('frame_decode')
    decoder = 'decord'
    decoder_error = None
    try:
        frames, fps, dur, ivl = _extract_frames_decord(video_path)
    except Exception as e:
        decoder_error = str(e)
        decoder = 'opencv_sequential'
        print(f'  decord unavailable/failed ({e}); using sequential OpenCV decode')
        frames, fps, dur, ivl = _extract_frames_opencv(video_path)
    PROFILER.stop({
        'frames': len(frames), 'duration_s': dur, 'sample_interval_s': ivl,
        'decoder': decoder, 'decoder_error': decoder_error,
    })
    print(f'  Extracted {len(frames)} frames from {dur}s video (interval={ivl}s, decoder={decoder}, in_memory={KEEP_FRAMES_IN_MEMORY})')
    return frames, fps, dur


def decode_frames_at_seconds(video_path, seconds):
    """Decode sparse extra frames for event-level validation only."""
    if not seconds:
        return []
    PROFILER.start('dense_event_decode')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for sec in sorted(set(int(s) for s in seconds if s >= 0)):
        target = min(total - 1, int(sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret:
            out.append({'timestamp': sec, 'frame_path': None, 'image': frame, 'is_dense_event_frame': True})
    cap.release()
    PROFILER.stop({'requested': len(set(seconds)), 'decoded': len(out)})
    return out

def attach_dense_event_frames(video_path, event_candidates, duration):
    """Attach +/- window frames to likely VLM candidates that need temporal visual context."""
    dense_cache = {}
    eligible = [c for c in event_candidates if c.needs_vlm and c.behavior in DENSE_EVENT_BEHAVIORS]
    priority = {'alertness': 0, 'nap': 1, 'writing': 2, 'distraction': 3}
    eligible.sort(key=lambda c: (priority.get(c.behavior, 99), -c.local_score))
    eligible = eligible[:max(VLM_MAX_EVENTS, 8)]
    for cand in eligible:
        start = max(0, int(cand.start_ts) - DENSE_EVENT_WINDOW_SECONDS)
        end = min(int(duration), int(cand.end_ts) + DENSE_EVENT_WINDOW_SECONDS)
        step = max(1, int(1 / DENSE_EVENT_FPS)) if DENSE_EVENT_FPS < 1 else 1
        seconds = list(range(start, end + 1, step))
        missing = [s for s in seconds if s not in dense_cache]
        for frame in decode_frames_at_seconds(video_path, missing):
            dense_cache[frame['timestamp']] = frame
        dense = [dense_cache[s] for s in seconds if s in dense_cache]
        if dense:
            cand.metadata['dense_seconds'] = [f['timestamp'] for f in dense]
            cand.metadata['dense_frames'] = dense
            cand.metadata['dense_window_s'] = [start, end]
    PROFILER.events.append({'stage': 'dense_event_attachment', 'eligible': len(eligible), 'unique_dense_frames': len(dense_cache)})
    return event_candidates

def materialize_frame(frame_item):
    if frame_item.get('frame_path') and os.path.exists(frame_item['frame_path']):
        return frame_item['frame_path']
    img = frame_item.get('image')
    if img is None:
        return None
    p = f"{FRAME_DIR}/frame_{frame_item['timestamp']}.jpg"
    t = time.time(); cv2.imwrite(p, img); PROFILER.add('jpeg_write_materialize', time.time()-t)
    frame_item['frame_path'] = p
    return p

print('extract_frames() defined.')

# CELL 10: Batched YOLO inference (GPU)
def _ensure_yolo_sources(frames):
    sources = []
    for f in frames:
        if isinstance(f, dict):
            if KEEP_FRAMES_IN_MEMORY and f.get('image') is not None:
                sources.append(f['image'])
            else:
                sources.append(materialize_frame(f))
        else:
            sources.append(f)
    return sources

def batch_yolo(frames_or_paths, model, class_map, conf=0.2, stage_name='yolo'):
    out = {}
    sources = _ensure_yolo_sources(frames_or_paths)
    if not sources:
        return out
    try:
        PROFILER.start(stage_name, sync_cuda=True)
        gen = model.predict(source=sources, batch=YOLO_BATCH_SIZE, verbose=False, conf=conf, stream=True)
        for i, r in enumerate(gen):
            dets = []
            for box in r.boxes:
                cid = int(box.cls[0])
                if cid in class_map:
                    dets.append({
                        'class_id': cid, 'class_name': class_map[cid],
                        'confidence': round(float(box.conf[0]), 3),
                        'bbox': [int(x) for x in box.xyxy[0].tolist()],
                    })
            out[i] = dets
        PROFILER.stop({'frames': len(sources), 'batch': YOLO_BATCH_SIZE, 'conf': conf, 'in_memory': KEEP_FRAMES_IN_MEMORY})
    except Exception as e:
        if PROFILER.stage_stack:
            PROFILER.stop({'error': str(e), 'fallback': True})
        print(f'  Batch YOLO error ({e}), falling back to sequential...')
        for i, source in enumerate(sources):
            try:
                t0 = time.time()
                r = model(source, verbose=False, conf=conf)[0]
                PROFILER.add(f'{stage_name}_sequential_frame', time.time()-t0)
                out[i] = [{'class_id':int(b.cls[0]),'class_name':class_map[int(b.cls[0])],
                            'confidence':round(float(b.conf[0]),3),
                            'bbox':[int(x) for x in b.xyxy[0].tolist()]}
                           for b in r.boxes if int(b.cls[0]) in class_map]
            except Exception:
                out[i] = []
    return out

print('batch_yolo() defined.')

# CELL 11: Parallel MediaPipe + frame assembly, delayed annotation
from concurrent.futures import ThreadPoolExecutor
import time as _time

def _worker_face(args):
    idx, ts, frame_item, yolo_dets = args
    lm = get_thread_landmarker()
    return idx, analyze_face_on_crops(frame_item, yolo_dets, ts, lm)

def create_annotated_frame(frame_item_or_path, all_dets, timestamp):
    img = _read_frame(frame_item_or_path)
    if img is None:
        return frame_item_or_path if isinstance(frame_item_or_path, str) else materialize_frame(frame_item_or_path)
    img = img.copy()
    H, W = img.shape[:2]
    CRITICAL = {'cell_phone','book','laptop','horn_active','brake_applied'}
    for d in all_dets:
        if d['confidence'] < 0.5:
            continue
        x1,y1,x2,y2 = d['bbox']
        cls = d['class_name']
        col = YOLO_COLORS.get(cls,(255,255,255))
        th  = 3 if cls in CRITICAL else 2
        cv2.rectangle(img,(x1,y1),(x2,y2),col,th)
        lbl = f"{cls.upper() if cls in CRITICAL else cls} {d['confidence']:.0%}"
        fs  = 0.55 if cls in CRITICAL else 0.45
        (lw,lh),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ly = max(y1, lh+10)
        cv2.rectangle(img,(x1,ly-lh-6),(x1+lw+6,ly+2),col,-1)
        cv2.putText(img,lbl,(x1+3,ly-3),cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),2)
    cv2.putText(img,f'T={timestamp}s',(5,H-8),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
    out = f'{ANNOTATED_DIR}/ann_{timestamp}.jpg'
    t0 = _time.time(); cv2.imwrite(out, img); PROFILER.add('jpeg_write_annotation', _time.time()-t0)
    return out

def ensure_annotated_frame(frame_analyses, idx):
    fa = frame_analyses[idx]
    if fa.get('annotated_path') and os.path.exists(fa['annotated_path']):
        return fa['annotated_path']
    all_dets = fa.get('yolo_detections', []) + fa.get('brake_horn_detections', [])
    fa['annotated_path'] = create_annotated_frame(fa.get('frame_item') or fa.get('frame_path'), all_dets, fa['timestamp'])
    return fa['annotated_path']

def save_evidence_image(frame_path, prefix):
    if not frame_path or not os.path.exists(frame_path):
        return None
    filename = f"{prefix}_{int(_time.time()*1000)}.jpg"
    out = f"{DETECTION_DIR}/{filename}"
    import shutil; shutil.copy2(frame_path, out)
    return f"/detections/{filename}"

def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    area_a = max(1, (ax2-ax1) * (ay2-ay1))
    area_b = max(1, (bx2-bx1) * (by2-by1))
    return inter / (area_a + area_b - inter + 1e-6)

def assign_person_tracks(frame_analyses, iou_threshold=0.25, max_missing=2):
    """Lightweight tracker for sampled frames. Adds track_id to person detections and active_track_ids per frame."""
    tracks = {}
    next_id = 1
    for idx, fa in enumerate(frame_analyses):
        persons = [d for d in fa.get('yolo_detections', []) if d['class_name'] == 'person']
        assigned_tracks = set()
        active_ids = []
        for det in sorted(persons, key=lambda d: -d['confidence']):
            best_id, best_iou = None, 0.0
            for tid, tr in tracks.items():
                if idx - tr['last_seen'] > max_missing or tid in assigned_tracks:
                    continue
                iou = _bbox_iou(det['bbox'], tr['bbox'])
                if iou > best_iou:
                    best_id, best_iou = tid, iou
            if best_id is None or best_iou < iou_threshold:
                best_id = next_id
                next_id += 1
            det['track_id'] = best_id
            tracks[best_id] = {'bbox': det['bbox'], 'last_seen': idx}
            assigned_tracks.add(best_id)
            active_ids.append(best_id)
        fa['active_track_ids'] = sorted(active_ids)
    return frame_analyses

def run_full_local_analysis(frames):
    print('  [1/3] Batched YOLO inference...')
    yolo_res = batch_yolo(frames, yolo_model, YOLO_CLASSES, conf=0.2, stage_name='yolo_general')
    bh_res   = batch_yolo(frames, brake_horn_model, BRAKE_HORN_CLASSES, conf=0.3, stage_name='yolo_brake_horn') if BRAKE_HORN_AVAILABLE else {i:[] for i in range(len(frames))}

    print(f'  [2/3] Parallel MediaPipe ({MEDIAPIPE_THREADS} threads)...')
    PROFILER.start('mediapipe_parallel')
    args = [(i, frames[i]['timestamp'], frames[i], yolo_res.get(i,[])) for i in range(len(frames))]
    face_res = {}
    with ThreadPoolExecutor(max_workers=MEDIAPIPE_THREADS) as ex:
        for idx, fd in ex.map(_worker_face, args):
            face_res[idx] = fd
    PROFILER.stop({'frames': len(frames), 'threads': MEDIAPIPE_THREADS})

    print('  [3/3] Assembling frame data...')
    PROFILER.start('frame_assembly')
    frame_analyses = []
    for i, f in enumerate(frames):
        ts  = f['timestamp']
        yd  = yolo_res.get(i,[])
        bhd = bh_res.get(i,[])
        fd  = face_res.get(i,{'faces_detected':0,'face_analyses':[]})
        cnt = {}
        for d in yd:
            cnt[d['class_name']] = cnt.get(d['class_name'],0)+1
        frame_analyses.append({
            'timestamp':ts, 'frame_path':f.get('frame_path'), 'frame_item':f,
            'annotated_path':None,
            'yolo_detections':yd, 'brake_horn_detections':bhd,
            'object_counts':cnt, 'person_count':cnt.get('person',0),
            'phone_detected':'cell_phone' in cnt,
            'bag_detected':any(k in cnt for k in ['backpack','handbag','suitcase']),
            'book_detected':'book' in cnt,
            'horn_active_detected':any(d['class_name']=='horn_active' for d in bhd),
            'brake_applied_detected':any(d['class_name']=='brake_applied' for d in bhd),
            'face_analysis':fd,
        })
    smooth_face_metrics(frame_analyses, window=3)
    assign_person_tracks(frame_analyses)
    PROFILER.stop({'frames': len(frame_analyses), 'person_tracks': max([max(f.get('active_track_ids', [0]) or [0]) for f in frame_analyses] or [0])})
    return frame_analyses

print('Parallel analysis functions defined.')

# ── CELL 12: Tier 2 — Local rule-based parameter engine ───────────────────────
# Report-focused revision: cleaner text, more conservative writing/mobile logic,
# and zero-person consecutive-frame leaving detection.

def find_max_consecutive(indices, total):
    if not indices:
        return 0
    mx = cur = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 1
    return mx

def _runs_local(indices, max_gap=1):
    indices = sorted(set(indices))
    if not indices:
        return []
    runs = [[indices[0]]]
    for idx in indices[1:]:
        if idx - runs[-1][-1] <= max_gap:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs

def _mid(lst):
    return lst[len(lst) // 2] if lst else None

def _event_duration(frame_analyses, indices):
    if not indices:
        return 0
    return max(0, frame_analyses[indices[-1]]['timestamp'] - frame_analyses[indices[0]]['timestamp'])

def _frame_hw_from_analysis(fa):
    img = _read_frame(fa.get('frame_item') or fa.get('frame_path'))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return h, w

def _actuator_run_stats(frame_analyses, run, class_name):
    boxes, confs, centers = [], [], []
    areas = []
    for i in run:
        dets = [d for d in frame_analyses[i].get('brake_horn_detections', []) if d['class_name'] == class_name]
        if not dets:
            continue
        best = max(dets, key=lambda d: d['confidence'])
        x1, y1, x2, y2 = best['bbox']
        h, w = _frame_hw_from_analysis(frame_analyses[i])
        if not h or not w:
            continue
        boxes.append(best['bbox'])
        confs.append(float(best['confidence']))
        centers.append(((x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h))
        areas.append(max(1.0, (x2 - x1) * (y2 - y1)) / float(w * h))
    if not boxes:
        return {
            'detections': 0, 'max_conf': 0.0, 'mean_conf': 0.0, 'median_iou': 0.0,
            'center_span_x': 1.0, 'center_span_y': 1.0, 'area_median': 0.0, 'stable': False,
        }
    pair_ious = [_bbox_iou(boxes[i - 1], boxes[i]) for i in range(1, len(boxes))]
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    area_sorted = sorted(areas)
    return {
        'detections': len(boxes),
        'max_conf': max(confs),
        'mean_conf': sum(confs) / len(confs),
        'median_iou': sorted(pair_ious)[len(pair_ious) // 2] if pair_ious else 0.0,
        'center_span_x': max(xs) - min(xs) if xs else 1.0,
        'center_span_y': max(ys) - min(ys) if ys else 1.0,
        'area_median': area_sorted[len(area_sorted) // 2] if area_sorted else 0.0,
        'stable': False,
    }

def _actuator_is_stable(stats, behavior):
    if stats['detections'] < 2:
        return False
    if behavior == 'brake_a9':
        return (
            stats['max_conf'] >= 0.78 and
            stats['median_iou'] >= 0.22 and
            stats['center_span_x'] <= 0.18 and
            stats['center_span_y'] <= 0.18 and
            0.0004 <= stats['area_median'] <= 0.08
        )
    return (
        stats['max_conf'] >= 0.72 and
        stats['median_iou'] >= 0.18 and
        stats['center_span_x'] <= 0.22 and
        stats['center_span_y'] <= 0.22 and
        0.0004 <= stats['area_median'] <= 0.10
    )

def _stamp_parameter(param, frame_analyses, indices):
    idx = _mid(indices)
    param['evidence_frame_idx'] = idx
    if idx is not None and 0 <= idx < len(frame_analyses):
        param['evidence_timestamp_s'] = frame_analyses[idx].get('timestamp')
        param['timestamp'] = f"{frame_analyses[idx].get('timestamp', 0)}s"
    else:
        param['evidence_timestamp_s'] = None
        param['timestamp'] = 'N/A'
    param['event_duration_s'] = _event_duration(frame_analyses, indices)
    return param

def compute_parameters_locally(frame_analyses):
    total = len(frame_analyses)
    if total == 0:
        return {}, 'No frames available for analysis.', ''

    eyes_closed, eyes_partial, head_droop = [], [], []
    head_turned, gaze_down, yawning = [], [], []
    phone_idx, bag_idx, book_idx = [], [], []
    horn_idx, brake_idx = [], []
    person_counts = []
    all_yaw, all_pitch = [], []

    for i, fa in enumerate(frame_analyses):
        person_counts.append(fa['person_count'])
        if fa['phone_detected']:
            phone_idx.append(i)
        if fa['bag_detected']:
            bag_idx.append(i)
        if fa['book_detected']:
            book_idx.append(i)
        if fa.get('horn_active_detected'):
            horn_idx.append(i)
        if fa.get('brake_applied_detected'):
            brake_idx.append(i)
        for f in fa['face_analysis']['face_analyses']:
            es = f['eye_status']
            if es == 'CLOSED':
                eyes_closed.append(i)
            elif es == 'PARTIALLY_CLOSED':
                eyes_partial.append(i)
            p = f.get('head_pitch_deg')
            y = f.get('head_yaw_deg')
            if p is not None:
                all_pitch.append(p)
            if y is not None:
                all_yaw.append(y)
            if p is not None and p < -25:
                head_droop.append(i)
            if f.get('gaze_direction') == 'down':
                gaze_down.append(i)
            if f.get('yawn_status') == 'YAWNING':
                yawning.append(i)

    base_yaw = sorted(all_yaw)[len(all_yaw) // 2] if all_yaw else 0.0
    for i, fa in enumerate(frame_analyses):
        for f in fa['face_analysis']['face_analyses']:
            y = f.get('head_yaw_deg')
            if y is not None and abs(y - base_yaw) > 30:
                head_turned.append(i)

    def dedup(lst):
        return sorted(set(lst))

    eyes_closed = dedup(eyes_closed)
    eyes_partial = dedup(eyes_partial)
    head_droop = dedup(head_droop)
    head_turned = dedup(head_turned)
    gaze_down = dedup(gaze_down)
    yawning = dedup(yawning)
    phone_idx = dedup(phone_idx)
    bag_idx = dedup(bag_idx)
    book_idx = dedup(book_idx)

    all_impaired = dedup(eyes_closed + eyes_partial)
    perclos = len(all_impaired) / max(1, total)
    consec_closed = find_max_consecutive(eyes_closed, total)
    consec_phone = find_max_consecutive(phone_idx, total)
    zero_idx = [i for i, c in enumerate(person_counts) if c == 0]
    zero_runs = _runs_local(zero_idx, max_gap=1)
    longest_zero_run = max(zero_runs, key=len) if zero_runs else []

    params = {}

    dp = len(head_droop) / total
    yp = len(yawning) / total
    ascore = perclos * 0.5 + dp * 0.3 + yp * 0.2
    ast = 'LOW' if ascore > 0.3 or perclos > 0.4 else 'MODERATE' if ascore > 0.12 or perclos > 0.15 else 'HIGH'
    adesc = {
        'LOW': 'Low. Extended periods of drooping head or closed eyes were observed.',
        'MODERATE': 'Moderate. Occasional signs of reduced alertness were observed.',
        'HIGH': 'High. Crew remained alert with upright posture in sampled frames.',
    }[ast]
    params['alertness'] = _stamp_parameter({
        'key': 'alertness',
        'label': 'Crew Alertness',
        'icon': '',
        'description': adesc,
        'status': ast,
        'confidence': 'HIGH' if total > 20 else 'MEDIUM',
        'image_url': None,
    }, frame_analyses, eyes_closed or head_droop or yawning)

    dp2 = len(head_turned) / total
    dst = 'HIGH' if dp2 > 0.3 else 'MODERATE' if dp2 > 0.15 else 'LOW' if dp2 > 0.08 else 'NONE'
    ddesc = {
        'HIGH': 'High. Crew frequently looked away from the forward view.',
        'MODERATE': 'Moderate. Occasional head turns away from the forward view were observed.',
        'LOW': 'Low. Minor head turns were observed and may be operational.',
        'NONE': 'None. Crew maintained forward focus in sampled frames.',
    }[dst]
    params['distraction'] = _stamp_parameter({
        'key': 'distraction',
        'label': 'Crew Distraction',
        'icon': '',
        'description': ddesc,
        'status': dst,
        'confidence': 'HIGH' if total > 20 else 'MEDIUM',
        'image_url': None,
    }, frame_analyses, head_turned)

    writing_focus = dedup(set(book_idx) & set(gaze_down))
    consec_writing = find_max_consecutive(writing_focus, total)
    if len(writing_focus) >= 3 and consec_writing >= 2:
        wst, wc, wd = 'OBSERVED', 'MEDIUM', 'Observed. Sustained paperwork-focused activity consistent with writing was detected.'
        writing_indices = writing_focus
    elif len(book_idx) >= 2:
        wst, wc, wd = 'NOT_OBSERVED', 'MEDIUM', 'Not Observed. Paper/document-like material was visible, but active writing was not confirmed.'
        writing_indices = book_idx
    elif len(book_idx) == 1:
        wst, wc, wd = 'NOT_OBSERVED', 'LOW', 'Not Observed. A paper/document-like object was briefly visible, but active writing was not confirmed.'
        writing_indices = book_idx
    else:
        wst, wc, wd = 'NOT_OBSERVED', 'HIGH', 'Not Observed. No sustained paperwork activity was detected.'
        writing_indices = []
    params['writing'] = _stamp_parameter({
        'key': 'writing',
        'label': 'Crew Doing Writing Work',
        'icon': '',
        'description': wd,
        'status': wst,
        'confidence': wc,
        'image_url': None,
        'writing_signal_frames': len(writing_focus),
        'document_visible_frames': len(book_idx),
    }, frame_analyses, writing_indices)

    if consec_closed >= 5:
        nst, nc, nd = 'OBSERVED', 'HIGH', 'Observed. Prolonged eye closure consistent with sleep was detected.'
    elif consec_closed >= 3:
        nst, nc, nd = 'OBSERVED', 'MEDIUM', 'Observed. Micro-sleep indicators were detected.'
    elif perclos > 0.4:
        nst, nc, nd = 'OBSERVED', 'MEDIUM', 'Observed. High PERCLOS suggests possible micro-sleep.'
    else:
        nst, nc, nd = 'NOT_OBSERVED', 'HIGH', 'Not Observed. No significant eye-closure pattern was detected.'
    params['nap'] = _stamp_parameter({
        'key': 'nap',
        'label': 'Crew Taking Nap / Micro-sleep',
        'icon': '',
        'description': nd,
        'status': nst,
        'confidence': nc,
        'image_url': None,
    }, frame_analyses, eyes_closed)

    if len(phone_idx) >= 3 and consec_phone >= 2:
        mst, mc, md = 'OBSERVED', 'HIGH', 'Observed. Sustained mobile phone handling/use was detected.'
        mobile_indices = phone_idx
    elif len(phone_idx) >= 1:
        mst, mc, md = 'NOT_OBSERVED', 'MEDIUM' if len(phone_idx) >= 2 else 'LOW', 'Not Observed. A phone-like object was visible, but active use was not confirmed.'
        mobile_indices = phone_idx
    else:
        mst, mc, md = 'NOT_OBSERVED', 'HIGH', 'Not Observed. No phone-like object or active phone use was detected.'
        mobile_indices = []
    params['mobile'] = _stamp_parameter({
        'key': 'mobile',
        'label': 'Crew Using Mobile',
        'icon': '',
        'description': md,
        'status': mst,
        'confidence': mc,
        'image_url': None,
        'phone_visible_frames': len(phone_idx),
    }, frame_analyses, mobile_indices)

    if bag_idx:
        pst, pc, pd = 'NOT_OBSERVED', 'LOW', 'Not Observed. Bags/personal items were visible but no active packing was confirmed.'
    else:
        pst, pc, pd = 'NOT_OBSERVED', 'HIGH', 'Not Observed. No bags or personal items were detected.'
    params['packing'] = _stamp_parameter({
        'key': 'packing',
        'label': 'Crew Packing Belongings',
        'icon': '',
        'description': pd,
        'status': pst,
        'confidence': pc,
        'image_url': None,
    }, frame_analyses, bag_idx)

    baseline_people_seen = sum(1 for c in person_counts[:min(10, total)] if c > 0)
    if baseline_people_seen >= 2 and len(longest_zero_run) >= 3:
        lst = 'OBSERVED'
        lc = 'HIGH' if len(longest_zero_run) >= 5 else 'MEDIUM'
        ld = f'Observed. Person count remained 0 for {len(longest_zero_run)} consecutive sampled frames, consistent with crew leaving position.'
        leaving_indices = longest_zero_run
    elif baseline_people_seen >= 2 and len(longest_zero_run) == 2:
        lst, lc = 'NOT_OBSERVED', 'LOW'
        ld = 'Not Observed. Brief empty-cabin frames were seen, but the run was too short to confirm leaving position.'
        leaving_indices = longest_zero_run
    elif baseline_people_seen >= 2:
        lst, lc, ld = 'NOT_OBSERVED', 'HIGH', 'Not Observed. The cabin did not remain empty for enough consecutive sampled frames to confirm leaving position.'
        leaving_indices = []
    else:
        lst, lc, ld = 'NOT_OBSERVED', 'LOW', 'Insufficient stable crew presence was available before empty-cabin checks.'
        leaving_indices = []
    params['leaving'] = _stamp_parameter({
        'key': 'leaving',
        'label': 'Crew Leaving Place',
        'icon': '',
        'description': ld,
        'status': lst,
        'confidence': lc,
        'image_url': None,
        'zero_person_run_frames': len(longest_zero_run),
    }, frame_analyses, leaving_indices)

    horn_idx = dedup(horn_idx)
    horn_runs = _runs_local(horn_idx, max_gap=1)
    horn_eval = [(run, _actuator_run_stats(frame_analyses, run, 'horn_active')) for run in horn_runs]
    best_horn_run, best_horn_stats = max(
        horn_eval,
        key=lambda item: (item[1]['stable'], len(item[0]), item[1]['max_conf'], item[1]['median_iou']),
        default=([], {'stable': False, 'detections': 0, 'max_conf': 0.0, 'median_iou': 0.0}),
    )
    if best_horn_run:
        best_horn_stats['stable'] = _actuator_is_stable(best_horn_stats, 'horn')
    ch2 = len(best_horn_run)
    if best_horn_stats.get('stable') and len(best_horn_run) >= 4:
        hst, hc = 'OBSERVED', 'HIGH'
    elif best_horn_stats.get('stable') and len(best_horn_run) >= 3:
        hst, hc = 'OBSERVED', 'MEDIUM'
    elif horn_idx:
        hst, hc = 'NOT_OBSERVED', 'LOW'
    else:
        hst, hc = 'NOT_OBSERVED', 'HIGH' if BRAKE_HORN_AVAILABLE else 'LOW'
    hd = (
        f'Observed. Horn lever activity was detected in {len(horn_idx)} sampled frames ({ch2} consecutive).'
        if hst == 'OBSERVED' else
        ('Not Observed. Brief horn-like detections were seen, but temporal support was insufficient to confirm operation.'
         if horn_idx else 'Not Observed. No horn activation was detected by the custom YOLO model.')
    )
    params['horn'] = _stamp_parameter({
        'key': 'horn',
        'label': 'Crew Operating Horn Lever/Switch',
        'icon': '',
        'description': hd,
        'status': hst,
        'confidence': hc,
        'image_url': None,
        'temporal_run_frames': len(best_horn_run),
        'detector_max_confidence': round(best_horn_stats.get('max_conf', 0.0), 3),
        'spatially_stable': bool(best_horn_stats.get('stable')),
    }, frame_analyses, best_horn_run or horn_idx)

    brake_idx = dedup(brake_idx)
    brake_runs = _runs_local(brake_idx, max_gap=1)
    brake_eval = [(run, _actuator_run_stats(frame_analyses, run, 'brake_applied')) for run in brake_runs]
    best_brake_run, best_brake_stats = max(
        brake_eval,
        key=lambda item: (item[1]['stable'], len(item[0]), item[1]['max_conf'], item[1]['median_iou']),
        default=([], {'stable': False, 'detections': 0, 'max_conf': 0.0, 'median_iou': 0.0}),
    )
    if best_brake_run:
        best_brake_stats['stable'] = _actuator_is_stable(best_brake_stats, 'brake_a9')
    cb2 = len(best_brake_run)
    if best_brake_stats.get('stable') and len(best_brake_run) >= 4:
        bst, bcc = 'OBSERVED', 'HIGH'
    elif best_brake_stats.get('stable') and len(best_brake_run) >= 3:
        bst, bcc = 'OBSERVED', 'MEDIUM'
    elif brake_idx:
        bst, bcc = 'NOT_OBSERVED', 'LOW'
    else:
        bst, bcc = 'NOT_OBSERVED', 'HIGH' if BRAKE_HORN_AVAILABLE else 'LOW'
    bd = (
        f'Observed. A9/SA9 brake activity was detected in {len(brake_idx)} sampled frames ({cb2} consecutive).'
        if bst == 'OBSERVED' else
        ('Not Observed. Brief brake-handle detections were seen, but temporal support was insufficient to confirm application.'
         if brake_idx else 'Not Observed. No A9/SA9 brake activity was detected by the custom YOLO model.')
    )
    params['brake_a9'] = _stamp_parameter({
        'key': 'brake_a9',
        'label': 'LP Applying A9/SA9 After Halt',
        'icon': '',
        'description': bd,
        'status': bst,
        'confidence': bcc,
        'image_url': None,
        'temporal_run_frames': len(best_brake_run),
        'detector_max_confidence': round(best_brake_stats.get('max_conf', 0.0), 3),
        'spatially_stable': bool(best_brake_stats.get('stable')),
    }, frame_analyses, best_brake_run or brake_idx)

    for key, param in params.items():
        idx = param.get('evidence_frame_idx')
        st = param.get('status', '')
        if st in ('NOT_OBSERVED', 'NONE') or (key == 'alertness' and st == 'HIGH'):
            param['evidence_frame_idx'] = None
            continue
        if idx is not None and 0 <= idx < total:
            src = frame_analyses[idx].get('annotated_path', frame_analyses[idx]['frame_path'])
            param['image_url'] = save_evidence_image(src, key)

    issues = []
    if params['alertness']['status'] == 'LOW':
        issues.append('low alertness')
    elif params['alertness']['status'] == 'MODERATE':
        issues.append('moderate alertness')
    if params['nap']['status'] == 'OBSERVED':
        issues.append('napping/micro-sleep')
    if params['mobile']['status'] == 'OBSERVED':
        issues.append('mobile phone use')
    if params['distraction']['status'] in ('HIGH', 'MODERATE', 'OBSERVED'):
        issues.append('distraction')
    if params['leaving']['status'] == 'OBSERVED':
        issues.append('crew left position')
    if params['writing']['status'] == 'OBSERVED':
        issues.append('writing during duty')
    if params['packing']['status'] == 'OBSERVED':
        issues.append('packing belongings')
    if params['horn']['status'] == 'OBSERVED':
        issues.append('horn operation')
    if params['brake_a9']['status'] == 'OBSERVED':
        issues.append('A9/SA9 brake')

    summary = f"Flagged {len(issues)} concern(s): {', '.join(issues)}. Review recommended." if issues else 'Safe crew behaviour. No critical violations.'
    recs = []
    if params['alertness']['status'] != 'HIGH':
        recs.append('- Monitor alertness indicators around the evidence timestamp.')
    if params['mobile']['status'] == 'OBSERVED':
        recs.append('- Review the mobile-use interval and enforce no-phone policy if confirmed.')
    if params['leaving']['status'] == 'OBSERVED':
        recs.append('- Review the empty-cabin interval to confirm crew leaving position.')
    if params['writing']['status'] == 'OBSERVED':
        recs.append('- Verify whether paperwork handling was operationally required.')
    if params['brake_a9']['status'] == 'OBSERVED':
        recs.append('- Confirm that A9/SA9 brake application matches procedure.')
    if not recs:
        recs.append('- Behaviour compliant. Continue monitoring.')
    return params, summary, '\n'.join(recs)



# CELL 13: Event candidate generation + reserved VLM event selection
def _runs(indices, max_gap=1):
    indices = sorted(set(indices))
    if not indices:
        return []
    runs, cur = [], [indices[0]]
    for idx in indices[1:]:
        if idx - cur[-1] <= max_gap:
            cur.append(idx)
        else:
            runs.append(cur)
            cur = [idx]
    runs.append(cur)
    return runs

def _ts(frame_analyses, idx):
    return frame_analyses[idx]['timestamp']

def _mid(run):
    return run[len(run) // 2] if run else None

def _det_conf(fa, names, source='yolo_detections'):
    vals = [d['confidence'] for d in fa.get(source, []) if d['class_name'] in names]
    return max(vals) if vals else 0.0

def _frame_hw_from_analysis(fa):
    img = _read_frame(fa.get('frame_item') or fa.get('frame_path'))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return h, w

def _actuator_run_stats(frame_analyses, run, class_name):
    boxes, confs, centers = [], [], []
    areas = []
    for i in run:
        dets = [d for d in frame_analyses[i].get('brake_horn_detections', []) if d['class_name'] == class_name]
        if not dets:
            continue
        best = max(dets, key=lambda d: d['confidence'])
        x1, y1, x2, y2 = best['bbox']
        h, w = _frame_hw_from_analysis(frame_analyses[i])
        if not h or not w:
            continue
        boxes.append(best['bbox'])
        confs.append(float(best['confidence']))
        centers.append(((x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h))
        areas.append(max(1.0, (x2 - x1) * (y2 - y1)) / float(w * h))
    if not boxes:
        return {
            'detections': 0, 'max_conf': 0.0, 'mean_conf': 0.0, 'median_iou': 0.0,
            'center_span_x': 1.0, 'center_span_y': 1.0, 'area_median': 0.0, 'stable': False,
        }
    pair_ious = [_bbox_iou(boxes[i - 1], boxes[i]) for i in range(1, len(boxes))]
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    area_sorted = sorted(areas)
    return {
        'detections': len(boxes),
        'max_conf': max(confs),
        'mean_conf': sum(confs) / len(confs),
        'median_iou': sorted(pair_ious)[len(pair_ious) // 2] if pair_ious else 0.0,
        'center_span_x': max(xs) - min(xs) if xs else 1.0,
        'center_span_y': max(ys) - min(ys) if ys else 1.0,
        'area_median': area_sorted[len(area_sorted) // 2] if area_sorted else 0.0,
        'stable': False,
    }

def _actuator_is_stable(stats, behavior):
    if stats['detections'] < 2:
        return False
    if behavior == 'brake_a9':
        return (
            stats['max_conf'] >= 0.78 and
            stats['median_iou'] >= 0.22 and
            stats['center_span_x'] <= 0.18 and
            stats['center_span_y'] <= 0.18 and
            0.0004 <= stats['area_median'] <= 0.08
        )
    return (
        stats['max_conf'] >= 0.72 and
        stats['median_iou'] >= 0.18 and
        stats['center_span_x'] <= 0.22 and
        stats['center_span_y'] <= 0.22 and
        0.0004 <= stats['area_median'] <= 0.10
    )

def _add_candidate(cands, frame_analyses, behavior, run, score, reason, needs_vlm=False, metadata=None):
    if not run:
        return
    tids = sorted({tid for i in run for tid in frame_analyses[i].get('active_track_ids', [])})
    cands.append(EventCandidate(
        behavior=behavior,
        start_ts=_ts(frame_analyses, run[0]),
        end_ts=_ts(frame_analyses, run[-1]),
        frame_indices=list(run),
        track_ids=tids,
        local_score=round(float(max(0.0, min(1.0, score))), 3),
        final_score=round(float(max(0.0, min(1.0, score))), 3),
        evidence_frame=_mid(run),
        reason=reason,
        needs_vlm=needs_vlm,
        metadata=metadata or {},
    ))

def _paper_roi_signal(frame_item):
    img = _read_frame(frame_item)
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    roi = img[int(h * 0.45):int(h * 0.92), int(w * 0.05):int(w * 0.58)]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 2] > 145).mean()
    low_sat = (hsv[:, :, 1] < 90).mean()
    edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 60, 140).mean() / 255.0
    return float(min(1.0, 0.45 * bright + 0.35 * low_sat + 0.60 * edges))

def _body_attention_signal(fa):
    persons = [d for d in fa.get('yolo_detections', []) if d['class_name'] == 'person']
    if not persons:
        return 0.0
    p = max(persons, key=lambda d: d['confidence'])
    x1, y1, x2, y2 = p['bbox']
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect = height / width
    conf = p['confidence']
    return float(min(1.0, 0.28 + 0.35 * conf + (0.12 if aspect < 1.8 else 0.0)))

def build_event_candidates(frame_analyses, local_params):
    total = len(frame_analyses)
    if total == 0:
        return []
    cands = []
    face_valid = [i for i, fa in enumerate(frame_analyses) if fa['face_analysis']['faces_detected'] > 0]
    face_rate = len(face_valid) / max(1, total)
    closed = [i for i, fa in enumerate(frame_analyses) if any(f['eye_status'] == 'CLOSED' for f in fa['face_analysis']['face_analyses'])]
    partial = [i for i, fa in enumerate(frame_analyses) if any(f['eye_status'] == 'PARTIALLY_CLOSED' for f in fa['face_analysis']['face_analyses'])]
    yawning = [i for i, fa in enumerate(frame_analyses) if any(f.get('yawn_status') in ('YAWNING', 'MOUTH_OPEN') for f in fa['face_analysis']['face_analyses'])]
    all_yaw = [f.get('head_yaw_deg_smooth', f.get('head_yaw_deg')) for fa in frame_analyses for f in fa['face_analysis']['face_analyses'] if f.get('head_yaw_deg') is not None]
    base_yaw = sorted(all_yaw)[len(all_yaw) // 2] if all_yaw else 0
    turned = [i for i, fa in enumerate(frame_analyses) if any(abs(f.get('head_yaw_deg_smooth', f.get('head_yaw_deg', base_yaw)) - base_yaw) > 30 for f in fa['face_analysis']['face_analyses'])]
    phone = [i for i, fa in enumerate(frame_analyses) if fa['phone_detected']]
    book = [i for i, fa in enumerate(frame_analyses) if fa['book_detected']]
    bag = [i for i, fa in enumerate(frame_analyses) if fa['bag_detected']]
    gaze_down = [i for i, fa in enumerate(frame_analyses) if any(f.get('gaze_direction') == 'down' for f in fa['face_analysis']['face_analyses'])]
    horn = [i for i, fa in enumerate(frame_analyses) if fa.get('horn_active_detected')]
    brake = [i for i, fa in enumerate(frame_analyses) if fa.get('brake_applied_detected')]
    person_c = [fa['person_count'] for fa in frame_analyses]
    zero_person = [i for i, c in enumerate(person_c) if c == 0]
    baseline_people_seen = sum(1 for c in person_c[:min(10, total)] if c > 0)

    face_den = max(1, len(face_valid))
    perclos = len(set(closed + partial)) / face_den
    for run in _runs(closed, max_gap=1):
        score = min(1.0, 0.35 + 0.13 * len(run) + 0.35 * perclos)
        _add_candidate(cands, frame_analyses, 'nap', run, score, 'closed-eye temporal run', needs_vlm=(0.35 <= score <= 0.85), metadata={'face_rate': round(face_rate, 3)})
    for run in _runs(yawning, max_gap=1):
        score = min(1.0, 0.42 + 0.10 * len(run))
        _add_candidate(cands, frame_analyses, 'attention_state', run, score, 'possible yawning / reduced alertness attention event', needs_vlm=True, metadata={'attention_hint': 'yawn', 'face_rate': round(face_rate, 3)})
    if not closed and perclos > 0.18:
        _add_candidate(cands, frame_analyses, 'attention_state', partial[:5], min(0.8, perclos + 0.20), 'elevated PERCLOS from valid face frames', needs_vlm=True, metadata={'attention_hint': 'perclos', 'face_rate': round(face_rate, 3)})

    for run in _runs(turned, max_gap=1):
        score = min(0.72, 0.20 + 0.08 * len(run) + len(turned) / max(1, total))
        _add_candidate(cands, frame_analyses, 'attention_state', run, score, 'head/body attention state requires operational-context VLM classification', needs_vlm=True, metadata={'attention_hint': 'head_yaw', 'head_yaw_only': True, 'face_rate': round(face_rate, 3)})

    if face_rate < 0.08:
        person_present = [i for i, fa in enumerate(frame_analyses) if fa['person_count'] > 0 and _body_attention_signal(fa) > 0.45]
        for run in _runs(person_present, max_gap=2):
            if len(run) < 2:
                continue
            step = max(2, len(run) // 3)
            for start in range(0, len(run), step):
                sub = run[start:start + step]
                if len(sub) >= 2:
                    score = min(0.66, 0.38 + 0.025 * len(sub))
                    _add_candidate(cands, frame_analyses, 'attention_state', sub, score, 'MediaPipe sparse; classify operational control-check vs distraction vs reduced alertness', needs_vlm=True, metadata={'attention_hint': 'body_orientation', 'face_rate': round(face_rate, 3)})
                if start >= step * 2:
                    break

    for run in _runs(phone, max_gap=1):
        conf = max(_det_conf(frame_analyses[i], {'cell_phone'}) for i in run)
        score = min(1.0, 0.22 + 0.16 * len(run) + 0.40 * conf)
        _add_candidate(cands, frame_analyses, 'mobile', run, score, 'phone-like object visible; verify active holding/use across the sequence', needs_vlm=True, metadata={'max_yolo_conf': conf, 'phone_visible_frames': len(run)})

    paper_like = [i for i, fa in enumerate(frame_analyses) if _paper_roi_signal(fa.get('frame_item') or fa.get('frame_path')) > 0.48]
    writing_seed = sorted((set(book) & set(gaze_down)) | (set(paper_like) & set(gaze_down)) | (set(book) & set(paper_like)))
    for run in _runs(writing_seed, max_gap=2):
        conf = max([_det_conf(frame_analyses[i], {'book', 'laptop'}) for i in run] or [0.0])
        gaze_overlap = len(set(run) & set(gaze_down)) / max(1, len(run))
        paper_score = max([_paper_roi_signal(frame_analyses[i].get('frame_item') or frame_analyses[i].get('frame_path')) for i in run] or [0.0])
        if len(run) < 2 and paper_score < 0.72:
            continue
        score = min(1.0, 0.18 + 0.09 * len(run) + 0.18 * conf + 0.28 * gaze_overlap + 0.28 * paper_score)
        _add_candidate(cands, frame_analyses, 'writing', run, score, 'paper/document visible with sustained downward attention; verify active writing versus normal console interaction', needs_vlm=True, metadata={'max_yolo_conf': conf, 'gaze_overlap': round(gaze_overlap, 2), 'paper_score': round(paper_score, 2), 'writing_signal_frames': len(run)})

    for run in _runs(bag, max_gap=2):
        score = min(0.65, 0.20 + 0.06 * len(run))
        _add_candidate(cands, frame_analyses, 'packing', run, score, 'bag/belonging visible; VLM must distinguish static vs active handling', needs_vlm=True)

    if baseline_people_seen >= 2:
        for run in _runs(zero_person, max_gap=1):
            if len(run) < 2:
                continue
            score = min(1.0, 0.32 + 0.15 * len(run))
            _add_candidate(cands, frame_analyses, 'leaving', run, score, f'cab empty (0 persons) for {len(run)} consecutive frames', needs_vlm=(len(run) < 4), metadata={'zero_person_run_frames': len(run), 'baseline_people_seen': baseline_people_seen})

    for behavior, idxs, src_name in [('horn', horn, 'horn_active'), ('brake_a9', brake, 'brake_applied')]:
        for run in _runs(idxs, max_gap=1):
            stats = _actuator_run_stats(frame_analyses, run, src_name)
            stats['stable'] = _actuator_is_stable(stats, behavior)
            conf = stats['max_conf']
            stability_bonus = 0.10 if stats['stable'] else -0.18
            score = min(1.0, 0.10 + 0.12 * len(run) + 0.45 * conf + 0.20 * stats['median_iou'] + stability_bonus)
            reason = (
                f'custom YOLO {src_name} run with {len(run)} frames, max confidence {conf:.2f}, '
                f'median IoU {stats["median_iou"]:.2f}, stable={stats["stable"]}'
            )
            if behavior == 'brake_a9' and not stats['stable'] and len(run) < 4:
                continue
            _add_candidate(
                cands,
                frame_analyses,
                behavior,
                run,
                score,
                reason,
                needs_vlm=(not stats['stable'] or len(run) < 3 or conf < 0.78 or score < 0.68),
                metadata={
                    'max_yolo_conf': conf,
                    'temporal_run_frames': len(run),
                    'median_iou': round(stats['median_iou'], 3),
                    'center_span_x': round(stats['center_span_x'], 3),
                    'center_span_y': round(stats['center_span_y'], 3),
                    'area_median': round(stats['area_median'], 4),
                    'spatially_stable': bool(stats['stable']),
                },
            )

    behavior_priority = {'attention_state': 0, 'alertness': 1, 'nap': 2, 'writing': 3, 'mobile': 4, 'leaving': 5, 'horn': 6, 'brake_a9': 7, 'packing': 8, 'distraction': 9}
    cands.sort(key=lambda c: (not c.needs_vlm, behavior_priority.get(c.behavior, 99), -c.local_score))
    print(f'  Event candidates: {len(cands)} total, {sum(c.needs_vlm for c in cands)} need VLM')
    for c in cands[:14]:
        tag = 'VLM' if c.needs_vlm else 'LOCAL'
        print(f'    [{tag}] {c.behavior}: t={c.start_ts}-{c.end_ts}s frames={c.frame_indices} local={c.local_score}')
    return cands

def select_vlm_events(event_candidates):
    selected, seen_ids = [], set()
    def add_for(behavior, limit):
        added = 0
        for c in event_candidates:
            if added >= limit or len(selected) >= VLM_MAX_EVENTS:
                break
            if not c.needs_vlm or c.behavior != behavior or id(c) in seen_ids:
                continue
            selected.append(c)
            seen_ids.add(id(c))
            added += 1
    add_for('attention_state', 3)
    add_for('nap', 1)
    add_for('writing', 2)
    add_for('mobile', 1)
    for behavior in ('leaving', 'horn', 'brake_a9', 'packing'):
        add_for(behavior, VLM_MAX_EVENTS)
    slot_usage = {}
    for c in selected:
        slot_usage[c.behavior] = slot_usage.get(c.behavior, 0) + 1
        c.metadata['vlm_slot_reserved'] = True
    print(f'  VLM event selection: {len(selected)}/{len(event_candidates)} candidates; slots={slot_usage}')
    PROFILER.events.append({'stage': 'vlm_slot_selection', 'slots': slot_usage, 'selected': len(selected)})
    return selected

def select_vlm_frames(frame_analyses, local_params):
    return {c.evidence_frame: [c.behavior] for c in select_vlm_events(build_event_candidates(frame_analyses, local_params)) if c.evidence_frame is not None}

print('Event candidate selection defined.')




# CELL 14: Event-level VLM analysis + calibrated fusion/evidence mapping
import re

BEHAVIOR_TO_PARAM = {
    'alertness': 'alertness', 'distraction': 'distraction', 'writing': 'writing', 'nap': 'nap',
    'mobile': 'mobile', 'packing': 'packing', 'leaving': 'leaving', 'horn': 'horn', 'brake_a9': 'brake_a9',
}
VLM_TASK_RULES = {
    'attention_state': 'Classify crew attention state across the sequence. Use one of: alert, reduced_alertness, yawning, drowsy, normal_control_check, writing, non_operational_distraction, uncertain. Normal checking/operating instrument panels and controls is NOT distraction.',
    'mobile': 'Decide whether a crew member is actively holding or using a mobile phone. Do not count reflective handles, radios, controls, or phone-like objects not in hand.',
    'writing': 'Use the sequence. Decide whether the hand is writing/marking paperwork for several seconds versus merely operating controls or touching a console.',
    'packing': 'Decide whether a crew member is actively gathering/packing belongings. A stationary bag alone is NO.',
    'leaving': 'Decide whether a crew member vacated the expected seat/position. Temporary occlusion or detector miss is UNCERTAIN/NO.',
    'nap': 'Classify the sequence as blink, yawn, microsleep, sleep, or none. Treat repeated yawning/long mouth-open as alertness evidence; treat sustained closed eyes/head droop as microsleep/sleep.',
    'distraction': 'Decide whether attention is away from track/controls for a sustained non-operational reason. Normal instrument/control-panel checking is NO or normal_control_check.',
    'horn': 'Decide whether the horn lever/switch is actually being operated. Prefer local custom detector if visible.',
    'brake_a9': 'Decide whether A9/SA9 brake handle is being applied. Prefer local custom detector if visible.',
    'alertness': 'Assess alertness from yawning, posture, eyes, and engagement with controls. Report yawn/mouth-open separately from microsleep.',
}
ATTENTION_STATES = {
    'alert', 'reduced_alertness', 'yawning', 'drowsy', 'normal_control_check',
    'writing', 'non_operational_distraction', 'uncertain'
}

def _safe_json(text):
    text = text.strip()
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        pass
    low = text.lower()
    subtype = None
    for state in list(ATTENTION_STATES) + ['blink', 'microsleep', 'sleep', 'none']:
        if state in low:
            subtype = state
            break
    decision = 'uncertain'
    if re.search(r'\byes\b|observed|active|microsleep|sleep|yawn|drowsy|reduced_alertness|non_operational_distraction|writing', low):
        decision = 'yes'
    if re.search(r'\bno\b|not observed|not active|none|normal_control_check|alert', low):
        decision = 'no' if subtype in (None, 'none', 'normal_control_check', 'alert') else decision
    return {'decision': decision, 'confidence_0_1': 0.45, 'evidence_image': 1, 'subtype': subtype, 'attention_state': subtype, 'reason': text[:180]}

def _frame_cv_summary(fa):
    dets = []
    for d in fa.get('yolo_detections', []) + fa.get('brake_horn_detections', []):
        if d['class_name'] != 'person':
            dets.append(f"{d['class_name']}:{d['confidence']:.2f}@{d['bbox']}")
    face_bits = []
    for f in fa.get('face_analysis', {}).get('face_analyses', [])[:2]:
        face_bits.append(f"eye={f.get('eye_status')} ear={f.get('ear_avg')} pitch={f.get('head_pitch_deg')} yaw={f.get('head_yaw_deg')} gaze={f.get('gaze_direction')} yawn={f.get('yawn_status')}")
    tracks = fa.get('active_track_ids', [])
    return f"t={fa['timestamp']}s persons={fa['person_count']} tracks={tracks} dets=[{'; '.join(dets) or 'none'}] faces=[{'; '.join(face_bits) or 'none'}]"

def _event_context_items(candidate, frame_analyses):
    dense = candidate.metadata.get('dense_frames') or []
    if dense:
        if len(dense) > VLM_CONTEXT_IMAGES:
            step = max(1, math.ceil(len(dense) / VLM_CONTEXT_IMAGES))
            dense = dense[::step][:VLM_CONTEXT_IMAGES]
        return [{'kind': 'dense', 'frame': f, 'summary': f"dense t={f['timestamp']}s"} for f in dense]
    total = len(frame_analyses)
    ev = candidate.evidence_frame if candidate.evidence_frame is not None else candidate.frame_indices[0]
    choices = [max(0, candidate.frame_indices[0] - 1), ev, min(total - 1, candidate.frame_indices[-1] + 1)]
    clean = None
    for i, fa in enumerate(frame_analyses):
        if i in candidate.frame_indices:
            continue
        if candidate.behavior == 'mobile' and not fa['phone_detected']:
            clean = i
            break
        if candidate.behavior == 'writing' and not fa['book_detected']:
            clean = i
            break
        if candidate.behavior == 'packing' and not fa['bag_detected']:
            clean = i
            break
        if candidate.behavior == 'leaving' and fa['person_count'] >= 1:
            clean = i
            break
    if clean is not None:
        choices.append(clean)
    out = []
    for i in choices:
        if i not in out and 0 <= i < total:
            out.append(i)
    return [{'kind': 'analysis', 'idx': i, 'summary': _frame_cv_summary(frame_analyses[i])} for i in out[:VLM_CONTEXT_IMAGES]]

def _pil_for_vlm_item(item, candidate, frame_analyses):
    if item['kind'] == 'dense':
        img = _read_frame(item['frame'])
    else:
        fa = frame_analyses[item['idx']]
        img = _read_frame(fa.get('frame_item') or fa.get('frame_path'))
        if img is None:
            img = cv2.imread(ensure_annotated_frame(frame_analyses, item['idx']))
    if candidate.behavior in ('mobile', 'writing', 'packing') and item['kind'] == 'analysis':
        fa = frame_analyses[item['idx']]
        names = {'mobile': {'cell_phone'}, 'writing': {'book', 'laptop'}, 'packing': {'backpack', 'handbag', 'suitcase'}}[candidate.behavior]
        boxes = [d['bbox'] for d in fa.get('yolo_detections', []) if d['class_name'] in names]
        if boxes:
            H, W = img.shape[:2]
            x1 = max(0, min(b[0] for b in boxes) - int(0.15 * W))
            y1 = max(0, min(b[1] for b in boxes) - int(0.15 * H))
            x2 = min(W, max(b[2] for b in boxes) + int(0.15 * W))
            y2 = min(H, max(b[3] for b in boxes) + int(0.15 * H))
            crop = img[y1:y2, x1:x2]
            if crop.size:
                img = crop
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).resize((VLM_IMAGE_SIZE, VLM_IMAGE_SIZE))

def build_event_prompt(candidate, frame_analyses, items):
    cv_lines = '\n'.join([f"Image {n + 1}: {it['summary']}" for n, it in enumerate(items)])
    near = frame_analyses[max(0, candidate.frame_indices[0] - 3):candidate.frame_indices[-1] + 4]
    person_timeline = ', '.join([f"{fa['timestamp']}s:{fa['person_count']}p" for fa in near])
    rule = VLM_TASK_RULES.get(candidate.behavior, 'Decide if the behavior is present.')
    return f"""You are validating one safety event candidate from locomotive cab CCTV.
Behavior: {candidate.behavior}
Candidate interval: {candidate.start_ts}-{candidate.end_ts}s
Local CV score: {candidate.local_score}
Local reason: {candidate.reason}
Candidate metadata: {json.dumps({k: v for k, v in candidate.metadata.items() if k not in ('dense_frames', 'evidence_frame_item')}, default=str)}
Person-count timeline near event: {person_timeline}

CV evidence per image:
{cv_lines}

Task rule: {rule}

Important railway context:
- Looking at or operating the control panel, gauges, switches, brake, horn, or normal instruments is normal_control_check, not distraction.
- Writing/paperwork is a separate behavior, not generic distraction.
- Reduced alertness includes yawning, drowsy posture, prolonged disengagement, or microsleep signs.

Return ONLY valid JSON:
{{"decision":"yes|no|uncertain", "confidence_0_1":0.0, "attention_state":"alert|reduced_alertness|yawning|drowsy|normal_control_check|writing|non_operational_distraction|uncertain", "subtype":"writing|distracted|yawn|blink|microsleep|sleep|none", "evidence_image":1, "reason":"short reason grounded in the image sequence"}}
"""

def _evidence_info_from_item(item, candidate):
    if item.get('kind') == 'dense':
        frame = item['frame']
        return {
            'source': 'dense_event_frame',
            'timestamp': frame.get('timestamp'),
            'frame_item': frame,
            'analysis_idx': None,
        }
    return {
        'source': 'sampled_analysis_frame',
        'timestamp': None,
        'frame_item': None,
        'analysis_idx': item.get('idx', candidate.evidence_frame),
    }

def run_vlm_on_event(candidate, frame_analyses):
    load_vlm_model()
    items = _event_context_items(candidate, frame_analyses)
    images = [_pil_for_vlm_item(item, candidate, frame_analyses) for item in items]
    prompt = build_event_prompt(candidate, frame_analyses, items)
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': im} for im in images] + [{'type': 'text', 'text': prompt}]}]
    PROFILER.start('vlm_preprocess', sync_cuda=True)
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = vlm_processor(text=[text], images=img_inputs, videos=vid_inputs, padding=True, return_tensors='pt').to(vlm_model.device)
    input_tokens = int(inputs.input_ids.shape[1])
    PROFILER.stop({'behavior': candidate.behavior, 'images': len(images), 'input_tokens': input_tokens, 'dense': bool(candidate.metadata.get('dense_frames'))})
    PROFILER.start('vlm_generate', sync_cuda=True)
    with torch.inference_mode():
        out_ids = vlm_model.generate(**inputs, max_new_tokens=VLM_MAX_TOKENS, do_sample=False)
    gen = out_ids[:, inputs.input_ids.shape[1]:]
    raw = vlm_processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    PROFILER.stop({'behavior': candidate.behavior, 'output_tokens': int(gen.shape[1])})
    parsed = _safe_json(raw)
    ev_img = parsed.get('evidence_image') or parsed.get('evidence_frame') or 1
    try:
        ev_item = items[max(0, min(len(items) - 1, int(ev_img) - 1))]
    except Exception:
        ev_item = items[0] if items else {'kind': 'analysis', 'idx': candidate.evidence_frame}
    evidence_info = _evidence_info_from_item(ev_item, candidate)
    ev_idx = evidence_info.get('analysis_idx')
    return parsed, raw, ev_idx, evidence_info

def _status_to_bool(key, status):
    if key == 'alertness':
        return status in ('LOW', 'MODERATE')
    if key == 'distraction':
        return status in ('LOW', 'MODERATE', 'HIGH', 'OBSERVED')
    return status == 'OBSERVED'

def _attention_state(parsed):
    state = str(parsed.get('attention_state') or parsed.get('subtype') or '').lower()
    if state == 'yawn':
        state = 'yawning'
    if state == 'distracted':
        state = 'non_operational_distraction'
    return state if state in ATTENTION_STATES else 'uncertain'

def _clean_reason(text):
    if not text:
        return ''
    return re.sub(r'\s*\|\s*VLM error:.*$', '', str(text)).strip()

def _fmt_seconds(seconds):
    if seconds is None:
        return 'N/A'
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:02d}'

def _behavior_description(key, status, candidate, param):
    reason = _clean_reason(candidate.reason if candidate else param.get('local_signal_summary') or '')
    runtime_note = param.get('vlm_runtime_note')
    if key == 'mobile':
        if status == 'OBSERVED':
            text = 'Observed. Active mobile phone use was confirmed over the event sequence.'
        elif param.get('phone_visible_frames'):
            text = 'Not Observed. A phone-like object was visible, but active holding/use was not confirmed.'
        else:
            text = 'Not Observed. No phone-like object or active use was confirmed.'
    elif key == 'writing':
        if status == 'OBSERVED':
            text = 'Observed. Sustained paperwork-focused activity consistent with writing was detected.'
        elif param.get('writing_signal_frames') or param.get('document_visible_frames'):
            text = 'Not Observed. Paper/document-like material was visible, but active writing was not confirmed.'
        else:
            text = 'Not Observed. No sustained paperwork activity was confirmed.'
    elif key == 'leaving':
        run_len = param.get('zero_person_run_frames') or candidate.metadata.get('zero_person_run_frames', 0) if candidate else param.get('zero_person_run_frames', 0)
        if status == 'OBSERVED':
            text = f'Observed. Person count remained 0 for {run_len} consecutive sampled frames, consistent with crew leaving position.'
        else:
            text = 'Not Observed. The cabin did not remain empty for enough consecutive sampled frames to confirm leaving position.'
    elif key in ('horn', 'brake_a9'):
        run_len = param.get('temporal_run_frames') or (candidate.metadata.get('temporal_run_frames', 0) if candidate else 0)
        stable = param.get('spatially_stable')
        label = 'Horn lever activity' if key == 'horn' else 'A9/SA9 brake activity'
        if status == 'OBSERVED':
            text = f'Observed. {label} was detected across {run_len} temporally consistent sampled frames.'
        else:
            reason_suffix = 'detected boxes were not spatially stable enough to confirm visibility.' if stable is False else 'temporal support was insufficient to confirm the action.'
            text = f'Not Observed. Brief {label.lower()} detections may have appeared, but {reason_suffix}'
    else:
        lead = {
            'OBSERVED': 'Observed.',
            'NOT_OBSERVED': 'Not Observed.',
            'HIGH': 'High.',
            'MODERATE': 'Moderate.',
            'LOW': 'Low.',
            'NONE': 'None.',
        }.get(status, f'{status}.')
        text = f'{lead} {reason}'.strip() if reason else lead
    if runtime_note:
        text = f'{text} {runtime_note}'.strip()
    return text

def fuse_event_score(candidate, vlm_result=None):
    local = candidate.local_score
    vlm_score = 0.0
    vlm_decision = 'none'
    vlm_conf = 0.0
    subtype = None
    state = None
    if vlm_result:
        vlm_decision = str(vlm_result.get('decision', 'uncertain')).lower()
        subtype = str(vlm_result.get('subtype') or '').lower() or None
        state = _attention_state(vlm_result)
        vlm_conf = float(vlm_result.get('confidence_0_1') or 0.0)
        if vlm_decision == 'yes':
            vlm_score = vlm_conf
        elif vlm_decision == 'no':
            vlm_score = -vlm_conf
    candidate.vlm_score = round(vlm_score, 3)
    candidate.metadata['vlm_decision'] = vlm_decision
    candidate.metadata['vlm_confidence'] = round(vlm_conf, 3)
    if subtype:
        candidate.metadata['vlm_subtype'] = subtype
    if state:
        candidate.metadata['attention_state'] = state

    if candidate.behavior == 'attention_state':
        if state in ('reduced_alertness', 'yawning', 'drowsy'):
            final = max(0.58, local * 0.35 + vlm_conf * 0.65)
            candidate.decision = 'LOW' if state == 'drowsy' and vlm_conf >= 0.75 else 'MODERATE'
        elif state == 'non_operational_distraction':
            final = max(0.62, local * 0.35 + vlm_conf * 0.65)
            candidate.decision = 'HIGH' if vlm_conf >= 0.82 and (candidate.end_ts - candidate.start_ts) >= 4 else 'MODERATE'
        elif state == 'writing':
            final = max(0.62, local * 0.30 + vlm_conf * 0.70)
            candidate.decision = 'WRITING'
        elif state in ('normal_control_check', 'alert'):
            final = min(0.35, local * 0.25)
            candidate.decision = 'NORMAL_OPERATION'
        else:
            final = local * 0.35 + max(0, vlm_score) * 0.25
            candidate.decision = 'UNCERTAIN'
        candidate.final_score = round(max(0.0, min(1.0, final)), 3)
        return candidate

    if candidate.behavior in ('horn', 'brake_a9'):
        run_len = int(candidate.metadata.get('temporal_run_frames') or len(candidate.frame_indices))
        conf_gate = float(candidate.metadata.get('max_yolo_conf') or 0.0)
        stable = bool(candidate.metadata.get('spatially_stable'))
        median_iou = float(candidate.metadata.get('median_iou') or 0.0)
        final = local * 0.70 + max(0, vlm_score) * 0.30
        if candidate.behavior == 'brake_a9':
            if stable and run_len >= 3 and conf_gate >= 0.80 and median_iou >= 0.22:
                final = max(final, 0.74)
            else:
                final = min(final, 0.42)
        else:
            if stable and run_len >= 2 and conf_gate >= 0.72:
                final = max(final, 0.68)
            elif run_len < 2 or conf_gate < 0.55 or not stable:
                final = min(final, 0.44)
    elif candidate.behavior == 'nap':
        if subtype in ('yawn', 'yawning'):
            candidate.metadata['alertness_from_yawn'] = True
            final = min(0.55, local * 0.40 + max(0, vlm_score) * 0.20)
        elif vlm_score >= 0.80 and local >= 0.35:
            final = max(0.72, local * 0.55 + vlm_score * 0.45)
        else:
            final = local * 0.65 + max(0, vlm_score) * 0.35
    elif candidate.behavior == 'distraction':
        if state == 'normal_control_check':
            final = 0.15
        elif state == 'non_operational_distraction':
            final = max(0.62, local * 0.35 + max(0, vlm_score) * 0.65)
        else:
            final = min(0.55, local * 0.40 + max(0, vlm_score) * 0.30)
        if candidate.metadata.get('head_yaw_only') and state != 'non_operational_distraction':
            final = min(final, 0.45)
    elif candidate.behavior == 'writing':
        sustained = bool(candidate.metadata.get('dense_frames')) or len(candidate.frame_indices) >= 2
        gaze_overlap = float(candidate.metadata.get('gaze_overlap') or 0.0)
        paper_score = float(candidate.metadata.get('paper_score') or 0.0)
        final = local * 0.35 + max(0, vlm_score) * 0.65
        if sustained and vlm_decision == 'yes':
            final = max(final, 0.72)
        elif sustained and gaze_overlap >= 0.60 and paper_score >= 0.55 and vlm_decision not in ('no',):
            final = max(final, 0.62)
        if vlm_score < -0.70:
            final *= 0.30
    elif candidate.behavior == 'alertness':
        final = local * 0.45 + max(0, vlm_score) * 0.55
        if subtype in ('yawn', 'yawning') and vlm_score >= 0.55:
            final = max(final, 0.55)
    elif candidate.behavior == 'leaving':
        zero_run = int(candidate.metadata.get('zero_person_run_frames') or len(candidate.frame_indices))
        final = local * 0.70 + max(0, vlm_score) * 0.30
        if zero_run >= 4:
            final = max(final, 0.72)
        if vlm_score < -0.65 and local < 0.80:
            final *= 0.50
    elif candidate.behavior == 'mobile':
        sustained = bool(candidate.metadata.get('dense_frames')) or len(candidate.frame_indices) >= 2
        final = local * 0.35 + max(0, vlm_score) * 0.65
        if vlm_decision == 'yes' and sustained:
            final = max(final, 0.72)
        elif vlm_decision == 'no':
            final = min(final, 0.42)
        elif not sustained:
            final = min(final, 0.48)
    else:
        final = local * 0.45 + max(0, vlm_score) * 0.55
        if vlm_score < -0.55:
            final *= 0.45
    candidate.final_score = round(max(0.0, min(1.0, final)), 3)
    threshold = FUSION_THRESHOLDS.get(candidate.behavior, 0.65)
    candidate.decision = 'OBSERVED' if candidate.final_score >= threshold else 'NOT_OBSERVED'
    if candidate.behavior == 'distraction' and candidate.decision == 'OBSERVED':
        candidate.decision = 'HIGH' if candidate.final_score >= 0.8 and state == 'non_operational_distraction' else 'MODERATE'
    if candidate.behavior == 'alertness' and candidate.final_score >= 0.48:
        candidate.decision = 'MODERATE' if candidate.final_score < 0.75 else 'LOW'
    return candidate

def _save_candidate_evidence(candidate, frame_analyses, key):
    src_item = candidate.metadata.get('evidence_frame_item')
    if src_item and src_item.get('image') is not None:
        filename = f"{key}_{int(time.time() * 1000)}.jpg"
        p = f"{DETECTION_DIR}/{filename}"
        cv2.imwrite(p, src_item['image'])
        return f"/detections/{filename}"
    idx = candidate.evidence_frame
    if idx is not None and 0 <= idx < len(frame_analyses):
        ann = ensure_annotated_frame(frame_analyses, idx)
        return save_evidence_image(ann, key)
    return None

def _apply_attention_state(parameters, attention_candidates):
    if not attention_candidates:
        return None, None
    valid = [c for c in attention_candidates if c.metadata.get('attention_state')]
    if not valid:
        return None, None
    priority = {
        'drowsy': 0, 'reduced_alertness': 1, 'yawning': 2,
        'non_operational_distraction': 3, 'writing': 4,
        'normal_control_check': 5, 'alert': 6, 'uncertain': 7,
    }
    best = sorted(valid, key=lambda c: (priority.get(c.metadata.get('attention_state'), 9), -c.final_score))[0]
    state = best.metadata.get('attention_state')
    if state in ('drowsy', 'reduced_alertness', 'yawning'):
        parameters['alertness']['status'] = 'LOW' if state == 'drowsy' and best.final_score >= 0.75 else 'MODERATE'
        parameters['alertness']['confidence'] = 'HIGH' if best.final_score >= 0.80 else 'MEDIUM'
        parameters['alertness']['description'] = f"{parameters['alertness']['status']}. Attention-state VLM classified {state}."
        parameters['alertness']['evidence_frame_idx'] = best.evidence_frame
        parameters['alertness']['_evidence_candidate'] = best
    elif state == 'non_operational_distraction':
        parameters['distraction']['status'] = 'HIGH' if best.final_score >= 0.80 and (best.end_ts - best.start_ts) >= 4 else 'MODERATE'
        parameters['distraction']['confidence'] = 'HIGH' if best.final_score >= 0.80 else 'MEDIUM'
        parameters['distraction']['description'] = 'Observed. Attention-state VLM classified non-operational distraction.'
        parameters['distraction']['evidence_frame_idx'] = best.evidence_frame
        parameters['distraction']['_evidence_candidate'] = best
    elif state == 'writing':
        if 'writing' in parameters:
            parameters['writing']['status'] = 'OBSERVED'
            parameters['writing']['confidence'] = 'HIGH' if best.final_score >= 0.80 else 'MEDIUM'
            parameters['writing']['description'] = 'Observed. Attention-state VLM classified writing/paperwork.'
            parameters['writing']['evidence_frame_idx'] = best.evidence_frame
            parameters['writing']['_evidence_candidate'] = best
        if parameters.get('distraction', {}).get('status') == 'HIGH':
            parameters['distraction']['status'] = 'LOW'
            parameters['distraction']['description'] = 'Low. Attention was associated with paperwork rather than generic non-operational distraction.'
    elif state in ('normal_control_check', 'alert'):
        if parameters.get('distraction', {}).get('status') in ('HIGH', 'MODERATE'):
            parameters['distraction']['status'] = 'NONE' if state == 'alert' else 'LOW'
            parameters['distraction']['confidence'] = 'HIGH' if best.metadata.get('vlm_confidence', 0) >= 0.75 else 'MEDIUM'
            parameters['distraction']['description'] = f"{parameters['distraction']['status']}. VLM classified the posture as {state}, not non-operational distraction."
            parameters['distraction']['evidence_frame_idx'] = None
    return best, state

def apply_event_fusion(parameters, event_candidates, frame_analyses):
    by_behavior = defaultdict(list)
    for c in event_candidates:
        by_behavior[c.behavior].append(c)

    _apply_attention_state(parameters, by_behavior.get('attention_state', []))

    for behavior, cands in by_behavior.items():
        if behavior == 'attention_state':
            continue
        pk = BEHAVIOR_TO_PARAM.get(behavior)
        if not pk or pk not in parameters:
            continue
        best = max(cands, key=lambda c: c.final_score)
        prior_status = parameters[pk].get('status')
        if behavior == 'alertness':
            if best.final_score >= 0.75:
                parameters[pk]['status'] = 'LOW'
            elif best.final_score >= 0.48 and parameters[pk]['status'] == 'HIGH':
                parameters[pk]['status'] = 'MODERATE'
        elif behavior == 'distraction':
            state = best.metadata.get('attention_state')
            if best.metadata.get('head_yaw_only') and state != 'non_operational_distraction':
                if parameters[pk]['status'] in ('HIGH', 'MODERATE'):
                    parameters[pk]['status'] = 'LOW'
                else:
                    parameters[pk]['status'] = 'NONE'
            else:
                parameters[pk]['status'] = best.decision if best.decision in ('HIGH', 'MODERATE') else ('NONE' if not _status_to_bool(pk, parameters[pk]['status']) else parameters[pk]['status'])
        elif behavior == 'writing' and prior_status == 'OBSERVED' and best.decision == 'NOT_OBSERVED':
            strong_no = best.metadata.get('vlm_decision') == 'no' and best.metadata.get('vlm_confidence', 0) >= 0.80 and not best.metadata.get('dense_frames')
            if strong_no:
                parameters[pk]['status'] = 'NOT_OBSERVED'
            else:
                parameters[pk]['status'] = 'OBSERVED'
                best.decision = 'OBSERVED'
                best.final_score = max(best.final_score, 0.50)
        else:
            parameters[pk]['status'] = best.decision
        parameters[pk]['confidence'] = 'HIGH' if best.final_score >= 0.80 else 'MEDIUM' if best.final_score >= 0.55 else 'LOW'
        parameters[pk]['evidence_type'] = 'vlm_sequence' if best.metadata.get('dense_frames') else ('roi_motion' if best.metadata.get('paper_score') else 'fusion')
        parameters[pk]['event_duration_s'] = max(0, best.end_ts - best.start_ts)
        parameters[pk]['vlm_decision'] = best.metadata.get('vlm_decision', 'not_run')
        parameters[pk]['vlm_subtype'] = best.metadata.get('vlm_subtype') or best.metadata.get('attention_state')
        parameters[pk]['local_signal_summary'] = _clean_reason(best.reason)
        parameters[pk]['evidence_timestamp_s'] = best.metadata.get('evidence_timestamp_s')
        parameters[pk]['evidence_source'] = best.metadata.get('evidence_source')
        parameters[pk]['phone_visible_frames'] = best.metadata.get('phone_visible_frames', parameters[pk].get('phone_visible_frames'))
        parameters[pk]['writing_signal_frames'] = best.metadata.get('writing_signal_frames', parameters[pk].get('writing_signal_frames'))
        parameters[pk]['document_visible_frames'] = best.metadata.get('document_visible_frames', parameters[pk].get('document_visible_frames'))
        parameters[pk]['zero_person_run_frames'] = best.metadata.get('zero_person_run_frames', parameters[pk].get('zero_person_run_frames'))
        parameters[pk]['temporal_run_frames'] = best.metadata.get('temporal_run_frames', parameters[pk].get('temporal_run_frames'))
        parameters[pk]['spatially_stable'] = best.metadata.get('spatially_stable', parameters[pk].get('spatially_stable'))
        parameters[pk]['detector_max_confidence'] = best.metadata.get('max_yolo_conf', parameters[pk].get('detector_max_confidence'))
        parameters[pk]['description'] = _behavior_description(pk, parameters[pk]['status'], best, parameters[pk])
        if best.evidence_frame is not None and _status_to_bool(pk, parameters[pk]['status']):
            parameters[pk]['evidence_frame_idx'] = best.evidence_frame
            parameters[pk]['_evidence_candidate'] = best

    face_frames = sum(1 for fa in frame_analyses if fa.get('face_analysis', {}).get('faces_detected', 0) > 0)
    face_rate = face_frames / max(1, len(frame_analyses))
    if face_rate < 0.08 and parameters.get('alertness', {}).get('status') == 'HIGH':
        parameters['alertness']['confidence'] = 'MEDIUM'
        parameters['alertness']['description'] = 'High. No reduced-alertness event was confirmed, but face visibility was sparse.'
        parameters['alertness']['local_signal_summary'] = f'sparse face detection rate={face_rate:.1%}'

    yawn_alert = [c for c in event_candidates if c.metadata.get('alertness_from_yawn') and c.vlm_score > 0.5]
    if yawn_alert and parameters.get('alertness', {}).get('status') == 'HIGH':
        best_yawn = max(yawn_alert, key=lambda c: c.vlm_score)
        parameters['alertness']['status'] = 'MODERATE'
        parameters['alertness']['confidence'] = 'MEDIUM'
        parameters['alertness']['description'] = 'Moderate. Yawning or mouth-open alertness evidence was detected in the event sequence.'
        parameters['alertness']['evidence_frame_idx'] = best_yawn.evidence_frame
        parameters['alertness']['_evidence_candidate'] = best_yawn
        parameters['alertness']['evidence_type'] = 'vlm_sequence'
        parameters['alertness']['event_duration_s'] = max(0, best_yawn.end_ts - best_yawn.start_ts)
        parameters['alertness']['vlm_decision'] = best_yawn.metadata.get('vlm_decision', 'yes')
        parameters['alertness']['vlm_subtype'] = best_yawn.metadata.get('vlm_subtype')
        parameters['alertness']['local_signal_summary'] = _clean_reason(best_yawn.reason)

    for key, param in parameters.items():
        cand = param.pop('_evidence_candidate', None)
        st = param.get('status', '')
        if st in ('NOT_OBSERVED', 'NONE') or (key == 'alertness' and st == 'HIGH'):
            param['evidence_frame_idx'] = None
            continue
        if cand is not None:
            param['image_url'] = _save_candidate_evidence(cand, frame_analyses, key)
            param['evidence_timestamp_s'] = cand.metadata.get('evidence_timestamp_s', param.get('evidence_timestamp_s'))
            param['evidence_source'] = cand.metadata.get('evidence_source', param.get('evidence_source'))
        else:
            idx = param.get('evidence_frame_idx')
            if idx is not None and 0 <= idx < len(frame_analyses):
                ann = ensure_annotated_frame(frame_analyses, idx)
                param['image_url'] = save_evidence_image(ann, key)
                param['evidence_timestamp_s'] = frame_analyses[idx].get('timestamp')
                param['evidence_source'] = 'sampled_analysis_frame'
        if param.get('evidence_timestamp_s') is not None:
            param['timestamp'] = _fmt_seconds(param['evidence_timestamp_s'])
    return parameters

def build_final_summary(parameters):
    issues = []
    if parameters.get('alertness', {}).get('status') == 'LOW':
        issues.append('low alertness')
    if parameters.get('alertness', {}).get('status') == 'MODERATE':
        issues.append('moderate alertness/yawning')
    if parameters.get('nap', {}).get('status') == 'OBSERVED':
        issues.append('napping/micro-sleep')
    if parameters.get('mobile', {}).get('status') == 'OBSERVED':
        issues.append('mobile phone use')
    if parameters.get('distraction', {}).get('status') in ('HIGH', 'MODERATE', 'OBSERVED'):
        issues.append('distraction')
    if parameters.get('leaving', {}).get('status') == 'OBSERVED':
        issues.append('crew left position')
    if parameters.get('writing', {}).get('status') == 'OBSERVED':
        issues.append('writing during duty')
    if parameters.get('packing', {}).get('status') == 'OBSERVED':
        issues.append('packing belongings')
    if parameters.get('horn', {}).get('status') == 'OBSERVED':
        issues.append('horn operation')
    if parameters.get('brake_a9', {}).get('status') == 'OBSERVED':
        issues.append('A9/SA9 brake')
    summary = f"Flagged {len(issues)} concern(s): {', '.join(issues)}. Review recommended." if issues else 'Safe crew behaviour. No critical violations.'
    recs = []
    alert_ts = parameters.get('alertness', {}).get('timestamp')
    if parameters.get('alertness', {}).get('status') != 'HIGH':
        recs.append(f"- Review alertness evidence around {alert_ts}.")
    if parameters.get('mobile', {}).get('status') == 'OBSERVED':
        recs.append(f"- Review the mobile-use interval around {parameters['mobile'].get('timestamp')}.")
    if parameters.get('leaving', {}).get('status') == 'OBSERVED':
        recs.append(f"- Review the empty-cabin interval around {parameters['leaving'].get('timestamp')}.")
    if parameters.get('writing', {}).get('status') == 'OBSERVED':
        recs.append(f"- Verify whether paperwork activity around {parameters['writing'].get('timestamp')} was operationally required.")
    if parameters.get('horn', {}).get('status') == 'OBSERVED':
        recs.append(f"- Verify horn operation around {parameters['horn'].get('timestamp')}.")
    if parameters.get('brake_a9', {}).get('status') == 'OBSERVED':
        recs.append(f"- Confirm A9/SA9 brake procedure around {parameters['brake_a9'].get('timestamp')}.")
    if not recs:
        recs.append('- Behaviour compliant. Continue monitoring.')
    return summary, '\n'.join(recs)

def analyze_with_qwen_vl(parameters, frame_analyses, event_candidates_or_map):
    event_candidates = build_event_candidates(frame_analyses, parameters) if isinstance(event_candidates_or_map, dict) else event_candidates_or_map
    selected = select_vlm_events(event_candidates) if ENABLE_VLM else []
    print(f'  Running event-level VLM on {len(selected)} candidates...')
    raw_outputs = []
    for n, cand in enumerate(selected, 1):
        try:
            parsed, raw, ev_idx, evidence_info = run_vlm_on_event(cand, frame_analyses)
            cand.evidence_frame = ev_idx if ev_idx is not None else cand.evidence_frame
            cand.metadata['evidence_source'] = evidence_info.get('source')
            ev_ts = evidence_info.get('timestamp')
            if ev_ts is None and ev_idx is not None and 0 <= ev_idx < len(frame_analyses):
                ev_ts = frame_analyses[ev_idx].get('timestamp')
            cand.metadata['evidence_timestamp_s'] = ev_ts
            if evidence_info.get('frame_item') is not None:
                cand.metadata['evidence_frame_item'] = evidence_info['frame_item']
            cand.reason = parsed.get('reason') or cand.reason
            raw_outputs.append({'candidate': cand.to_dict(), 'raw': raw, 'parsed': parsed})
            fuse_event_score(cand, parsed)
            print(f"[{n}/{len(selected)}] {cand.behavior} t={cand.start_ts}-{cand.end_ts}s local={cand.local_score:.2f} vlm={cand.vlm_score:.2f} final={cand.final_score:.2f} state={cand.metadata.get('attention_state')} -> {cand.decision}")
        except Exception as e:
            cand.metadata['vlm_runtime_note'] = 'VLM verification was unavailable for this event; local evidence was used.'
            cand.metadata['vlm_error'] = str(e)
            fuse_event_score(cand, None)
            print(f'    [{n}/{len(selected)}] {cand.behavior}: VLM error - {e}')
    for cand in event_candidates:
        if cand not in selected:
            fuse_event_score(cand, None)
    parameters = apply_event_fusion(parameters, event_candidates, frame_analyses)
    for cand in event_candidates:
        pk = BEHAVIOR_TO_PARAM.get(cand.behavior)
        if pk in parameters and cand.metadata.get('vlm_runtime_note') and parameters[pk].get('status') == cand.decision:
            parameters[pk]['vlm_runtime_note'] = cand.metadata['vlm_runtime_note']
            parameters[pk]['description'] = _behavior_description(pk, parameters[pk]['status'], cand, parameters[pk])
    summary, recs = build_final_summary(parameters)
    return parameters, summary, recs, event_candidates, raw_outputs

print('Event-level VLM and fusion defined.')

# ── CELL 15: PDF report generation ────────────────────────────────────────────
from fpdf import FPDF

def strip_emoji(t):
    return t.encode('latin-1', 'ignore').decode('latin-1') if t else t

def _fmt_report_seconds(seconds):
    if seconds is None:
        return 'N/A'
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:02d}'

class CrewReportPDF(FPDF):
    def __init__(self, video_info):
        super().__init__()
        self._vi = video_info
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(30, 64, 175)
        self.cell(0, 10, 'Crew Behavior Monitoring Report', align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 9)
        self.set_text_color(120, 120, 120)
        vi = self._vi
        self.cell(
            0,
            5,
            strip_emoji(
                f"Generated: {vi.get('analysis_time', 'N/A')}  |  File: {vi.get('filename', 'N/A')}  "
                f"|  Duration: {vi.get('duration', 0)}s  |  Frames: {vi.get('frames_analyzed', 0)}"
            ),
            align='C',
            new_x='LMARGIN',
            new_y='NEXT'
        )
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Qwen2.5-VL + YOLOv8 + MediaPipe  |  Page {self.page_no()}/{{nb}}', align='C')

def generate_pdf_report(parameters, overall_summary, recommendations, video_info):
    pdf = CrewReportPDF(video_info)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    def status_color(key, st):
        s = st.upper()
        if s == 'LOW' or (key in ('nap', 'mobile') and s == 'OBSERVED'):
            return (220, 38, 38)
        if s in ('NOT_OBSERVED', 'NONE') or (key == 'alertness' and s == 'HIGH'):
            return (34, 197, 94)
        if s == 'OBSERVED':
            return (59, 130, 246)
        return (245, 158, 11)

    for n, (key, p) in enumerate(parameters.items(), 1):
        if pdf.get_y() > 220:
            pdf.add_page()
        pdf.set_fill_color(24, 30, 50)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, strip_emoji(f'  #{n}  {p.get("label", key)}'), fill=True, new_x='LMARGIN', new_y='NEXT')
        st = p.get('status', 'N/A')
        r, g, b = status_color(key, st)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(r, g, b)
        pdf.cell(22, 7, '  Status: ', new_x='END')
        pdf.cell(0, 7, st, new_x='LMARGIN', new_y='NEXT')

        if p.get('description'):
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5, strip_emoji(p['description']), new_x='LMARGIN', new_y='NEXT')

        details = [f"Confidence: {p.get('confidence', 'N/A')}"]
        if p.get('evidence_timestamp_s') is not None:
            details.append(f"Evidence time: {_fmt_report_seconds(p.get('evidence_timestamp_s'))}")
        if p.get('event_duration_s'):
            details.append(f"Event duration: {_fmt_report_seconds(p.get('event_duration_s'))}")
        if p.get('evidence_source'):
            details.append(f"Evidence source: {str(p.get('evidence_source')).replace('_', ' ')}")
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(0, 5, '  |  '.join(details), new_x='LMARGIN', new_y='NEXT')

        img_url = p.get('image_url')
        if img_url:
            img_path = str(BASE_DIR / img_url.lstrip("/"))
            if os.path.exists(img_path):
                try:
                    if pdf.get_y() > 210:
                        pdf.add_page()
                    pdf.image(img_path, x=15, w=80, h=55)
                    pdf.ln(2)
                except Exception:
                    pass
        pdf.ln(4)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    if overall_summary:
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, '  Overall Safety Assessment', fill=True, new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(50, 50, 50)
        pdf.ln(2)
        pdf.multi_cell(0, 5, strip_emoji(overall_summary), new_x='LMARGIN', new_y='NEXT')
        pdf.ln(4)

    if recommendations:
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, '  Recommendations', fill=True, new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(50, 50, 50)
        pdf.ln(2)
        pdf.multi_cell(0, 5, strip_emoji(recommendations), new_x='LMARGIN', new_y='NEXT')

    pdf_path = f'{DETECTION_DIR}/crew_monitoring_report.pdf'
    pdf.output(pdf_path)
    return pdf_path

# =========================
# FastAPI UI wrapper using main_optimized.py template style
# =========================
app = FastAPI(title="Loco Pilot Crew Behavior Monitoring")
TEMPLATE_DIR = BASE_DIR / "templates"
TEMPLATE_INDEX = TEMPLATE_DIR / "index.html"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/detections", StaticFiles(directory=DETECTION_DIR), name="detections")

last_report = {
    "parameters": {},
    "overall_summary": "",
    "recommendations": "",
    "video_info": {},
    "result_text": "",
    "pdf_path": None,
}

def _reset_runtime_dirs():
    for d in [FRAME_DIR, ANNOTATED_DIR, DETECTION_DIR, CROP_DIR, PROFILE_DIR]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

def run_local_pipeline(video_path: str, video_filename: str):
    global PROFILER
    _reset_runtime_dirs()
    PROFILER = PipelineProfiler()
    pipeline_start = time.time()

    print("=" * 60)
    print("STEP 1: Frame extraction")
    frames, fps, duration = extract_frames(video_path)

    print("=" * 60)
    print("STEP 2: Local CV (batched YOLO + parallel MediaPipe)")
    PROFILER.start("local_cv_total")
    frame_analyses = run_full_local_analysis(frames)
    PROFILER.stop({"frames": len(frame_analyses)})
    print(f"  Faces detected: {sum(f['face_analysis']['faces_detected'] for f in frame_analyses)}")
    print(f"  Phone frames : {sum(1 for f in frame_analyses if f['phone_detected'])}")
    print(f"  Horn frames  : {sum(1 for f in frame_analyses if f['horn_active_detected'])}")

    print("=" * 60)
    print("STEP 3: Rule-based parameter computation")
    PROFILER.start("rule_engine")
    parameters, local_summary, local_recs = compute_parameters_locally(frame_analyses)
    PROFILER.stop()
    for k, p in parameters.items():
        icon = 'ALERT' if p['status'] in ('LOW', 'OBSERVED') else '  ok'
        print(f"  [{icon}] {p['label']}: {p['status']} (conf={p['confidence']})")

    print("=" * 60)
    print("STEP 4: Event candidate generation")
    PROFILER.start("event_candidate_generation")
    event_candidates = build_event_candidates(frame_analyses, parameters)
    event_candidates = attach_dense_event_frames(video_path, event_candidates, duration)
    PROFILER.stop({
        "events": len(event_candidates),
        "dense_events": sum(1 for c in event_candidates if c.metadata.get("dense_frames")),
    })

    print("=" * 60)
    print("STEP 5: Event-level Qwen VLM cross-validation")
    PROFILER.start("vlm_total", sync_cuda=True)
    if ENABLE_VLM:
        parameters, final_summary, final_recs, event_candidates, vlm_raw_outputs = analyze_with_qwen_vl(
            parameters, frame_analyses, event_candidates
        )
    else:
        final_summary, final_recs, vlm_raw_outputs = local_summary, local_recs, []
    PROFILER.stop({
        "selected_events": sum(1 for c in event_candidates if c.needs_vlm),
        "max_events": VLM_MAX_EVENTS,
    })

    video_info = {
        "filename": video_filename,
        "duration": duration,
        "fps": fps,
        "frames_analyzed": len(frames),
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 60)
    print("STEP 6: Generating PDF report")
    PROFILER.start("pdf")
    pdf_path = generate_pdf_report(parameters, final_summary, final_recs, video_info)
    PROFILER.stop()
    total_time = time.time() - pipeline_start
    PROFILER.add("TOTAL", total_time)

    profile_payload = PROFILER.report()
    profile_payload.update({
        'video_info': video_info,
        'final_parameters': parameters,
        'event_candidates': [c.to_dict() for c in event_candidates],
        'vlm_raw_outputs': vlm_raw_outputs,
    })
    profile_path = f'{PROFILE_DIR}/analysis_profile.json'
    with open(profile_path, 'w') as f:
        json.dump(profile_payload, f, indent=2)

    print("=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for row in profile_payload['timings']:
        print(f"  {row['stage']:<24}: {row['seconds']:7.2f}s ({row['calls']} calls)")
    print()
    print("FINAL RESULTS:")
    for k, p in parameters.items():
        icon = 'ALERT' if p['status'] in ('LOW', 'OBSERVED') else '  ok'
        print(f"  [{icon}] {p['label']}: {p['status']} ({p['confidence']})")
    print()
    print(f"SUMMARY: {final_summary}")
    print(f"PDF: {pdf_path}")
    print(f"Profile JSON: {profile_path}")
    print(f"Pipeline complete in {total_time:.1f}s")

    result_text = f"Analysis complete in {total_time:.1f}s. {final_summary}"
    return parameters, final_summary, final_recs, video_info, result_text, pdf_path

def render_index(request: Request, **context):
    ctx = {"request": request}
    ctx.update(context)
    if TEMPLATE_INDEX.exists():
        return templates.TemplateResponse("index.html", ctx)
    return HTMLResponse(build_fallback_html(ctx))

def _esc(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build_fallback_html(ctx):
    parameters = ctx.get("parameters") or {}
    video_info = ctx.get("video_info") or {}
    cards = []
    for key, p in parameters.items():
        img = p.get("image_url")
        img_html = f'<img src="/{_esc(img)}" alt="evidence">' if img else ""
        cards.append(f"""
        <section class='card'>
          <div class='row'><h3>{_esc(p.get('label', key))}</h3><span>{_esc(p.get('status', 'N/A'))}</span></div>
          <p>{_esc(p.get('description', ''))}</p>
          <small>Confidence: {_esc(p.get('confidence', 'N/A'))} | Time: {_esc(p.get('timestamp', 'N/A'))}</small>
          {img_html}
        </section>""")
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>Loco Pilot Crew Monitoring</title><style>
    body{{margin:0;background:#f4f7fb;color:#111827;font-family:Arial,sans-serif}}.wrap{{max-width:1120px;margin:auto;padding:30px 20px}}
    h1{{margin:0 0 16px}}form,.panel,.card{{background:white;border:1px solid #dbe3ef;border-radius:8px;padding:16px}}form{{display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
    input[type=file]{{flex:1;min-width:260px}}button,.btn{{background:#1d4ed8;color:white;border:0;border-radius:6px;padding:10px 14px;text-decoration:none}}
    .panel{{margin:16px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}}.row{{display:flex;justify-content:space-between;gap:10px}}
    .row h3{{margin:0;font-size:16px}}.row span{{background:#e0ecff;color:#1d4ed8;border-radius:999px;padding:4px 8px;font-size:12px}}img{{width:100%;border-radius:6px;margin-top:10px}}
    pre{{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;padding:12px;border-radius:6px;overflow:auto}}</style></head><body><main class='wrap'>
    <h1>Loco Pilot Crew Monitoring</h1>
    <form action='/analyze' method='post' enctype='multipart/form-data'><input type='file' name='video' accept='video/*' required><button type='submit'>Analyze Video</button><a class='btn' href='/download-report'>Download PDF</a></form>
    <section class='panel'><strong>{_esc(video_info.get('filename',''))}</strong><p>{_esc(ctx.get('overall_summary') or 'Upload a video to start analysis.')}</p><pre>{_esc(ctx.get('recommendations') or ctx.get('result') or '')}</pre></section>
    <section class='grid'>{''.join(cards)}</section></main></body></html>"""

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render_index(request)

@app.post("/analyze")
async def analyze_video(request: Request, video: UploadFile = File(...)):
    upload_path = BASE_DIR / f"temp_{int(time.time())}_{video.filename}"
    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        parameters, summary, recs, video_info, result_text, pdf_path = run_local_pipeline(str(upload_path), video.filename)
    except Exception as exc:
        traceback.print_exc()
        parameters, summary, recs, video_info, result_text, pdf_path = {}, "", "", {"filename": video.filename}, f"Error: {exc}", None
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass

    last_report.update({
        "parameters": parameters,
        "overall_summary": summary,
        "recommendations": recs,
        "video_info": video_info,
        "result_text": result_text,
        "pdf_path": pdf_path,
    })
    return render_index(
        request,
        result=result_text,
        parameters=parameters,
        overall_summary=summary,
        recommendations=recs,
        video_info=video_info,
    )

@app.get("/download-report")
async def download_report():
    pdf_path = last_report.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        return HTMLResponse("<h3>No report available. Analyze a video first.</h3>", status_code=400)
    vi = last_report.get("video_info", {})
    filename = f"crew_report_{Path(vi.get('filename', 'video')).stem}.pdf"
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))