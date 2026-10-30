"""YOLO + MediaPipe inference, face metrics (EAR/MAR/head pose) and person tracking."""
from __future__ import annotations

import cv2
import math
import mediapipe as mp
import numpy as np
import os
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from .config import BRAKE_HORN_CLASSES, BRAKE_HORN_MODEL_PATH, FACE_LANDMARKER_PATH, FACE_LANDMARKER_URL, FACE_ROI_MIN_SIZE, KEEP_FRAMES_IN_MEMORY, MEDIAPIPE_MIN_PERSON_CONF, MEDIAPIPE_THREADS, YOLO_BATCH_SIZE, YOLO_CLASSES
from .frames import _read_frame, materialize_frame
from .profiler import PROFILER
from .utils import _bbox_iou


if not os.path.exists(FACE_LANDMARKER_PATH):
    print("Downloading face_landmarker.task (~3.7 MB)...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
    print(f"face_landmarker.task ready ({os.path.getsize(FACE_LANDMARKER_PATH)/1e6:.1f} MB)")

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
        ear = round((el+er)/2, 3) if el is not None and er is not None else None
        eye = 'CLOSED' if ear is not None and ear<0.18 else 'PARTIALLY_CLOSED' if ear is not None and ear<0.23 else 'OPEN'
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

def _worker_face(args):
    idx, ts, frame_item, yolo_dets = args
    lm = get_thread_landmarker()
    return idx, analyze_face_on_crops(frame_item, yolo_dets, ts, lm)

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
