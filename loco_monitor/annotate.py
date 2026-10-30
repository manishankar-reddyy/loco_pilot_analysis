"""Annotated frames and evidence images."""
from __future__ import annotations

import cv2
import os
import time as _time
from .config import ANNOTATED_DIR, DETECTION_DIR, YOLO_COLORS
from .frames import _read_frame, materialize_frame
from .profiler import PROFILER


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
