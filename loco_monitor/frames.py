"""Frame sampling and decoding."""
from __future__ import annotations

import cv2
import os
import time
from .config import DENSE_EVENT_BEHAVIORS, DENSE_EVENT_FPS, DENSE_EVENT_WINDOW_SECONDS, END_DENSE_SECONDS, FRAME_DIR, KEEP_FRAMES_IN_MEMORY, MAX_FRAMES, SAMPLE_INTERVAL, VLM_MAX_EVENTS, WRITE_EXTRACTED_FRAMES
from .profiler import PROFILER


def _read_frame(frame_item):
    if isinstance(frame_item, dict):
        if frame_item.get('image') is not None:
            return frame_item['image']
        if frame_item.get('frame_path'):
            return cv2.imread(frame_item['frame_path'])
    if isinstance(frame_item, str):
        return cv2.imread(frame_item)
    return None

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
