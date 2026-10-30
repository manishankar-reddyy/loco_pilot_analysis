
"""Event candidate generation and VLM slot selection."""
from __future__ import annotations

import cv2
from .actuators import _actuator_is_stable, _actuator_run_stats
from .config import VLM_MAX_EVENTS
from .frames import _read_frame
from .models import EventCandidate
from .profiler import PROFILER
from .utils import _mid, _runs, _ts


def _det_conf(fa, names, source='yolo_detections'):
    vals = [d['confidence'] for d in fa.get(source, []) if d['class_name'] in names]
    return max(vals) if vals else 0.0

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

    paper_scores = {i: _paper_roi_signal(fa.get('frame_item') or fa.get('frame_path')) for i, fa in enumerate(frame_analyses)}
    paper_like = [i for i, s in paper_scores.items() if s > 0.48]
    writing_seed = sorted((set(book) & set(gaze_down)) | (set(paper_like) & set(gaze_down)) | (set(book) & set(paper_like)))
    for run in _runs(writing_seed, max_gap=2):
        conf = max([_det_conf(frame_analyses[i], {'book', 'laptop'}) for i in run] or [0.0])
        gaze_overlap = len(set(run) & set(gaze_down)) / max(1, len(run))
        paper_score = max([paper_scores[i] for i in run] or [0.0])
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
