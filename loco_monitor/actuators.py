"""Horn / A9 brake detection stability checks."""
from __future__ import annotations

from .frames import _read_frame
from .utils import _bbox_iou


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
