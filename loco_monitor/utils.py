"""Small shared helpers."""
from __future__ import annotations


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

def _event_duration(frame_analyses, indices):
    if not indices:
        return 0
    return max(0, frame_analyses[indices[-1]]['timestamp'] - frame_analyses[indices[0]]['timestamp'])

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
