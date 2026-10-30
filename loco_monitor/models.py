"""Data classes shared across stages."""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field


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
