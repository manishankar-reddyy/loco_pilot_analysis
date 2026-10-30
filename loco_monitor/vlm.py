"""Qwen2.5-VL loading, prompting and event-level verification."""
from __future__ import annotations

import cv2
import json
import math
import re
import torch
from PIL import Image
from .annotate import ensure_annotated_frame
from .config import ENABLE_VLM, VLM_CONTEXT_IMAGES, VLM_IMAGE_SIZE, VLM_MAX_TOKENS, VLM_MODEL_ID
from .frames import _read_frame
from .profiler import PROFILER


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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
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


def preload_vlm() -> None:
    """Load the VLM at startup (eager, as in the original notebook); never crash the app."""
    # Load VLM at startup (matches notebook Cell 7 — eager load)
    if ENABLE_VLM:
        try:
            load_vlm_model()
        except Exception as _vlm_err:
            # Keep the app up (e.g. no GPU); each event retries and falls back to local-only evidence.
            print(f'WARNING: VLM preload failed ({_vlm_err}). Analysis will fall back to local CV evidence.')
