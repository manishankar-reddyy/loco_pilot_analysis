"""Fusion of local CV scores and VLM verdicts into final parameters and summary."""
from __future__ import annotations

import cv2
import re
import time
from collections import defaultdict
from .annotate import ensure_annotated_frame, save_evidence_image
from .config import DETECTION_DIR, ENABLE_VLM, FUSION_THRESHOLDS
from .events import build_event_candidates, select_vlm_events
from .vlm import ATTENTION_STATES, run_vlm_on_event


BEHAVIOR_TO_PARAM = {
    'alertness': 'alertness', 'distraction': 'distraction', 'writing': 'writing', 'nap': 'nap',
    'mobile': 'mobile', 'packing': 'packing', 'leaving': 'leaving', 'horn': 'horn', 'brake_a9': 'brake_a9',
}

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
    selected_ids = {id(c) for c in selected}
    for cand in event_candidates:
        if id(cand) not in selected_ids:
            fuse_event_score(cand, None)
    parameters = apply_event_fusion(parameters, event_candidates, frame_analyses)
    for cand in event_candidates:
        pk = BEHAVIOR_TO_PARAM.get(cand.behavior)
        if pk in parameters and cand.metadata.get('vlm_runtime_note') and parameters[pk].get('status') == cand.decision:
            parameters[pk]['vlm_runtime_note'] = cand.metadata['vlm_runtime_note']
            parameters[pk]['description'] = _behavior_description(pk, parameters[pk]['status'], cand, parameters[pk])
    summary, recs = build_final_summary(parameters)
    return parameters, summary, recs, event_candidates, raw_outputs
