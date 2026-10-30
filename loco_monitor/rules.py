"""Rule-based parameter engine (tier 2)."""
from __future__ import annotations

from .actuators import _actuator_is_stable, _actuator_run_stats
from .annotate import ensure_annotated_frame, save_evidence_image
from .utils import _event_duration, _mid, _runs_local, find_max_consecutive
from .vision import BRAKE_HORN_AVAILABLE


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
            src = ensure_annotated_frame(frame_analyses, idx)
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
