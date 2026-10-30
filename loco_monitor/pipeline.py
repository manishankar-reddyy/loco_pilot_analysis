"""End-to-end pipeline run for one video."""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from .config import ANNOTATED_DIR, CROP_DIR, DETECTION_DIR, ENABLE_VLM, FRAME_DIR, PROFILE_DIR, VLM_MAX_EVENTS
from .events import build_event_candidates
from .frames import attach_dense_event_frames, extract_frames
from .fusion import analyze_with_qwen_vl
from .profiler import PROFILER
from .report import generate_pdf_report
from .rules import compute_parameters_locally
from .vision import run_full_local_analysis


def _reset_runtime_dirs():
    for d in [FRAME_DIR, ANNOTATED_DIR, DETECTION_DIR, CROP_DIR, PROFILE_DIR]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

def _preprocess_video(video_path: str) -> str:
    """Fix broken H.264 metadata using ffmpeg stream-copy so frame seeking is deterministic."""
    import subprocess
    fixed_path = video_path + '_fixed.mp4'
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-fflags', '+genpts+igndts', '-i', video_path,
             '-c', 'copy', '-avoid_negative_ts', 'make_zero', fixed_path],
            capture_output=True, timeout=300
        )
        if result.returncode == 0 and os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 0:
            print(f'  Video metadata fixed → using {os.path.basename(fixed_path)}')
            return fixed_path
    except Exception as e:
        print(f'  ffmpeg fix skipped ({e}); using original')
    return video_path

def run_local_pipeline(video_path: str, video_filename: str):
    _reset_runtime_dirs()
    PROFILER.reset()
    pipeline_start = time.time()

    print("=" * 60)
    print("STEP 1: Frame extraction")
    video_path = _preprocess_video(video_path)
    frames, fps, duration = extract_frames(video_path)

    print("=" * 60)
    print("STEP 2: Local CV (batched YOLO + parallel MediaPipe)")
    PROFILER.start("local_cv_total")
    frame_analyses = run_full_local_analysis(frames)
    PROFILER.stop({"frames": len(frame_analyses)})
    print(f"  Faces detected: {sum(f['face_analysis']['faces_detected'] for f in frame_analyses)}")
    print(f"  Phone frames : {sum(1 for f in frame_analyses if f['phone_detected'])}")
    print(f"  Horn frames  : {sum(1 for f in frame_analyses if f['horn_active_detected'])}")

    print("=" * 60)
    print("STEP 3: Rule-based parameter computation")
    PROFILER.start("rule_engine")
    parameters, local_summary, local_recs = compute_parameters_locally(frame_analyses)
    PROFILER.stop()
    for k, p in parameters.items():
        icon = 'ALERT' if p['status'] in ('LOW', 'OBSERVED') else '  ok'
        print(f"  [{icon}] {p['label']}: {p['status']} (conf={p['confidence']})")

    print("=" * 60)
    print("STEP 4: Event candidate generation")
    PROFILER.start("event_candidate_generation")
    event_candidates = build_event_candidates(frame_analyses, parameters)
    event_candidates = attach_dense_event_frames(video_path, event_candidates, duration)
    PROFILER.stop({
        "events": len(event_candidates),
        "dense_events": sum(1 for c in event_candidates if c.metadata.get("dense_frames")),
    })

    print("=" * 60)
    print("STEP 5: Event-level Qwen VLM cross-validation")
    PROFILER.start("vlm_total", sync_cuda=True)
    if ENABLE_VLM:
        parameters, final_summary, final_recs, event_candidates, vlm_raw_outputs = analyze_with_qwen_vl(
            parameters, frame_analyses, event_candidates
        )
    else:
        final_summary, final_recs, vlm_raw_outputs = local_summary, local_recs, []
    PROFILER.stop({
        "selected_events": sum(1 for c in event_candidates if c.needs_vlm),
        "max_events": VLM_MAX_EVENTS,
    })

    video_info = {
        "filename": video_filename,
        "duration": duration,
        "fps": fps,
        "frames_analyzed": len(frames),
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 60)
    print("STEP 6: Generating PDF report")
    PROFILER.start("pdf")
    pdf_path = generate_pdf_report(parameters, final_summary, final_recs, video_info)
    PROFILER.stop()
    total_time = time.time() - pipeline_start
    PROFILER.add("TOTAL", total_time)

    profile_payload = PROFILER.report()
    profile_payload.update({
        'video_info': video_info,
        'final_parameters': parameters,
        'event_candidates': [c.to_dict() for c in event_candidates],
        'vlm_raw_outputs': vlm_raw_outputs,
    })
    profile_path = f'{PROFILE_DIR}/analysis_profile.json'
    with open(profile_path, 'w') as f:
        json.dump(profile_payload, f, indent=2)

    print("=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for row in profile_payload['timings']:
        print(f"  {row['stage']:<24}: {row['seconds']:7.2f}s ({row['calls']} calls)")
    print()
    print("FINAL RESULTS:")
    for k, p in parameters.items():
        icon = 'ALERT' if p['status'] in ('LOW', 'OBSERVED') else '  ok'
        print(f"  [{icon}] {p['label']}: {p['status']} ({p['confidence']})")
    print()
    print(f"SUMMARY: {final_summary}")
    print(f"PDF: {pdf_path}")
    print(f"Profile JSON: {profile_path}")
    print(f"Pipeline complete in {total_time:.1f}s")

    result_text = f"Analysis complete in {total_time:.1f}s. {final_summary}"
    return parameters, final_summary, final_recs, video_info, result_text, pdf_path
