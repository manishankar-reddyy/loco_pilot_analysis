"""FastAPI app: upload, results page, report download."""
from __future__ import annotations

import os
import threading
import traceback
import uuid
from fastapi import FastAPI
from fastapi import File
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from .config import ALLOWED_VIDEO_EXTS, BASE_DIR, DETECTION_DIR, MAX_UPLOAD_MB
from .pipeline import run_local_pipeline
from .vlm import preload_vlm


# Load the VLM at startup (eager, as in the original notebook).
preload_vlm()


# =========================
# FastAPI UI wrapper using main_optimized.py template style
# =========================
app = FastAPI(title="Loco Pilot Crew Behavior Monitoring")

TEMPLATE_DIR = BASE_DIR / "templates"

TEMPLATE_INDEX = TEMPLATE_DIR / "index.html"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

app.mount("/detections", StaticFiles(directory=DETECTION_DIR), name="detections")

last_report = {
    "parameters": {},
    "overall_summary": "",
    "recommendations": "",
    "video_info": {},
    "result_text": "",
    "pdf_path": None,
}

def render_index(request: Request, **context):
    ctx = {"request": request}
    ctx.update(context)
    if TEMPLATE_INDEX.exists():
        return templates.TemplateResponse("index.html", ctx)
    return HTMLResponse(build_fallback_html(ctx))

def _esc(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build_fallback_html(ctx):
    parameters = ctx.get("parameters") or {}
    video_info = ctx.get("video_info") or {}
    cards = []
    for key, p in parameters.items():
        img = p.get("image_url")
        img_html = f'<img src="/{_esc(img)}" alt="evidence">' if img else ""
        cards.append(f"""
        <section class='card'>
          <div class='row'><h3>{_esc(p.get('label', key))}</h3><span>{_esc(p.get('status', 'N/A'))}</span></div>
          <p>{_esc(p.get('description', ''))}</p>
          <small>Confidence: {_esc(p.get('confidence', 'N/A'))} | Time: {_esc(p.get('timestamp', 'N/A'))}</small>
          {img_html}
        </section>""")
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>Loco Pilot Crew Monitoring</title><style>
    body{{margin:0;background:#f4f7fb;color:#111827;font-family:Arial,sans-serif}}.wrap{{max-width:1120px;margin:auto;padding:30px 20px}}
    h1{{margin:0 0 16px}}form,.panel,.card{{background:white;border:1px solid #dbe3ef;border-radius:8px;padding:16px}}form{{display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
    input[type=file]{{flex:1;min-width:260px}}button,.btn{{background:#1d4ed8;color:white;border:0;border-radius:6px;padding:10px 14px;text-decoration:none}}
    .panel{{margin:16px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}}.row{{display:flex;justify-content:space-between;gap:10px}}
    .row h3{{margin:0;font-size:16px}}.row span{{background:#e0ecff;color:#1d4ed8;border-radius:999px;padding:4px 8px;font-size:12px}}img{{width:100%;border-radius:6px;margin-top:10px}}
    pre{{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;padding:12px;border-radius:6px;overflow:auto}}</style></head><body><main class='wrap'>
    <h1>Loco Pilot Crew Monitoring</h1>
    <form action='/analyze' method='post' enctype='multipart/form-data'><input type='file' name='video' accept='video/*' required><button type='submit'>Analyze Video</button><a class='btn' href='/download-report'>Download PDF</a></form>
    <section class='panel'><strong>{_esc(video_info.get('filename',''))}</strong><p>{_esc(ctx.get('overall_summary') or 'Upload a video to start analysis.')}</p><pre>{_esc(ctx.get('recommendations') or ctx.get('result') or '')}</pre></section>
    <section class='grid'>{''.join(cards)}</section></main></body></html>"""

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render_index(request)

# Pipeline state (frames/, detections/, last_report) is shared, so only one analysis runs at a time.
_pipeline_lock = threading.Lock()

# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so the long analysis
# no longer blocks the event loop (health checks and the UI stay responsive).
@app.post("/analyze")
def analyze_video(request: Request, video: UploadFile = File(...)):
    safe_name = Path(video.filename or "video").name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return HTMLResponse(f"<h3>Unsupported file type '{_esc(ext)}'.</h3>", status_code=400)
    if not _pipeline_lock.acquire(blocking=False):
        return HTMLResponse("<h3>Another analysis is already running. Please retry when it finishes.</h3>", status_code=429)

    upload_path = BASE_DIR / f"temp_{uuid.uuid4().hex}{ext}"
    try:
        written, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with open(upload_path, "wb") as f:
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise ValueError(f"Upload exceeds {MAX_UPLOAD_MB} MB limit")
                f.write(chunk)
        parameters, summary, recs, video_info, result_text, pdf_path = run_local_pipeline(str(upload_path), safe_name)
    except Exception as exc:
        traceback.print_exc()
        parameters, summary, recs, video_info, result_text, pdf_path = {}, "", "", {"filename": safe_name}, f"Error: {exc}", None
    finally:
        # _preprocess_video() may leave a '<upload>_fixed.mp4' copy next to the upload; remove both.
        for p in (upload_path, Path(str(upload_path) + "_fixed.mp4")):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        _pipeline_lock.release()

    last_report.update({
        "parameters": parameters,
        "overall_summary": summary,
        "recommendations": recs,
        "video_info": video_info,
        "result_text": result_text,
        "pdf_path": pdf_path,
    })
    return render_index(
        request,
        result=result_text,
        parameters=parameters,
        overall_summary=summary,
        recommendations=recs,
        video_info=video_info,
    )

@app.get("/download-report")
async def download_report():
    pdf_path = last_report.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        return HTMLResponse("<h3>No report available. Analyze a video first.</h3>", status_code=400)
    vi = last_report.get("video_info", {})
    filename = f"crew_report_{Path(vi.get('filename', 'video')).stem}.pdf"
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)
