from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "videos" / "uploads"
DETECTIONS_DIR = BASE_DIR / "detections"
REPORTS_DIR = BASE_DIR / "reports"
YOLO_WEIGHTS = BASE_DIR / "yolov8s.pt"
CUSTOM_WEIGHTS = BASE_DIR / "models" / "brake_horn_detection.pt"

SAMPLE_INTERVAL_SECONDS = 4.0
YOLO_GENERAL_CONF = 0.20
YOLO_BRAKE_HORN_CONF = 0.35
ENABLE_QWEN = os.getenv("ENABLE_QWEN", "1").strip().lower() in {"1", "true", "yes", "on"}
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
AUTO_INSTALL_QWEN = os.getenv("AUTO_INSTALL_QWEN", "1").strip().lower() in {"1", "true", "yes", "on"}
QWEN_LOAD_IN_4BIT = os.getenv("QWEN_LOAD_IN_4BIT", "1").strip().lower() in {"1", "true", "yes", "on"}
QWEN_IMAGE_SIZE = int(os.getenv("QWEN_IMAGE_SIZE", "768"))
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "160"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Loco Pilot Crew Behavior Monitoring")
app.mount("/detections", StaticFiles(directory=str(DETECTIONS_DIR)), name="detections")


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    source: str


@dataclass
class FaceMetrics:
    eye_aspect_ratio: float | None = None
    mouth_aspect_ratio: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None


@dataclass
class VlmSignal:
    parameter: str
    value: str
    reason: str


@dataclass
class FrameObservation:
    frame_index: int
    timestamp: float
    image_path: Path | None = None
    detections: list[Detection] = field(default_factory=list)
    face: FaceMetrics = field(default_factory=FaceMetrics)
    vlm_signals: list[VlmSignal] = field(default_factory=list)


@dataclass
class ParameterResult:
    name: str
    status: str
    score: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    video_path: Path
    report_id: str
    sampled_frames: int
    duration_seconds: float
    parameters: list[ParameterResult]
    observations: list[FrameObservation]
    pdf_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FaceAnalyzer:
    def __init__(self) -> None:
        self.enabled = False
        self.face_mesh = None
        try:
            import mediapipe as mp

            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.45,
            )
            self.enabled = True
        except Exception:
            self.face_mesh = None
            self.enabled = False

    def analyze(self, frame_bgr: np.ndarray) -> FaceMetrics:
        if not self.enabled or self.face_mesh is None:
            return FaceMetrics()
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(image_rgb)
        if not result.multi_face_landmarks:
            return FaceMetrics()
        landmarks = result.multi_face_landmarks[0].landmark
        height, width = frame_bgr.shape[:2]
        points = np.array([(lm.x * width, lm.y * height, lm.z * width) for lm in landmarks], dtype=np.float64)
        left_ear = _eye_aspect_ratio(points, [33, 160, 158, 133, 153, 144])
        right_ear = _eye_aspect_ratio(points, [362, 385, 387, 263, 373, 380])
        yaw, pitch, roll = _head_pose(points, width, height)
        return FaceMetrics(
            eye_aspect_ratio=float((left_ear + right_ear) / 2),
            mouth_aspect_ratio=float(_mouth_aspect_ratio(points)),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )


class QwenFrameValidator:
    prompt = """You are validating one sampled locomotive cab CCTV frame.
Use only visible evidence. If something is unclear, answer uncertain.
Return ONLY valid JSON:
{"attention_state":"alert|reduced_alertness|yawning|drowsy|normal_control_check|writing|non_operational_distraction|uncertain",
"mobile":"yes|no|uncertain","writing":"yes|no|uncertain","packing":"yes|no|uncertain",
"leaving":"yes|no|uncertain","horn":"yes|no|uncertain","brake_a9":"yes|no|uncertain",
"nap":"yes|no|uncertain","confidence_0_1":0.0,"reason":"short reason grounded in the frame"}"""

    def __init__(self, model_id: str = QWEN_MODEL_ID) -> None:
        self.enabled = False
        self.model_id = model_id
        self.model: Any = None
        self.processor: Any = None
        self.process_vision_info: Any = None
        if ENABLE_QWEN:
            self._load()

    def _load(self) -> None:
        _ensure_qwen_dependencies()
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen is enabled, but CUDA GPU is not available. Install CUDA PyTorch or set ENABLE_QWEN=0.")

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLClass
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as QwenVLClass

        print(f"Loading Qwen VLM on GPU: {self.model_id}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        self.model = None
        if QWEN_LOAD_IN_4BIT:
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = QwenVLClass.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    quantization_config=quant_config,
                )
                print("Qwen loaded in 4-bit NF4 mode.")
            except Exception as exc:
                print(f"4-bit Qwen load failed, retrying FP16 GPU load: {exc}")

        if self.model is None:
            self.model = QwenVLClass.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            print("Qwen loaded in FP16 mode.")

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.process_vision_info = process_vision_info
        self.enabled = True

    def validate_frame(self, image_path: Path, context: str) -> dict[str, Any]:
        if not self.enabled:
            return {}
        import torch

        image = self._load_image(image_path)
        prompt = f"{context}\n\n{self.prompt}"
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS, do_sample=False)
        generated = out_ids[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        parsed = _safe_json(raw)
        parsed["raw"] = raw
        return parsed

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read frame for Qwen validation: {image_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).resize((QWEN_IMAGE_SIZE, QWEN_IMAGE_SIZE))


class LocoPilotAnalyzer:
    def __init__(self) -> None:
        self.standard_model = YOLO(str(YOLO_WEIGHTS))
        self.custom_model = YOLO(str(CUSTOM_WEIGHTS)) if CUSTOM_WEIGHTS.exists() else None
        self.face_analyzer = FaceAnalyzer()
        self.qwen = QwenFrameValidator() if ENABLE_QWEN else None

    def analyze_video(self, video_path: Path) -> AnalysisReport:
        report_id = uuid.uuid4().hex[:12]
        report_frame_dir = DETECTIONS_DIR / report_id
        report_frame_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count else 0.0
        observations: list[FrameObservation] = []

        for sample_time in _sample_times(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            detections = self._detect(frame)
            face_metrics = self.face_analyzer.analyze(frame)
            annotated = _annotate_frame(frame.copy(), detections, sample_time, face_metrics.yaw, face_metrics.pitch)
            image_path = report_frame_dir / f"frame_{len(observations):04d}_{sample_time:.1f}s.jpg"
            cv2.imwrite(str(image_path), annotated)

            vlm_signals: list[VlmSignal] = []
            if self.qwen and self.qwen.enabled:
                context = _frame_context(sample_time, detections, face_metrics)
                try:
                    vlm_signals = _signals_from_qwen(self.qwen.validate_frame(image_path, context))
                except Exception as exc:
                    vlm_signals = [VlmSignal(parameter="Qwen", value="uncertain", reason=f"Qwen validation failed: {exc}")]

            observations.append(
                FrameObservation(
                    frame_index=frame_index,
                    timestamp=sample_time,
                    image_path=image_path,
                    detections=detections,
                    face=face_metrics,
                    vlm_signals=vlm_signals,
                )
            )

        cap.release()
        report = AnalysisReport(
            video_path=video_path,
            report_id=report_id,
            sampled_frames=len(observations),
            duration_seconds=duration,
            parameters=compute_parameters(observations),
            observations=observations,
            metadata={
                "standard_model": str(YOLO_WEIGHTS),
                "custom_model": str(CUSTOM_WEIGHTS) if self.custom_model else "not configured",
                "mediapipe_enabled": self.face_analyzer.enabled,
                "qwen_enabled": bool(self.qwen and self.qwen.enabled),
                "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            },
        )
        report.pdf_path = write_pdf_report(report)
        return report

    def _detect(self, frame: np.ndarray) -> list[Detection]:
        detections = _run_yolo(self.standard_model, frame, source="yolov8", conf=YOLO_GENERAL_CONF)
        if self.custom_model is not None:
            detections.extend(_run_yolo(self.custom_model, frame, source="custom_brake_horn", conf=YOLO_BRAKE_HORN_CONF))
        return detections


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Loco Pilot Monitor</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; color: #17202a; background: #f5f7f8; }
    main { max-width: 1100px; margin: 0 auto; padding: 32px 18px; }
    h1 { font-size: 30px; margin: 0 0 10px; }
    p { line-height: 1.5; }
    form { background: white; border: 1px solid #d7dee2; border-radius: 8px; padding: 18px; margin: 22px 0; }
    input, button { font-size: 16px; }
    button { border: 0; border-radius: 8px; background: #146c5f; color: white; padding: 10px 16px; cursor: pointer; }
    .hint { color: #53656d; }
  </style>
</head>
<body>
  <main>
    <h1>Loco Pilot Crew Behavior Monitoring</h1>
    <p>Upload cab CCTV footage to analyze alertness, distraction, phone use, leaving place, horn operation, and brake application.</p>
    <form action="/analyze" method="post" enctype="multipart/form-data">
      <p><input type="file" name="video" accept="video/*" required></p>
      <button type="submit">Analyze Video</button>
      <p class="hint">Runs locally with YOLO + MediaPipe + Qwen2.5-VL on GPU. Set ENABLE_QWEN=0 before starting the server to disable Qwen.</p>
    </form>
  </main>
</body>
</html>
"""


@app.post("/analyze", response_class=HTMLResponse)
def analyze(video: UploadFile = File(...)) -> str:
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with upload_path.open("wb") as file:
        shutil.copyfileobj(video.file, file)

    try:
        report = LocoPilotAnalyzer().analyze_video(upload_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    rows = "\n".join(
        "<tr><td>{}</td><td><strong>{}</strong></td><td>{:.2f}</td><td>{}</td></tr>".format(
            html.escape(item.name),
            html.escape(item.status),
            item.score,
            "<br>".join(html.escape(evidence) for evidence in item.evidence),
        )
        for item in report.parameters
    )
    images = "\n".join(
        '<figure><img src="/detections/{}/{}" alt="Frame at {:.1f}s"><figcaption>{:.1f}s</figcaption></figure>'.format(
            report.report_id,
            html.escape(obs.image_path.name),
            obs.timestamp,
            obs.timestamp,
        )
        for obs in report.observations[:8]
        if obs.image_path
    )
    qwen_enabled = bool(report.metadata.get("qwen_enabled"))
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analysis Result</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #17202a; background: #f5f7f8; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 28px 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee2; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #d7dee2; text-align: left; vertical-align: top; }}
    th {{ background: #e8eeee; }}
    a.button {{ display: inline-block; border-radius: 8px; background: #146c5f; color: white; padding: 10px 14px; text-decoration: none; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 18px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d7dee2; border-radius: 8px; padding: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
    .note {{ color: #53656d; }}
  </style>
</head>
<body>
  <main>
    <h1>Analysis Result</h1>
    <p>Video duration: {report.duration_seconds:.1f}s | Sampled frames: {report.sampled_frames}</p>
    <p class="note">Qwen validation: {"enabled" if qwen_enabled else "disabled"}</p>
    <p><a class="button" href="/reports/{report.report_id}.pdf">Download PDF Report</a> <a href="/">Analyze another video</a></p>
    <table>
      <thead><tr><th>Parameter</th><th>Status</th><th>Score</th><th>Evidence</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <h2>Sampled Frames</h2>
    <div class="grid">{images}</div>
  </main>
</body>
</html>
"""


@app.get("/reports/{report_id}.pdf")
def report_pdf(report_id: str) -> FileResponse:
    pdf_path = REPORTS_DIR / f"{report_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)


def compute_parameters(observations: list[FrameObservation]) -> list[ParameterResult]:
    if not observations:
        return [ParameterResult(name=name, status="UNKNOWN", score=0.0, evidence=["No frames could be sampled."]) for name in _parameter_names()]

    total = len(observations)
    person_counts = [_count_labels(obs, "person") for obs in observations]
    yaw_values = [obs.face.yaw for obs in observations if obs.face.yaw is not None]
    pitch_values = [obs.face.pitch for obs in observations if obs.face.pitch is not None]
    yaw_baseline = median(yaw_values) if yaw_values else 0.0
    pitch_baseline = median(pitch_values) if pitch_values else 0.0

    eye_closed = [obs for obs in observations if obs.face.eye_aspect_ratio is not None and obs.face.eye_aspect_ratio < 0.18]
    partial_eye = [obs for obs in observations if obs.face.eye_aspect_ratio is not None and 0.18 <= obs.face.eye_aspect_ratio < 0.23]
    yawning = [obs for obs in observations if obs.face.mouth_aspect_ratio is not None and obs.face.mouth_aspect_ratio > 0.62]
    drooping = [obs for obs in observations if obs.face.pitch is not None and obs.face.pitch - pitch_baseline > 16]
    distracted = [obs for obs in observations if obs.face.yaw is not None and abs(obs.face.yaw - yaw_baseline) > 25]

    phone_visible = [obs for obs in observations if _has_any(obs, {"cell phone", "cell_phone", "mobile", "phone"})]
    paper_visible = [obs for obs in observations if _has_any(obs, {"book", "paper", "notebook", "laptop"})]
    gaze_down = [obs for obs in observations if obs.face.pitch is not None and obs.face.pitch - pitch_baseline > 8]
    writing_local = _intersection_by_time(paper_visible, gaze_down)
    packing_visible = [obs for obs in observations if _has_any(obs, {"backpack", "handbag", "suitcase", "bag"})]
    zero_person = [obs for obs, count in zip(observations, person_counts) if count == 0]
    horn_local = [obs for obs in observations if _has_any(obs, {"horn_active", "horn", "horn_lever"})]
    brake_local = [obs for obs in observations if _has_any(obs, {"brake_applied", "a9_applied", "sa9_applied", "a9", "sa9"})]

    has_vlm = any(obs.vlm_signals for obs in observations)
    vlm_yes = _vlm_yes_counts(observations)
    vlm_attention = _vlm_attention_counts(observations)

    alert_score = max(
        (len(eye_closed) + len(partial_eye) + len(yawning) + len(drooping)) / max(1, total * 4),
        (vlm_attention["drowsy"] + vlm_attention["reduced_alertness"] + vlm_attention["yawning"]) / max(1, total),
    )
    distraction_score = max(len(distracted) / total, vlm_attention["non_operational_distraction"] / max(1, total))
    mobile_hits = _vlm_confirmed_or_local(vlm_yes["Mobile Use"], phone_visible) if has_vlm else _local_mobile_hits(phone_visible, observations)
    writing_hits = _vlm_confirmed_or_local(vlm_yes["Writing"], writing_local) if has_vlm else _local_writing_hits(writing_local, observations)
    packing_hits = _vlm_confirmed_or_local(vlm_yes["Packing belongings"], packing_visible) if has_vlm else []
    leaving_hits = zero_person if _longest_run([obs in zero_person for obs in observations]) >= 3 else []
    horn_hits = _confirmed_actuator(horn_local, observations, "horn")
    brake_hits = _confirmed_actuator(brake_local, observations, "brake")
    nap_hits = eye_closed if _longest_run([obs in eye_closed for obs in observations]) >= 3 else vlm_yes["Nap/Micro-sleep"]

    return [
        _result("Crew Alertness", "RISK" if alert_score >= 0.18 else "OK", alert_score, _evidence("Alertness cues", eye_closed + yawning + drooping) + _vlm_evidence("Crew Alertness", observations)),
        _result("Distraction", "DETECTED" if distraction_score >= 0.15 else "OK", distraction_score, _evidence("Non-operational attention shift", distracted) + _vlm_evidence("Distraction", observations)),
        _result("Writing", "DETECTED" if writing_hits else "OK", len(writing_hits) / total, _evidence("Paperwork with downward gaze / VLM confirmation", writing_hits) or _negative_paper_evidence(paper_visible)),
        _result("Nap/Micro-sleep", "DETECTED" if nap_hits else "OK", len(nap_hits) / total, _evidence("Closed-eye or VLM sleep cues", nap_hits)),
        _result("Mobile Use", "DETECTED" if mobile_hits else "OK", len(mobile_hits) / total, _evidence("Active phone use confirmed", mobile_hits) or _negative_phone_evidence(phone_visible)),
        _result("Packing belongings", "DETECTED" if len(packing_hits) >= 2 else "OK", len(packing_hits) / total, _evidence("Active packing or bag handling", packing_hits) or _negative_bag_evidence(packing_visible)),
        _result("Leaving Place", "DETECTED" if leaving_hits else "OK", len(leaving_hits) / total, _evidence("Cab empty for consecutive sampled frames", leaving_hits)),
        _result("Horn Lever Operation", "DETECTED" if horn_hits else "OK", len(horn_hits) / total, _evidence("Stable horn operation evidence", horn_hits) or _negative_actuator_evidence(horn_local, "horn")),
        _result("A9/SA9 Brake Application", "DETECTED" if brake_hits else "OK", len(brake_hits) / total, _evidence("Stable brake application evidence", brake_hits) or _negative_actuator_evidence(brake_local, "brake")),
    ]


def write_pdf_report(report: AnalysisReport) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORTS_DIR / f"{report.report_id}.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Loco Pilot Crew Behavior Monitoring Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, _latin(f"Video: {report.video_path.name}"), ln=True)
    pdf.cell(0, 8, f"Duration: {report.duration_seconds:.1f}s | Sampled frames: {report.sampled_frames}", ln=True)
    pdf.cell(0, 8, f"Qwen validation: {'enabled' if report.metadata.get('qwen_enabled') else 'disabled'}", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Behavior Summary", ln=True)
    for parameter in report.parameters:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, _latin(f"{parameter.name}: {parameter.status} ({parameter.score:.2f})"), ln=True)
        pdf.set_font("Helvetica", "", 9)
        for item in parameter.evidence[:4]:
            pdf.multi_cell(0, 5, _latin(f"- {item}"))

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sampled Frame Notes", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for obs in report.observations[:12]:
        labels = ", ".join(sorted({det.label for det in obs.detections})) or "none"
        vlm = ", ".join(f"{s.parameter}:{s.value}" for s in obs.vlm_signals if s.value == "yes")
        suffix = f" | VLM={vlm}" if vlm else ""
        pdf.multi_cell(0, 5, _latin(f"{obs.timestamp:.1f}s: detections={labels}{suffix}"))

    pdf.output(str(pdf_path))
    return pdf_path


def _sample_times(duration: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    times: list[float] = []
    current = 0.0
    while current < duration:
        times.append(round(current, 2))
        current += SAMPLE_INTERVAL_SECONDS
    return times


def _run_yolo(model: YOLO, frame: np.ndarray, source: str, conf: float) -> list[Detection]:
    output: list[Detection] = []
    results = model.predict(frame, verbose=False, conf=conf)
    if not results or results[0].boxes is None:
        return output
    names = results[0].names
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        label = str(names.get(cls_id, cls_id))
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        output.append(Detection(label=label, confidence=confidence, box=(x1, y1, x2, y2), source=source))
    return output


def _annotate_frame(frame: np.ndarray, detections: list[Detection], timestamp: float, yaw: float | None, pitch: float | None) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = (0, 180, 0) if detection.source == "yolov8" else (0, 90, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{detection.label} {detection.confidence:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    pose = f" yaw={yaw:.1f} pitch={pitch:.1f}" if yaw is not None and pitch is not None else ""
    cv2.putText(frame, f"t={timestamp:.1f}s{pose}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return frame


def _eye_aspect_ratio(points: np.ndarray, indexes: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [points[index][:2] for index in indexes]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    return float(vertical / horizontal) if horizontal else 0.0


def _mouth_aspect_ratio(points: np.ndarray) -> float:
    left = points[61][:2]
    right = points[291][:2]
    top = points[13][:2]
    bottom = points[14][:2]
    horizontal = np.linalg.norm(left - right)
    return float(np.linalg.norm(top - bottom) / horizontal) if horizontal else 0.0


def _head_pose(points: np.ndarray, width: int, height: int) -> tuple[float | None, float | None, float | None]:
    image_points = np.array([points[1][:2], points[152][:2], points[33][:2], points[263][:2], points[61][:2], points[291][:2]], dtype=np.float64)
    model_points = np.array(
        [(0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)],
        dtype=np.float64,
    )
    camera_matrix = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
    success, rotation_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4, 1)))
    if not success:
        return None, None, None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy < 1e-6:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _frame_context(timestamp: float, detections: list[Detection], face: FaceMetrics) -> str:
    labels = ", ".join(f"{d.label}:{d.confidence:.2f}" for d in detections) or "none"
    return f"timestamp={timestamp:.1f}s detections=[{labels}] face=[ear={face.eye_aspect_ratio} mar={face.mouth_aspect_ratio} yaw={face.yaw} pitch={face.pitch}]"


def _signals_from_qwen(parsed: dict[str, Any]) -> list[VlmSignal]:
    reason = str(parsed.get("reason") or parsed.get("raw") or "")
    signals: list[VlmSignal] = []
    mapping = {
        "mobile": "Mobile Use",
        "writing": "Writing",
        "packing": "Packing belongings",
        "leaving": "Leaving Place",
        "horn": "Horn Lever Operation",
        "brake_a9": "A9/SA9 Brake Application",
        "nap": "Nap/Micro-sleep",
    }
    for key, name in mapping.items():
        value = str(parsed.get(key, "uncertain")).lower()
        if value in {"yes", "no", "uncertain"}:
            signals.append(VlmSignal(parameter=name, value=value, reason=reason))
    attention = str(parsed.get("attention_state", "uncertain")).lower()
    if attention in {"reduced_alertness", "yawning", "drowsy"}:
        signals.append(VlmSignal(parameter="Crew Alertness", value="yes", reason=f"{attention}: {reason}"))
    if attention == "non_operational_distraction":
        signals.append(VlmSignal(parameter="Distraction", value="yes", reason=reason))
    return signals


def _safe_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {
        "attention_state": "uncertain",
        "mobile": "uncertain",
        "writing": "uncertain",
        "packing": "uncertain",
        "leaving": "uncertain",
        "horn": "uncertain",
        "brake_a9": "uncertain",
        "nap": "uncertain",
        "confidence_0_1": 0.0,
        "reason": text[:220],
    }


def _ensure_qwen_dependencies() -> None:
    missing: list[str] = []
    checks = [
        ("transformers", "transformers>=4.52.0"),
        ("accelerate", "accelerate>=0.34.0"),
        ("qwen_vl_utils", "qwen-vl-utils>=0.0.8"),
    ]
    if QWEN_LOAD_IN_4BIT:
        checks.append(("bitsandbytes", "bitsandbytes>=0.44.0"))

    for module_name, package_spec in checks:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_spec)

    if not missing:
        return
    if not AUTO_INSTALL_QWEN:
        raise RuntimeError(
            "Missing Qwen dependencies: "
            + ", ".join(missing)
            + ". Install them manually or set AUTO_INSTALL_QWEN=1."
        )

    print("Installing missing Qwen dependencies:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def _parameter_names() -> list[str]:
    return ["Crew Alertness", "Distraction", "Writing", "Nap/Micro-sleep", "Mobile Use", "Packing belongings", "Leaving Place", "Horn Lever Operation", "A9/SA9 Brake Application"]


def _result(name: str, status: str, score: float, evidence: list[str]) -> ParameterResult:
    return ParameterResult(name=name, status=status, score=round(max(0.0, min(1.0, score)), 3), evidence=evidence or ["No strong evidence."])


def _count_labels(obs: FrameObservation, label: str) -> int:
    return sum(1 for det in obs.detections if det.label.lower() == label)


def _has_any(obs: FrameObservation, labels: set[str]) -> bool:
    normalized = {item.lower().replace(" ", "_") for item in labels}
    return any(det.label.lower() in labels or det.label.lower().replace(" ", "_") in normalized for det in obs.detections)


def _intersection_by_time(a: list[FrameObservation], b: list[FrameObservation]) -> list[FrameObservation]:
    b_times = {round(obs.timestamp, 2) for obs in b}
    return [obs for obs in a if round(obs.timestamp, 2) in b_times]


def _evidence(prefix: str, frames: list[FrameObservation]) -> list[str]:
    if not frames:
        return []
    shown = ", ".join(f"{frame.timestamp:.1f}s" for frame in frames[:5])
    suffix = "" if len(frames) <= 5 else f" (+{len(frames) - 5} more)"
    return [f"{prefix} at {shown}{suffix}."]


def _vlm_yes_counts(observations: list[FrameObservation]) -> dict[str, list[FrameObservation]]:
    out: dict[str, list[FrameObservation]] = {name: [] for name in _parameter_names()}
    for obs in observations:
        for signal in obs.vlm_signals:
            if signal.value == "yes" and signal.parameter in out:
                out[signal.parameter].append(obs)
    return out


def _vlm_attention_counts(observations: list[FrameObservation]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for obs in observations:
        for signal in obs.vlm_signals:
            if signal.parameter == "Crew Alertness" and signal.value == "yes":
                lower = signal.reason.lower()
                if "drowsy" in lower:
                    counts["drowsy"] += 1
                elif "yawning" in lower:
                    counts["yawning"] += 1
                else:
                    counts["reduced_alertness"] += 1
            if signal.parameter == "Distraction" and signal.value == "yes":
                counts["non_operational_distraction"] += 1
    return counts


def _vlm_evidence(parameter: str, observations: list[FrameObservation]) -> list[str]:
    frames = [obs for obs in observations for signal in obs.vlm_signals if signal.parameter == parameter and signal.value == "yes"]
    return _evidence(f"Qwen validated {parameter.lower()}", frames)


def _vlm_confirmed_or_local(vlm_frames: list[FrameObservation], local_frames: list[FrameObservation]) -> list[FrameObservation]:
    if vlm_frames and local_frames:
        return _intersection_by_time(vlm_frames, local_frames) or vlm_frames
    return vlm_frames


def _local_mobile_hits(phone_visible: list[FrameObservation], observations: list[FrameObservation]) -> list[FrameObservation]:
    indices = [observations.index(obs) for obs in phone_visible if obs in observations]
    if len(indices) >= 3 and _longest_run([i in indices for i in range(len(observations))]) >= 2:
        return phone_visible
    return []


def _local_writing_hits(writing_frames: list[FrameObservation], observations: list[FrameObservation]) -> list[FrameObservation]:
    indices = [observations.index(obs) for obs in writing_frames if obs in observations]
    if len(indices) >= 3 and _longest_run([i in indices for i in range(len(observations))]) >= 2:
        return writing_frames
    return []


def _confirmed_actuator(local_frames: list[FrameObservation], observations: list[FrameObservation], kind: str) -> list[FrameObservation]:
    if not local_frames:
        return []
    labels = {"horn": {"horn_active", "horn", "horn_lever"}, "brake": {"brake_applied", "a9_applied", "sa9_applied", "a9", "sa9"}}[kind]
    runs = _runs([observations.index(obs) for obs in local_frames if obs in observations])
    confirmed: list[FrameObservation] = []
    for run in runs:
        frames = [observations[i] for i in run]
        if len(frames) < 3:
            continue
        stats = _actuator_stats(frames, labels)
        if kind == "brake":
            ok = stats["max_conf"] >= 0.78 and stats["median_iou"] >= 0.22 and stats["center_span_x"] <= 0.18 and stats["center_span_y"] <= 0.18
        else:
            ok = stats["max_conf"] >= 0.72 and stats["median_iou"] >= 0.18 and stats["center_span_x"] <= 0.22 and stats["center_span_y"] <= 0.22
        if ok:
            confirmed.extend(frames)
    return confirmed


def _runs(indices: list[int], max_gap: int = 1) -> list[list[int]]:
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


def _actuator_stats(frames: list[FrameObservation], labels: set[str]) -> dict[str, float]:
    boxes: list[tuple[int, int, int, int]] = []
    confs: list[float] = []
    centers: list[tuple[float, float]] = []
    for obs in frames:
        dets = [d for d in obs.detections if d.label.lower().replace(" ", "_") in labels]
        if not dets:
            continue
        det = max(dets, key=lambda item: item.confidence)
        x1, y1, x2, y2 = det.box
        boxes.append(det.box)
        confs.append(det.confidence)
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    if not boxes:
        return {"max_conf": 0.0, "median_iou": 0.0, "center_span_x": 999.0, "center_span_y": 999.0}
    ious = [_bbox_iou(boxes[i - 1], boxes[i]) for i in range(1, len(boxes))]
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    width_norm = max(1.0, max(x2 for _, _, x2, _ in boxes))
    height_norm = max(1.0, max(y2 for _, _, _, y2 in boxes))
    return {"max_conf": max(confs), "median_iou": median(ious) if ious else 0.0, "center_span_x": (max(xs) - min(xs)) / width_norm, "center_span_y": (max(ys) - min(ys)) / height_norm}


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def _longest_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _negative_phone_evidence(frames: list[FrameObservation]) -> list[str]:
    return _evidence("Phone-like object visible locally, active use not confirmed", frames)


def _negative_paper_evidence(frames: list[FrameObservation]) -> list[str]:
    return _evidence("Paper/document visible locally, writing not confirmed", frames)


def _negative_bag_evidence(frames: list[FrameObservation]) -> list[str]:
    return _evidence("Bags visible but active packing not confirmed", frames)


def _negative_actuator_evidence(frames: list[FrameObservation], name: str) -> list[str]:
    return _evidence(f"Brief {name} detections were not stable enough to confirm action", frames)


def _latin(text: str) -> str:
    return text.encode("latin-1", "ignore").decode("latin-1")
