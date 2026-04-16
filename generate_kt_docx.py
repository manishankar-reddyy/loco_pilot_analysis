from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def add_heading(doc, text, level):
    h = doc.add_heading(text, level)
    return h

def add_paragraph(doc, text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    return p

def add_bullet(doc, text, bold_start=None, rest_text=None):
    p = doc.add_paragraph(style='List Bullet')
    if bold_start:
        p.add_run(bold_start).bold = True
        if rest_text:
            p.add_run(rest_text)
    else:
        p.add_run(text)
    return p

def main():
    doc = Document()

    # Title
    title = doc.add_heading('Knowledge Transfer Document: Loco Pilot Crew Behavior Monitoring', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 1. Project Overview
    doc.add_heading('1. Project Overview', level=1)
    doc.add_paragraph('The Loco Pilot Crew Behavior Monitoring project is an AI-powered safety monitoring system that analyzes CCTV footage from locomotive cabs. It detects 9 critical crew behaviors—ranging from napping and mobile phone usage to horn and brake operation—using a multi-tiered Artificial Intelligence pipeline (Local CV + LLM Validation).')

    # 2. Tech Stack & Architecture
    doc.add_heading('2. Tech Stack & Architecture', level=1)
    add_bullet(doc, None, 'Web Framework: ', 'FastAPI, Uvicorn, Jinja2 templates (Frontend).')
    
    p = doc.add_paragraph(style='List Bullet')
    p.add_run('Computer Vision (Local Edge processing):').bold = True
    add_bullet(doc, 'OpenCV (Image processing & fallback DNN face detection).').paragraph_format.left_indent = Inches(0.5)
    add_bullet(doc, 'MediaPipe Tasks API (FaceLandmarker for precise EAR, yawning, and head pose estimation).').paragraph_format.left_indent = Inches(0.5)
    add_bullet(doc, 'Ultralytics YOLOv8 (for object detection like persons, phones, bags).').paragraph_format.left_indent = Inches(0.5)
    
    add_bullet(doc, None, 'Custom Object Detection: ', 'Custom-trained YOLOv8 model for Loco cabin specifics (Horn & Brake application).')
    add_bullet(doc, None, 'Vision-Language Model (VLM): ', 'Google Gemini (gemini-2.5-flash) for multi-modal contextual cross-validation of behaviors.')
    add_bullet(doc, None, 'Reporting: ', 'Python fpdf2 for dynamic PDF report generation.')
    add_bullet(doc, None, 'Containerization: ', 'Docker & Docker Compose (python:3.11-slim).')

    doc.add_paragraph()

    # 3. The 3-Tier AI Analysis Pipeline  
    doc.add_heading('3. The 3-Tier AI Analysis Pipeline', level=1)
    doc.add_paragraph('The core of the system lies in main.py which triggers when a video is posted to the /analyze endpoint.')

    doc.add_heading('Tier 1: Local Computer Vision Extraction (Dense Sampling)', level=2)
    add_bullet(doc, None, 'Frame Extraction: ', 'Extracts frames every 4 seconds, and every 1 second in the last 8 seconds.')
    add_bullet(doc, None, 'YOLO Detection (yolov8s.pt): ', 'Detects standard COCO objects like person, cell_phone, backpack, book, etc.')
    add_bullet(doc, None, 'Custom Brake/Horn YOLO (brake_horn_detection.pt): ', 'Detects horn_active, brake_applied, etc.')
    add_bullet(doc, None, 'MediaPipe FaceLandmarker: ', 'Crops YOLO person detections and passes them to MediaPipe to extract:')
    
    add_bullet(doc, None, 'EAR (Eye Aspect Ratio) & Blink Blendshapes: ', 'For sleep/micro-sleep detection.').paragraph_format.left_indent = Inches(0.5)
    add_bullet(doc, None, 'MAR (Mouth Aspect Ratio): ', 'For yawn detection.').paragraph_format.left_indent = Inches(0.5)
    add_bullet(doc, None, 'SolvePnP Head Pose: ', 'Yaw, Pitch, Roll to detect looking away (distraction) or drooping head.').paragraph_format.left_indent = Inches(0.5)

    doc.add_heading('Tier 2: Local Rule-Based Engine', level=2)
    doc.add_paragraph('Calculates parameters locally, analyzing sequences over the frames:')
    add_bullet(doc, 'Validates the camera baselines (calculating median yaw/pitch to adjust for side-mounted CCTV cameras).')
    add_bullet(doc, 'Tracks person count drops across consecutive frames to flag "Leaving Place".')
    add_bullet(doc, 'Matches "hands-on horn bounding boxes" to flag Horn Operations.')

    doc.add_heading('Tier 3: LLM Cross-Validation (Gemini)', level=2)
    doc.add_paragraph('To reduce false positives and capture complex context, frames are passed to gemini-2.5-flash every 7 seconds.')
    add_bullet(doc, 'It uses a specialized highly descriptive prompt (see GEMINI_PER_FRAME_PROMPT in main.py).')
    add_bullet(doc, 'It parses structured LLM feedback (e.g., PARAMETER: VALUE | reason).')
    add_bullet(doc, 'Final statuses are a consensus of YOLO/MediaPipe alerts + Gemini context (especially for Phone usage, Alertness, and Distraction).')
    add_bullet(doc, 'Custom Brake & Horn triggers rely purely on the specialized local model since VLMs struggle with specific loco mechanics.')

    doc.add_paragraph()

    # 4. Key Behaviors Monitored (The 9 Parameters)
    doc.add_heading('4. Key Behaviors Monitored (The 9 Parameters)', level=1)
    doc.add_paragraph('The system actively scores:')
    doc.add_paragraph('1. Crew Alertness: Blended score of Eye Closure (PERCLOS), drooping head, and yawning.', style='List Number')
    doc.add_paragraph('2. Distraction: Sustained gaze away from the camera baseline.', style='List Number')
    doc.add_paragraph('3. Writing: Detection of books/papers combined with downward gaze.', style='List Number')
    doc.add_paragraph('4. Nap/Micro-sleep: Consecutive frames with eyes closed.', style='List Number')
    doc.add_paragraph('5. Mobile Use: YOLO cell phone detection + Gemini active usage verification.', style='List Number')
    doc.add_paragraph('6. Packing belongings: Detecting handling of bags/backpacks.', style='List Number')
    doc.add_paragraph('7. Leaving Place: Drops in person count (from 2 to 1 or 0).', style='List Number')
    doc.add_paragraph('8. Horn Lever Operation: Custom YOLO detection of horn_active.', style='List Number')
    doc.add_paragraph('9. A9/SA9 Brake Application: Custom YOLO detection of brake_applied.', style='List Number')

    doc.add_paragraph()

    # 5. Model Training Pipeline
    doc.add_heading('5. Model Training Pipeline', level=1)
    doc.add_paragraph('The repository includes dedicated scripts for retraining the Brake & Horn custom model on new cab layouts:')

    doc.add_heading('prepare_yolo_dataset.py', level=2)
    add_bullet(doc, 'Converts standard LabelMe JSON annotations into YOLO normalized bounding boxes.')
    add_bullet(doc, 'Class mapping: horn -> horn_active, no_brake -> brake_released.')
    add_bullet(doc, 'Shuffles and automatically splits data (80% Train, 20% Val).')
    add_bullet(doc, 'Creates the necessary data.yaml automatically.')

    doc.add_heading('train_yolo.py', level=2)
    add_bullet(doc, 'Training script utilizing Ultralytics YOLOv8 engine.')
    add_bullet(doc, 'Base Model: yolov8s.pt (Small variant offers a good speed/accuracy trade-off).')
    add_bullet(doc, 'Embedded augmentations geared toward Loco cabs (Hue mapping, minor 5-degree rotation, NO vertical flip since cab orientation is fixed).')
    add_bullet(doc, 'Model outputs save to ./runs/brake_horn_detection/weights/best.pt.')

    doc.add_paragraph()

    # 6. Setup & Deployment Instructions
    doc.add_heading('6. Setup & Deployment Instructions', level=1)

    doc.add_heading('Prerequisites', level=2)
    add_bullet(doc, 'Docker & Docker Compose')
    add_bullet(doc, 'Google Gemini API Key')

    doc.add_heading('Running Locally (Docker)', level=2)
    doc.add_paragraph('1. Add your Gemini API key to a .env file (or pass it via environment):', style='List Number')
    doc.add_paragraph('   GEMINI_API_KEY=your_key_here')
    doc.add_paragraph('2. Start the container:', style='List Number')
    doc.add_paragraph('   docker-compose up -d --build')
    doc.add_paragraph('3. Access the web interface at: http://localhost:8000', style='List Number')

    doc.add_heading('Important File Paths', level=2)
    add_bullet(doc, None, 'Uploaded Videos: ', 'Mounted to ./videos')
    add_bullet(doc, None, 'Output Detections / Images: ', 'Mounted to ./detections')

    doc.add_paragraph()

    # 7. Troubleshooting & Maintenance Tips
    doc.add_heading('7. Troubleshooting & Maintenance Tips', level=1)
    add_bullet(doc, None, 'MediaPipe Failing to Load Faces: ', 'The model relies on CCTV resolution. The pipeline automatically tries cropping standard YOLO person boundaries and upscaling them to 400x400px before feeding to MediaPipe to ensure accurate eye/mouth extraction.')
    add_bullet(doc, None, 'High False Positive Mobile Usage: ', 'Tweak GEMINI_PER_FRAME_PROMPT to restrict Gemini from flagging objects that look like phones (e.g., reflective handles).')
    
    p = doc.add_paragraph(style='List Bullet')
    p.add_run('Adding New Safety Parameters:').bold = True
    doc.add_paragraph('1. Add UI Config in PARAMETER_CONFIG dict.', style='List Number').paragraph_format.left_indent = Inches(0.5)
    doc.add_paragraph('2. Implement CV rule logic in compute_parameters_locally.', style='List Number').paragraph_format.left_indent = Inches(0.5)
    doc.add_paragraph('3. Expand the GEMINI_PER_FRAME_PROMPT prompt format.', style='List Number').paragraph_format.left_indent = Inches(0.5)

    add_bullet(doc, None, 'Reporting Issues: ', 'Formatting problems in the PDF are usually due to non-Latin-1 characters breaking fpdf2. Keep the strip_emoji function up-to-date.')

    # Save
    docx_path = r'd:\Projects\loco_pilot_analysis\Loco_Pilot_KT_Document.docx'
    doc.save(docx_path)
    print(f'Document saved to {docx_path}')

if __name__ == "__main__":
    main()
