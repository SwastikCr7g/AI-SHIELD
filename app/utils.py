import os
import uuid
import shutil
from typing import List
from werkzeug.utils import secure_filename

# 🚀 PRO FEATURE: OpenCV is core for deepfake frame extraction
try:
    import cv2
    import numpy as np
except Exception as e:
    cv2 = None

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# 🎯 Pre-loading Face Detector (Haar Cascade) for focused forensic scanning
face_cascade = None
if cv2 is not None:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)


def is_allowed_file(filename: str, allowed: set) -> bool:
    """Checks if the uploaded file extension is supported."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed


def save_upload(file_storage, upload_dir: str) -> str:
    """Save upload with unique ID and ensures directory exists."""
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)

    filename = secure_filename(file_storage.filename)
    _, ext = os.path.splitext(filename)
    # UUID prevents filename collision in high-traffic scans
    unique = f"{uuid.uuid4().hex[:12]}{ext}"
    path = os.path.join(upload_dir, unique)
    file_storage.save(path)
    return path


def is_frame_blurry(frame, threshold: float = 70.0) -> bool:
    """
    RESEARCH UPGRADE: Strict Blur Detection.
    Increased threshold (70.0) to filter out motion blur which hides AI grid artifacts.
    """
    if cv2 is None: return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Laplacian variance measures the edge sharpness (high = sharp)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def extract_video_frames(video_path: str, out_dir: str, max_frames: int = 20) -> List[str]:
    """
    Next-Gen Research Upgrade: Focused ROI (Region of Interest) Extraction.
    Prioritizes face regions for Deepfake forensics while maintaining data integrity.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV not found. Install via: pip install opencv-python")

    if not os.path.isfile(video_path):
        return []

    # 🚨 Cleanup extracted frames to prevent result overlap
    frame_dir = os.path.join(out_dir, "extracted_frames")
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    # 🛡️ SMART SAMPLING: Skip 10% from start and end to avoid transitions/logos
    start_buffer = int(total_frames * 0.1)
    end_buffer = int(total_frames * 0.9)
    scan_range = end_buffer - start_buffer

    # Sample density increased to find the highest quality face frames
    step = max(scan_range // (max_frames * 2), 1)

    saved = []
    current_frame = start_buffer

    while len(saved) < max_frames and current_frame < end_buffer:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        # Check for frame sharpness
        if not is_frame_blurry(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect Faces for localized deepfake analysis
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

            if len(faces) > 0:
                # 🎯 Crop and Extract Face (The area where Deepfakes are most prominent)
                for (x, y, w, h) in faces:
                    # Dynamic padding to preserve surrounding texture for FFT check
                    p = int(w * 0.25)
                    y1, y2 = max(0, y - p), min(frame.shape[0], y + h + p)
                    x1, x2 = max(0, x - p), min(frame.shape[1], x + w + p)
                    roi = frame[y1:y2, x1:x2]

                    name = f"roi_scan_{len(saved)}_{uuid.uuid4().hex[:6]}.jpg"
                    out_path = os.path.join(frame_dir, name)
                    cv2.imwrite(out_path, roi)
                    saved.append(out_path)
                    break  # Focus on the dominant face per frame
            else:
                # 🖼️ Fallback: Save full frame if it's high quality but no face detected
                name = f"full_frame_{len(saved)}_{uuid.uuid4().hex[:6]}.jpg"
                out_path = os.path.join(frame_dir, name)
                cv2.imwrite(out_path, frame)
                saved.append(out_path)

        current_frame += step

    cap.release()

    # Emergency Fallback if no frames were saved during scan
    if not saved and total_frames > 0:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ok, frame = cap.read()
        if ok:
            out_path = os.path.join(frame_dir, "fallback_mid_frame.jpg")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
        cap.release()

    return saved