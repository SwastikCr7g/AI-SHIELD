import os
from flask import Blueprint, render_template, request, current_app, jsonify
from PIL import Image
from .utils import (
    is_allowed_file,
    save_upload,
    extract_video_frames,
    ALLOWED_IMAGE_EXT,
    ALLOWED_VIDEO_EXT,
)
from .services.local_model import classify_image_file

main = Blueprint("main", __name__)

def get_image_metadata(path):
    """Scans image DNA for AI signatures in metadata."""
    try:
        img = Image.open(path)
        info = img.info
        metadata_findings = []
        ai_sigs = ['midjourney', 'dall-e', 'stable diffusion', 'adobe firefly', 'photoshop', 'gan', 'invoked']

        for key, value in info.items():
            content = str(value).lower()
            for sig in ai_sigs:
                if sig in content:
                    metadata_findings.append(f"Trace Found: {sig.upper()}")
        return metadata_findings
    except:
        return []

@main.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@main.route("/detect-image", methods=["GET"])
def detect_image_page():
    return render_template("detect_image.html", result=None, meta=None, error=None)

@main.route("/detect-image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return render_template("detect_image.html", error="No file uploaded.")

    file = request.files["image"]
    if not file or file.filename == "":
        return render_template("detect_image.html", error="No file selected.")

    if not is_allowed_file(file.filename, ALLOWED_IMAGE_EXT):
        return render_template("detect_image.html", error="Invalid file type.")

    saved_path = save_upload(file, current_app.config["UPLOAD_FOLDER"])

    try:
        meta = get_image_metadata(saved_path)
        res = classify_image_file(saved_path)

        # 🛡️ Hybrid Decision Override
        if meta or res.get('fft_score', 0) > 9.5:
            res['verdict'] = "⚠️ AI-GENERATED (Forensic Match)"
            res['risk'] = "Critical"
            res['color'] = "#ff4d4d"
            res['confidence'] = max(res['confidence'], 99.12)

        forensic_tags = []
        if meta: forensic_tags.extend(meta)
        if 'fft_score' in res: forensic_tags.append(f"FFT Grid: {res['fft_score']}")
        if 'ela_score' in res: forensic_tags.append(f"ELA Stability: {res['ela_score']}")
        if 'noise_score' in res: forensic_tags.append(f"Noise Level: {res['noise_score']}")

        return render_template("detect_image.html", result=res, meta=forensic_tags, filename=os.path.basename(saved_path))
    except Exception as e:
        return render_template("detect_image.html", error=f"Analysis Failed: {e}")

@main.route("/detect-video", methods=["GET"])
def detect_video_page():
    return render_template("detect_video.html", verdict=None, timeline=None, error=None)

@main.route("/detect-video", methods=["POST"])
def detect_video():
    if "video" not in request.files:
        return render_template("detect_video.html", error="No file uploaded.")

    file = request.files["video"]
    saved_path = save_upload(file, current_app.config["UPLOAD_FOLDER"])

    try:
        frames = extract_video_frames(saved_path, current_app.config["UPLOAD_FOLDER"], max_frames=20)
        if not frames:
            return render_template("detect_video.html", error="Could not extract clear frames.")

        frame_confidences = []
        ai_votes = 0
        total_fft = 0
        total_noise = 0

        for fp in frames:
            r = classify_image_file(fp)
            frame_confidences.append(r['confidence'])
            total_fft += r.get('fft_score', 0)
            total_noise += r.get('noise_score', 0)

            # Voting Logic
            if (r['risk'] in ["Critical", "High"] or r.get('fft_score', 0) > 9.0) and r.get('noise_score', 0) < 180:
                ai_votes += 1

        avg_fft = total_fft / len(frames)
        avg_noise = total_noise / len(frames)
        ai_ratio = ai_votes / len(frames)

        if (ai_ratio >= 0.25 or avg_fft > 10.0) and avg_noise < 250:
            verdict = "⚠️ DEEPFAKE DETECTED"
            v_color = "#ff4d4d"
        else:
            verdict = "✅ AUTHENTIC (LIKELY REAL)"
            v_color = "#00f2ff"

        return render_template("detect_video.html", verdict=verdict, verdict_color=v_color, timeline=frame_confidences, ratio=round(ai_ratio * 100, 2))
    except Exception as e:
        return render_template("detect_video.html", error=f"Forensic Error: {e}")