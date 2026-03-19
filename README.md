# AI SHIELD: Next-Gen Deepfake & AI Image Detector 🛡️

AI SHIELD is a digital forensic web application designed to detect AI-generated images and video deepfakes. Developed during my AI/ML internship, this tool uses a **Hybrid Forensic Engine** that combines neural networks with mathematical pixel-DNA analysis.

## 🚀 Key Features
- **Video Deepfake Detection**: Scans video frames for temporal inconsistencies and facial warping.
- **FFT Analysis**: Detects periodic "checkerboard" artifacts in the frequency domain.
- **ELA (Error Level Analysis)**: Identifies compression inconsistencies in manipulated regions.
- **Sensor Noise Analysis**: Measures Laplacian variance to distinguish real camera grain from AI smoothness.
- **Metadata DNA Scan**: Traces digital signatures from Midjourney, DALL-E, and Photoshop.

## 🧠 Technical Architecture
The system operates on a **Four-Layer Defense Matrix**:
1. **Neural Layer**: Uses `EfficientNet-B0` (via `timm`) to extract deep semantic features.
2. **Frequency Layer (FFT)**: Converts pixels into frequencies to find artificial grid patterns.
3. **Forensic Layer (ELA)**: Re-compresses images to find high-error manipulation zones.
4. **Noise Layer**: Checks for physical sensor noise signature (PRNU-inspired).

## 🛠️ Installation & Setup

1. **Clone the repo**:
   ```bash
   git clone [https://github.com/SwastikCr7g/AI-SHIELD.git](https://github.com/SwastikCr7g/AI-SHIELD.git)
   cd AI-SHIELD
Create Virtual Environment:

Bash
python -m venv venv
source venv/bin/scripts/activate  # For Windows: venv\Scripts\activate
Install Dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
python run.py
📊 Accuracy Metrics
Optimized to minimize False Positives on high-detail real-world photography (e.g., sports, outdoor scenes) by prioritizing sensor noise signatures.

👤 Author
Swastik Gahukar AI/ML Intern | Nagpur, India

