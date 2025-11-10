# 🧠 Human Action Recognition (Image Processing Toolbox) — Python Edition

This project provides a **Tkinter GUI application** for image processing and human action recognition.  
It integrates four modular components, replacing the original MATLAB instructions with a **fully Python-based workflow**.

---

## 📦 Modules Overview
1. **Module 1 — Image Enhancement**  
   Histogram equalization, histogram matching, smoothing (mean/median), and sharpening (Laplacian, high-pass), including frequency domain filtering.

2. **Module 2 — Segmentation & Edge Detection**  
   Implements Otsu thresholding, Sobel/Prewitt/Canny edge detection, HSV-based color segmentation, and morphological operations (dilate, erode, open, close).

3. **Module 3 — Geometric Transformations & Interpolation**  
   Supports translation, scaling, rotation, and radial distortion correction with multiple interpolation methods.

4. **Module 4 — Human Action Recognition (NEW)**  
   Uses **MediaPipe Pose** to detect and label human actions:
   - **Standing**
   - **Sitting**
   - **Squatting**
   - **Meditation**
   - **Hand-Up**
   - (Detects “No person” if none found)

---

## 🐍 Recommended Python Version
Use **Python 3.11** (tested on macOS).  
> Note: Python 3.13 is currently **not supported** by MediaPipe.

---

## ⚙️ Setup Instructions
```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the GUI app
python app.py
