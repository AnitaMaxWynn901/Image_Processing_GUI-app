# Human Action Recognition (Image Processing Toolbox) — Python Edition

This repo provides a **Tkinter GUI** that integrates three modules:
1. **Module 1 — Image Enhancement**
2. **Module 2 — Segmentation & Edge Detection**
3. **Module 3 — Geometric Transformations & Interpolation**

> This replaces the MATLAB instruction with a Python workflow.

## Recommended Python Version
Use **Python 3.10** (or 3.11). Tested with 3.10.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

## Project Structure
```
image_toolbox/
├─ app.py                      # Tkinter GUI
├─ requirements.txt
├─ README.md
└─ modules/
   ├─ __init__.py
   ├─ module1_enhance.py       # Enhancement functions
   ├─ module2_segment.py       # Segmentation & edges
   └─ module3_transform.py     # Geometric xforms & interpolation
```

## Notes
- Place demo images into `assets/` (or load any image at runtime).
- You can **replace** functions in `modules/` with your own implementations as long as signatures remain the same.
- Histogram **matching** uses `skimage.exposure.match_histograms` (optional feature).
# Image_Processing_GUI-app
