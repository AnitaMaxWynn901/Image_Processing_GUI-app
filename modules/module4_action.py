# modules/module4_action.py
# Fast-start version: preload MediaPipe Pose once, reuse for all calls.

import os
# Quiet TensorFlow/MediaPipe logs in terminal (nice for demos)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# If startup is still slow on your machine, uncomment to force pure CPU:
# os.environ["MEDIAPIPE_DISABLE_TFLITE_DELEGATES"] = "1"
# If you want to avoid GPU init on macOS/Metal, uncomment:
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# -------------------- Pose model (preloaded once) --------------------
# Load at import time to avoid delays during the first button click.
# model_complexity=1 is a good balance, set to 0 for fastest or 2 for best.
_pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def _pt(landmarks, name):
    """Return (x, y) for a PoseLandmark name, in normalized [0..1] coords."""
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return np.array([lm.x, lm.y], dtype=np.float32)

def _angle_deg(a, b, c):
    """Angle ABC in degrees using 2D points (numpy arrays)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1) + 1e-6
    n2 = np.linalg.norm(v2) + 1e-6
    cosang = np.clip((v1 @ v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _draw_label(img_bgr, text, org=(30, 50)):
    """Draw label with a filled background for readability."""
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    cv2.rectangle(img_bgr, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                  (0, 0, 0), thickness=-1)
    cv2.putText(img_bgr, text, (x, y), font, scale, (0, 255, 0), thick, cv2.LINE_AA)
    return img_bgr

def detect_pose_and_label(img_bgr):
    """
    Rule-based single-image labeling with these classes:
      - Hand-Up     (any wrist above shoulders)
      - Squatting   (knee angle small; hips lowered)
      - Meditation  (sitting + hands resting near lap)
      - Sitting     (hips close to knees)
      - Standing    (fallback upright)
      - No person detected
    Returns: (annotated_bgr_image, label_str)
    """
    global _pose

    # Convert once; MediaPipe expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Reuse the preloaded pose model (fast)
    try:
        res = _pose.process(img_rgb)
    except Exception:
        # In rare cases MP can drop; recreate once to recover gracefully.
        _pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        res = _pose.process(img_rgb)

    if not res.pose_landmarks:
        return _draw_label(img_bgr, "No person detected"), "No person detected"

    # Draw landmarks
    mp_drawing.draw_landmarks(img_bgr, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    lms = res.pose_landmarks.landmark
    # Keypoints (2D, normalized)
    L_SH = _pt(lms, "LEFT_SHOULDER")
    R_SH = _pt(lms, "RIGHT_SHOULDER")
    L_WR = _pt(lms, "LEFT_WRIST")
    R_WR = _pt(lms, "RIGHT_WRIST")
    L_HIP = _pt(lms, "LEFT_HIP")
    R_HIP = _pt(lms, "RIGHT_HIP")
    L_KNEE = _pt(lms, "LEFT_KNEE")
    R_KNEE = _pt(lms, "RIGHT_KNEE")
    L_ANK = _pt(lms, "LEFT_ANKLE")
    R_ANK = _pt(lms, "RIGHT_ANKLE")

    SH_AVG = (L_SH + R_SH) * 0.5
    HIP_AVG = (L_HIP + R_HIP) * 0.5
    KNEE_AVG = (L_KNEE + R_KNEE) * 0.5

    # ---------- 1) HAND-UP ----------
    y_sh_min = min(L_SH[1], R_SH[1])
    if min(L_WR[1], R_WR[1]) < y_sh_min - 0.05:
        return _draw_label(img_bgr, "Action: Hand-Up"), "Hand-Up"

    # ---------- Helper measures ----------
    ang_l_knee = _angle_deg(L_HIP, L_KNEE, L_ANK)
    ang_r_knee = _angle_deg(R_HIP, R_KNEE, R_ANK)
    knee_ang_min = min(ang_l_knee, ang_r_knee)

    hip_knee_gap = abs(((L_HIP[1] + R_HIP[1]) * 0.5) - ((L_KNEE[1] + R_KNEE[1]) * 0.5))
    hip_ank_gap  = abs(((L_HIP[1] + R_HIP[1]) * 0.5) - ((L_ANK[1] + R_ANK[1]) * 0.5))

    lap_center = (HIP_AVG + KNEE_AVG) * 0.5
    d_wrist_lap = min(np.linalg.norm(L_WR - lap_center),
                      np.linalg.norm(R_WR - lap_center))

    # ---------- 2) SQUATTING ----------
    if (knee_ang_min < 110.0) and (hip_ank_gap < 0.35) and (hip_knee_gap < 0.20):
        return _draw_label(img_bgr, "Action: Squatting"), "Squatting"

    # ---------- 3) MEDITATION ----------
    shoulders_level = abs(L_SH[1] - R_SH[1]) < 0.05
    sitting_like = hip_knee_gap < 0.12
    hands_in_lap = d_wrist_lap < 0.12
    if sitting_like and hands_in_lap and shoulders_level:
        return _draw_label(img_bgr, "Action: Meditation"), "Meditation"

    # ---------- 4) SITTING ----------
    if hip_knee_gap < 0.08:
        return _draw_label(img_bgr, "Action: Sitting"), "Sitting"

    # ---------- 5) Default ----------
    return _draw_label(img_bgr, "Action: Standing"), "Standing"



# Optional helpers you can call from app.py if you want explicit control:
def warmup_pose():
    """Call once at app startup to ensure the model is ready before the demo."""
    # A tiny black image just to trigger interpreter init
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    _ = _pose.process(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))

def close_pose():
    """Release resources explicitly when exiting the app."""
    global _pose
    if _pose is not None:
        _pose.close()
        _pose = None
