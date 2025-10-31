import numpy as np
import cv2

interp_map = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC
}

def translate(img, tx=0, ty=0):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))

def scale(img, sx=1.0, sy=1.0, method="bilinear"):
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(w*sx)), max(1, int(h*sy))
    return cv2.resize(img, (new_w, new_h), interpolation=interp_map.get(method, cv2.INTER_LINEAR))

def rotate(img, angle_deg=0, method="bilinear"):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=interp_map.get(method, cv2.INTER_LINEAR))

def affine(img, src_pts, dst_pts):
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))

def resize_interpolate(img, new_w, new_h, method="bilinear"):
    return cv2.resize(img, (int(new_w), int(new_h)), interpolation=interp_map.get(method, cv2.INTER_LINEAR))

def radial_correction(img, k1=-1e-6, k2=0.0):
    """Simple radial distortion correction model: x-prime = x*(1 + k1*r^2 + k2*r^4)."""
    h, w = img.shape[:2]
    cx, cy = w/2.0, h/2.0
    fx = fy = max(w, h)  # crude focal length assumption
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, K, dist, None, newK)
