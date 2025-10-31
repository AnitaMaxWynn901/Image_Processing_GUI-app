import numpy as np
import cv2

def to_gray(img):
    if img is None:
        return img
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def sobel_edges(img):
    g = to_gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag

def prewitt_edges(img):
    g = to_gray(img).astype(np.float32)
    kx = np.array([[ -1, 0, 1],
                   [ -1, 0, 1],
                   [ -1, 0, 1]], dtype=np.float32)
    ky = kx.T
    gx = cv2.filter2D(g, -1, kx)
    gy = cv2.filter2D(g, -1, ky)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag

def canny_edges(img, t1=100, t2=200):
    g = to_gray(img)
    return cv2.Canny(g, t1, t2)

def otsu_threshold(img):
    g = to_gray(img)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def color_segmentation_hsv(img, lower=(35,50,50), upper=(85,255,255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    seg = cv2.bitwise_and(img, img, mask=mask)
    return seg

def morphology(img, op="dilate", ksize=3, iterations=1):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    if op == "dilate":
        res = cv2.dilate(g, kernel, iterations=iterations)
    elif op == "erode":
        res = cv2.erode(g, kernel, iterations=iterations)
    elif op == "open":
        res = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == "close":
        res = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        res = g
    return res
