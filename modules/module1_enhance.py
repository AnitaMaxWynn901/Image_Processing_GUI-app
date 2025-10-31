import numpy as np
import cv2
from skimage import exposure

def _to_bgr(img):
    if img is None:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def to_gray(img):
    if img is None:
        return img
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def hist_equalize(img):
    """Apply histogram equalization (per-channel for color)."""
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    # YCrCb approach: equalize Y only
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def hist_match(img, reference):
    """Histogram matching using skimage; if reference is None, return img."""
    if img is None or reference is None:
        return img
    matched = exposure.match_histograms(img, reference, channel_axis=-1 if img.ndim==3 else None)
    if img.ndim == 3:
        matched = np.clip(matched, 0, 255).astype(np.uint8)
    return matched

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """alpha: contrast (>1 increases contrast); beta: brightness bias"""
    if img is None:
        return None
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out

def mean_filter(img, ksize=3):
    if img is None:
        return None
    return cv2.blur(img, (ksize, ksize))

def mean_filter(img, ksize=3):
    """Apply mean (average) filter for noise reduction."""
    if img is None:
        return None
    return cv2.blur(img, (ksize, ksize))


def median_filter(img, ksize=3):
    """Apply median filter (odd kernel only)."""
    if img is None:
        return None
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)
    if len(img.shape) == 3:
        channels = cv2.split(img)
        filtered_channels = [cv2.medianBlur(ch, ksize) for ch in channels]
        return cv2.merge(filtered_channels)
    else:
        return cv2.medianBlur(img, ksize)


def laplacian_sharpen(img):
    if img is None:
        return None
    gray = to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    sharp = cv2.addWeighted(_to_bgr(img), 1.0, _to_bgr(lap), 0.7, 0)
    return sharp

def high_pass_filter(img):
    if img is None:
        return None
    kernel = np.array([[ -1,-1,-1],
                       [ -1, 8,-1],
                       [ -1,-1,-1]], dtype=np.float32)
    hp = cv2.filter2D(img, -1, kernel)
    return cv2.convertScaleAbs(hp)

def _fft2_gray(img):
    gray = to_gray(img).astype(np.float32)
    dft = np.fft.fftshift(np.fft.fft2(gray))
    return gray, dft

def _ifft2_to_uint8(dft):
    img_back = np.fft.ifft2(np.fft.ifftshift(dft))
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255)
    return img_back.astype(np.uint8)

def _ideal_lowpass_mask(shape, d0):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    mask = (dist <= d0).astype(np.float32)
    return mask

def _ideal_highpass_mask(shape, d0):
    return 1.0 - _ideal_lowpass_mask(shape, d0)

def _ideal_bandpass_mask(shape, d_low, d_high):
    return _ideal_lowpass_mask(shape, d_high) * _ideal_highpass_mask(shape, d_low)

def freq_lowpass(img, d0=30):
    if img is None:
        return None
    gray, dft = _fft2_gray(img)
    mask = _ideal_lowpass_mask(gray.shape, d0)
    filtered = dft * mask
    out = _ifft2_to_uint8(filtered)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

def freq_highpass(img, d0=30):
    if img is None:
        return None
    gray, dft = _fft2_gray(img)
    mask = _ideal_highpass_mask(gray.shape, d0)
    filtered = dft * mask
    out = _ifft2_to_uint8(filtered)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

def freq_bandpass(img, d_low=10, d_high=60):
    if img is None:
        return None
    gray, dft = _fft2_gray(img)
    mask = _ideal_bandpass_mask(gray.shape, d_low, d_high)
    filtered = dft * mask
    out = _ifft2_to_uint8(filtered)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
