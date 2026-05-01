"""OpenCV-based statistical anomaly detector.

Combines four normalised signals:
  - Laplacian variance  -> low value = blur / focus defect
  - Canny edge density  -> very low or very high vs. baseline
  - Hot/dead pixel ratio (extreme luminance)
  - Color saturation spread (HSV S-channel std)

Returns a 0-1 anomaly score plus a heatmap (uint8 numpy array).
"""

from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np


def _normalise(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def score_opencv(img_bgr: np.ndarray) -> Tuple[float, np.ndarray, dict]:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("empty image")

    # standardise size
    h, w = img_bgr.shape[:2]
    scale = 512 / max(h, w) if max(h, w) > 512 else 1.0
    if scale != 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) blur
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_signal = 1.0 - _normalise(lap_var, 30.0, 800.0)         # less variance -> more blurry -> higher anomaly

    # 2) edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean()) / 255.0
    # treat both extremes as anomalous (target ~0.05-0.20)
    edge_signal = abs(edge_density - 0.12) / 0.12
    edge_signal = float(np.clip(edge_signal, 0.0, 1.0))

    # 3) hot/dead pixel ratio
    hot = float((gray > 250).mean())
    dead = float((gray < 5).mean())
    hotpix_ratio = hot + dead
    hotpix_signal = _normalise(hotpix_ratio, 0.001, 0.10)

    # 4) saturation spread
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_std = float(hsv[:, :, 1].std()) / 255.0
    sat_signal = abs(sat_std - 0.18) / 0.18
    sat_signal = float(np.clip(sat_signal, 0.0, 1.0))

    score = float(np.clip(
        0.40 * blur_signal + 0.25 * edge_signal +
        0.20 * hotpix_signal + 0.15 * sat_signal,
        0.0, 1.0,
    ))

    # heatmap: |gray - blurred| highlights local anomalies
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    diff = cv2.absdiff(gray, blurred)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    features = {
        "lap_var": lap_var,
        "edge_density": edge_density,
        "hotpix_ratio": hotpix_ratio,
        "sat_std": sat_std,
    }
    return score, heat, features
