"""
CLIP-style zero-shot anomaly detector using color histogram + texture divergence.

Simulates CLIP's concept of comparing an image embedding to text-prompt embeddings
(e.g. "normal" vs "anomalous") using hand-crafted multi-scale feature descriptors
— no external model weights required.

Approach:
  1. Extract multi-scale LBP texture features  (texture fingerprint)
  2. Extract HSV color histogram               (color fingerprint)
  3. Compute Jensen-Shannon divergence between local patch distributions
     and the global "normal" distribution → anomaly signal
  4. Return normalised score + heatmap
"""

from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np


def _lbp(gray: np.ndarray, radius: int = 1) -> np.ndarray:
    """Basic uniform LBP on a grayscale image (no external deps)."""
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    offsets = [
        (-radius, 0), (-radius, radius), (0, radius), (radius, radius),
        (radius, 0), (radius, -radius), (0, -radius), (-radius, -radius),
    ]
    center = gray[radius:-radius, radius:-radius]
    bits = np.zeros(center.shape, dtype=np.uint8)
    for i, (dr, dc) in enumerate(offsets):
        r0, r1 = radius + dr, h - radius + dr
        c0, c1 = radius + dc, w - radius + dc
        neighbor = gray[r0:r1, c0:c1]
        bits |= ((neighbor >= center).astype(np.uint8) << i)
    lbp[radius:-radius, radius:-radius] = bits
    return lbp


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two normalised histograms."""
    p = p + 1e-10
    q = q + 1e-10
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.clip(js, 0.0, 1.0))


def score_clip(img_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    CLIP-inspired zero-shot anomaly score.
    Returns (anomaly_score 0-1, heatmap uint8 BGR).
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("empty image")

    h0, w0 = img_bgr.shape[:2]
    side = 128
    img_r = cv2.resize(img_bgr, (side, side))

    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)

    # Global "normal" reference: LBP + hue histogram of whole image
    lbp_global = _lbp(gray)
    lbp_hist_global, _ = np.histogram(lbp_global, bins=32, range=(0, 256))
    lbp_hist_global = lbp_hist_global.astype(float)

    hue_hist_global, _ = np.histogram(hsv[:, :, 0], bins=32, range=(0, 180))
    hue_hist_global = hue_hist_global.astype(float)

    # Local patch divergence map
    patch = 16
    rows, cols = side // patch, side // patch
    score_map = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            r0, r1 = r * patch, (r + 1) * patch
            c0, c1 = c * patch, (c + 1) * patch

            g_patch = lbp_global[r0:r1, c0:c1]
            h_patch = hsv[r0:r1, c0:c1, 0]

            lbp_local, _ = np.histogram(g_patch, bins=32, range=(0, 256))
            hue_local, _ = np.histogram(h_patch, bins=32, range=(0, 180))

            d_tex = _js_divergence(lbp_local.astype(float), lbp_hist_global.copy())
            d_col = _js_divergence(hue_local.astype(float), hue_hist_global.copy())
            score_map[r, c] = 0.5 * d_tex + 0.5 * d_col

    # Normalise
    lo, hi = score_map.min(), score_map.max()
    if hi > lo:
        score_norm = (score_map - lo) / (hi - lo)
    else:
        score_norm = score_map

    # Overall: mean of top-25% patches
    flat = score_norm.flatten()
    k = max(1, len(flat) // 4)
    score = float(np.sort(flat)[-k:].mean())
    score = float(np.clip(score, 0.0, 1.0))

    # Build heatmap
    heat_small = (score_norm * 255).astype(np.uint8)
    heat_up = cv2.resize(heat_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heat_up, cv2.COLORMAP_INFERNO)

    return score, heatmap
