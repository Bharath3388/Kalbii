"""
Autoencoder-style anomaly detector using PCA reconstruction error (NumPy only).

Simulates the core idea of a convolutional autoencoder — project into a
compressed representation (via SVD/PCA on image patches), reconstruct, and
measure the per-pixel reconstruction error as the anomaly map.

No PyTorch required — this is a deterministic, dependency-free implementation
that produces meaningful anomaly scores based on the same underlying principle.
"""

from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np


_PATCH = 8          # patch size for patch-PCA
_N_COMPONENTS = 6   # PCA components kept (compression bottleneck)
_ANOMALY_THRESH = 0.15   # per-patch MSE threshold for anomaly


def _extract_patches(gray: np.ndarray, patch: int) -> np.ndarray:
    """Slide a patch window and return (N, patch*patch) array."""
    h, w = gray.shape
    h2, w2 = (h // patch) * patch, (w // patch) * patch
    g = gray[:h2, :w2].astype(np.float32) / 255.0
    rows, cols = h2 // patch, w2 // patch
    patches = g.reshape(rows, patch, cols, patch).transpose(0, 2, 1, 3)
    return patches.reshape(-1, patch * patch), rows, cols


def score_autoencoder(img_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    PCA-based reconstruction anomaly score.
    Returns (anomaly_score 0-1, heatmap uint8 BGR).
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("empty image")

    h0, w0 = img_bgr.shape[:2]
    # Resize to fixed square for consistent PCA
    side = 128
    img_r = cv2.resize(img_bgr, (side, side))
    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    patches, rows, cols = _extract_patches(gray, _PATCH)  # (N, 64)

    # Center
    mean = patches.mean(axis=0)
    X = patches - mean

    # SVD as PCA
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback to random projection if SVD fails
        Vt = np.random.default_rng(0).standard_normal((_N_COMPONENTS, X.shape[1])).astype(np.float32)

    components = Vt[:_N_COMPONENTS]           # (k, 64)
    encoded = X @ components.T                # (N, k) — encoder
    reconstructed = encoded @ components + mean  # (N, 64) — decoder

    # Per-patch MSE
    mse = ((patches - reconstructed) ** 2).mean(axis=1)  # (N,)
    mse_map = mse.reshape(rows, cols).astype(np.float32)

    # Normalise to [0,1]
    lo, hi = mse_map.min(), mse_map.max()
    if hi > lo:
        mse_norm = (mse_map - lo) / (hi - lo)
    else:
        mse_norm = mse_map

    # Overall score = mean of top-20% patches (focus on worst regions)
    flat = mse_norm.flatten()
    k = max(1, len(flat) // 5)
    score = float(np.sort(flat)[-k:].mean())
    score = float(np.clip(score, 0.0, 1.0))

    # Build heatmap at original size
    heat_small = (mse_norm * 255).astype(np.uint8)
    heat_up = cv2.resize(heat_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heat_up, cv2.COLORMAP_JET)

    return score, heatmap
