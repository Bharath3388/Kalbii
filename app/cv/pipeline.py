"""CV pipeline orchestrator. Picks a backend via env CV_BACKEND."""

from __future__ import annotations
import os
import pathlib
import uuid
from typing import Any, Dict

import cv2
import numpy as np

from .opencv_anomaly import score_opencv


def analyze(image_bytes: bytes, save_dir: str) -> Dict[str, Any]:
    if not image_bytes:
        return {"anomaly_score": 0.0, "method": "noop", "error": "empty_input",
                "image_path": None, "heatmap_path": None,
                "edge_density": 0.0, "hotpix_ratio": 0.0}

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"anomaly_score": 0.0, "method": "noop", "error": "decode_failed",
                "image_path": None, "heatmap_path": None,
                "edge_density": 0.0, "hotpix_ratio": 0.0}

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    base = uuid.uuid4().hex[:12]
    image_path = str(pathlib.Path(save_dir) / f"{base}.png")
    heatmap_path = str(pathlib.Path(save_dir) / f"{base}_heat.png")
    cv2.imwrite(image_path, img)

    backend = os.getenv("CV_BACKEND", "opencv").lower()
    score, heat, feats = score_opencv(img)         # default + always available
    method = "opencv_stats"

    if backend == "autoencoder":
        try:
            from .autoencoder.model import score_autoencoder        # type: ignore
            score, heat = score_autoencoder(img)
            method = "autoencoder_mse"
        except Exception as e:                                       # noqa: BLE001
            method = f"autoencoder_unavailable_fallback_opencv ({e.__class__.__name__})"
    elif backend == "clip":
        try:
            from .clip_zeroshot import score_clip                   # type: ignore
            score, heat = score_clip(img)
            method = "clip_zeroshot"
        except Exception as e:                                       # noqa: BLE001
            method = f"clip_unavailable_fallback_opencv ({e.__class__.__name__})"

    cv2.imwrite(heatmap_path, heat)

    return {
        "anomaly_score": round(float(score), 4),
        "method": method,
        "image_path": image_path,
        "heatmap_path": heatmap_path,
        "edge_density": round(float(feats.get("edge_density", 0.0)), 4),
        "hotpix_ratio": round(float(feats.get("hotpix_ratio", 0.0)), 4),
    }
