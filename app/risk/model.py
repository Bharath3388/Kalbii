"""Inference wrapper for the unified risk model."""

from __future__ import annotations
from functools import lru_cache
from typing import Any, Dict

import joblib
import numpy as np

from .train import ARTIFACT_PATH, FEATURE_COLUMNS, train as _train


@lru_cache(maxsize=1)
def _load() -> Dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        bundle, _ = _train(save=True)
        return bundle
    return joblib.load(ARTIFACT_PATH)


def _build_features(nlp_out: Dict[str, Any], cv_out: Dict[str, Any]) -> Dict[str, float]:
    sent = nlp_out.get("sentiment", {}) or {}
    sent_neg = float(sent.get("score", 0.0)) if sent.get("label") == "NEGATIVE" else 0.0
    text_risk_sub = float(nlp_out.get("text_risk_sub", 0.0))
    image_anomaly = float(cv_out.get("anomaly_score", 0.0))
    feats = {
        "text_risk_sub":         text_risk_sub,
        "sent_neg_score":        sent_neg,
        "n_risk_terms_critical": float(nlp_out.get("n_risk_terms_critical", 0)),
        "n_risk_terms_high":     float(nlp_out.get("n_risk_terms_high", 0)),
        "n_entities":            float(len(nlp_out.get("entities") or [])),
        "text_length":           float(nlp_out.get("text_length", 0)),
        "image_anomaly":         image_anomaly,
        "image_edge_density":    float(cv_out.get("edge_density", 0.0)),
        "image_hotpix_ratio":    float(cv_out.get("hotpix_ratio", 0.0)),
        "cross_term":            text_risk_sub * image_anomaly,
    }
    return feats


def predict(nlp_out: Dict[str, Any], cv_out: Dict[str, Any]) -> Dict[str, Any]:
    bundle = _load()
    feats = _build_features(nlp_out, cv_out)
    cols = bundle["columns"]
    x = np.array([[feats[c] for c in cols]], dtype=float)

    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x)[0, 1])
    else:                                                       # SVM etc.
        proba = float(1.0 / (1.0 + np.exp(-model.decision_function(x)[0])))

    score_100 = round(100.0 * proba, 1)
    label = "LOW" if score_100 < 33 else "MEDIUM" if score_100 < 66 else "HIGH"
    return {
        "risk_score": score_100,
        "risk_label": label,
        "proba": round(proba, 4),
        "features": feats,
        "model_backend": bundle.get("backend", "unknown"),
    }
