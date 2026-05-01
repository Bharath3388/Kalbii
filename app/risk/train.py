"""Train the unified-risk fusion model on synthetic data.

Run via:  python -m app.risk.train
Outputs:  app/risk/artifacts/risk_model.joblib
"""

from __future__ import annotations
import pathlib
from typing import List

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = pathlib.Path(__file__).parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "risk_model.joblib"

FEATURE_COLUMNS: List[str] = [
    "text_risk_sub",
    "sent_neg_score",
    "n_risk_terms_critical",
    "n_risk_terms_high",
    "n_entities",
    "text_length",
    "image_anomaly",
    "image_edge_density",
    "image_hotpix_ratio",
    "cross_term",
]


def make_synthetic(n: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "text_risk_sub":         rng.beta(2, 5, n),
        "sent_neg_score":        rng.beta(2, 4, n),
        "n_risk_terms_critical": rng.poisson(0.3, n),
        "n_risk_terms_high":     rng.poisson(0.6, n),
        "n_entities":            rng.poisson(2.0, n),
        "text_length":           rng.integers(20, 600, n),
        "image_anomaly":         rng.beta(2, 5, n),
        "image_edge_density":    rng.beta(2, 5, n),
        "image_hotpix_ratio":    rng.beta(1, 9, n),
    })
    df["cross_term"] = df["text_risk_sub"] * df["image_anomaly"]

    logit = (
        3.0 * df["text_risk_sub"]
        + 2.5 * df["image_anomaly"]
        + 1.8 * df["cross_term"]
        + 0.6 * df["n_risk_terms_critical"]
        + 0.3 * df["n_risk_terms_high"]
        + 1.2 * df["sent_neg_score"]
        - 2.5
    )
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = (proba > rng.uniform(size=n)).astype(int)
    return df[FEATURE_COLUMNS], y


def train(save: bool = True):
    X, y = make_synthetic()
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", n_jobs=-1, tree_method="hist",
        )
        backend = "xgboost"
    except Exception:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
        backend = "logreg"

    clf.fit(X, y)
    train_acc = float((clf.predict(X) == y).mean())

    bundle = {"model": clf, "columns": FEATURE_COLUMNS, "backend": backend, "train_acc": train_acc}
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, ARTIFACT_PATH)
    return bundle, train_acc


if __name__ == "__main__":
    bundle, acc = train()
    print(f"trained backend={bundle['backend']} train_acc={acc:.4f} -> {ARTIFACT_PATH}")
