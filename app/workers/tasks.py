"""Celery task: full processing pipeline.

Stage 1 — decrypt
Stage 2 — NLP
Stage 3 — CV
Stage 4 — Risk fusion
Stage 5 — persist to Mongo
"""

from __future__ import annotations
import base64
import datetime as dt
from typing import Any, Dict

from app.core.config import get_settings
from app.crypto.cipher import CustomCipher
from app.nlp.pipeline import analyze as nlp_analyze
from app.cv.pipeline import analyze as cv_analyze
from app.risk.model import predict as risk_predict
from app.db.repositories import RecordsRepo
from app.workers.celery_app import celery_app


def process_payload_sync(
    job_id: str,
    ciphertext_text: str,
    ciphertext_image: str,
    metadata: Dict[str, Any] | None = None,
    cv_backend: str | None = None,
) -> Dict[str, Any]:
    settings = get_settings()
    cipher = CustomCipher.from_str(settings.KMI_PASSPHRASE)

    # Stage 1: decrypt
    decrypted_text = cipher.decrypt_text(ciphertext_text) if ciphertext_text else ""
    image_bytes = cipher.decrypt(ciphertext_image) if ciphertext_image else b""

    # Stage 2: NLP
    nlp_out = nlp_analyze(decrypted_text)

    # Stage 3: CV (allow per-request backend override)
    import os
    old_cv = os.environ.get("CV_BACKEND")
    if cv_backend:
        os.environ["CV_BACKEND"] = cv_backend
    cv_out = cv_analyze(image_bytes, save_dir=settings.DATA_DIR)
    if old_cv is not None:
        os.environ["CV_BACKEND"] = old_cv
    elif cv_backend:
        os.environ.pop("CV_BACKEND", None)

    # Stage 4: risk fusion
    risk_out = risk_predict(nlp_out, cv_out)

    # Stage 5: persist
    record = {
        "job_id": job_id,
        "decrypted_text": decrypted_text,
        "image_path": cv_out.get("image_path"),
        "heatmap_path": cv_out.get("heatmap_path"),
        "nlp": nlp_out,
        "cv": cv_out,
        "risk_score": risk_out["risk_score"],
        "risk_label": risk_out["risk_label"],
        "model_backend": risk_out["model_backend"],
        "metadata": metadata or {},
        "created_at": dt.datetime.utcnow(),
        "status": "done",
    }
    RecordsRepo().insert(record)
    record["_id"] = record.get("_id") or job_id
    return record


@celery_app.task(name="kalbii.process_payload", bind=True)
def process_payload(self, ciphertext_text: str, ciphertext_image: str,
                    metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return process_payload_sync(self.request.id, ciphertext_text, ciphertext_image, metadata)
