"""FastAPI gateway.

If Redis is reachable, jobs are dispatched to Celery workers; otherwise the
pipeline runs **synchronously** inline so the API still works on a single-
process deployment (e.g. Render free tier without Redis add-on).
"""

from __future__ import annotations
import base64
import os
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    EncryptIn, EncryptOut, IngestIn, IngestOut, RecordOut,
)
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.security import api_key_auth
from app.crypto.cipher import CustomCipher
from app.db.repositories import RecordsRepo
from app.workers.tasks import process_payload_sync

configure_logging()
log = get_logger("kalbii.api")
settings = get_settings()

app = FastAPI(
    title="Encrypted Multi-Modal Intelligence System",
    version="1.0.0",
    description="Custom-encrypted text+image risk analytics service.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _celery_available() -> bool:
    if os.getenv("KMI_FORCE_SYNC") == "1":
        return False
    try:
        from app.workers.celery_app import celery_app
        # quick non-blocking ping
        with celery_app.connection_or_acquire() as conn:
            conn.ensure_connection(max_retries=1, timeout=1)
        return True
    except Exception:                                   # noqa: BLE001
        return False


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "mongo": RecordsRepo().ping(),
        "celery": _celery_available(),
        "cv_backend": settings.CV_BACKEND,
        "nlp_backend": settings.NLP_BACKEND,
    }


@app.post("/encrypt", response_model=EncryptOut, dependencies=[Depends(api_key_auth)])
def encrypt(payload: EncryptIn):
    cipher = CustomCipher.from_str(settings.KMI_PASSPHRASE)
    ct_text = cipher.encrypt_text(payload.text)
    ct_image = None
    if payload.image_b64:
        try:
            img_bytes = base64.b64decode(payload.image_b64)
        except Exception as e:                          # noqa: BLE001
            raise HTTPException(400, f"image_b64 not valid base64: {e}")
        ct_image = cipher.encrypt(img_bytes)
    return EncryptOut(ciphertext_text=ct_text, ciphertext_image=ct_image)


@app.post("/ingest", response_model=IngestOut, status_code=status.HTTP_202_ACCEPTED,
          dependencies=[Depends(api_key_auth)])
def ingest(payload: IngestIn):
    job_id = uuid.uuid4().hex
    if _celery_available():
        from app.workers.tasks import process_payload
        async_res = process_payload.apply_async(
            args=[payload.ciphertext_text, payload.ciphertext_image or "", payload.metadata or {}],
            task_id=job_id,
        )
        log.info("enqueued", job_id=async_res.id)
        return IngestOut(job_id=async_res.id, status="queued",
                         status_url=f"/result/{async_res.id}")

    # synchronous fallback
    try:
        process_payload_sync(job_id, payload.ciphertext_text,
                             payload.ciphertext_image or "", payload.metadata or {})
    except ValueError as e:
        raise HTTPException(400, f"processing failed: {e}")
    return IngestOut(job_id=job_id, status="done", status_url=f"/result/{job_id}")


@app.get("/result/{job_id}", response_model=RecordOut, dependencies=[Depends(api_key_auth)])
def get_result(job_id: str):
    rec = RecordsRepo().by_job(job_id)
    if not rec:
        raise HTTPException(404, "unknown or not ready")
    rec["created_at"] = rec.get("created_at").isoformat() if rec.get("created_at") else None
    return rec


@app.get("/records", dependencies=[Depends(api_key_auth)])
def list_records(limit: int = Query(50, ge=1, le=500),
                 risk_label: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH)$")):
    items = RecordsRepo().recent(limit=limit, risk_label=risk_label)
    for it in items:
        ca = it.get("created_at")
        if ca and not isinstance(ca, str):
            it["created_at"] = ca.isoformat()
    return {"count": len(items), "items": items}
