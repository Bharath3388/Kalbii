from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class EncryptIn(BaseModel):
    text: str = Field(..., description="Plain text to encrypt (demo helper)")
    image_b64: Optional[str] = Field(None, description="Base64-encoded image bytes")


class EncryptOut(BaseModel):
    ciphertext_text: str
    ciphertext_image: Optional[str] = None


class IngestIn(BaseModel):
    ciphertext_text: str = Field(..., description="Cipher token from /encrypt")
    ciphertext_image: Optional[str] = Field(None, description="Cipher token for image bytes")
    cv_backend: Optional[str] = Field(None, pattern="^(opencv|autoencoder|clip)$",
                                       description="CV backend override: opencv, autoencoder, or clip")
    metadata: Optional[Dict[str, Any]] = None


class IngestOut(BaseModel):
    job_id: str
    status: str = "queued"
    status_url: str


class RecordOut(BaseModel):
    job_id: str
    status: str
    decrypted_text: Optional[str] = None
    image_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    nlp: Optional[Dict[str, Any]] = None
    cv: Optional[Dict[str, Any]] = None
    risk_score: Optional[float] = None
    risk_label: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
