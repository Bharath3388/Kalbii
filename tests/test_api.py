import base64
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)
HEAD = {"X-API-Key": "test-key"}


def _img_b64() -> str:
    img = np.full((128, 128, 3), 200, dtype=np.uint8)
    cv2.circle(img, (64, 64), 20, (0, 0, 255), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_auth_required():
    r = client.post("/encrypt", json={"text": "hi"})
    assert r.status_code == 401


def test_full_pipeline_end_to_end():
    # 1) encrypt
    r = client.post("/encrypt", headers=HEAD,
                    json={"text": "URGENT: gas leak and fire detected — fatal hazard!",
                          "image_b64": _img_b64()})
    assert r.status_code == 200, r.text
    enc = r.json()
    assert enc["ciphertext_text"] and enc["ciphertext_image"]

    # 2) ingest
    r = client.post("/ingest", headers=HEAD,
                    json={"ciphertext_text":  enc["ciphertext_text"],
                          "ciphertext_image": enc["ciphertext_image"],
                          "metadata": {"source": "pytest"}})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    # 3) result
    r = client.get(f"/result/{job_id}", headers=HEAD)
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["risk_label"] in {"LOW", "MEDIUM", "HIGH"}
    assert "fire" in rec["decrypted_text"].lower()
    assert rec["risk_score"] >= 0


def test_records_listing():
    r = client.get("/records?limit=5", headers=HEAD)
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert body["count"] <= 5


def test_bad_ciphertext_rejected():
    r = client.post("/ingest", headers=HEAD,
                    json={"ciphertext_text": "not-a-real-token"})
    assert r.status_code in (400, 422)
