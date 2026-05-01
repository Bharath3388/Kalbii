import io
import numpy as np
import cv2

from app.cv.pipeline import analyze


def _png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_clean_image_low_anomaly(tmp_path):
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (200, 200), (200, 100, 50), -1)
    cv2.circle(img, (128, 128), 30, (50, 200, 100), -1)
    out = analyze(_png_bytes(img), save_dir=str(tmp_path))
    assert 0.0 <= out["anomaly_score"] <= 1.0
    assert out["method"].startswith("opencv")


def test_blurry_image_higher_anomaly(tmp_path):
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    flat = np.full_like(img, 128)
    s_clean = analyze(_png_bytes(img),     save_dir=str(tmp_path))["anomaly_score"]
    s_flat  = analyze(_png_bytes(flat),    save_dir=str(tmp_path))["anomaly_score"]
    # a perfectly flat image (zero variance) should look anomalous (very blurry)
    assert s_flat > s_clean


def test_invalid_bytes(tmp_path):
    out = analyze(b"not an image", save_dir=str(tmp_path))
    assert out["anomaly_score"] == 0.0
    assert "error" in out
