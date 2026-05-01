import os
import sys
import pathlib

# ensure project root on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# tests use mongomock + sync mode by default
os.environ.setdefault("KMI_USE_MONGOMOCK", "1")
os.environ.setdefault("KMI_FORCE_SYNC", "1")
os.environ.setdefault("KMI_PASSPHRASE", "test-passphrase-123")
os.environ.setdefault("KMI_API_KEYS", "test-key")
os.environ.setdefault("DATA_DIR", str(ROOT / "data" / "images"))
os.environ.setdefault("CV_BACKEND", "opencv")
os.environ.setdefault("NLP_BACKEND", "vader")
