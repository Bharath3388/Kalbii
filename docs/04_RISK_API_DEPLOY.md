# 04 — Risk Scoring · API · Deployment

## A. Unified Risk Scoring Model

### A.1 Feature vector (fed to the ML model)

| # | Feature | Source |
|---|---------|--------|
| 1 | `text_risk_sub` | NLP module |
| 2 | `sent_neg_score` | NLP sentiment (0 if positive) |
| 3 | `n_risk_terms_critical` | NLP lexicon hits |
| 4 | `n_risk_terms_high` | NLP lexicon hits |
| 5 | `n_entities` | spaCy NER count |
| 6 | `text_length` | len(decrypted_text) |
| 7 | `image_anomaly` | CV module |
| 8 | `image_edge_density` | CV opencv stats |
| 9 | `image_hotpix_ratio` | CV opencv stats |
| 10 | `cross_term` | `text_risk_sub * image_anomaly` (interaction) |

### A.2 Model choice

**XGBoost classifier** (binary: HIGH-risk yes/no) → output `predict_proba`
gives a calibrated `risk_score ∈ [0,1]`. We then map to **0–100** and bucket:

```
risk_score_100 = round(100 * proba, 1)
risk_label     = "LOW"    if risk_score_100 < 33
                 "MEDIUM" if risk_score_100 < 66
                 "HIGH"   otherwise
```

Why XGBoost: best-in-class on small tabular data, supports
`feature_importances_` and SHAP for explainability. Logistic Regression is
the fallback if XGBoost cannot be installed (e.g. arm64 wheels).

### A.3 Training data strategy (no public dataset needed)

Generate a **synthetic but realistic** training set in `app/risk/train.py`:

```python
import numpy as np, pandas as pd, joblib
from xgboost import XGBClassifier

rng = np.random.default_rng(42)
N = 5000
X = pd.DataFrame({
    "text_risk_sub":           rng.beta(2, 5, N),
    "sent_neg_score":          rng.beta(2, 4, N),
    "n_risk_terms_critical":   rng.poisson(0.3, N),
    "n_risk_terms_high":       rng.poisson(0.6, N),
    "n_entities":              rng.poisson(2.0, N),
    "text_length":             rng.integers(20, 600, N),
    "image_anomaly":           rng.beta(2, 5, N),
    "image_edge_density":      rng.beta(2, 5, N),
    "image_hotpix_ratio":      rng.beta(1, 9, N),
})
X["cross_term"] = X.text_risk_sub * X.image_anomaly

# ground-truth rule (kept hidden from the model)
logit = (
    3.0*X.text_risk_sub + 2.5*X.image_anomaly + 1.8*X.cross_term +
    0.6*X.n_risk_terms_critical + 0.3*X.n_risk_terms_high +
    1.2*X.sent_neg_score - 2.5
)
y = (1/(1+np.exp(-logit)) > rng.uniform(size=N)).astype(int)

clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                    eval_metric="logloss", n_jobs=-1)
clf.fit(X, y)
joblib.dump({"model": clf, "columns": list(X.columns)},
            "app/risk/artifacts/risk_model.joblib")
print("AUC", clf.score(X, y))
```

The synthetic generator is documented so reviewers can see *exactly* what
the ground truth is — this is the opposite of a black box.

### A.4 Inference wrapper

```python
# app/risk/model.py
import joblib, numpy as np
from functools import lru_cache

@lru_cache(maxsize=1)
def _load(): return joblib.load("app/risk/artifacts/risk_model.joblib")

def predict(features: dict) -> dict:
    bundle = _load()
    cols = bundle["columns"]
    x = np.array([[features.get(c, 0) for c in cols]])
    proba = float(bundle["model"].predict_proba(x)[0, 1])
    score = round(100 * proba, 1)
    label = "LOW" if score < 33 else "MEDIUM" if score < 66 else "HIGH"
    return {"risk_score": score, "risk_label": label, "proba": proba}
```

---

## B. API System (FastAPI)

### B.1 Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/encrypt` | (helper) accepts `{text, image_b64}`, returns ciphertext pair |
| `POST` | `/ingest` | accepts ciphertext, returns `job_id` (202) |
| `GET`  | `/result/{job_id}` | poll job status / final record |
| `GET`  | `/records` | paginated history; supports `risk_label`, `from`, `to` filters |
| `GET`  | `/stream` | SSE live feed of new records (bonus) |
| `GET`  | `/healthz` | liveness/readiness |
| `GET`  | `/docs` | Swagger UI auto-generated |

### B.2 Auth & rate-limit

* `X-API-Key` header validated against `KMI_API_KEYS` env (comma list).
* `slowapi` middleware: 60 req/min/key.

### B.3 Sample request / response

```bash
# 1) Encrypt (optional helper, for demo only)
curl -X POST $API/encrypt \
  -H "X-API-Key: demo" -H "Content-Type: application/json" \
  -d '{"text":"Crack on weld joint","image_b64":"<...>"}'
# → {"ciphertext_text":"...","ciphertext_image":"..."}

# 2) Ingest
curl -X POST $API/ingest -H "X-API-Key: demo" -H "Content-Type: application/json" \
  -d '{"ciphertext_text":"...","ciphertext_image":"..."}'
# → 202  {"job_id":"8c1e...","status_url":"/result/8c1e..."}

# 3) Poll
curl $API/result/8c1e... -H "X-API-Key: demo"
# → 200  full record (see 01_ARCHITECTURE.md §3)
```

### B.4 FastAPI sketch

```python
# app/api/main.py
from fastapi import FastAPI, Depends, HTTPException, status
from .schemas import IngestIn, IngestOut, RecordOut
from app.workers.tasks import process_payload
from app.db.repositories import RecordsRepo
from app.core.security import api_key_auth

app = FastAPI(title="Encrypted Multi-Modal Intelligence System", version="1.0.0")

@app.post("/ingest", response_model=IngestOut, status_code=202,
          dependencies=[Depends(api_key_auth)])
async def ingest(payload: IngestIn):
    task = process_payload.delay(payload.ciphertext_text,
                                 payload.ciphertext_image,
                                 payload.metadata or {})
    return IngestOut(job_id=task.id, status_url=f"/result/{task.id}")

@app.get("/result/{job_id}", response_model=RecordOut,
         dependencies=[Depends(api_key_auth)])
async def result(job_id: str, repo: RecordsRepo = Depends()):
    rec = await repo.by_job(job_id)
    if not rec: raise HTTPException(404, "not ready or unknown")
    return rec
```

---

## C. Dashboard (Streamlit)

```python
# dashboard/streamlit_app.py
import streamlit as st, pandas as pd, plotly.express as px
from app.db.repositories import RecordsRepo

st.set_page_config(page_title="Risk Insights", layout="wide")
records = RecordsRepo().recent_sync(limit=500)
df = pd.DataFrame(records)

c1, c2, c3 = st.columns(3)
c1.metric("Records (24h)", len(df))
c2.metric("Avg risk", f"{df.risk_score.mean():.1f}")
c3.metric("HIGH risk %", f"{(df.risk_label=='HIGH').mean()*100:.1f}%")

st.plotly_chart(px.line(df.sort_values("created_at"), x="created_at", y="risk_score"))
st.dataframe(df[["created_at","risk_label","risk_score","decrypted_text"]].head(50))
```

---

## D. Deployment

### D.1 Docker Compose (local + Render)

```yaml
# docker-compose.yml
services:
  api:
    build: { context: ., dockerfile: docker/api.Dockerfile }
    env_file: .env
    ports: ["8000:8000"]
    depends_on: [redis]
  worker:
    build: { context: ., dockerfile: docker/worker.Dockerfile }
    env_file: .env
    depends_on: [redis]
  dashboard:
    build: { context: ., dockerfile: docker/dashboard.Dockerfile }
    env_file: .env
    ports: ["8501:8501"]
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

### D.2 Render

* **api** — Web Service, Docker, port 8000, env vars from Render dashboard.
* **worker** — Background Worker, same image, command
  `celery -A app.workers.celery_app worker -l INFO`.
* **redis** — Render Redis add-on (or Upstash free tier).
* **dashboard** — separate Web Service (Docker), port 8501.
* **MongoDB** — Atlas free M0 cluster; whitelist Render egress IPs.

### D.3 Kubernetes (bonus)

`k8s/` contains:
* `api-deployment.yaml` (replicas: 2, readiness on `/healthz`)
* `worker-deployment.yaml` (replicas: 2)
* `redis.yaml` (StatefulSet, 1Gi PVC)
* `hpa.yaml` (HPA on api: cpu 70 %; worker: queue length via custom metric)

### D.4 CI/CD (GitHub Actions, bonus)

`.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: ruff check . && pytest -q
  build:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with: { context: ., file: docker/api.Dockerfile, push: true,
                tags: ghcr.io/${{ github.repository }}/api:latest }
      - name: Deploy hook
        run: curl -fsSL -X POST $RENDER_DEPLOY_HOOK
        env: { RENDER_DEPLOY_HOOK: ${{ secrets.RENDER_DEPLOY_HOOK }} }
```

---

## E. Acceptance checklist (map to the brief)

- [x] Accepts encrypted text + image (`/ingest`)
- [x] Custom decryption (XOR + shift + scramble + Base64, **not** AES/Fernet only)
- [x] NLP: sentiment, NER, keywords, risk-keyword detection
- [x] CV: anomaly detection (OpenCV baseline + Autoencoder bonus + CLIP optional)
- [x] Unified risk score via XGBoost (LR fallback)
- [x] Insights dashboard (Streamlit)
- [x] Scalable + queryable API (FastAPI + Celery + Redis + Mongo indexes)
- [x] Storage of decrypted text, image path, risk score, timestamp
- [x] Bonus: async, Docker, K8s, CI/CD, advanced anomaly, SSE streaming
- [x] No black-box: every transformation/model is documented in this `docs/` folder
