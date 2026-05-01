# 01 — System Architecture

## 1. Logical components

```
┌────────────────────────────────────────────────────────────────────────┐
│                            CLIENT / TESTER                             │
│   (curl, Postman, dashboard, or external producer of encrypted data)   │
└──────────────────────┬─────────────────────────────────────────────────┘
                       │ HTTPS  (JSON: {ciphertext_text, ciphertext_image})
                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       API GATEWAY (FastAPI)                            │
│   • POST /encrypt        – helper: returns ciphertext (demo aid)       │
│   • POST /ingest         – enqueues a job, returns job_id              │
│   • GET  /result/{id}    – returns risk record when ready              │
│   • GET  /records?limit  – paginated history (dashboard)               │
│   • GET  /stream         – SSE live feed (bonus)                       │
│   • GET  /healthz /docs                                                │
└──────────────────────┬─────────────────────────────────────────────────┘
                       │ celery.send_task("process_payload", …)
                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    BROKER + RESULT BACKEND (Redis)                     │
└──────────────────────┬─────────────────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          CELERY WORKER(S)                              │
│  Stage 1  CustomCipher.decrypt(text), CustomCipher.decrypt_bytes(img)  │
│  Stage 2  NLPProcessor.analyze(text)  →  features_text                 │
│  Stage 3  CVProcessor.analyze(image)  →  features_image                │
│  Stage 4  RiskModel.predict(features) →  unified_risk_score (0-100)   │
│  Stage 5  Mongo.insert(record)                                         │
│  Stage 6  publish to Redis pubsub channel "risk_events" (for SSE)      │
└──────────────────────┬─────────────────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      MONGODB ATLAS  (collection: records)              │
│   { _id, job_id, decrypted_text, image_path, nlp, cv,                  │
│     risk_score, risk_label, created_at }                               │
└──────────────────────┬─────────────────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                                │
│   • KPI cards (avg/median risk, count today)                           │
│   • Time-series chart (Plotly)                                         │
│   • Recent records table with text + thumbnail                         │
│   • Live tail via SSE (bonus)                                          │
└────────────────────────────────────────────────────────────────────────┘
```

## 2. Repository layout

```
kalbii/
├── app/
│   ├── api/
│   │   ├── main.py             # FastAPI app & routers
│   │   ├── schemas.py          # Pydantic models
│   │   └── deps.py             # DB / settings dependencies
│   ├── core/
│   │   ├── config.py           # pydantic-settings (.env)
│   │   ├── logging.py
│   │   └── security.py         # API key auth (optional)
│   ├── crypto/
│   │   ├── cipher.py           # CustomCipher class
│   │   └── kdf.py              # PBKDF2-HMAC key derivation
│   ├── nlp/
│   │   ├── pipeline.py         # NLPProcessor
│   │   ├── risk_lexicon.yml
│   │   └── keywords.py         # KeyBERT wrapper
│   ├── cv/
│   │   ├── pipeline.py         # CVProcessor (orchestrates)
│   │   ├── opencv_anomaly.py
│   │   └── autoencoder/
│   │       ├── model.py        # ConvAutoencoder (PyTorch)
│   │       ├── train.py
│   │       └── weights.pt
│   ├── risk/
│   │   ├── model.py            # XGBoost wrapper
│   │   ├── train.py            # synthetic data + fit
│   │   └── artifacts/risk_model.joblib
│   ├── workers/
│   │   ├── celery_app.py
│   │   └── tasks.py            # process_payload task
│   ├── db/
│   │   ├── mongo.py            # Motor / PyMongo client
│   │   └── repositories.py
│   └── utils/
│       └── images.py           # base64 ↔ ndarray helpers
├── dashboard/
│   └── streamlit_app.py
├── tests/
│   ├── test_cipher.py
│   ├── test_nlp.py
│   ├── test_cv.py
│   ├── test_risk.py
│   └── test_api.py
├── docker/
│   ├── api.Dockerfile
│   ├── worker.Dockerfile
│   └── dashboard.Dockerfile
├── k8s/                        # bonus
│   ├── api-deployment.yaml
│   ├── worker-deployment.yaml
│   ├── redis.yaml
│   └── hpa.yaml
├── .github/workflows/ci.yml    # bonus
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── docs/                       # (this folder)
```

## 3. Data flow & contracts

**Ingest request**
```jsonc
POST /ingest
{
  "ciphertext_text":  "<base64 string>",
  "ciphertext_image": "<base64 string>",   // image bytes encrypted then b64
  "metadata": { "source": "device-01" }    // optional
}
→ 202 Accepted
{ "job_id": "8c1e…", "status_url": "/result/8c1e…" }
```

**Result response**
```jsonc
GET /result/8c1e…
{
  "job_id": "8c1e…",
  "status": "done",
  "decrypted_text": "Hairline crack near weld joint, urgent.",
  "image_path": "s3://… or /data/images/8c1e.png",
  "nlp": {
    "sentiment": { "label": "NEGATIVE", "score": 0.987 },
    "entities":  [ {"text": "weld joint", "label": "PRODUCT"} ],
    "keywords":  ["hairline crack", "weld joint", "urgent"],
    "risk_terms":["crack", "urgent"],
    "text_risk_sub": 0.74
  },
  "cv": {
    "anomaly_score": 0.61,
    "method": "autoencoder_mse",
    "heatmap_path": "/data/images/8c1e_heat.png"
  },
  "risk_score": 78.4,
  "risk_label": "HIGH",
  "created_at": "2026-05-01T12:34:56Z"
}
```

## 4. Mongo schema (collection `records`)

```jsonc
{
  "_id":            ObjectId,
  "job_id":         "uuid4",
  "decrypted_text": String,
  "image_path":     String,
  "nlp":            Object,    // see above
  "cv":             Object,
  "risk_score":     Number,    // 0-100
  "risk_label":     "LOW" | "MEDIUM" | "HIGH",
  "created_at":     ISODate,
  "metadata":       Object
}
```
Indexes: `{created_at: -1}`, `{risk_label: 1}`, `{job_id: 1}` unique.

## 5. Scaling & failure modes

| Concern | Mitigation |
|---------|-----------|
| Slow CV inference | Celery concurrency = vCPU count; HPA on CPU>70% |
| Large images | Reject >5 MB at API; downsize to 512×512 in worker |
| Model cold-start | Load models at worker boot (`@worker_process_init`) |
| DB outage | API returns 202 + retries write with exponential backoff |
| Bad ciphertext | Validate HMAC prefix; return 400 with structured error |
| Replay attacks | Cipher includes per-message nonce + timestamp |

## 6. Observability

* **Structured logs**: `structlog` JSON to stdout (Render captures).
* **Metrics**: `/metrics` Prometheus endpoint via `prometheus-fastapi-instrumentator`.
* **Tracing (optional)**: OpenTelemetry → console exporter for dev.
