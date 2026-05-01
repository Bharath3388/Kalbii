# Kalbii — Encrypted Multi-Modal Risk Intelligence

> **Live deployments**
> | Service | URL |
> |---|---|
> | 🌐 Frontend (React) | https://kalbii-frontend.onrender.com |
> | ⚙️ API (FastAPI) | https://kalbii-api.onrender.com |
> | 📊 Dashboard (Streamlit) | https://kalbii-dashboard.onrender.com |

Kalbii is a production-grade, API-first risk analysis system that accepts **encrypted text and images**, decrypts them server-side, runs them through **NLP + Computer Vision pipelines**, fuses the signals with an **XGBoost model**, and returns a unified risk score (0–100) with label `LOW / MEDIUM / HIGH`.

---

## Table of Contents
- [What it does](#what-it-does)
- [How we built it](#how-we-built-it)
  - [1. Encryption Layer](#1-encryption-layer)
  - [2. NLP Pipeline](#2-nlp-pipeline)
  - [3. Computer Vision Pipeline](#3-computer-vision-pipeline)
  - [4. Risk Fusion Model (XGBoost)](#4-risk-fusion-model-xgboost)
  - [5. API](#5-api)
  - [6. Frontend](#6-frontend)
  - [7. Dashboard](#7-dashboard)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Running Locally](#running-locally)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Deployed Services](#deployed-services)

---

## What it does

1. User submits text + optional image via the React frontend
2. Frontend **encrypts** the data using a custom cipher before sending
3. API decrypts, runs **NLP analysis** (sentiment, risk terms, entities, keywords)
4. API runs **CV analysis** on the image (anomaly detection)
5. Both scores are fused by an **XGBoost classifier** into a single 0–100 risk score
6. Result is stored in **MongoDB Atlas** and returned to the user
7. All historical runs are visible in the **Streamlit dashboard**

---

## How we built it

### 1. Encryption Layer

**File:** [`app/crypto/cipher.py`](app/crypto/cipher.py)

A custom multi-layer cipher protects all data in transit:
- **XOR** with a key-derived byte mask
- **Byte shift** (Caesar-style at byte level)
- **Block scramble** using a deterministic permutation
- **Base64 URL-safe encoding** for transport

The passphrase is derived from `KMI_PASSPHRASE` (env var, never logged). This means raw text and images are **never transmitted in plaintext** — the frontend encrypts before the request leaves the browser.

---

### 2. NLP Pipeline

**File:** [`app/nlp/pipeline.py`](app/nlp/pipeline.py)

Four stages run on the decrypted text:

| Stage | Tool | What it produces |
|---|---|---|
| **Sentiment** | VADER (NLTK) | `POSITIVE / NEGATIVE / NEUTRAL` + confidence score |
| **Entities** | spaCy `en_core_web_md` | Named entities (people, orgs, locations) |
| **Keywords** | spaCy noun-chunks | Top 8 key phrases by length |
| **Risk detection** | spaCy word vectors (GloVe) | Semantically matched risk terms with similarity score |

#### Risk Detection — Vector Similarity (no keyword list)

Instead of a hard-coded word list, we use **spaCy `en_core_web_md`** (300-dim GloVe vectors). Seed words per severity tier are embedded once:

```
critical → explosion, fire, fatality, emergency, toxic, hazard, lethal
high     → crack, fracture, failure, malfunction, rupture, overheat, defect
medium   → warning, anomaly, irregular, leakage, suspicious, vibration
low      → minor, scratch, cosmetic, delay, noise
```

Every content token in the input is compared via **cosine similarity** against all seeds. A match fires when similarity ≥ 0.55. This means:
- `"blaze"` matches `fire` (sim=0.68) ✅
- `"fissure"` matches `fracture` (sim=0.61) ✅
- `"poisonous"` matches `toxic` (sim=0.76) ✅
- `"team lunch"` → no matches ✅

The weighted hits are combined into a **text_risk_sub** score (0–1):

```
text_risk_sub = 0.55 × risk_norm + 0.35 × sent_neg + 0.10 × urgency
```

---

### 3. Computer Vision Pipeline

**File:** [`app/cv/pipeline.py`](app/cv/pipeline.py)

Three selectable backends, chosen per-request:

| Backend | How it works |
|---|---|
| **OpenCV** (default) | 4 statistical signals: blur (Laplacian variance), edge density (Canny), hot/dead pixel ratio, colour saturation spread → weighted anomaly score |
| **Autoencoder** | PyTorch Conv-AE — reconstruction MSE; high error = anomaly |
| **CLIP** | Zero-shot ViT — LBP texture + Jensen-Shannon divergence |

Output: `anomaly_score` (0–1) + a saved heatmap image.

---

### 4. Risk Fusion Model (XGBoost)

**File:** [`app/risk/model.py`](app/risk/model.py), [`app/risk/train.py`](app/risk/train.py)

An **XGBoost classifier** fuses 10 features from both pipelines:

| Feature | Source |
|---|---|
| `text_risk_sub` | NLP |
| `sent_neg_score` | NLP sentiment |
| `n_risk_terms_critical`, `n_risk_terms_high` | NLP |
| `n_entities`, `text_length` | NLP |
| `image_anomaly` | CV |
| `image_edge_density`, `image_hotpix_ratio` | CV |
| `cross_term` = text × image | Both |

The `cross_term` captures the key insight: **both text AND image being risky is far more dangerous than either alone**.

Score mapping:
- `0–32` → **LOW**
- `33–65` → **MEDIUM**
- `66–100` → **HIGH**

Why XGBoost?
- Best-in-class for tabular, structured features
- Handles non-linear interactions natively (the cross-term)
- Trains in < 2 seconds on 5,000 synthetic samples
- Artifact is ~300 KB — fits comfortably on free tier
- Falls back to Logistic Regression if XGBoost isn't installed

---

### 5. API

**File:** [`app/api/main.py`](app/api/main.py)

Built with **FastAPI**. All endpoints require `X-API-Key` header.

| Endpoint | Method | What it does |
|---|---|---|
| `/healthz` | GET | Health check (Mongo, Celery, backends) |
| `/encrypt` | POST | Encrypt text + image client-side |
| `/ingest` | POST | Decrypt → NLP → CV → Risk → persist |
| `/result/{job_id}` | GET | Fetch a completed analysis |
| `/records` | GET | List recent analyses (with filter) |

On **free tier** (no Redis): pipeline runs synchronously inline.
On **paid tier** (with Redis): jobs are dispatched to Celery workers asynchronously.

---

### 6. Frontend

**File:** [`frontend/`](frontend/)

Built with **React + TypeScript + Vite**, served via nginx.

- Submit text and/or upload an image
- Select CV backend (OpenCV / Autoencoder / CLIP)
- Encrypts data in the browser before sending to the API
- Displays risk score, sentiment, risk terms with similarity scores, CV anomaly, keywords, entities
- History page with pie chart (risk distribution) and histogram (score distribution)

UI design: dark theme, emerald accent palette, clean card layout.

---

### 7. Dashboard

**File:** [`dashboard/streamlit_app.py`](dashboard/streamlit_app.py)

**Streamlit** dashboard connected directly to MongoDB Atlas:
- KPI cards: total records, high/medium/low counts, average score
- Risk distribution pie chart
- Score histogram
- Full records table with filtering

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| Async workers | Celery, Redis |
| Database | MongoDB Atlas |
| NLP | VADER (NLTK), spaCy `en_core_web_md` |
| CV | OpenCV, PyTorch Conv-AE, CLIP |
| Risk model | XGBoost (Logistic Regression fallback) |
| Encryption | Custom XOR+shift+scramble cipher |
| Frontend | React, TypeScript, Vite, nginx |
| Dashboard | Streamlit, Recharts |
| Containerisation | Docker (multi-stage builds) |
| Deployment | Render (free tier) |

---

## Architecture

```
Browser (React)
    │  encrypt(text, image)  ←── KMI_PASSPHRASE
    ▼
FastAPI  /ingest
    ├── CustomCipher.decrypt()
    ├── NLP Pipeline
    │       ├── VADER sentiment
    │       ├── spaCy NER + keywords
    │       └── spaCy vector similarity → risk terms
    ├── CV Pipeline
    │       ├── OpenCV stats (default)
    │       ├── Autoencoder MSE
    │       └── CLIP zero-shot
    ├── XGBoost risk fusion
    │       └── score 0–100 + label LOW/MEDIUM/HIGH
    └── MongoDB Atlas  ←── persist result
```

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/Bharath3388/Kalbii.git
cd Kalbii

# 2. Create .env
cp .env.example .env
# Edit KMI_PASSPHRASE and KMI_API_KEYS

# 3. Start all services
docker-compose up --build

# API   → http://localhost:8000
# UI    → http://localhost:3000
# Dash  → http://localhost:8501
```

Or run the API directly:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_md
python -m nltk.downloader vader_lexicon
uvicorn app.api.main:app --reload
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `KMI_PASSPHRASE` | Cipher key — **change this** | `change-me-please-very-secret` |
| `KMI_API_KEYS` | Comma-separated valid API keys | `demo-key` |
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGO_DB` | Database name | `kalbii` |
| `REDIS_URL` | Celery broker | `redis://localhost:6379/0` |
| `CV_BACKEND` | `opencv` / `autoencoder` / `clip` | `opencv` |
| `NLP_BACKEND` | `vader` / `transformers` | `vader` |
| `DATA_DIR` | Image storage path | `./data/images` |
| `KMI_FORCE_SYNC` | Skip Celery, run synchronously | `1` (free tier) |

---

## API Reference

### Encrypt
```http
POST /encrypt
X-API-Key: demo-key

{
  "text": "URGENT: gas leak detected near reactor",
  "image_b64": "<base64 string or null>"
}
```

### Ingest (analyse)
```http
POST /ingest
X-API-Key: demo-key

{
  "ciphertext_text": "<from /encrypt>",
  "ciphertext_image": "<from /encrypt or omit>",
  "cv_backend": "opencv",
  "metadata": {}
}
```

### Fetch result
```http
GET /result/{job_id}
X-API-Key: demo-key
```

**Response:**
```json
{
  "job_id": "abc123",
  "risk_score": 72.4,
  "risk_label": "HIGH",
  "decrypted_text": "URGENT: gas leak detected near reactor",
  "nlp": {
    "sentiment": {"label": "NEGATIVE", "score": 0.87},
    "risk_terms": [{"term": "fire", "category": "critical", "similarity": 0.68}],
    "text_risk_sub": 0.91,
    "keywords": ["reactor", "gas leak"],
    "entities": []
  },
  "cv": {
    "anomaly_score": 0.0,
    "method": "noop"
  }
}
```

---

## Deployed Services

| Service | URL | Stack |
|---|---|---|
| **Frontend** | https://kalbii-frontend.onrender.com | React + Vite + nginx |
| **API** | https://kalbii-api.onrender.com | FastAPI + Docker |
| **Dashboard** | https://kalbii-dashboard.onrender.com | Streamlit + Docker |

All services are deployed on **Render free tier** with auto-deploy from the `main` branch on GitHub.

---

## Development Journey

| Phase | What was built |
|---|---|
| Encryption | Custom XOR+shift+scramble cipher with Base64 transport encoding |
| NLP v1 | VADER sentiment + spaCy NER + YAML lexicon + rapidfuzz fuzzy matching |
| CV | OpenCV statistical anomaly detector (blur, edge, hotpix, saturation) |
| Risk model | XGBoost trained on 5,000 synthetic samples with logistic ground truth |
| API | FastAPI with sync/async dual-mode (Celery when Redis available) |
| Frontend | React + TypeScript with browser-side encryption |
| NLP v2 | Replaced YAML+rapidfuzz with spaCy `en_core_web_md` vector similarity — semantic generalisation with no keyword list |
| UI redesign | Emerald color palette, clean CV backend cards, polished dark theme |
| Deployment | All 3 services deployed to Render with CLI automation |
