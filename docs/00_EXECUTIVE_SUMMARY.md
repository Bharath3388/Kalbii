# Encrypted Multi-Modal Intelligence System — Executive Summary

**Role:** Senior Data Scientist analysis & solution blueprint
**Timeline target:** 12 hours (MVP) + bonus stretch
**Stack class:** Python · FastAPI · MongoDB Atlas · HuggingFace · OpenCV/PyTorch · Docker · Render

---

## 1. Problem (re-stated in one line)

Build a **scalable, API-first** service that ingests **encrypted text + image**, custom-decrypts them, runs **NLP + CV** analysis, fuses signals into a **unified risk score** via a trained ML model, persists everything in MongoDB Atlas, and surfaces it through a **dashboard**.

## 2. North-star design principles

| # | Principle | Why it matters here |
|---|-----------|--------------------|
| 1 | **Modularity** — encryption / NLP / CV / risk / API are independent packages | Lets evaluators inspect each piece (mandatory: "no black-box") |
| 2 | **Async by default** — FastAPI + Celery + Redis | Image inference is slow; satisfies bonus + scalability |
| 3 | **Stateless API, stateful DB** | Horizontal scale on Render / K8s |
| 4 | **Reproducibility** — pinned models, deterministic seeds, dockerized | Required for evaluation re-runs |
| 5 | **Explainability over accuracy** for the risk model | Logistic Regression / RF expose feature weights |

## 3. High-level architecture

```
                ┌──────────────────────────────────────────────────────────┐
   Client ─►    │  FastAPI Gateway  (/encrypt /ingest /result /records)    │
                └─────────┬────────────────────────────────┬───────────────┘
                          │ enqueue                        │ query
                          ▼                                ▼
                ┌───────────────────────┐         ┌──────────────────┐
                │ Redis broker + Celery │         │  MongoDB Atlas   │
                └─────────┬─────────────┘         └────────▲─────────┘
                          ▼                                │
        ┌─────────────────┴──────────────────┐             │
        │           Worker pool              │             │
        │  1. CustomCipher.decrypt           │             │
        │  2. NLP pipeline (DistilBERT+spaCy)│  writes ────┘
        │  3. CV pipeline (CLIP / Autoencoder│
        │  4. RiskModel.predict (XGBoost)    │
        └────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │  Streamlit Dashboard │  ── reads from MongoDB
                └──────────────────────┘
```

## 4. Component-to-tech mapping (best open-source picks)

| Layer | Recommended OSS choice | Why this over alternatives |
|-------|------------------------|----------------------------|
| API | **FastAPI + Uvicorn** | Async-native, OpenAPI docs free, fastest Python web framework |
| Queue | **Celery + Redis** | Mandated bonus; mature, easy to deploy on Render |
| DB | **MongoDB Atlas (free M0)** | Mandated; flexible schema fits multi-modal records |
| Encryption | **Custom (PBKDF2 → keystream + XOR + byte-shift + permutation + Base64)** | Satisfies "not AES/Fernet only" rule; deterministically reversible |
| Sentiment | **`distilbert-base-uncased-finetuned-sst-2-english`** | 67M params, runs CPU in <100 ms; strong baseline |
| NER / keywords | **spaCy `en_core_web_sm` + KeyBERT (`all-MiniLM-L6-v2`)** | spaCy NER + MiniLM embeddings give entity + semantic keywords |
| Risk lexicon | **Curated YAML list + fuzzy match (`rapidfuzz`)** | Transparent, tweak-able, no black-box |
| CV anomaly (basic) | **OpenCV: Laplacian variance + edge density + color histogram outlier** | Zero training data needed |
| CV anomaly (advanced ⭐) | **PyTorch convolutional autoencoder trained on MVTec-AD subset** | Real anomaly score = reconstruction MSE; bonus points |
| CV embedding (alt) | **`openai/clip-vit-base-patch32`** via HF | Zero-shot defect captioning + cosine-distance anomaly |
| Risk fusion model | **XGBoost classifier** trained on synthetic labelled fusion features | Best tabular accuracy + `feature_importances_` for explainability |
| Dashboard | **Streamlit** (or Next.js if time permits) | Fastest to ship "risk score / recent records / charts" |
| Container | **Docker (multi-stage) + docker-compose** | Bonus + Render deploy |
| Orchestration (bonus) | **Kubernetes manifests (Deployment + Service + HPA)** | Bonus checkbox |
| CI/CD | **GitHub Actions** → lint → test → build image → deploy hook to Render | Bonus checkbox |
| Streaming (bonus) | **Server-Sent Events (`/stream`) over FastAPI** | Lighter than Kafka, fits 12h budget |

## 5. 12-hour delivery plan

| Hour | Deliverable |
|------|-------------|
| 0–1 | Repo scaffold, `pyproject.toml`, Docker skeleton, MongoDB Atlas cluster |
| 1–2 | Custom cipher (encrypt/decrypt) + unit tests (round-trip on text+image) |
| 2–4 | NLP pipeline module + tests |
| 4–6 | CV pipeline module (OpenCV baseline first, autoencoder if time) |
| 6–7 | Risk scoring model — train on synthetic data, persist with `joblib` |
| 7–9 | FastAPI endpoints + MongoDB persistence + Celery worker |
| 9–10 | Streamlit dashboard |
| 10–11 | Dockerize, write README + architecture doc, push to GitHub |
| 11–12 | Deploy to Render, smoke test, record demo links |

## 6. Documents in this folder

1. `01_ARCHITECTURE.md` — detailed system & data-flow diagrams
2. `02_ENCRYPTION_DESIGN.md` — custom cipher spec + reversibility proof
3. `03_NLP_AND_CV_DESIGN.md` — model choices, prompts, scoring formulas
4. `04_RISK_API_DEPLOY.md` — risk model, API contract, deployment runbook
