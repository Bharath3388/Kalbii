# Encrypted Multi-Modal Intelligence System (Kalbii)

A scalable, API-first service that ingests **encrypted text + image**,
custom-decrypts them, runs **NLP + CV** analysis, fuses the signals into a
**unified risk score** with an XGBoost model, persists everything in
MongoDB Atlas, and surfaces insights through a Streamlit dashboard.

> Full design lives in [`docs/`](docs/). Start with
> [docs/00_EXECUTIVE_SUMMARY.md](docs/00_EXECUTIVE_SUMMARY.md).

## Documentation index

| # | Doc | Contents |
|---|-----|----------|
| 0 | [Executive Summary](docs/00_EXECUTIVE_SUMMARY.md) | Approach, model picks, 12-hour plan |
| 1 | [Architecture](docs/01_ARCHITECTURE.md) | Component diagram, repo layout, API contracts, Mongo schema |
| 2 | [Encryption Design](docs/02_ENCRYPTION_DESIGN.md) | Custom cipher (XOR + shift + scramble + Base64) with code |
| 3 | [NLP & CV Design](docs/03_NLP_AND_CV_DESIGN.md) | Best OSS models, scoring formulas, code |
| 4 | [Risk · API · Deploy](docs/04_RISK_API_DEPLOY.md) | XGBoost risk model, FastAPI endpoints, Docker / K8s / CI-CD |

## Tech stack at a glance

- **API** FastAPI · Uvicorn
- **Async** Celery · Redis
- **DB** MongoDB Atlas
- **NLP** HuggingFace `distilbert-sst2`, spaCy `en_core_web_sm`, KeyBERT (`all-MiniLM-L6-v2`), `rapidfuzz`
- **CV** OpenCV baseline, PyTorch Conv-Autoencoder (MVTec-AD), optional CLIP zero-shot
- **Risk** XGBoost (Logistic Regression fallback)
- **Dashboard** Streamlit + Plotly
- **Deploy** Docker · Render · Kubernetes (bonus) · GitHub Actions (bonus)
