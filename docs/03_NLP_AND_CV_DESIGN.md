# 03 — NLP & Computer Vision Design

## A. Text → NLP analysis

### A.1 Pipeline

```
decrypted_text
   │
   ├─► clean & normalise (lowercase only for matching, keep original for display)
   │
   ├─► (1) Sentiment      : DistilBERT-SST2 (HF transformers)
   │
   ├─► (2) NER            : spaCy en_core_web_sm
   │
   ├─► (3) Keyphrases     : KeyBERT with all-MiniLM-L6-v2 embeddings
   │
   ├─► (4) Risk-term hits : YAML lexicon + rapidfuzz partial_ratio ≥ 85
   │
   └─► aggregate → features_text
```

### A.2 Best open-source model picks (and why)

| Task | Model | Size | Why chosen |
|------|-------|------|-----------|
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | 67 M | Best accuracy/latency trade-off on CPU; widely benchmarked |
| NER (default) | spaCy `en_core_web_sm` | 12 MB | Fast, no GPU; covers PERSON/ORG/PRODUCT/GPE |
| NER (heavier alt) | `dslim/bert-base-NER` | 110 M | If GPU available, better recall on uncommon entities |
| Embeddings for KeyBERT | `sentence-transformers/all-MiniLM-L6-v2` | 22 M | De-facto standard for semantic similarity, 384-dim |
| Zero-shot risk classifier (optional) | `facebook/bart-large-mnli` | 407 M | If you want labels like "safety", "fraud", "fault" without training |

> Avoid massive LLMs (Llama-3, Mistral-7B) for this task — overkill, slow on
> Render free tier, and the rule says "no black-box".

### A.3 Risk lexicon (`app/nlp/risk_lexicon.yml`)

```yaml
critical:   [explosion, fire, fatal, fatality, leak, hazard, urgent, emergency, breach]
high:       [crack, fracture, defect, malfunction, failure, broken, corrosion, overheat]
medium:     [delay, anomaly, irregular, vibration, noise, warning, suspicious]
low:        [minor, scratch, dust, cosmetic]
```

Weights: `critical=1.0, high=0.7, medium=0.4, low=0.15`.

### A.4 Text-risk sub-score formula

```
sent_neg   = sentiment_score if label == NEGATIVE else 0
risk_hits  = Σ weight(category) * count(matches in category)
risk_norm  = min(1.0, risk_hits / 3.0)             # saturate at 3 weighted hits
text_risk_sub = clip(0.55 * risk_norm + 0.35 * sent_neg + 0.10 * urgency_flag, 0, 1)
```
where `urgency_flag` = 1 if any "critical"-category term matched.

### A.5 Reference module

```python
# app/nlp/pipeline.py
from functools import lru_cache
from transformers import pipeline as hf_pipeline
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import spacy, yaml, pathlib

@lru_cache(maxsize=1)
def _sentiment():  return hf_pipeline("sentiment-analysis",
                       model="distilbert-base-uncased-finetuned-sst-2-english")
@lru_cache(maxsize=1)
def _spacy():      return spacy.load("en_core_web_sm")
@lru_cache(maxsize=1)
def _kb():         return KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
@lru_cache(maxsize=1)
def _lexicon():    return yaml.safe_load(pathlib.Path(__file__).with_name("risk_lexicon.yml").read_text())

WEIGHTS = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.15}

def analyze(text: str) -> dict:
    text = text.strip()
    sent = _sentiment()(text[:512])[0]
    doc  = _spacy()(text)
    ents = [{"text": e.text, "label": e.label_} for e in doc.ents]
    kws  = [k for k, _ in _kb().extract_keywords(text, keyphrase_ngram_range=(1, 2),
                                                 stop_words="english", top_n=8)]

    lex, hits, weighted, urgency = _lexicon(), [], 0.0, 0
    low_text = text.lower()
    for cat, words in lex.items():
        for w in words:
            if fuzz.partial_ratio(w, low_text) >= 90:
                hits.append({"term": w, "category": cat})
                weighted += WEIGHTS[cat]
                if cat == "critical": urgency = 1

    risk_norm  = min(1.0, weighted / 3.0)
    sent_neg   = sent["score"] if sent["label"] == "NEGATIVE" else 0.0
    text_risk  = max(0.0, min(1.0, 0.55*risk_norm + 0.35*sent_neg + 0.10*urgency))

    return {
        "sentiment":  sent,
        "entities":   ents,
        "keywords":   kws,
        "risk_terms": hits,
        "text_risk_sub": round(text_risk, 4),
    }
```

---

## B. Image → Anomaly / Defect Detection

We provide **two interchangeable backends** behind `CVProcessor`. The
`CV_BACKEND` env var picks one (`opencv` | `autoencoder` | `clip`).

### B.1 Backend 1 — OpenCV statistical baseline (fast, no training)

Signal features:
1. **Laplacian variance** — low value ⇒ blur / focus defect.
2. **Edge density** (Canny) — abnormally high or low vs. baseline.
3. **Color histogram KL-divergence** vs. a reference profile (computed once
   from a folder of "normal" images you provide).
4. **Hot-pixel ratio** — fraction of pixels with luminance > 250 or < 5.

Combine into `anomaly_score ∈ [0, 1]` via min-max normalised weighted sum.
Generate a heatmap by thresholding |gray − blur(gray)|.

### B.2 Backend 2 — Convolutional Autoencoder (PyTorch, **bonus**)

Architecture (small, CPU-trainable):

```
Input 1×128×128 (grayscale)
  Conv 1→16 k3 s2  + ReLU      → 16×64×64
  Conv 16→32 k3 s2 + ReLU      → 32×32×32
  Conv 32→64 k3 s2 + ReLU      → 64×16×16   (bottleneck)
  ConvT 64→32 k3 s2 + ReLU     → 32×32×32
  ConvT 32→16 k3 s2 + ReLU     → 16×64×64
  ConvT 16→1  k3 s2 + Sigmoid  → 1×128×128
```

Train on the **MVTec-AD** "good" subset of one category (e.g. `bottle`) for
quick demo (≈2 min on CPU for 30 epochs at 64 batch). Anomaly score =
per-image MSE between input and reconstruction; heatmap = per-pixel squared
error, blurred and normalised.

Why MVTec-AD? It is the standard public benchmark for industrial anomaly
detection (CC-BY-NC-SA 4.0 — fine for a portfolio task; cite it in README).

### B.3 Backend 3 — CLIP zero-shot (creative bonus)

Use `openai/clip-vit-base-patch32`:

```
prompts = ["a photo of a defective part", "a photo of a normal part"]
score   = softmax(image⋅text_embeds)[0]   # P(defective)
```

No training required, surprisingly strong baseline, fully transparent.

### B.4 Reference orchestrator

```python
# app/cv/pipeline.py
import os, cv2, numpy as np
from .opencv_anomaly import score_opencv
# from .autoencoder.model import score_autoencoder
# from .clip_zeroshot   import score_clip

BACKEND = os.getenv("CV_BACKEND", "opencv")

def analyze(image_bytes: bytes, save_dir: str) -> dict:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"anomaly_score": 0.0, "method": "noop", "error": "decode_failed"}

    if BACKEND == "opencv":
        score, heatmap = score_opencv(img)
        method = "opencv_stats"
    elif BACKEND == "autoencoder":
        from .autoencoder.model import score_autoencoder
        score, heatmap = score_autoencoder(img)
        method = "autoencoder_mse"
    else:
        from .clip_zeroshot import score_clip
        score, heatmap = score_clip(img)
        method = "clip_zeroshot"

    heatmap_path = f"{save_dir}/heatmap.png"
    cv2.imwrite(heatmap_path, heatmap)
    return {"anomaly_score": float(score), "method": method, "heatmap_path": heatmap_path}
```

### B.5 Image-risk sub-score

```
image_risk_sub = clip(anomaly_score, 0, 1)
```
Already normalised by each backend.
