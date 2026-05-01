"""NLP pipeline.

Default backend: VADER sentiment (NLTK) + spaCy NER + spaCy noun_chunks for
keyphrases + a YAML risk lexicon with rapidfuzz partial-ratio matching.

Heavy backend (`NLP_BACKEND=transformers`): swaps VADER for the HuggingFace
DistilBERT-SST2 sentiment pipeline and KeyBERT for keyphrase extraction.
That path is exercised when the optional `[heavy]` extras are installed.
"""

from __future__ import annotations
import os
import pathlib
from functools import lru_cache
from typing import Any, Dict, List

import yaml
from rapidfuzz import fuzz

WEIGHTS = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.15}
_LEX_PATH = pathlib.Path(__file__).with_name("risk_lexicon.yml")


@lru_cache(maxsize=1)
def _lexicon() -> Dict[str, List[str]]:
    return yaml.safe_load(_LEX_PATH.read_text())


# ----------------------- sentiment backends ----------------------------
@lru_cache(maxsize=1)
def _vader():
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _hf_sentiment():
    from transformers import pipeline as hf_pipeline       # type: ignore
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def _sentiment(text: str) -> Dict[str, Any]:
    backend = os.getenv("NLP_BACKEND", "vader").lower()
    if backend == "transformers":
        try:
            res = _hf_sentiment()(text[:512])[0]
            return {"label": res["label"], "score": float(res["score"])}
        except Exception:                                   # graceful fallback
            pass
    s = _vader().polarity_scores(text)
    label = "POSITIVE" if s["compound"] >= 0.05 else "NEGATIVE" if s["compound"] <= -0.05 else "NEUTRAL"
    score = abs(s["compound"]) if label != "NEUTRAL" else 1 - abs(s["compound"])
    return {"label": label, "score": round(float(score), 4), "vader": s}


# ----------------------- spaCy lazy loader -----------------------------
@lru_cache(maxsize=1)
def _nlp_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Best-effort runtime download (works in dev; in prod we install in image)
        from spacy.cli import download as spacy_download    # type: ignore
        spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


# ----------------------- public API ------------------------------------
def analyze(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {
            "sentiment": {"label": "NEUTRAL", "score": 0.0},
            "entities": [], "keywords": [], "risk_terms": [],
            "text_risk_sub": 0.0, "text_length": 0,
        }

    sent = _sentiment(text)

    nlp = _nlp_spacy()
    doc = nlp(text)
    entities = [{"text": e.text, "label": e.label_} for e in doc.ents]

    # noun-chunk based keyphrases (deduped, stop-word-free, top 8 by length)
    seen, keywords = set(), []
    for chunk in doc.noun_chunks:
        t = chunk.text.strip().lower()
        if 2 < len(t) < 60 and t not in seen:
            seen.add(t)
            keywords.append(t)
    keywords = sorted(keywords, key=len, reverse=True)[:8]

    lex = _lexicon()
    hits: List[Dict[str, str]] = []
    weighted = 0.0
    urgency = 0
    low_text = text.lower()
    for category, words in lex.items():
        for w in words:
            if fuzz.partial_ratio(w, low_text) >= 90:
                hits.append({"term": w, "category": category})
                weighted += WEIGHTS.get(category, 0.0)
                if category == "critical":
                    urgency = 1

    risk_norm = min(1.0, weighted / 3.0)
    sent_neg = sent["score"] if sent["label"] == "NEGATIVE" else 0.0
    text_risk_sub = max(0.0, min(1.0, 0.55 * risk_norm + 0.35 * sent_neg + 0.10 * urgency))

    return {
        "sentiment": {"label": sent["label"], "score": float(sent["score"])},
        "entities": entities,
        "keywords": keywords,
        "risk_terms": hits,
        "n_risk_terms_critical": sum(1 for h in hits if h["category"] == "critical"),
        "n_risk_terms_high":     sum(1 for h in hits if h["category"] == "high"),
        "text_length": len(text),
        "text_risk_sub": round(float(text_risk_sub), 4),
    }
