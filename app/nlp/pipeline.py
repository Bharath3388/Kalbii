"""NLP pipeline.

Default backend: VADER sentiment (NLTK) + spaCy NER + spaCy noun_chunks for
keyphrases + spaCy word-vector similarity for risk detection (en_core_web_md).

Heavy backend (`NLP_BACKEND=transformers`): swaps VADER for the HuggingFace
DistilBERT-SST2 sentiment pipeline and KeyBERT for keyphrase extraction.
That path is exercised when the optional `[heavy]` extras are installed.
"""

from __future__ import annotations
import os
import nltk
from functools import lru_cache
from typing import Any, Dict, List

# --------------- risk concept seeds (no external file needed) ----------
# These are the seed words per severity tier. spaCy word vectors handle
# semantic generalisation — e.g. "blaze" matches "fire", "rupture" matches
# "crack", etc. — without any fuzzy string matching or YAML lookup.
RISK_SEEDS: Dict[str, List[str]] = {
    "critical": ["explosion", "fire", "fatality", "emergency", "toxic", "hazard", "lethal"],
    "high":     ["crack", "fracture", "failure", "malfunction", "rupture", "overheat", "defect"],
    "medium":   ["warning", "anomaly", "irregular", "leakage", "suspicious", "vibration"],
    "low":      ["minor", "scratch", "cosmetic", "delay", "noise"],
}
WEIGHTS = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.15}
SIM_THRESHOLD = 0.55   # cosine similarity threshold for a "hit"


# ----------------------- sentiment backends ----------------------------
@lru_cache(maxsize=1)
def _vader():
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
        return spacy.load("en_core_web_md")
    except OSError:
        from spacy.cli import download as spacy_download    # type: ignore
        spacy_download("en_core_web_md")
        return spacy.load("en_core_web_md")


# --------- pre-compute seed vectors once at first call ----------------
@lru_cache(maxsize=1)
def _seed_docs() -> Dict[str, List[Any]]:
    """Returns {category: [spaCy Token, ...]} for all seeds."""
    nlp = _nlp_spacy()
    return {
        cat: [nlp(seed)[0] for seed in seeds]
        for cat, seeds in RISK_SEEDS.items()
    }


# ----------------------- vector-based risk detection ------------------
def _risk_from_vectors(doc) -> tuple[List[Dict[str, Any]], float, int]:
    """
    For every content token in the doc, compare against risk seed vectors.
    Each seed registers at most one hit (the highest-similarity doc token).
    Returns (hits, risk_norm, urgency_flag).
    """
    content_tokens = [
        t for t in doc
        if not t.is_stop and not t.is_punct and t.has_vector and len(t.text) > 1
    ]
    if not content_tokens:
        return [], 0.0, 0

    seeds_by_cat = _seed_docs()
    hits: List[Dict[str, Any]] = []
    seen_seeds: set = set()
    weighted = 0.0
    urgency = 0

    for category, seed_tokens in seeds_by_cat.items():
        for seed_tok in seed_tokens:
            if not seed_tok.has_vector or seed_tok.text in seen_seeds:
                continue
            best_sim = max(tok.similarity(seed_tok) for tok in content_tokens)
            if best_sim >= SIM_THRESHOLD:
                seen_seeds.add(seed_tok.text)
                hits.append({
                    "term": seed_tok.text,
                    "category": category,
                    "similarity": round(float(best_sim), 3),
                })
                weighted += WEIGHTS[category] * float(best_sim)
                if category == "critical":
                    urgency = 1

    return hits, min(1.0, weighted / 3.0), urgency


# ----------------------- public API ------------------------------------
def analyze(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {
            "sentiment": {"label": "NEUTRAL", "score": 0.0},
            "entities": [], "keywords": [], "risk_terms": [],
            "text_risk_sub": 0.0, "text_length": 0,
            "n_risk_terms_critical": 0, "n_risk_terms_high": 0,
        }

    sent = _sentiment(text)

    nlp = _nlp_spacy()
    doc = nlp(text)
    entities = [{"text": e.text, "label": e.label_} for e in doc.ents]

    # noun-chunk based keyphrases (deduped, top 8 by length)
    seen, keywords = set(), []
    for chunk in doc.noun_chunks:
        t = chunk.text.strip().lower()
        if 2 < len(t) < 60 and t not in seen:
            seen.add(t)
            keywords.append(t)
    keywords = sorted(keywords, key=len, reverse=True)[:8]

    # vector-based risk detection — no YAML, no fuzzy strings
    hits, risk_norm, urgency = _risk_from_vectors(doc)

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
