"""
engine.py — Sentilytics AI Core NLP Engine
============================================
Architecture: Groq LLM (primary) + DistilBERT (fallback) + SpaCy (aspects)
             + AuthenticityDetector (novel contribution #2)

  • Primary:  Groq / Llama-3-70b — structured JSON output for sentiment,
              emotion, and aspect analysis in a single inference call.
              Handles sarcasm, negation, and context-dependent nuances.
  • Fallback: DistilBERT transformers pipelines (no API required, runs offline).
  • Aspects:  SpaCy dependency parsing always runs for granular noun-adj linking.
  • Novel #1: Trust Score — composite reliability metric across 4 axes.
  • Novel #2: Authenticity Score — 6-signal fake review detector.
  • Novel #3: Integrity Score — unified reliability framework (Trust × Authenticity).
"""

import os
import json
import torch
from transformers import pipeline
import spacy
from authenticity import AuthenticityDetector

# ── Load environment variables (.env) ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — falls back to system env vars

# ── Groq client (optional — graceful fallback if not installed / no key) ───────
try:
    from groq import Groq as GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Prompt template for Groq structured analysis
_GROQ_SYSTEM_PROMPT = """You are an expert e-commerce review analyst.
When given a customer review, respond ONLY with a valid JSON object in exactly this schema:

{
  "sentiment": "Positive" | "Negative" | "Neutral",
  "sentiment_confidence": <float 0.0–1.0>,
  "emotions": [
    {"label": "Joy"|"Anger"|"Sadness"|"Fear"|"Surprise"|"Excitement", "score": <float 0.0–1.0>},
    ...
  ],
  "aspects": [
    {"aspect": "<product feature>", "sentiment_word": "<descriptor>"},
    ...
  ],
  "sarcasm_detected": <true|false>,
  "reasoning": "<one sentence explaining your verdict>"
}

Rules:
- emotions array must contain all 6 labels, scores must sum to ~1.0
- aspects: extract real product features mentioned (battery, screen, delivery, etc.)
- If sarcasm is detected, flip the naive sentiment accordingly
- No markdown, no explanation outside the JSON object
"""


class SentilyticsEngine:
    """
    Core NLP engine for Sentilytics AI.

    Analysis pipeline:
      1. Groq / Llama-3-70b  — primary LLM path (rich, context-aware)
      2. DistilBERT fallback  — offline transformer path
      3. SpaCy                — always runs for fine-grained dependency-parsed aspects
      4. Trust Score          — composite reliability metric (novel contribution #1)
      5. Authenticity Score   — 6-signal fake review detector   (novel contribution #2)
      6. Integrity Score      — unified reliability framework   (novel contribution #3)
    """

    def __init__(self):
        # ── Groq client ────────────────────────────────────────────────────────
        self.groq_client = None
        if _GROQ_AVAILABLE and GROQ_API_KEY:
            try:
                self.groq_client = GroqClient(api_key=GROQ_API_KEY)
                print("✅ Groq LLM client initialised (Llama-3-70b primary engine).")
            except Exception as e:
                print(f"⚠️  Groq init failed ({e}) — will use DistilBERT fallback.")

        # ── DistilBERT fallback pipelines ──────────────────────────────────────
        print("Loading DistilBERT sentiment model (fallback)…")
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        print("Loading DistilBERT emotion model (fallback)…")
        self.emotion_pipe = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None
        )

        # ── SpaCy aspect extractor (always active) ────────────────────────────
        print("Loading SpaCy NLP for dependency-parsed aspect extraction…")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # ── Trust Score weights ────────────────────────────────────────────────
        self.trust_weights = {
            "alpha": 0.40,   # sentiment confidence
            "beta":  0.25,   # top emotion score
            "gamma": 0.25,   # aspect richness
            "delta": 0.10,   # review detail (length)
        }

        # ── Integrity Score weights ────────────────────────────────────────────
        self.integrity_weights = {
            "trust":         0.60,   # Trust Score (review quality)
            "authenticity":  0.40,   # Authenticity Score (review genuineness)
        }

        # ── Authenticity Detector (Novel Contribution #2) ──────────────────────
        self.auth_detector = AuthenticityDetector()

        print("✅ SentilyticsEngine ready.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY PATH — Groq / Llama-3-70b
    # ══════════════════════════════════════════════════════════════════════════
    def _analyze_with_groq(self, text: str) -> dict | None:
        """
        Sends the review to Groq (Llama-3-70b-8192) and returns a parsed dict.
        Returns None on any failure so the caller can fall back to DistilBERT.
        """
        if not self.groq_client:
            return None

        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": _GROQ_SYSTEM_PROMPT},
                    {"role": "user",   "content": text[:1500]}  # guard token limit
                ],
                response_format={"type": "json_object"},
                temperature=0.1,   # deterministic for research reproducibility
                max_tokens=512,
            )

            raw  = response.choices[0].message.content
            data = json.loads(raw)

            # Validate & normalise required keys
            sentiment  = data.get("sentiment", "Neutral")
            confidence = float(data.get("sentiment_confidence", 0.8))
            emotions   = data.get("emotions", [])
            aspects    = data.get("aspects", [])
            sarcasm    = data.get("sarcasm_detected", False)
            reasoning  = data.get("reasoning", "")

            # Normalise emotion scores to sum to 1.0
            total = sum(e.get("score", 0) for e in emotions)
            if total > 0:
                for e in emotions:
                    e["score"] = round(e["score"] / total, 4)

            return {
                "sentiment":        sentiment,
                "sentiment_score":  confidence,
                "emotions":         sorted(emotions, key=lambda x: x["score"], reverse=True),
                "aspects":          aspects,
                "sarcasm_detected": sarcasm,
                "reasoning":        reasoning,
                "engine":           "Groq / Llama-3-70b",
            }

        except Exception as e:
            print(f"⚠️  Groq analysis failed: {e} — switching to DistilBERT.")
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # FALLBACK PATH — DistilBERT transformers
    # ══════════════════════════════════════════════════════════════════════════
    def _analyze_sentiment_distilbert(self, text: str):
        result = self.sentiment_pipe(text[:512])[0]
        label  = result["label"].title()
        score  = result["score"]
        if score < 0.60:
            label = "Neutral"
        return label, score

    def _analyze_emotions_distilbert(self, text: str):
        raw = self.emotion_pipe(text[:512])[0]
        label_map = {
            "joy":      "Joy",
            "anger":    "Anger",
            "sadness":  "Sadness",
            "fear":     "Fear",
            "surprise": "Surprise",
            "love":     "Excitement",
        }
        mapped = [
            {"label": label_map[r["label"]], "score": round(r["score"], 4)}
            for r in raw if r["label"] in label_map
        ]
        return sorted(mapped, key=lambda x: x["score"], reverse=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ASPECT EXTRACTION — SpaCy (always active, supplements Groq aspects)
    # ══════════════════════════════════════════════════════════════════════════
    def extract_aspects(self, text: str) -> list:
        """
        Hybrid aspect extraction:
          1. SpaCy dependency parsing for NOUN-ADJ pairs and acomp patterns.
          2. Merged with Groq-extracted aspects (if available) with deduplication.
        """
        doc    = self.nlp(text)
        aspects = []
        seen    = set()

        for token in doc:
            if token.pos_ == "NOUN":
                modifiers = [w.text for w in token.children if w.pos_ == "ADJ"]
                if modifiers:
                    key = (token.text.lower(), ", ".join(m.lower() for m in modifiers))
                    if key not in seen:
                        seen.add(key)
                        aspects.append({
                            "aspect":         token.text,
                            "sentiment_word": ", ".join(modifiers)
                        })

            if token.pos_ == "ADJ" and token.dep_ == "acomp":
                subjects = [w.text for w in token.head.children if w.dep_ == "nsubj"]
                if subjects:
                    key = (subjects[0].lower(), token.text.lower())
                    if key not in seen:
                        seen.add(key)
                        aspects.append({
                            "aspect":         subjects[0],
                            "sentiment_word": token.text
                        })

        return aspects

    def _merge_aspects(self, groq_aspects: list, spacy_aspects: list) -> list:
        """Merges Groq and SpaCy aspects, deduplicating by aspect name."""
        seen   = set()
        merged = []
        for a in groq_aspects + spacy_aspects:
            key = a.get("aspect", "").lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(a)
        return merged

    # ══════════════════════════════════════════════════════════════════════════
    # TRUST SCORE — Novel Contribution #1
    # ══════════════════════════════════════════════════════════════════════════
    def compute_trust_score(
        self,
        sentiment_confidence: float,
        top_emotion_score:    float,
        aspects:              list,
        review_text:          str,
        sarcasm_detected:     bool = False,
    ) -> dict:
        """
        Trust Score formula (novel metric — Sentilytics AI paper):

        TrustScore = α·Cₛ + β·Eₜₒₚ + γ·(|A|/5) + δ·(|W|/100)

        A sarcasm penalty of −10 pts is applied when sarcasm is detected,
        reflecting reduced reliability of face-value sentiment inference.
        """
        w = self.trust_weights

        aspect_density = min(len(aspects) / 5.0, 1.0)
        word_count     = len(review_text.split())
        length_factor  = min(word_count / 100.0, 1.0)

        raw_score  = (
            w["alpha"] * sentiment_confidence +
            w["beta"]  * top_emotion_score +
            w["gamma"] * aspect_density +
            w["delta"] * length_factor
        )
        trust_score = round(raw_score * 100, 1)

        if sarcasm_detected:
            trust_score = max(0.0, trust_score - 10)

        if trust_score >= 75:
            label, color = "Highly Trustworthy",     "#39ff14"
        elif trust_score >= 50:
            label, color = "Moderately Trustworthy", "#ffcc00"
        else:
            label, color = "Low Reliability",        "#ff003c"

        return {
            "score": trust_score,
            "label": label,
            "color": color,
            "sarcasm_penalty": -10 if sarcasm_detected else 0,
            "breakdown": {
                "Sentiment Confidence": round(w["alpha"] * sentiment_confidence * 100, 1),
                "Emotion Intensity":    round(w["beta"]  * top_emotion_score    * 100, 1),
                "Aspect Richness":      round(w["gamma"] * aspect_density        * 100, 1),
                "Review Detail":        round(w["delta"] * length_factor         * 100, 1),
            }
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INTEGRITY SCORE — Novel Contribution #3 (Trust + Authenticity)
    # ══════════════════════════════════════════════════════════════════════════
    def compute_integrity_score(
        self,
        trust_score:  float,
        auth_score:   float,
    ) -> dict:
        """
        Integrity Score — unified two-dimensional reliability framework.

        Formula:
          IntegrityScore = w_t · TrustScore + w_a · AuthenticityScore
          where w_t = 0.60, w_a = 0.40

        Combines:
          • Trust Score   → Is the review well-written and emotionally coherent?
          • Auth Score    → Is the review genuinely written by a real customer?

        Together these answer: "Can this review be trusted by a business?"
        """
        w = self.integrity_weights
        integrity = round(
            w["trust"]        * trust_score
            + w["authenticity"] * auth_score,
            1,
        )

        if integrity >= 75:
            label, color = "High Integrity",     "#39ff14"
        elif integrity >= 50:
            label, color = "Moderate Integrity", "#ffcc00"
        else:
            label, color = "Low Integrity",      "#ff003c"

        return {
            "score":               integrity,
            "label":               label,
            "color":               color,
            "trust_contribution":  round(w["trust"]        * trust_score, 1),
            "auth_contribution":   round(w["authenticity"] * auth_score,  1),
            "formula":             "IntegrityScore = 0.60 × TrustScore + 0.40 × AuthenticityScore",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # FULL PIPELINE — public entry point
    # ══════════════════════════════════════════════════════════════════════════
    def full_analysis(self, text: str, star_rating: float = None) -> dict:
        """
        Runs the complete Sentilytics AI pipeline:
          1. Try Groq (Llama-3-70b) → rich, sarcasm-aware structured output.
          2. If Groq unavailable/fails → DistilBERT fallback.
          3. Always run SpaCy for fine-grained aspects, merge with LLM aspects.
          4. Compute Trust Score (novel contribution #1).
          5. Compute Authenticity Score (novel contribution #2).
          6. Compute Integrity Score (novel contribution #3).

        Args:
            text:        The review text to analyze.
            star_rating: Optional star rating (1–5) for authenticity mismatch check.
        """
        spacy_aspects    = self.extract_aspects(text)
        groq_result      = self._analyze_with_groq(text)
        sarcasm_detected = False
        reasoning        = ""

        if groq_result:
            # ── Groq path ──────────────────────────────────────────────────
            sentiment        = groq_result["sentiment"]
            s_score          = groq_result["sentiment_score"]
            emotions         = groq_result["emotions"]
            sarcasm_detected = groq_result.get("sarcasm_detected", False)
            reasoning        = groq_result.get("reasoning", "")
            groq_aspects     = groq_result.get("aspects", [])
            aspects          = self._merge_aspects(groq_aspects, spacy_aspects)
            engine_used      = groq_result["engine"]
        else:
            # ── DistilBERT fallback path ───────────────────────────────────
            sentiment, s_score = self._analyze_sentiment_distilbert(text)
            emotions           = self._analyze_emotions_distilbert(text)
            aspects            = spacy_aspects
            engine_used        = "DistilBERT (offline fallback)"

        top_emotion_score = emotions[0]["score"] if emotions else 0.0

        # ── Trust Score (novel #1) ─────────────────────────────────────────────
        trust = self.compute_trust_score(
            sentiment_confidence=s_score,
            top_emotion_score=top_emotion_score,
            aspects=aspects,
            review_text=text,
            sarcasm_detected=sarcasm_detected,
        )

        # ── Authenticity Score (novel #2) ──────────────────────────────────────
        auth = self.auth_detector.score(text, star_rating=star_rating)

        # ── Integrity Score (novel #3) ─────────────────────────────────────────
        integrity = self.compute_integrity_score(
            trust_score=trust["score"],
            auth_score=auth["score"],
        )

        return {
            "text":             text,
            "sentiment":        sentiment,
            "sentiment_score":  s_score,
            "top_emotion":      emotions[0]["label"] if emotions else "Unknown",
            "emotions":         emotions,
            "aspects":          aspects,
            "trust_score":      trust,
            "authenticity":     auth,
            "integrity_score":  integrity,
            "sarcasm_detected": sarcasm_detected,
            "reasoning":        reasoning,
            "engine_used":      engine_used,
        }
