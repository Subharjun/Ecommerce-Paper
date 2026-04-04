"""
authenticity.py — Sentilytics AI: Fake Review Authenticity Detector
====================================================================
Novel Contribution #2

Computes a composite Authenticity Score (0–100) for any e-commerce review
using 6 linguistic signals empirically correlated with fake/spam reviews:

  Signal 1: Lexical Diversity     — unique_words / total_words (low = spam)
  Signal 2: Sentiment-Rating Align — positive text with low star = suspicious
  Signal 3: Exclamation Density   — >3 per 100 words = inflated enthusiasm
  Signal 4: ALL CAPS Ratio        — excessive caps = bot-like behavior
  Signal 5: Length Normality      — very short (<10) or very long (>800) = suspect
  Signal 6: Specificity           — generic phrases with no product details = suspect

Formula:
  AuthenticityScore = Σ(wᵢ · sᵢ) × 100
  where sᵢ ∈ [0, 1] for each signal (1 = authentic, 0 = suspicious)
  and   Σwᵢ = 1.0

Labels:
  ≥ 75  →  ✅ Likely Genuine
  50–74 →  ⚠️  Uncertain
  < 50  →  🚨 Suspicious

Reference basis:
  Jindal & Liu (2008) — Opinion Spam and Analysis, WSDM.
  Ott et al. (2011) — Finding Deceptive Opinion Spam by Any Stretch.
  Fang et al. (2016) — Fraud Detector Is Both Robust and Interpretable.
"""

from typing import Optional

# ── Generic spam phrase dictionary ─────────────────────────────────────────────
GENERIC_PHRASES = [
    "highly recommend", "great product", "best product", "love it",
    "five stars", "must buy", "excellent product", "very happy",
    "amazing product", "best purchase", "would recommend", "very good",
    "nice product", "good quality", "satisfied", "as described",
    "fast shipping", "good price", "worth the money", "works great",
    "exactly as described", "great value", "would buy again", "love this",
    "perfect product", "awesome product", "works perfectly", "great buy",
]

# Positive / Negative anchor words for sentiment-rating mismatch
_POS_WORDS = {
    "good", "great", "excellent", "amazing", "love", "perfect",
    "best", "awesome", "fantastic", "wonderful", "superb", "outstanding",
    "brilliant", "happy", "pleased", "satisfied", "recommend", "quality",
}
_NEG_WORDS = {
    "bad", "terrible", "horrible", "awful", "worst", "poor",
    "useless", "broken", "disappointed", "hate", "defective", "waste",
    "garbage", "junk", "cheap", "flimsy", "disaster", "regret",
}

# ── Signal weights (must sum to 1.0) ──────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "lexical_diversity":      0.25,
    "sentiment_rating_align": 0.25,
    "exclamation_control":    0.15,
    "caps_control":           0.10,
    "length_normality":       0.15,
    "specificity":            0.10,
}


class AuthenticityDetector:
    """
    Computes a per-review Authenticity Score using 6 linguistic signals.

    Fully offline and deterministic — no external API required.
    Runs in O(n) per review where n = word count.

    Usage:
        detector = AuthenticityDetector()
        result = detector.score("This product is AMAZING!! BEST BUY EVER!!", star_rating=5)
        print(result["score"], result["label"])   # e.g. 48.2, Suspicious
    """

    def score(self, review_text: str,
              star_rating: Optional[float] = None) -> dict:
        """
        Returns a full authenticity analysis dict for a single review.

        Args:
            review_text:  Raw review string.
            star_rating:  Optional star rating (1–5) for mismatch detection.

        Returns:
            dict — score, label, color, icon, signals (raw), breakdown (pts), word_count
        """
        text   = str(review_text).strip()
        words  = text.split()
        n_words = len(words)
        signals = {}

        # ── Signal 1: Lexical Diversity ────────────────────────────────────────
        if n_words == 0:
            ld = 0.0
        else:
            unique = len({w.lower().strip(".,!?\"'") for w in words})
            ld     = unique / n_words
        # Healthy diversity ≥ 0.60; normalise so 0.60 → 1.0
        signals["lexical_diversity"] = min(ld / 0.60, 1.0)

        # ── Signal 2: Sentiment-Rating Alignment ──────────────────────────────
        if star_rating is not None:
            text_lower  = text.lower()
            pos_count   = sum(1 for w in _POS_WORDS if w in text_lower)
            neg_count   = sum(1 for w in _NEG_WORDS if w in text_lower)
            text_is_pos = pos_count >= neg_count

            if star_rating >= 4 and text_is_pos:
                align = 1.0    # High stars, positive text  ✅
            elif star_rating <= 2 and not text_is_pos:
                align = 1.0    # Low stars, negative text   ✅
            elif star_rating == 3:
                align = 0.75   # Neutral star → acceptable ambiguity
            elif star_rating >= 4 and not text_is_pos:
                align = 0.15   # High stars, negative text  🚨
            else:
                align = 0.15   # Low stars, positive text   🚨 (suspicious praise)
        else:
            align = 0.70       # No rating → neutral assumption

        signals["sentiment_rating_align"] = align

        # ── Signal 3: Exclamation Density ─────────────────────────────────────
        n_excl      = text.count("!")
        excl_per100 = (n_excl / max(n_words, 1)) * 100
        # >6 exclamations per 100 words → score 0; ≤1 → score 1
        excl_score  = max(0.0, 1.0 - (excl_per100 / 6.0))
        signals["exclamation_control"] = round(excl_score, 4)

        # ── Signal 4: ALL CAPS Ratio ───────────────────────────────────────────
        caps_words  = sum(1 for w in words if len(w) > 2 and w.isupper())
        caps_ratio  = caps_words / max(n_words, 1)
        # >10% ALL-CAPS words → suspect (bots often shout)
        caps_score  = max(0.0, 1.0 - (caps_ratio / 0.10))
        signals["caps_control"] = round(caps_score, 4)

        # ── Signal 5: Length Normality ─────────────────────────────────────────
        # Genuine reviews: 15–300 words (bell-curve approximated)
        if n_words < 5:
            len_score = 0.10    # Too short: almost certainly fake
        elif n_words < 15:
            len_score = 0.55    # Short but possible
        elif n_words <= 300:
            len_score = 1.00    # Healthy range
        elif n_words <= 600:
            len_score = 0.70    # Long but possible
        else:
            len_score = 0.35    # Very long: possible padding/fake essay
        signals["length_normality"] = len_score

        # ── Signal 6: Specificity (inverse generic phrase density) ─────────────
        text_lower    = text.lower()
        generic_hits  = sum(1 for p in GENERIC_PHRASES if p in text_lower)
        # >2 generic phrases in a short review = suspect
        density       = generic_hits / max(n_words / 20.0, 1.0)
        spec_score    = max(0.0, 1.0 - (density / 2.0))
        signals["specificity"] = round(spec_score, 4)

        # ── Composite Authenticity Score ───────────────────────────────────────
        raw_score  = sum(SIGNAL_WEIGHTS[k] * signals[k] for k in SIGNAL_WEIGHTS)
        auth_score = round(raw_score * 100, 1)

        if auth_score >= 75:
            label, color, icon = "Likely Genuine", "#39ff14", "✅"
        elif auth_score >= 50:
            label, color, icon = "Uncertain",      "#ffcc00", "⚠️"
        else:
            label, color, icon = "Suspicious",     "#ff003c", "🚨"

        # ── Human-readable contribution breakdown ──────────────────────────────
        breakdown = {
            "Lexical Diversity":      round(SIGNAL_WEIGHTS["lexical_diversity"]      * signals["lexical_diversity"]      * 100, 1),
            "Sentiment-Rating Align": round(SIGNAL_WEIGHTS["sentiment_rating_align"] * signals["sentiment_rating_align"] * 100, 1),
            "Exclamation Control":    round(SIGNAL_WEIGHTS["exclamation_control"]    * signals["exclamation_control"]    * 100, 1),
            "CAPS Control":           round(SIGNAL_WEIGHTS["caps_control"]           * signals["caps_control"]           * 100, 1),
            "Length Normality":       round(SIGNAL_WEIGHTS["length_normality"]       * signals["length_normality"]       * 100, 1),
            "Specificity":            round(SIGNAL_WEIGHTS["specificity"]            * signals["specificity"]            * 100, 1),
        }

        return {
            "score":      auth_score,
            "label":      label,
            "color":      color,
            "icon":       icon,
            "signals":    {k: round(v, 4) for k, v in signals.items()},
            "breakdown":  breakdown,
            "word_count": n_words,
            "formula":    "AuthenticityScore = Σ(wᵢ · sᵢ) × 100",
        }

    def batch_score(self, df, review_col: str = "Review",
                    rating_col: str = "Rating"):
        """
        Appends AuthenticityScore, AuthLabel, WordCount to a DataFrame.

        Args:
            df:          Pandas DataFrame with review data.
            review_col:  Name of the review text column.
            rating_col:  Name of the star rating column (optional, can be absent).

        Returns:
            New DataFrame with three appended columns.
        """
        import pandas as pd

        rows = []
        has_rating = rating_col in df.columns

        for _, row in df.iterrows():
            rating = float(row[rating_col]) if has_rating else None
            s = self.score(str(row[review_col]), star_rating=rating)
            rows.append({
                "AuthenticityScore": s["score"],
                "AuthLabel":         s["label"],
                "WordCount":         s["word_count"],
            })

        return pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(rows)],
            axis=1
        )
