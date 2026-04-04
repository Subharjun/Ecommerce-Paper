"""
evaluate.py — Sentilytics AI Evaluation Script
================================================
Runs the Sentilytics NLP engine and two baseline models (VADER, TextBlob)
against a stratified holdout set drawn from the real dataset and computes:

  • Accuracy, Precision, Recall, F1-Score (macro & per-class)
  • Confusion Matrix
  • Live VADER baseline  (computed on the same test set — NOT hardcoded)
  • Live TextBlob baseline (computed on the same test set — NOT hardcoded)
  • SVM + TF-IDF, LSTM, Vanilla BERT — from published benchmarks (cited in paper)

Outputs:
  metrics.json                — machine-readable metrics (read by app.py)
  performance.png             — bar chart comparing all models
  performance_confusion.png   — confusion matrix heatmap

Usage:
  source venv/bin/activate
  python evaluate.py
"""

import json
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from engine import SentilyticsEngine

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_PATH   = "ecommerce_data_real.csv"
METRICS_OUTPUT = "metrics.json"
SAMPLE_SIZE    = 300   # per-model test rows (keeps total runtime < 5 min on CPU)
RANDOM_SEED    = 42


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_test_split(path: str, n_sample: int, seed: int) -> pd.DataFrame:
    """Loads the dataset and returns a stratified test sample."""
    df = pd.read_csv(path).dropna(subset=["Review", "Sentiment"])

    _, test_df = train_test_split(
        df, test_size=0.30, stratify=df["Sentiment"], random_state=seed
    )
    sample_df = test_df.groupby("Sentiment", group_keys=False).apply(
        lambda x: x.sample(
            min(len(x), max(1, int(n_sample * len(x) / len(test_df)))),
            random_state=seed,
        )
    )
    return sample_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SENTILYTICS ENGINE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def run_engine_evaluation(engine: SentilyticsEngine,
                          test_df: pd.DataFrame):
    """Runs Sentilytics on every row; returns y_true, y_pred, latencies."""
    y_true, y_pred, latencies = [], [], []
    total = len(test_df)
    print(f"\n[Sentilytics] Evaluating on {total} test reviews…")

    for i, (_, row) in enumerate(test_df.iterrows()):
        t0 = time.time()
        result = engine.full_analysis(str(row["Review"]))
        latencies.append((time.time() - t0) * 1000)
        y_true.append(row["Sentiment"])
        y_pred.append(result["sentiment"])
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{total}] avg latency: {np.mean(latencies):.1f} ms")

    return y_true, y_pred, latencies


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE EVALUATIONS — LIVE COMPUTED (not hardcoded)
# ══════════════════════════════════════════════════════════════════════════════
def run_vader_baseline(test_df: pd.DataFrame) -> float:
    """
    Computes VADER accuracy on the test set.
    Returns accuracy as a percentage (e.g. 76.4).
    VADER maps compound score ≥ 0.05 → Positive, ≤ -0.05 → Negative, else Neutral.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("⚠️  vaderSentiment not installed. Run: pip install vaderSentiment")
        return None

    vader = SentimentIntensityAnalyzer()
    print(f"\n[VADER] Evaluating on {len(test_df)} reviews…")

    def vader_predict(text: str) -> str:
        s = vader.polarity_scores(str(text))
        if   s["compound"] >= 0.05:  return "Positive"
        elif s["compound"] <= -0.05: return "Negative"
        else:                        return "Neutral"

    y_true = test_df["Sentiment"].tolist()
    y_pred = [vader_predict(r) for r in test_df["Review"]]
    acc    = accuracy_score(y_true, y_pred) * 100
    print(f"  VADER Accuracy: {acc:.2f}%")
    return round(acc, 2)


def run_textblob_baseline(test_df: pd.DataFrame) -> float:
    """
    Computes TextBlob accuracy on the test set.
    TextBlob maps polarity > 0.1 → Positive, < -0.1 → Negative, else Neutral.
    """
    try:
        from textblob import TextBlob
    except ImportError:
        print("⚠️  textblob not installed. Run: pip install textblob")
        return None

    print(f"\n[TextBlob] Evaluating on {len(test_df)} reviews…")

    def textblob_predict(text: str) -> str:
        p = TextBlob(str(text)).sentiment.polarity
        if   p > 0.10:  return "Positive"
        elif p < -0.10: return "Negative"
        else:           return "Neutral"

    y_true = test_df["Sentiment"].tolist()
    y_pred = [textblob_predict(r) for r in test_df["Review"]]
    acc    = accuracy_score(y_true, y_pred) * 100
    print(f"  TextBlob Accuracy: {acc:.2f}%")
    return round(acc, 2)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, latencies,
                    vader_acc: float = None,
                    textblob_acc: float = None) -> tuple:
    """Computes all sklearn metrics and returns a unified metrics dict."""
    labels_present = sorted(set(y_true) | set(y_pred))

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro",
                                labels=labels_present, zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro",
                             labels=labels_present, zero_division=0)
    f1        = f1_score(y_true, y_pred, average="macro",
                         labels=labels_present, zero_division=0)

    per_class_report = classification_report(
        y_true, y_pred, labels=labels_present,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)

    # ── Baselines ──
    #   VADER, TextBlob  → live computed on real test set (see above)
    #   SVM / LSTM / BERT → benchmark values from published literature
    #   (cited in references.bib as standard benchmarks)
    baselines = {
        "VADER (live)":    vader_acc    if vader_acc    is not None else "N/A",
        "TextBlob (live)": textblob_acc if textblob_acc is not None else "N/A",
        "SVM + TF-IDF†":  81.4,    # He et al. 2017; Pon14 (SemEval ABSA)
        "LSTM†":           84.5,    # Wan16; Wang et al. 2016 EMNLP
        "Vanilla BERT†":   90.1,    # Dev19; Devlin et al. 2019 NAACL
    }

    metrics = {
        "accuracy":           round(accuracy * 100, 2),
        "precision_macro":    round(precision * 100, 2),
        "recall_macro":       round(recall   * 100, 2),
        "f1_macro":           round(f1       * 100, 2),
        "avg_latency_ms":     round(float(np.mean(latencies)), 2),
        "reviews_per_minute": round(60_000 / float(np.mean(latencies)), 0),
        "test_sample_size":   len(y_true),
        "baseline_note":      "† Published benchmark values from cited literature (references.bib). "
                              "VADER and TextBlob are live-computed on the same holdout set.",
        "per_class": {
            k: {
                "precision": round(v["precision"] * 100, 2),
                "recall":    round(v["recall"]    * 100, 2),
                "f1":        round(v["f1-score"]  * 100, 2),
                "support":   int(v["support"])
            }
            for k, v in per_class_report.items()
            if k in labels_present
        },
        "confusion_matrix": {
            "labels": labels_present,
            "values": cm.tolist()
        },
        "baselines": baselines,
    }
    return metrics, cm, labels_present


# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_performance_chart(metrics: dict, output_path: str = "performance.png"):
    """Generates the performance comparison bar chart (all baselines live)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    baselines  = {
        k: v for k, v in metrics["baselines"].items()
        if isinstance(v, (int, float))
    }
    systems    = list(baselines.keys()) + ["Sentilytics AI (Ours)"]
    accuracies = list(baselines.values()) + [metrics["accuracy"]]
    colors     = ["#4a5568"] * len(baselines) + ["#00ff88"]

    bars = ax.barh(systems, accuracies, color=colors, edgecolor="none", height=0.55)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%", va="center", ha="left",
            color="white", fontsize=11, fontweight="bold"
        )

    ax.set_xlim(55, 100)
    ax.set_xlabel("Accuracy (%)", color="white", fontsize=12)
    ax.set_title(
        "Sentilytics AI vs. Baseline Models\n"
        "(VADER + TextBlob: live-computed  |  SVM/LSTM/BERT: published benchmarks†)",
        color="white", fontsize=13, fontweight="bold", pad=15
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.xaxis.label.set_color("white")

    legend_us = mpatches.Patch(color="#00ff88", label="Sentilytics AI (Ours)")
    legend_bl = mpatches.Patch(color="#4a5568", label="Baseline Systems")
    ax.legend(handles=[legend_us, legend_bl],
              facecolor="#161b22", labelcolor="white", loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✅ Saved: {output_path}")
    plt.close(fig)


def plot_confusion_matrix(cm, labels,
                          output_path: str = "performance_confusion.png"):
    """Saves a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGn",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#222", ax=ax
    )
    ax.set_xlabel("Predicted Label", color="white", fontsize=11)
    ax.set_ylabel("True Label",      color="white", fontsize=11)
    ax.set_title("Confusion Matrix — Sentiment Classification",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✅ Saved: {output_path}")
    plt.close(fig)


def plot_sentiment_distribution(df: pd.DataFrame,
                                output_path: str = "sentiment.png"):
    """Generates the overall sentiment distribution chart from real data."""
    counts = df["Sentiment"].value_counts()
    colors = {"Positive": "#00ff88", "Negative": "#ff4b4b", "Neutral": "#ffd700"}

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    wedge_colors = [colors.get(label, "#aaa") for label in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=wedge_colors, startangle=90,
        wedgeprops=dict(edgecolor="#0e1117", linewidth=2)
    )
    for t in texts + autotexts:
        t.set_color("white")
        t.set_fontsize(12)

    ax.set_title(f"Sentiment Distribution — {len(df):,} Real Reviews",
                 color="white", fontsize=13, pad=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✅ Saved: {output_path}")
    plt.close(fig)


def plot_category_sentiment(df: pd.DataFrame, output_path: str = "sentiment2.png"):
    """Generates the category × sentiment breakdown chart."""
    cat_sent = df.groupby(["Category", "Sentiment"]).size().unstack(fill_value=0)
    colors   = {"Positive": "#00ff88", "Negative": "#ff4b4b", "Neutral": "#ffd700"}

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    x      = np.arange(len(cat_sent.index))
    width  = 0.25
    offset = -width

    for sentiment in ["Positive", "Negative", "Neutral"]:
        if sentiment in cat_sent.columns:
            ax.bar(x + offset, cat_sent[sentiment],
                   width, label=sentiment,
                   color=colors.get(sentiment, "#aaa"), edgecolor="none")
            offset += width

    ax.set_xticks(x)
    ax.set_xticklabels(cat_sent.index, color="white", fontsize=11)
    ax.set_ylabel("Review Count", color="white", fontsize=11)
    ax.set_title("Sentiment by Product Category (Keyword-Classified)",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(facecolor="#161b22", labelcolor="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✅ Saved: {output_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Sentilytics AI — Full Evaluation Pipeline")
    print("  (Live baselines: VADER + TextBlob | Literature: SVM/LSTM/BERT)")
    print("=" * 65)

    # 1. Load full dataset for chart generation
    print(f"\nLoading dataset from {DATASET_PATH}…")
    full_df = pd.read_csv(DATASET_PATH).dropna(subset=["Review", "Sentiment"])
    print(f"Total rows loaded: {len(full_df):,}")

    # 2. Build stratified test split (shared across all models for fairness)
    test_df = load_test_split(DATASET_PATH, SAMPLE_SIZE, RANDOM_SEED)
    print(f"Stratified test sample: {len(test_df)} reviews")
    print("Sentiment distribution in test set:")
    print(test_df["Sentiment"].value_counts().to_string())

    # 3. Run VADER baseline (live)
    vader_acc = run_vader_baseline(test_df)

    # 4. Run TextBlob baseline (live)
    textblob_acc = run_textblob_baseline(test_df)

    # 5. Initialise and run Sentilytics engine
    print("\nInitialising Sentilytics NLP engine…")
    engine = SentilyticsEngine()
    y_true, y_pred, latencies = run_engine_evaluation(engine, test_df)

    # 6. Compute all metrics
    metrics, cm, labels_present = compute_metrics(
        y_true, y_pred, latencies,
        vader_acc=vader_acc,
        textblob_acc=textblob_acc,
    )

    # 7. Save metrics.json
    with open(METRICS_OUTPUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Saved: {METRICS_OUTPUT}")

    # 8. Print summary
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Accuracy   : {metrics['accuracy']:.2f}%")
    print(f"  Precision  : {metrics['precision_macro']:.2f}%  (macro)")
    print(f"  Recall     : {metrics['recall_macro']:.2f}%  (macro)")
    print(f"  F1-Score   : {metrics['f1_macro']:.2f}%  (macro)")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f} ms / review")
    print(f"  Throughput : ~{int(metrics['reviews_per_minute'])} reviews/min")
    print(f"\n  VADER (live)    : {vader_acc}%")
    print(f"  TextBlob (live) : {textblob_acc}%")
    print(f"  Sentilytics     : {metrics['accuracy']}%")
    print("=" * 65)

    # 9. Generate charts
    print("\nGenerating charts…")
    plot_performance_chart(metrics)
    plot_confusion_matrix(cm, labels_present)
    plot_sentiment_distribution(full_df)
    plot_category_sentiment(full_df)

    print("\n🎉 Evaluation complete. metrics.json and all charts updated.")
    print("   Now run: streamlit run app.py")
