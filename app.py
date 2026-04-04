import json
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from engine import SentilyticsEngine
from authenticity import AuthenticityDetector
from temporal import compute_drift
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentilytics AI | E-Commerce Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS — OS95 Retro Terminal UI ─────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

    html, body, [class*="css"]  { font-family: 'VT323', monospace !important; letter-spacing: 0.05em; }
    .main        { background-color: #000000; }
    .stApp       { background-color: #000000; }

    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 2px dashed #39ff14;
    }

    h1, h2, h3, h4, h5, p, span, div { color: #39ff14 !important; }

    /* Metric cards */
    .metric-card {
        background: #000000;
        padding: 20px 24px;
        border-radius: 0px;
        border: 2px solid #39ff14;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 4px 4px 0px #005500;
    }
    .metric-value { font-size: 2.5rem; font-weight: 400; color: #39ff14 !important; text-shadow: 0 0 5px #39ff14; }
    .metric-label { font-size: 1.2rem; color: #39ff14 !important; text-transform: uppercase; margin-top: 4px; }
    .metric-delta { font-size: 1rem; color: #00aa00 !important; margin-top: 2px; }

    /* Analysis result cards */
    .result-card {
        background: #000000;
        padding: 24px;
        border-radius: 0px;
        border: 2px dashed #00aa00;
        margin-bottom: 16px;
        transition: all 0.1s ease;
    }
    .result-card:hover {
        border-color: #39ff14; border-style: solid;
        box-shadow: 6px 6px 0px #004400;
    }

    /* Score bar */
    .score-bar-wrap {
        background: #000000;
        border: 1px solid #39ff14;
        border-radius: 0px;
        height: 16px;
        overflow: hidden;
        margin: 8px 0 4px;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 0px;
        background: repeating-linear-gradient(
          45deg,
          #39ff14,
          #39ff14 10px,
          #000000 10px,
          #000000 20px
        );
        transition: width 0.6s ease;
    }

    /* Sentiment badges */
    .badge-pos { background:#000000; border: 1px solid #39ff14; color:#39ff14 !important; padding:4px 14px; box-shadow: 2px 2px 0px #39ff14; font-weight:400; display:inline-block; }
    .badge-neg { background:#000000; border: 1px solid #ff003c; color:#ff003c !important; padding:4px 14px; box-shadow: 2px 2px 0px #ff003c; font-weight:400; display:inline-block; }
    .badge-neu { background:#000000; border: 1px solid #ffcc00; color:#ffcc00 !important; padding:4px 14px; box-shadow: 2px 2px 0px #ffcc00; font-weight:400; display:inline-block; }

    /* Novel badge */
    .novel-badge {
        background: #000000;
        border: 1px dashed #39ff14;
        color: #39ff14 !important; padding: 2px 10px; border-radius: 0px;
        font-size: 1rem; font-weight: 400; letter-spacing: 0.05em; text-transform: uppercase;
        display: inline-block; margin-left: 8px; vertical-align: middle;
    }

    .stProgress > div > div { background-color: #39ff14; border-radius: 0px; }

    /* UI Inputs */
    .stTextArea textarea { background-color: #000000 !important; color: #39ff14 !important; border: 2px solid #39ff14 !important; font-family: 'VT323', monospace !important; border-radius: 0px !important; font-size: 1.2rem;}
    .stSlider > div > div > div { background-color: #39ff14 !important; }
    .stButton button { background-color: #000000 !important; color: #39ff14 !important; border: 2px solid #39ff14 !important; border-radius: 0px !important; font-family: 'VT323', monospace !important; font-size: 1.5rem; text-transform: uppercase; box-shadow: 4px 4px 0px #006600 !important; transition: all 0.1s;}
    .stButton button:hover { background-color: #39ff14 !important; color: #000000 !important; box-shadow: none !important; transform: translate(4px, 4px);}
    .stDataFrame { border: 1px solid #39ff14; font-family: 'VT323', monospace !important; }
    </style>
""", unsafe_allow_html=True)


# ── Cached Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return SentilyticsEngine()


@st.cache_resource
def load_auth_detector():
    return AuthenticityDetector()


@st.cache_data
def load_data():
    try:
        return pd.read_csv("ecommerce_data_real.csv")
    except Exception:
        return pd.DataFrame(columns=["Review", "Sentiment", "Category", "Rating"])


@st.cache_data
def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return {
        "accuracy":          "AWAITING_RUN",
        "precision_macro":   "AWAITING_RUN",
        "recall_macro":      "AWAITING_RUN",
        "f1_macro":          "AWAITING_RUN",
        "avg_latency_ms":    "WAITING",
        "reviews_per_minute": "WAITING",
        "_note": "SYSTEM: Run `python evaluate.py` to stream live metrics data.",
    }


@st.cache_data
def compute_dataset_auth_stats(df):
    """Batch authenticity on 500-review sample — cached."""
    detector = AuthenticityDetector()
    sample   = df.sample(min(500, len(df)), random_state=42)
    return detector.batch_score(sample)


@st.cache_data
def compute_drift_cached(df):
    return compute_drift(df)


engine    = load_engine()
auth_det  = load_auth_detector()
data_df   = load_data()
metrics   = load_metrics()


# ── Helpers ────────────────────────────────────────────────────────────────────
def metric_card(value, label, delta=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)


def sentiment_badge(label):
    cls = {"Positive": "badge-pos", "Negative": "badge-neg"}.get(label, "badge-neu")
    return f'<span class="{cls}">{label}</span>'


def score_bar(score, color):
    return f"""
    <div class="score-bar-wrap">
        <div class="score-bar-fill" style="width:{score}%; background:{color};"></div>
    </div>
    """


def novel_badge():
    return '<span class="novel-badge">NOVEL</span>'


# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=72)
    st.title("Sentilytics AI")
    st.caption("E-Commerce Review Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🔍 Real-Time Analysis",
            "📊 Batch Insights",
            "🕵️ Authenticity Lab",
            "📈 Evaluation Report",
            "⚙️ System Info",
        ]
    )

    st.markdown("---")
    acc = metrics.get("accuracy", "—")
    acc_display = f"{acc}%" if isinstance(acc, (int, float)) else str(acc)
    st.markdown(f"**System Accuracy:** `{acc_display}`")
    st.markdown(f"**Dataset:** `{len(data_df):,}` reviews")
    st.markdown("---")
    st.info("By Mandira Banik & Subharjun Bose\nGuru Nanak Institute of Technology\nACMESGA 2k26")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Real-Time Analysis
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Real-Time Analysis":
    st.title("🔍 Deep Sentiment & Emotion Analysis")
    st.markdown(
        "Analyze any customer review with transformer-based precision. "
        "Get sentiment, emotions, aspects, **Trust Score**, **Authenticity Score**, "
        "and our unified **Integrity Score**."
    )

    col_input, col_meta = st.columns([3, 1])
    with col_input:
        review_text = st.text_area(
            "Enter Customer Review:",
            placeholder="The battery life is amazing but the camera quality is disappointing...",
            height=130
        )
    with col_meta:
        star_rating = st.slider("Star Rating (optional)", 1, 5, 5)
        use_rating  = st.checkbox("Use rating for authenticity check", value=True)

    col_btn, _, _ = st.columns([1, 1, 4])
    with col_btn:
        analyze_btn = st.button("⚡ Analyze", type="primary", width='stretch')

    if analyze_btn and review_text.strip():
        rating_input = float(star_rating) if use_rating else None

        with st.spinner("Processing with Sentilytics AI…"):
            t0      = time.time()
            results = engine.full_analysis(review_text, star_rating=rating_input)
            latency = (time.time() - t0) * 1000

        st.markdown("---")

        # ── Engine + Sarcasm banner ──────────────────────────────────────────
        eng_col, sarc_col = st.columns([3, 1])
        with eng_col:
            engine_used = results.get("engine_used", "DistilBERT")
            if "Groq" in engine_used:
                st.success(f"⚡ Powered by **{engine_used}** via Groq")
            else:
                st.info(f"🔧 Engine: {engine_used}")
        with sarc_col:
            if results.get("sarcasm_detected"):
                st.error("🎭 **Sarcasm Detected** — sentiment adjusted")

        # ── Row 1: Sentiment + Trust Score ──────────────────────────────────
        col_sent, col_trust = st.columns(2)

        with col_sent:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Core Sentiment")
            badge = sentiment_badge(results["sentiment"])
            st.markdown(f"**Verdict:** {badge}", unsafe_allow_html=True)
            st.progress(results["sentiment_score"])
            st.caption(f"Model confidence: {results['sentiment_score']:.2%}")
            if results.get("reasoning"):
                with st.expander("💡 LLM Reasoning"):
                    st.markdown(f"*{results['reasoning']}*")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_trust:
            ts    = results["trust_score"]
            score = ts["score"]
            color = ts["color"]
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"**Trust Score** {novel_badge()}", unsafe_allow_html=True)
            st.markdown(
                f"<span style='font-size:2.2rem;font-weight:700;color:{color};'>"
                f"{score}/100</span> &nbsp; "
                f"<span style='color:{color};font-weight:600;'>{ts['label']}</span>",
                unsafe_allow_html=True
            )
            st.markdown(score_bar(score, color), unsafe_allow_html=True)
            breakdown_df = pd.DataFrame([
                {"Component": k, "Pts": v}
                for k, v in ts["breakdown"].items()
            ])
            st.dataframe(breakdown_df, hide_index=True, width='stretch')
            st.caption("TrustScore = α·Cₛ + β·Eₜₒₚ + γ·(|A|/5) + δ·(|W|/100)")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 2: Authenticity + Integrity ─────────────────────────────────
        col_auth, col_integ = st.columns(2)

        with col_auth:
            auth  = results["authenticity"]
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"**Authenticity Score** {novel_badge()}", unsafe_allow_html=True)
            st.markdown(
                f"<span style='font-size:2.2rem;font-weight:700;color:{auth['color']};'>"
                f"{auth['score']}/100</span> &nbsp; "
                f"<span style='color:{auth['color']};font-weight:600;'>"
                f"{auth['icon']} {auth['label']}</span>",
                unsafe_allow_html=True
            )
            st.markdown(score_bar(auth["score"], auth["color"]), unsafe_allow_html=True)
            auth_df = pd.DataFrame([
                {"Signal": k, "Pts": v} for k, v in auth["breakdown"].items()
            ])
            st.dataframe(auth_df, hide_index=True, width='stretch')
            st.caption(auth["formula"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col_integ:
            integ = results["integrity_score"]
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"**Integrity Score** {novel_badge()}", unsafe_allow_html=True)
            st.markdown(
                f"<span style='font-size:2.6rem;font-weight:700;color:{integ['color']};'>"
                f"{integ['score']}/100</span><br>"
                f"<span style='color:{integ['color']};font-weight:600;font-size:1.1rem;'>"
                f"{integ['label']}</span>",
                unsafe_allow_html=True
            )
            st.markdown(score_bar(integ["score"], integ["color"]), unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-top:12px; font-size:0.85rem; color:#9ca3af;">
                Trust contribution &nbsp;&nbsp;&nbsp; <b style="color:white;">{integ['trust_contribution']} pts</b><br>
                Auth contribution &nbsp;&nbsp;&nbsp;&nbsp; <b style="color:white;">{integ['auth_contribution']} pts</b>
            </div>
            """, unsafe_allow_html=True)
            st.caption(integ["formula"])
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 3: Emotions + Aspects ────────────────────────────────────────
        col_emo, col_asp = st.columns(2)

        with col_emo:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Emotion Breakdown")
            emo_df = pd.DataFrame(results["emotions"])
            if not emo_df.empty:
                fig_emo = px.bar(
                    emo_df, x="score", y="label", orientation="h",
                    color="label", template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_emo.update_layout(
                    showlegend=False, height=280,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_emo, width='stretch')
            st.caption(f"Dominant emotion: **{results['top_emotion']}**")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_asp:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("E-Commerce Aspects")
            aspects = results["aspects"]
            if aspects:
                asp_df = pd.DataFrame(aspects)
                st.dataframe(asp_df, hide_index=True, width='stretch')
                st.caption(f"{len(aspects)} product feature(s) via dependency parsing.")
            else:
                st.info("No specific product aspects detected.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            f"⏱ Latency: **{latency:.1f} ms** &nbsp;|&nbsp; "
            f"Engine: {engine_used} &nbsp;|&nbsp; "
            f"Aspects: SpaCy en_core_web_sm"
        )

    elif analyze_btn:
        st.warning("Please enter a review to analyze.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Batch Insights":
    st.title("📊 E-Commerce Sentiment Trends")
    st.markdown("Aggregated view of **real customer feedback** across product categories.")

    if data_df.empty:
        st.warning("Dataset not found. Run `python fetch_real_data.py` first.")
        st.stop()

    total_rev  = len(data_df)
    pos_ratio  = (data_df["Sentiment"] == "Positive").mean() * 100
    neg_ratio  = (data_df["Sentiment"] == "Negative").mean() * 100
    avg_rating = data_df["Rating"].mean()
    acc_val    = metrics.get("accuracy", None)
    acc_display = f"{acc_val:.1f}%" if isinstance(acc_val, float) else "—"

    # ── Top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(f"{total_rev:,}",    "Total Reviews",    "Amazon dataset")
    with c2: metric_card(acc_display,          "System Accuracy",  "Real evaluation")
    with c3: metric_card(f"{pos_ratio:.1f}%",  "Positive Sentiment", "of all reviews")
    with c4: metric_card(f"{avg_rating:.2f} ⭐", "Avg Star Rating", f"n={total_rev:,}")

    st.markdown("---")

    # ── Charts ───────────────────────────────────────────────────────────────
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("Authentic Sentiment Distribution")
        dist = data_df["Sentiment"].value_counts().reset_index()
        dist.columns = ["Label", "Count"]
        fig_pie = px.pie(
            dist, values="Count", names="Label", color="Label",
            color_discrete_map={
                "Positive": "#39ff14", "Negative": "#ff003c", "Neutral": "#ffcc00"
            },
            hole=0.42, template="plotly_dark"
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, width='stretch')

    with col_bar:
        st.subheader("Keyword-Classified Category Breakdown")
        cat_sent = data_df.groupby(["Category", "Sentiment"]).size().unstack(fill_value=0)
        fig_bar  = go.Figure()
        palette  = {"Positive": "#39ff14", "Negative": "#ff003c", "Neutral": "#ffcc00"}
        for s in ["Positive", "Negative", "Neutral"]:
            if s in cat_sent.columns:
                fig_bar.add_trace(go.Bar(
                    name=s, x=cat_sent.index, y=cat_sent[s],
                    marker_color=palette[s]
                ))
        fig_bar.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, width='stretch')

    # ── Word cloud ───────────────────────────────────────────────────────────
    st.subheader("Word Cloud — Real Review Vocabulary")
    sample_text = " ".join(data_df["Review"].astype(str).tolist()[:500])
    wc = WordCloud(width=900, height=320, background_color="#000000",
                   colormap="summer").generate(sample_text)
    fig_wc, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig_wc.patch.set_facecolor("#000000")
    st.pyplot(fig_wc)

    # ── Temporal Sentiment Drift ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### ⏱️ Temporal Sentiment Drift  {novel_badge()}", unsafe_allow_html=True)
    st.markdown(
        "We use **star rating cohorts as temporal proxies** to model how sentiment "
        "evolves as product quality changes — an established NLP technique adapted "
        "from McAuley & Leskovec (2013, RecSys)."
    )

    drift = compute_drift_cached(data_df)

    col_sds, col_desc = st.columns([1, 2])
    with col_sds:
        st.markdown(f"""
        <div class="result-card" style="text-align:center">
            <div style="font-size:3rem;font-weight:700;color:{drift['color']}">
                {drift['sds']}
            </div>
            <div style="color:{drift['color']};font-size:1.1rem;margin-top:6px">
                {drift['label']}
            </div>
            <div style="font-size:0.75rem;color:#9ca3af;margin-top:10px">
                Sentiment Drift Score (SDS)
            </div>
            <div style="font-size:0.7rem;color:#6b7280;margin-top:4px">
                {drift['formula']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_desc:
        st.info(drift["interpretation"])
        st.dataframe(drift["cohort_df"], hide_index=True, width='stretch')

    if not drift["cat_drift_df"].empty:
        fig_drift = px.bar(
            drift["cat_drift_df"], x="Category", y="Drift Score",
            template="plotly_dark", color="Drift Score",
            color_continuous_scale=[[0, "#39ff14"], [0.5, "#ffcc00"], [1.0, "#ff003c"]],
            title="Sentiment Drift Score by Product Category",
            hover_data=["Reviews", "Status"],
        )
        fig_drift.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_drift, width='stretch')

    # ── Sample reviews table ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Sample Authentic Reviews")
    display_df = data_df[["Category", "Sentiment", "Rating", "Review"]].head(25)
    st.dataframe(display_df, hide_index=True, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Authenticity Lab (NEW)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🕵️ Authenticity Lab":
    st.title("🕵️ Fake Review Authenticity Lab")
    st.markdown(
        f"**Novel Contribution #2** {novel_badge()} — Detect suspicious or fake reviews "
        "using 6 independently weighted linguistic signals. "
        "Based on Jindal & Liu (2008, WSDM) and Ott et al. (2011).",
        unsafe_allow_html=True,
    )

    # ── Formula explainer ────────────────────────────────────────────────────
    with st.expander("📐 How Does the Authenticity Score Work?"):
        st.markdown("""
        | # | Signal | Weight | What It Detects |
        |---|--------|--------|-----------------|
        | 1 | **Lexical Diversity** | 25% | Low unique word ratio = spam repetition |
        | 2 | **Sentiment-Rating Align** | 25% | Positive text + 1★ = suspicious mismatch |
        | 3 | **Exclamation Control** | 15% | >6 exclamation marks per 100 words |
        | 4 | **CAPS Control** | 10% | >10% ALL-CAPS words = bot-like behavior |
        | 5 | **Length Normality** | 15% | <5 or >600 words = abnormal pattern |
        | 6 | **Specificity** | 10% | Generic phrases with no product details |

        **Formula:** `AuthenticityScore = Σ(wᵢ · sᵢ) × 100`
        - ≥ 75 → ✅ Likely Genuine
        - 50–74 → ⚠️ Uncertain
        - < 50 → 🚨 Suspicious
        """)

    st.markdown("---")

    # ── Single review analysis ────────────────────────────────────────────────
    st.subheader("Analyze a Review")
    col_rev, col_rate = st.columns([3, 1])
    with col_rev:
        auth_review = st.text_area(
            "Paste any review:",
            placeholder="This product is AMAZING!! BEST BUY EVER!! HIGHLY RECOMMEND TO EVERYONE!! TOP PRODUCT!!",
            height=120,
            key="auth_input"
        )
    with col_rate:
        auth_star   = st.slider("Star Rating", 1, 5, 5, key="auth_star")
        use_auth_r  = st.checkbox("Include rating", value=True, key="auth_use_r")

    auth_btn = st.button("🔍 Check Authenticity", type="primary")

    if auth_btn and auth_review.strip():
        rating = float(auth_star) if use_auth_r else None
        result = auth_det.score(auth_review, star_rating=rating)

        col_score, col_break = st.columns(2)

        with col_score:
            st.markdown(f"""
            <div class="result-card" style="text-align:center">
                <div style="font-size:3.5rem;font-weight:700;color:{result['color']}">
                    {result['score']}/100
                </div>
                <div style="font-size:1.3rem;color:{result['color']};margin-top:8px">
                    {result['icon']} {result['label']}
                </div>
                <div class="score-bar-wrap" style="margin:16px 0">
                    <div class="score-bar-fill"
                         style="width:{result['score']}%;background:{result['color']}">
                    </div>
                </div>
                <div style="font-size:0.75rem;color:#9ca3af;margin-top:8px">
                    Word count: {result['word_count']}
                </div>
                <div style="font-size:0.7rem;color:#6b7280;margin-top:4px">
                    {result['formula']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_break:
            st.markdown("**Signal Breakdown**")
            bd_df = pd.DataFrame([
                {"Signal": k, "Contribution (pts)": v}
                for k, v in result["breakdown"].items()
            ])
            st.dataframe(bd_df, hide_index=True, width='stretch')

            # Radar chart
            signals_raw = result["signals"]
            fig_radar = go.Figure(go.Scatterpolar(
                r=list(signals_raw.values()),
                theta=list(signals_raw.keys()),
                fill="toself",
                line_color=result["color"],
                fillcolor=result["color"] + "33",
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    bgcolor="rgba(0,0,0,0)",
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_radar, width='stretch')

    elif auth_btn:
        st.warning("Please enter a review.")

    # ── Dataset-level stats ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Dataset Authenticity Overview")
    st.caption("Computed on a stratified 500-review random sample from the real dataset.")

    if not data_df.empty:
        with st.spinner("Scoring 500 reviews for authenticity…"):
            scored_df = compute_dataset_auth_stats(data_df)

        genuine_pct    = (scored_df["AuthLabel"] == "Likely Genuine").mean() * 100
        uncertain_pct  = (scored_df["AuthLabel"] == "Uncertain").mean() * 100
        suspicious_pct = (scored_df["AuthLabel"] == "Suspicious").mean() * 100

        c1, c2, c3 = st.columns(3)
        with c1: metric_card(f"{genuine_pct:.1f}%",    "✅ Likely Genuine",
                              f"~{int(genuine_pct * 346):,} of dataset")
        with c2: metric_card(f"{uncertain_pct:.1f}%",  "⚠️ Uncertain",
                              f"~{int(uncertain_pct * 346):,} of dataset")
        with c3: metric_card(f"{suspicious_pct:.1f}%", "🚨 Suspicious",
                             f"~{int(suspicious_pct * 346):,} of dataset")

        col_pie2, col_hist = st.columns(2)

        with col_pie2:
            auth_dist = scored_df["AuthLabel"].value_counts().reset_index()
            auth_dist.columns = ["Label", "Count"]
            fig_ap = px.pie(
                auth_dist, values="Count", names="Label", color="Label",
                color_discrete_map={
                    "Likely Genuine": "#39ff14",
                    "Uncertain":      "#ffcc00",
                    "Suspicious":     "#ff003c",
                },
                hole=0.42, template="plotly_dark",
                title="Authenticity Label Distribution"
            )
            fig_ap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_ap, width='stretch')

        with col_hist:
            fig_hist = px.histogram(
                scored_df, x="AuthenticityScore", nbins=20,
                template="plotly_dark",
                title="Authenticity Score Distribution",
                color_discrete_sequence=["#39ff14"],
                labels={"AuthenticityScore": "Authenticity Score (0–100)"}
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_hist, width='stretch')

        # Score by category
        if "Category" in scored_df.columns:
            cat_auth = scored_df.groupby("Category")["AuthenticityScore"].mean().reset_index()
            cat_auth.columns = ["Category", "Avg Authenticity Score"]
            fig_cat = px.bar(
                cat_auth, x="Category", y="Avg Authenticity Score",
                template="plotly_dark", color="Avg Authenticity Score",
                color_continuous_scale=[[0, "#ff003c"], [0.5, "#ffcc00"], [1.0, "#39ff14"]],
                range_color=[85, 100],
                title="Average Authenticity Score by Category",
            )
            # Zoom in the y-axis to actually see the variance instead of a flat block
            fig_cat.update_yaxes(range=[80, 100])
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_cat, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Evaluation Report
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Evaluation Report":
    st.title("📈 Real Evaluation Report")

    if not os.path.exists("metrics.json"):
        st.error(
            "**`metrics.json` not found.** Please run the evaluation pipeline:\n\n"
            "```bash\nsource venv/bin/activate\npython evaluate.py\n```"
        )
        st.stop()

    st.markdown(
        "All metrics below are **computed on the real dataset holdout set** — "
        "VADER and TextBlob baselines are live-evaluated, not hardcoded."
    )
    st.markdown("---")

    # ── Summary metrics ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(f"{metrics['accuracy']}%",       "Accuracy",  "macro")
    with c2: metric_card(f"{metrics['precision_macro']}%", "Precision", "macro")
    with c3: metric_card(f"{metrics['recall_macro']}%",   "Recall",    "macro")
    with c4: metric_card(f"{metrics['f1_macro']}%",        "F1-Score",  "macro")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # ── Performance chart ────────────────────────────────────────────────────
    with col_l:
        st.subheader("vs. Baseline Models")
        if os.path.exists("performance.png"):
            st.image("performance.png", width='stretch')
        else:
            baselines = {
                k: v for k, v in metrics.get("baselines", {}).items()
                if isinstance(v, (int, float))
            }
            all_sys = list(baselines.keys()) + ["Sentilytics AI"]
            all_acc = list(baselines.values()) + [metrics["accuracy"]]
            fig_cmp = px.bar(
                x=all_sys, y=all_acc, template="plotly_dark",
                color=all_sys,
                color_discrete_sequence=["#006600"] * len(baselines) + ["#39ff14"],
                labels={"x": "System", "y": "Accuracy (%)"}
            )
            fig_cmp.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_cmp, width='stretch')
        note = metrics.get("baseline_note", "")
        if note:
            st.caption(note)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    with col_r:
        st.subheader("Confusion Matrix")
        if os.path.exists("performance_confusion.png"):
            st.image("performance_confusion.png", width='stretch')
        else:
            st.info("Run `python evaluate.py` to generate the confusion matrix chart.")

    # ── Per-class table ──────────────────────────────────────────────────────
    st.subheader("Per-Class Performance")
    per_class = metrics.get("per_class", {})
    if per_class:
        pc_df = pd.DataFrame([
            {"Class": cls, **vals} for cls, vals in per_class.items()
        ])
        st.dataframe(pc_df, hide_index=True, width='stretch')

    # ── Latency ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Runtime Performance")
    l1, l2 = st.columns(2)
    with l1:
        metric_card(
            f"{metrics.get('avg_latency_ms', '—')} ms",
            "Avg Inference Latency", "per review on CPU"
        )
    with l2:
        metric_card(
            f"~{int(metrics.get('reviews_per_minute', 0)):,}",
            "Reviews / Minute", "CPU throughput"
        )

    # ── Novel Contributions ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 Novel Contributions Summary")
    st.markdown("""
    #### Contribution #1 — Trust Score
    $$\\text{TrustScore} = \\alpha \\cdot C_s + \\beta \\cdot E_{top} + \\gamma \\cdot \\frac{|A|}{5} + \\delta \\cdot \\frac{|W|}{100}$$

    | Symbol | Meaning | Weight |
    |--------|---------|--------|
    | $C_s$ | Sentiment model confidence | α = 0.40 |
    | $E_{top}$ | Top emotion score | β = 0.25 |
    | $|A|$ | Extracted aspects (capped at 5) | γ = 0.25 |
    | $|W|$ | Word count (capped at 100) | δ = 0.10 |

    ---
    #### Contribution #2 — Authenticity Score
    $$\\text{AuthScore} = \\sum_{i=1}^{6} w_i \\cdot s_i \\times 100$$

    6 signals: Lexical Diversity (0.25), Sentiment-Rating Alignment (0.25),
    Exclamation Control (0.15), CAPS Control (0.10), Length Normality (0.15),
    Specificity (0.10).  Sarcasm penalty −10 pts.

    ---
    #### Contribution #3 — Integrity Score
    $$\\text{IntegrityScore} = 0.60 \\cdot \\text{TrustScore} + 0.40 \\cdot \\text{AuthScore}$$

    Unified business-usability metric combining review quality with review genuineness.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — System Info
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ System Info":
    st.title("⚙️ Sentilytics AI — System Architecture")

    st.markdown("""
    ### Core Methodology
    | Component | Detail |
    |-----------|--------|
    | **Primary Engine** | Groq / Llama-3-70b via Groq API (sarcasm-aware) |
    | **Fallback Engine** | DistilBERT-base-uncased-finetuned-sst-2-english |
    | **Emotion Model** | distilbert-base-uncased-emotion (6 classes) |
    | **Aspect Extractor** | SpaCy en_core_web_sm — NOUN-ADJ + acomp parsing |
    | **Novel #1** | Trust Score (4-axis composite reliability metric) |
    | **Novel #2** | Authenticity Score (6-signal fake review detector) |
    | **Novel #3** | Integrity Score (Trust × Authenticity unified metric) |
    | **Frontend** | Streamlit with custom OS95 Terminal CSS |
    | **Visualization** | Plotly, Matplotlib, WordCloud |

    ### Dataset
    | Field | Value |
    |-------|-------|
    | Source | Amazon product reviews (public GitHub — Arjun-Mota repo) |
    | Total Reviews | 34,659 authentic customer reviews |
    | Category Assignment | Keyword NLP classifier (Electronics, Books, Fashion, Appliances) |
    | Train / Val / Test | 70% / 15% / 15% stratified split |

    ### Baselines (Evaluation)
    | Model | Evaluation Method |
    |-------|-------------------|
    | VADER | ✅ Live-computed on real holdout set |
    | TextBlob | ✅ Live-computed on real holdout set |
    | SVM + TF-IDF | 📚 Published benchmark (cited in references.bib) |
    | LSTM | 📚 Published benchmark (cited in references.bib) |
    | Vanilla BERT | 📚 Published benchmark (cited in references.bib) |

    ### Temporal Analysis
    | Field | Value |
    |-------|-------|
    | Method | Rating-cohort proxies as temporal surrogates |
    | Basis | McAuley & Leskovec (2013, ACM RecSys) |
    | Metric | Sentiment Drift Score (SDS) = Pos%(5★) − Pos%(1-2★) |

    ### Research Publication
    > *Sentilytics AI: Transformer-Based Sentiment, Emotion, and Authenticity
    > Analysis for E-Commerce Reviews*
    > **Authors:** Mandira Banik, Subharjun Bose — GNIT Kolkata
    > **Conference:** ACMESGA 2k26, JIS College of Engineering, Kalyani
    """)

    st.markdown("---")
    acc = metrics.get("accuracy", "—")
    acc_str = f"{acc}%" if isinstance(acc, (int, float)) else str(acc)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Accuracy",       acc_str, help="Computed on real holdout set")
    with c2: st.metric("F1-Score (macro)", f"{metrics.get('f1_macro', '—')}%")
    with c3: st.metric("Avg Latency",    f"{metrics.get('avg_latency_ms', '—')} ms / review")

    if "_note" in metrics:
        st.warning(metrics["_note"])
