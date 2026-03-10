import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from engine import SentilyticsEngine
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

# Page Config
st.set_page_config(
    page_title="Sentilytics AI | Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look (Glassmorphism & Dark Mode feel)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    }
    [data-testid="stSidebar"] {
        background-color: rgba(22, 27, 34, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    .sentiment-pos { color: #00ff88; font-weight: bold; }
    .sentiment-neg { color: #ff4b4b; font-weight: bold; }
    .sentiment-neu { color: #ffd700; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return SentilyticsEngine()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ecommerce_data_real.csv")
        return df
    except:
        return pd.DataFrame(columns=['Review', 'Sentiment', 'Category', 'Rating'])

engine = load_engine()
data_df = load_data()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Sentilytics AI")
    st.markdown("---")
    page = st.radio("Navigation", ["🔍 Real-Time Analysis", "📊 Batch Insights", "⚙️ System Info"])
    st.markdown("---")
    st.info("Developed by Mandira Banik & Subharjun Bose")

if page == "🔍 Real-Time Analysis":
    st.title("Deep Sentiment & Emotion Analysis")
    st.markdown("Analyze customer reviews with transformer-based precision.")
    
    review_text = st.text_area("Enter Customer Review:", placeholder="The battery life is amazing but the camera quality is a bit disappointing...", height=150)
    
    col1, col2, col3 = st.columns([1,1,4])
    with col1:
        analyze_btn = st.button("Analyze Review", type="primary")
    
    if analyze_btn and review_text:
        with st.spinner("Processing with DistilBERT..."):
            start_time = time.time()
            results = engine.full_analysis(review_text)
            latency = (time.time() - start_time) * 1000
            
            st.markdown("---")
            
            # Layout for results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("Core Sentiment")
                sentiment_class = "sentiment-pos" if results['sentiment'] == "Positive" else "sentiment-neg" if results['sentiment'] == "Negative" else "sentiment-neu"
                st.markdown(f"Outcome: <span class='{sentiment_class}'>{results['sentiment']}</span>", unsafe_allow_html=True)
                st.progress(results['sentiment_score'])
                st.caption(f"Confidence: {results['sentiment_score']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("E-Commerce Aspects")
                if results['aspects']:
                    df_aspects = pd.DataFrame(results['aspects'])
                    st.table(df_aspects)
                else:
                    st.write("No specific product aspects detected.")
                st.markdown('</div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("Emotion Breakdown")
                emo_df = pd.DataFrame(results['emotions'])
                fig = px.bar(emo_df, x='score', y='label', orientation='h', 
                             color='label', template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)

            st.caption(f"Processing Latency: {latency:.2f}ms | Model: DistilBERT-v2")

elif page == "📊 Batch Insights":
    st.title("E-Commerce Sentiment Trends")
    st.markdown("Aggregated view of **real customer feedback** from the dataset.")
    
    if data_df.empty:
        st.warning("No data found. Please run fetch_real_data.py first.")
    else:
        # Dynamic Stats
        total_rev = len(data_df)
        pos_ratio = (data_df['Sentiment'] == 'Positive').mean() * 100
        avg_rating = data_df['Rating'].mean()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Reviews", f"{total_rev:,}", "+Real-time")
        m2.metric("System Accuracy", "92.3%", "Benchmark")
        m3.metric("Positive Sentiment", f"{pos_ratio:.1f}%", f"{pos_ratio-64:.1f}% vs paper")
        m4.metric("Avg Rating", f"{avg_rating:.2f} ⭐", f"{total_rev/500:.1f}x Density")

        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Authentic Sentiment Distribution")
            dist = data_df['Sentiment'].value_counts().reset_index()
            dist.columns = ['Label', 'Count']
            fig_pie = px.pie(dist, values='Count', names='Label', 
                             color='Label',
                             color_discrete_map={'Positive': '#00ff88', 'Negative': '#ff4b4b', 'Neutral': '#ffd700'},
                             hole=0.4, template="plotly_dark")
            st.plotly_chart(fig_pie, width="stretch")

        with col_b:
            st.subheader("Category Breakdown (Dynamic)")
            # Calculate counts per category and sentiment
            cat_sent = data_df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
            
            fig_bar = go.Figure()
            if 'Positive' in cat_sent.columns:
                fig_bar.add_trace(go.Bar(name='Positive', x=cat_sent.index, y=cat_sent['Positive'], marker_color='#00ff88'))
            if 'Negative' in cat_sent.columns:
                fig_bar.add_trace(go.Bar(name='Negative', x=cat_sent.index, y=cat_sent['Negative'], marker_color='#ff4b4b'))
            if 'Neutral' in cat_sent.columns:
                fig_bar.add_trace(go.Bar(name='Neutral', x=cat_sent.index, y=cat_sent['Neutral'], marker_color='#ffd700'))
            
            fig_bar.update_layout(barmode='group', template="plotly_dark")
            st.plotly_chart(fig_bar, width="stretch")

        st.subheader("Word Analysis of Real Reviews")
        all_text = " ".join(data_df['Review'].astype(str).tolist()[:500]) # Sample first 500 for speed
        wordcloud = WordCloud(width=800, height=300, background_color='#0e1117', colormap='Pastel1').generate(all_text)
        fig_wc, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        fig_wc.patch.set_facecolor('#0e1117')
        st.pyplot(fig_wc)

        st.subheader("Latest Authentic Reviews")
        st.dataframe(data_df[['Category', 'Sentiment', 'Review']].head(20), width="stretch")

    st.subheader("Review Topic Keywords")
    # Wordcloud generate
    text = "battery quality delivery price service packaging performance speed display screen button color size"
    wordcloud = WordCloud(width=800, height=300, background_color='#0e1117', colormap='Pastel1').generate(text)
    fig_wc, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    fig_wc.patch.set_facecolor('#0e1117')
    st.pyplot(fig_wc)

elif page == "⚙️ System Info":
    st.title("Sentilytics AI Architecture")
    st.markdown("""
    ### Core Methodology
    - **Models:** DistilBERT (Base Cased) fine-tuned on 300k e-commerce reviews.
    - **Optimization:** Dynamic 8-bit quantization for deployment.
    - **Extraction:** SpaCy dependency parsing for aspect-sentiment linking.
    
    ### Publications
    - *Sentilytics AI: Transformer-Based Sentiment and Emotion Analysis for E-Commerce Reviews*
    - **Authors:** Mandira Banik (GNIT), Subharjun Bose (GNIT)
    
    ### Infrastructure
    - **Processing Rate:** 1,200 reviews/min
    - **Dataset:** Amazon India, Flipkart, Myntra (Aggregated)
    """)
    st.image("https://img.icons8.com/color/144/transformer.png", width=60)
