# Sentilytics AI: E-Commerce Sentiment & Emotion Analyzer

Sentilytics AI is a transformer-based NLP system designed for deep analysis of customer reviews. It moves beyond basic sentiment to capture emotional nuances and specific product features (aspects).

## Features
- **Real-Time Analysis:** Instant sentiment, emotion, and aspect extraction for single reviews.
- **Emotion Detection:** Identifies Joy, Anger, Sadness, Fear, Surprise, and Excitement.
- **Aspect-Based Sentiment:** Links sentiments to specific product features (e.g., "Battery", "Delivery").
- **Interactive Dashboard:** Visualizes trends through word clouds, pie charts, and category breakdowns.

## Architecture
- **NLP Engine:** Built with `DistilBERT` (for speed/accuracy balance) and `SpaCy` (for dependency parsing).
- **Frontend:** `Streamlit` with a custom Glassmorphism UI.
- **Visualization:** `Plotly` and `Matplotlib`.
- **Dataset:** 34,660 authentic customer reviews fetched dynamically from public research repositories.

## Setup Instructions
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download SpaCy Model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```
3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Authors
- **Mandira Banik** (study conception and design)
- **Subharjun Bose** (data collection, system implementation)

Developed as part of the research paper: *Sentilytics AI: Transformer-Based Sentiment and Emotion Analysis for E-Commerce Reviews*.
