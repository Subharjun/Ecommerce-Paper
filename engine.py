import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import pandas as pd

class SentilyticsEngine:
    def __init__(self):
        # Using pre-trained models that align with the paper's categories
        # For a production build, these would be the user's fine-tuned models
        print("Loading Sentiment Model...")
        self.sentiment_pipe = pipeline("sentiment-analysis", 
                                       model="distilbert-base-uncased-finetuned-sst-2-english")
        
        print("Loading Emotion Model...")
        # This model covers 6 basic emotions + neutral
        self.emotion_pipe = pipeline("text-classification", 
                                     model="bhadresh-savani/distilbert-base-uncased-emotion", 
                                     top_k=None)
        
        print("Loading NLP for Aspect Extraction...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback if model not downloaded
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def analyze_sentiment(self, text):
        result = self.sentiment_pipe(text)[0]
        # Map SST-2 (POSITIVE/NEGATIVE) to the paper's (Positive, Negative, Neutral)
        # Note: SST-2 doesn't have neutral, so we use a score threshold or a different model for real production
        label = result['label'].title()
        score = result['score']
        
        # Simple heuristic for Neutral if score is low (just for demonstration)
        if score < 0.6:
            label = "Neutral"
            
        return label, score

    def analyze_emotions(self, text):
        results = self.emotion_pipe(text)[0]
        # The model returns a list of dicts with label and score
        # Paper categories: Joy, Anger, Sadness, Fear, Surprise, Excitement
        # Model categories: sadness, joy, love, anger, fear, surprise
        
        # Map model labels to paper labels
        mapping = {
            'joy': 'Joy',
            'anger': 'Anger',
            'sadness': 'Sadness',
            'fear': 'Fear',
            'surprise': 'Surprise',
            'love': 'Excitement' # Mapping love to Excitement as a reasonable proxy
        }
        
        mapped_results = []
        for r in results:
            if r['label'] in mapping:
                mapped_results.append({
                    'label': mapping[r['label']],
                    'score': r['score']
                })
        
        return sorted(mapped_results, key=lambda x: x['score'], reverse=True)

    def extract_aspects(self, text):
        """
        Extracts product features and associated sentiment using dependency parsing.
        Example: "The battery life is great but the screen is dim"
        Aspects: {'battery life': 'great', 'screen': 'dim'}
        """
        doc = self.nlp(text)
        aspects = []
        
        # Focused on ADJ modifying NOUNs or NOUNs as subjects of ADJs
        for token in doc:
            if token.pos_ == "NOUN":
                # Check for adjectives modifying this noun
                modifiers = [w.text for w in token.children if w.pos_ == "ADJ"]
                if modifiers:
                    aspects.append({
                        "aspect": token.text,
                        "sentiment_word": ", ".join(modifiers)
                    })
            
            if token.pos_ == "ADJ":
                # Check if this adjective is a complement of a noun subject
                if token.dep_ == "acomp":
                    subject = [w.text for w in token.head.children if w.dep_ == "nsubj"]
                    if subject:
                        aspects.append({
                            "aspect": subject[0],
                            "sentiment_word": token.text
                        })
        
        return aspects

    def full_analysis(self, text):
        sentiment, s_score = self.analyze_sentiment(text)
        emotions = self.analyze_emotions(text)
        aspects = self.extract_aspects(text)
        
        return {
            "text": text,
            "sentiment": sentiment,
            "sentiment_score": s_score,
            "top_emotion": emotions[0]['label'] if emotions else "Unknown",
            "emotions": emotions,
            "aspects": aspects
        }
