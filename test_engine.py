#!/usr/bin/env python3
"""
Quick test script to verify the Sentilytics AI engine is working correctly
"""

from engine import SentilyticsEngine
import time

def test_engine():
    print("=" * 60)
    print("🧪 Testing Sentilytics AI Engine")
    print("=" * 60)
    print()
    
    # Initialize engine
    print("📦 Loading models...")
    start = time.time()
    engine = SentilyticsEngine()
    load_time = time.time() - start
    print(f"✅ Models loaded in {load_time:.2f} seconds")
    print()
    
    # Test cases
    test_reviews = [
        "The battery life is amazing but the camera quality is disappointing",
        "Absolutely love this product! Fast delivery and great quality!",
        "Terrible experience. The product broke after 2 days.",
        "It's okay, nothing special but does the job."
    ]
    
    print("🔍 Running test analyses...")
    print("-" * 60)
    
    for i, review in enumerate(test_reviews, 1):
        print(f"\n📝 Test {i}: {review[:50]}...")
        
        start = time.time()
        results = engine.full_analysis(review)
        latency = (time.time() - start) * 1000
        
        print(f"   Sentiment: {results['sentiment']} (confidence: {results['sentiment_score']:.2%})")
        print(f"   Top Emotion: {results['top_emotion']}")
        
        if results['aspects']:
            print(f"   Aspects detected: {len(results['aspects'])}")
            for aspect in results['aspects'][:2]:  # Show first 2
                print(f"      - {aspect['aspect']}: {aspect['sentiment_word']}")
        else:
            print(f"   Aspects detected: None")
        
        print(f"   Processing time: {latency:.2f}ms")
    
    print()
    print("=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    print()
    print("💡 Next step: Run 'streamlit run app.py' to launch the dashboard")

if __name__ == "__main__":
    try:
        test_engine()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Activated the virtual environment: source venv/bin/activate")
        print("2. Installed all dependencies: pip install -r requirements.txt")
        print("3. Downloaded SpaCy model: python -m spacy download en_core_web_sm")
