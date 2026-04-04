import pandas as pd
import requests
import io
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Keyword-based category classifier (Priority 3 Fix)
CATEGORY_KEYWORDS = {
    "Electronics": [
        "tablet", "phone", "battery", "screen", "kindle", "fire", "usb",
        "charger", "cable", "wifi", "bluetooth", "device", "gadget", "camera",
        "display", "processor", "ram", "storage", "app", "software", "android",
        "ios", "laptop", "computer", "headphone", "speaker", "remote", "hdmi"
    ],
    "Books": [
        "book", "read", "author", "novel", "chapter", "story", "pages",
        "writing", "plot", "narrative", "fiction", "nonfiction", "textbook",
        "publish", "kindle edition", "paperback", "hardcover", "literature"
    ],
    "Fashion": [
        "wear", "cloth", "shirt", "dress", "size", "fit", "fabric", "colour",
        "color", "style", "fashion", "jeans", "shoe", "jacket", "material",
        "wash", "comfortable", "cotton", "polyester", "design", "look", "outfit"
    ],
    "Home Appliances": [
        "kitchen", "clean", "vacuum", "cook", "appliance", "home", "fridge",
        "microwave", "oven", "blender", "mixer", "filter", "air", "purifier",
        "washing", "iron", "fan", "heater", "cooler", "utensil", "pan", "pot"
    ]
}


def classify_category(review_text: str) -> str:
    """
    Assigns a product category based on keyword frequency in the review text.
    Falls back to 'Electronics' if no strong signal is found (dataset is
    predominantly Amazon electronics reviews).
    """
    if not isinstance(review_text, str):
        return "Electronics"

    text_lower = review_text.lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[category] += 1

    best_category = max(scores, key=scores.get)

    # If no keyword matched at all, default to Electronics
    if scores[best_category] == 0:
        return "Electronics"

    return best_category


def fetch_authentic_github_data():
    url = "https://raw.githubusercontent.com/Arjun-Mota/amazon-product-reviews-sentiment-analysis/master/1429_1.csv"
    print(f"Downloading authentic dataset from GitHub...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        analyzer = SentimentIntensityAnalyzer()


        df = pd.read_csv(io.StringIO(response.text))
        print(f"Downloaded {len(df)} raw rows.")

        # Column detection
        text_col = "reviews.text" if "reviews.text" in df.columns else df.columns[0]
        rating_col = "reviews.rating" if "reviews.rating" in df.columns else df.columns[1]

        data = []
        for idx, row in df.iterrows():
            review_text = row[text_col]
            if pd.isna(review_text):
                continue

            rating = row[rating_col]
            # Map real sentiment organically using VADER (creates realistic variance for Temporal Engine)
            vs = analyzer.polarity_scores(str(review_text))
            if vs['compound'] >= 0.05:
                sent = "Positive"
            elif vs['compound'] <= -0.05:
                sent = "Negative"
            else:
                sent = "Neutral"

            # ✅ Priority 3: Keyword-driven category (no more random assignment)
            category = classify_category(str(review_text))

            data.append({
                "Review": str(review_text),
                "Sentiment": sent,
                "Category": category,
                "Rating": rating
            })

        final_df = pd.DataFrame(data)
        final_df.to_csv("ecommerce_data_real.csv", index=False)

        # Summary report
        print(f"\n✅ Processed {len(final_df)} reviews → saved to ecommerce_data_real.csv")
        print("\nSentiment distribution:")
        print(final_df["Sentiment"].value_counts().to_string())
        print("\nCategory distribution (keyword-classified):")
        print(final_df["Category"].value_counts().to_string())

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        print("If running offline, the existing ecommerce_data_real.csv will be used.")


if __name__ == "__main__":
    fetch_authentic_github_data()
