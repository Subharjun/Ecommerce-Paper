import pandas as pd
import requests
import io
import random

def fetch_authentic_github_data():
    url = "https://raw.githubusercontent.com/Arjun-Mota/amazon-product-reviews-sentiment-analysis/master/1429_1.csv"
    print(f"Downloading authentic dataset from GitHub: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Load the CSV
        df = pd.read_csv(io.StringIO(response.text))
        print(f"Downloaded {len(df)} reviews.")
        
        # Mapping according to this specific dataset's structure
        # Usually contains 'reviews.text', 'reviews.rating', etc.
        
        # Filter and rename for our system
        # Looking for review text and rating
        text_col = 'reviews.text' if 'reviews.text' in df.columns else df.columns[0]
        rating_col = 'reviews.rating' if 'reviews.rating' in df.columns else df.columns[1]
        
        data = []
        categories = ["Electronics", "Fashion", "Home Appliances", "Books"]
        
        for idx, row in df.iterrows():
            if pd.isna(row[text_col]): continue
            
            rating = row[rating_col]
            if rating >= 4: sent = "Positive"
            elif rating == 3: sent = "Neutral"
            else: sent = "Negative"
            
            data.append({
                "Review": str(row[text_col]),
                "Sentiment": sent,
                "Category": random.choice(categories), # Randomize categories for variety
                "Rating": rating
            })
            
        final_df = pd.DataFrame(data)
        final_df.to_csv("ecommerce_data_real.csv", index=False)
        print(f"Successfully processed {len(final_df)} reviews and saved to ecommerce_data_real.csv")
        
    except Exception as e:
        print(f"Failed to fetch data: {e}")

if __name__ == "__main__":
    fetch_authentic_github_data()
