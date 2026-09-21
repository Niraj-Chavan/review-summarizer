import numpy as np
import pandas as pd
from typing import List, Dict, Any
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class FeatureExtractor:
    def __init__(self, embedding_model_name: str = 'intfloat/multilingual-e5-small'):
        self.embedder = SentenceTransformer(embedding_model_name)
        # Some generic templates often used in fake reviews
        self.templates = ["great product", "highly recommend", "very good", "will buy again", "excellent quality", "waste of money"]

    def extract_features(self, reviews: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Expects reviews to have:
        - reviewer_id, product_id, rating, verified_purchase, timestamp, text
        """
        df = pd.DataFrame(reviews)
        
        # 1. Behavioral Features
        
        # Reviewer's total review count
        df['reviewer_review_count'] = df.groupby('reviewer_id')['review_id'].transform('count')
        
        # Deviation from product average rating
        df['product_avg_rating'] = df.groupby('product_id')['rating'].transform('mean')
        df['rating_deviation'] = abs(df['rating'] - df['product_avg_rating'])
        
        # Posting burstiness (reviews by the same user within a short time)
        # Simplified: max number of reviews by this user in any 1-day window
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['reviewer_id', 'timestamp'])
        
        def calc_burstiness(group):
            # Number of reviews in the same day (naive burstiness)
            return group['timestamp'].dt.date.map(group['timestamp'].dt.date.value_counts())
        
        df['posting_burstiness'] = df.groupby('reviewer_id').apply(calc_burstiness).reset_index(level=0, drop=True)
        
        # Verified purchase flag
        df['is_verified'] = df['verified_purchase'].astype(int)

        # 2. Linguistic Features
        
        # Sentiment Polarity Extremity (using TextBlob, mostly works for EN but okay as a weak signal)
        df['sentiment_extremity'] = df['text'].apply(lambda x: abs(TextBlob(x).sentiment.polarity))
        
        # Generic/templated phrase detection
        df['is_templated'] = df['text'].apply(lambda x: int(any(t in x.lower() for t in self.templates)))
        
        # Near-duplicate text detection (using embeddings)
        # For a large dataset, we'd do this via LSH or FAISS, but for simplicity we do pairwise in product groups
        embeddings = self.embedder.encode(df['text'].tolist())
        
        is_duplicate = np.zeros(len(df))
        # Very naive O(N^2) duplicate detection for demonstration
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0) # ignore self
        max_sims = sim_matrix.max(axis=1)
        # Flag as duplicate if similarity > 0.9
        df['is_near_duplicate'] = (max_sims > 0.9).astype(int)
        
        return df
