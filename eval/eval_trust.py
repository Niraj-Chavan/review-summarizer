import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from src.trust.features import FeatureExtractor
from src.trust.graph import CollusionGraph
from src.trust.classifier import TrustClassifier

def evaluate_trust_model():
    print("Initializing components...")
    extractor = FeatureExtractor()
    graph = CollusionGraph()
    classifier = TrustClassifier()

    # Mock dataset representing 100 reviews with some obvious fake signals
    print("Generating mock data...")
    mock_reviews = []
    for i in range(100):
        # We inject some fakes
        is_fake_mock = (i % 5 == 0) # 20% fake
        mock_reviews.append({
            "review_id": f"r{i}",
            "reviewer_id": f"u{i % 10 if is_fake_mock else i}", # fakes share users
            "product_id": f"p{i % 5 if is_fake_mock else i}", # fakes hit same products
            "rating": 5 if is_fake_mock else (i % 5 + 1),
            "verified_purchase": False if is_fake_mock else True,
            "timestamp": f"2023-10-01T12:{i%60:02d}:00", # bursty timestamps
            "text": "great product highly recommend" if is_fake_mock else "It is an okay product, battery life is fine.",
            "label": 1 if is_fake_mock else 0 # Ground truth
        })

    # 1. Feature Extraction
    print("Extracting features...")
    df = extractor.extract_features(mock_reviews)
    
    # 2. Graph Collusion Detection
    print("Building collusion graph...")
    graph.build_graph(mock_reviews)
    risk_scores = graph.detect_collusion_clusters()
    
    # Map graph scores back to df
    df['graph_collusion_risk'] = df['reviewer_id'].map(risk_scores).fillna(0.0)

    # 3. Train/Test Split
    print("Training model...")
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    classifier.train(train_df)
    
    # 4. Evaluation
    print("Evaluating...")
    # Get continuous trust scores
    trust_scores = classifier.predict_trust_score(test_df)
    
    # To calculate F1, we convert trust score back to binary prediction (Fake = Trust < 0.5)
    y_true = test_df['label']
    y_pred = (trust_scores < 0.5).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- Trust Model Evaluation ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
if __name__ == "__main__":
    evaluate_trust_model()
