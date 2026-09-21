import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from typing import List, Dict, Any, Tuple

class TrustClassifier:
    def __init__(self):
        # We use a probabilistic output to get a continuous trust score
        self.model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        self.feature_cols = [
            'reviewer_review_count', 'rating_deviation', 'posting_burstiness', 
            'is_verified', 'sentiment_extremity', 'is_templated', 'is_near_duplicate',
            'graph_collusion_risk'
        ]

    def _prepare_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Creates weak labels if no ground truth is provided.
        ASSUMPTION: We assume reviews are FAKE (label=1) if they are NOT verified 
        AND (are near duplicates OR highly templated OR in high collusion clusters).
        Otherwise GENUINE (label=0).
        """
        if 'label' in df.columns:
            return df['label']
            
        # Weak labeling logic
        is_fake = (
            (df['is_verified'] == 0) & 
            ((df['is_near_duplicate'] == 1) | 
             (df['is_templated'] == 1) | 
             (df['graph_collusion_risk'] > 0.8))
        )
        return is_fake.astype(int)

    def train(self, df: pd.DataFrame):
        """
        Trains the XGBoost classifier on the feature dataframe.
        """
        X = df[self.feature_cols]
        y = self._prepare_labels(df)
        
        self.model.fit(X, y)

    def predict_trust_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns a trust score between 0 and 1.
        Since model predicts probability of being FAKE (1),
        Trust Score = 1 - P(Fake).
        """
        X = df[self.feature_cols]
        p_fake = self.model.predict_proba(X)[:, 1]
        return 1.0 - p_fake
