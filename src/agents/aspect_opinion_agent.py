from typing import List, Dict, Any
from .llm_client import LLMClient
import json

class AspectOpinionAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    def extract_and_score(self, reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Combines aspect extraction and opinion scoring into a single LLM prompt 
        to reduce round-trip latency.
        """
        if not reviews:
            return {}
            
        reviews_text = "\n".join([f"Review ID {r['review_id']}: {r['text']}" for r in reviews])
        
        messages = [
            {
                "role": "system",
                "content": "You are an aspect and sentiment extraction assistant. Read the reviews and extract the distinct aspects discussed (e.g., battery, delivery). For each aspect, return an average sentiment score from -1.0 (very negative) to 1.0 (very positive). Return ONLY valid JSON: an object where keys are aspect names and values are sentiment scores (floats)."
            },
            {
                "role": "user",
                "content": f"Reviews:\n{reviews_text}\n\nExtract aspects and average sentiment."
            }
        ]
        
        response = self.client.chat_structured(messages)
        
        raw_aspect_scores = response if isinstance(response, dict) else {}
        
        # Apply trust weighting based on the reviews
        # Since the LLM returns an average, we scale it by the average trust score of the reviews that likely mentioned it.
        # This is an approximation to save latency. For exact per-review weighting, we'd need a larger JSON output.
        avg_trust = sum(r.get('trust_score', 1.0) for r in reviews) / len(reviews)
        
        trust_adjusted_scores = {}
        for aspect, score in raw_aspect_scores.items():
            if isinstance(score, (int, float)):
                # Moderately weight the score by the average trust of the batch
                trust_adjusted_scores[aspect] = float(score) * avg_trust
                
        return trust_adjusted_scores
