from typing import List, Dict, Any
from .llm_client import LLMClient
import json

class SummarizationAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    def generate_summary(self, reviews: List[Dict[str, Any]], aspect_scores: Dict[str, float]) -> Dict[str, Any]:
        if not reviews:
            return {"summary_text": "No reviews available for summary."}

        # Calculate ratings
        valid_reviews = [r for r in reviews if 'rating' in r]
        
        raw_avg = 0.0
        trust_adj = 0.0
        if valid_reviews:
            raw_avg = sum(r['rating'] for r in valid_reviews) / len(valid_reviews)
            
            total_trust = sum(r.get('trust_score', 1.0) for r in valid_reviews)
            if total_trust > 0:
                trust_adj = sum(r['rating'] * r.get('trust_score', 1.0) for r in valid_reviews) / total_trust
            else:
                trust_adj = raw_avg
                
        down_weighted_count = sum(1 for r in reviews if r.get('trust_score', 1.0) < 0.5)
        
        # Prepare aspects string
        aspect_info = ", ".join([f"{k}: {v:.2f}" for k, v in aspect_scores.items()])
        
        messages = [
            {
                "role": "system",
                "content": "You are a product review summarization assistant. Given the aspect sentiment scores (range -1 to 1) and rating info, generate a 2-4 sentence natural-language opinion summary of the product. Return ONLY valid JSON with a single key 'summary_text'."
            },
            {
                "role": "user",
                "content": f"Aspects and Sentiment Scores: {aspect_info}\nRaw Rating: {raw_avg:.2f}/5\nTrust-Adjusted Rating: {trust_adj:.2f}/5\nDown-weighted reviews: {down_weighted_count}\n\nGenerate the summary."
            }
        ]
        
        response = self.client.chat_structured(messages)
        summary_text = response.get('summary_text', "Summary generation failed.")
        
        return {
            "raw_rating": round(raw_avg, 2),
            "trust_adjusted_rating": round(trust_adj, 2),
            "aspects": [{"name": k, "sentiment_score": round(v, 2)} for k, v in aspect_scores.items()],
            "summary_text": summary_text,
            "down_weighted_count": down_weighted_count
        }
