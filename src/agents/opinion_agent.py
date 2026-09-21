from typing import List, Dict, Any
from .llm_client import LLMClient

class OpinionAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    def score_opinions(self, reviews: List[Dict[str, Any]], aspects: List[Dict[str, Any]]) -> Dict[str, float]:
        if not aspects or not reviews:
            return {}
            
        aspect_names = [a.get('aspect', '') for a in aspects if a.get('aspect')]
        reviews_text = "\n".join([f"Review ID {r['review_id']}: {r['text']}" for r in reviews])
        
        messages = [
            {
                "role": "system",
                "content": f"You are an opinion scoring assistant. For these aspects: {', '.join(aspect_names)}, score the sentiment of each review. Return ONLY valid JSON: a list of objects with keys 'review_id', 'aspect', and 'sentiment' (a float from -1.0 to 1.0, where 0.0 means neutral or not mentioned)."
            },
            {
                "role": "user",
                "content": f"Reviews:\n{reviews_text}\n\nExtract sentiments."
            }
        ]
        
        response = self.client.chat_structured(messages)
        scores = response if isinstance(response, list) else response.get('scores', [])
        if isinstance(response, dict) and not scores:
             # handle possible dict wrapper
             for k, v in response.items():
                 if isinstance(v, list):
                     scores = v
                     break
                     
        aspect_weighted_scores = {}
        aspect_weights = {}
        trust_lookup = {str(r['review_id']): r.get('trust_score', 1.0) for r in reviews}
        
        for item in scores:
            r_id = str(item.get('review_id'))
            aspect = item.get('aspect')
            sentiment = item.get('sentiment', 0.0)
            
            if r_id in trust_lookup and aspect and sentiment != 0.0:
                trust = trust_lookup[r_id]
                if aspect not in aspect_weighted_scores:
                    aspect_weighted_scores[aspect] = 0.0
                    aspect_weights[aspect] = 0.0
                
                aspect_weighted_scores[aspect] += sentiment * trust
                aspect_weights[aspect] += trust

        final_aspect_scores = {}
        for a in aspect_weighted_scores:
            if aspect_weights[a] > 0:
                final_aspect_scores[a] = aspect_weighted_scores[a] / aspect_weights[a]
            else:
                final_aspect_scores[a] = 0.0
                
        return final_aspect_scores
