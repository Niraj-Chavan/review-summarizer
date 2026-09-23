from typing import List, Dict, Any
from .llm_client import LLMClient
import json

class AspectOpinionAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    def extract_and_score(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detailed aspect breakdown — now with AI explanation per aspect.
        Returns dict {aspect: {score, explanation, evidence, review_count}} or fallback to {aspect: score}
        """
        if not reviews:
            return {}
            
        reviews_text = "\n".join([f"Review ID {r['review_id']} (trust={r.get('trust_score',1.0):.2f}): {r['text']}" for r in reviews])
        
        messages = [
            {
                "role": "system",
                "content": "You are a detailed aspect and sentiment extraction assistant. Read the reviews (each has trust score). Extract 3-8 distinct aspects (e.g., battery, build quality, pin, waterproof). For each aspect, return: sentiment_score (-1.0 very negative to 1.0 very positive), explanation (1 sentence explaining WHY — what is the problem/praise), evidence (short quote from a review), and review_count (how many reviews mention it). Return ONLY valid JSON: {\"aspects\": [{\"name\": \"battery\", \"sentiment_score\": -0.5, \"explanation\": \"Battery dies quickly after 2 days\", \"evidence\": \"The battery died after two days\", \"review_count\": 3}]}. Do not add markdown."
            },
            {
                "role": "user",
                "content": f"Reviews:\n{reviews_text}\n\nExtract detailed aspect breakdown with explanation per aspect."
            }
        ]
        
        response = self.client.chat_structured(messages)
        
        # Support both new format {"aspects": [...]} and old {aspect: score}
        detailed = {}
        avg_trust = sum(r.get('trust_score', 1.0) for r in reviews) / max(len(reviews),1)

        if isinstance(response, dict) and "aspects" in response and isinstance(response["aspects"], list):
            for item in response["aspects"]:
                if not isinstance(item, dict): continue
                name = item.get("name") or item.get("aspect")
                if not name: continue
                score = item.get("sentiment_score", item.get("score", 0))
                try: score = float(score)
                except: continue
                detailed[name.lower().strip()] = {
                    "score": float(score) * avg_trust,
                    "explanation": item.get("explanation") or item.get("reason") or (" praised" if score>0.2 else " has issues" if score<-0.2 else " mixed"),
                    "evidence": item.get("evidence") or item.get("quote") or "",
                    "review_count": int(item.get("review_count") or item.get("count") or 1)
                }
            return detailed
        # Fallback old format
        raw = response if isinstance(response, dict) else {}
        fallback = {}
        for aspect, score in raw.items():
            if isinstance(score, (int, float)):
                fallback[aspect] = {"score": float(score) * avg_trust, "explanation": f"{aspect} sentiment is {'positive' if score>0.2 else 'negative' if score<-0.2 else 'neutral'} based on reviews.", "evidence": "", "review_count": 1}
            elif isinstance(score, dict) and "sentiment_score" in score:
                try:
                    sc = float(score["sentiment_score"]) * avg_trust
                    fallback[aspect] = {"score": sc, "explanation": score.get("explanation",""), "evidence": score.get("evidence",""), "review_count": int(score.get("review_count",1))}
                except: pass
        return fallback
