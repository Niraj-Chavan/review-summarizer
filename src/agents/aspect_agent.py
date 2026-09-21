from typing import List, Dict, Any
from .llm_client import LLMClient

class AspectAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    def extract_aspects(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not reviews:
            return []
            
        reviews_text = "\n".join([f"- {r['text']}" for r in reviews])
        
        messages = [
            {
                "role": "system",
                "content": "You are a product aspect extraction assistant. Extract distinct aspects (e.g., battery, camera, delivery) discussed in the reviews. Return ONLY valid JSON: a list of objects with keys 'aspect' (string) and 'example_spans' (list of strings)."
            },
            {
                "role": "user",
                "content": f"Reviews:\n{reviews_text}\n\nExtract the aspects."
            }
        ]
        
        response = self.client.chat_structured(messages)
        
        if isinstance(response, dict):
            # Sometimes LLMs wrap the list in a dict
            for key, val in response.items():
                if isinstance(val, list):
                    return val
            return []
        elif isinstance(response, list):
            return response
        return []
