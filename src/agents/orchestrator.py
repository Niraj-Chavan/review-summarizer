from typing import List, Dict, Any, Optional
import pandas as pd
from .llm_client import LLMClient
from .aspect_opinion_agent import AspectOpinionAgent
from .summarization_agent import SummarizationAgent
import time

class PipelineOrchestrator:
    def __init__(self, retriever, feature_extractor, trust_classifier, llm_client: LLMClient = None):
        self.retriever = retriever
        self.feature_extractor = feature_extractor
        self.trust_classifier = trust_classifier
        
        self.client = llm_client or LLMClient()
        self.aspect_opinion_agent = AspectOpinionAgent(self.client)
        self.summary_agent = SummarizationAgent(self.client)

    def run(self, product_id: str, query: str = "") -> Dict[str, Any]:
        start_time = time.time()
        print(f"[{self.client.mode.upper()}] Starting pipeline for product: {product_id}, query: '{query}'")
        
        # 1. Retrieval
        if query:
            retrieved_docs = self.retriever.search(query=query, product_id=product_id, top_k=10)
        else:
            retrieved_docs = [
                {"content": self.retriever.original_texts[i], "metadata": meta}
                for i, meta in enumerate(self.retriever.metadata)
                if meta.get("product_id") == product_id
            ][:10]
            
        if not retrieved_docs:
            return {"error": "No reviews found."}
            
        reviews = []
        for doc in retrieved_docs:
            rev = doc["metadata"].copy()
            rev["text"] = doc["content"]
            reviews.append(rev)

        # 2. Trust Scoring
        try:
            df = self.feature_extractor.extract_features(reviews)
            if 'graph_collusion_risk' not in df.columns:
                df['graph_collusion_risk'] = 0.0
                
            trust_scores = self.trust_classifier.predict_trust_score(df)
            for i, rev in enumerate(reviews):
                rev["trust_score"] = float(trust_scores[i])
        except Exception as e:
            for rev in reviews:
                rev["trust_score"] = 1.0
                
        # 3. Aspect & Opinion (Optimized into one call)
        print(f"[{self.client.mode.upper()}] Extracting aspects and scoring opinions...")
        aspect_scores = self.aspect_opinion_agent.extract_and_score(reviews)
        
        # 4. Summarization
        print(f"[{self.client.mode.upper()}] Generating summary...")
        final_summary = self.summary_agent.generate_summary(reviews, aspect_scores)
        
        elapsed = time.time() - start_time
        final_summary["latency_seconds"] = round(elapsed, 2)
        print(f"Pipeline completed in {elapsed:.2f}s")
        
        return final_summary
