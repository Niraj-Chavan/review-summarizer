from typing import List, Dict, Any, Optional
import pandas as pd
from .llm_client import LLMClient
from .aspect_opinion_agent import AspectOpinionAgent
from .summarization_agent import SummarizationAgent
import time

class PipelineOrchestrator:
    def __init__(self, retriever, feature_extractor, trust_classifier, llm_client: LLMClient = None, collusion_graph=None):
        self.retriever = retriever
        self.feature_extractor = feature_extractor
        self.trust_classifier = trust_classifier
        self.collusion_graph = collusion_graph
        
        self.client = llm_client or LLMClient()
        self.aspect_opinion_agent = AspectOpinionAgent(self.client)
        self.summary_agent = SummarizationAgent(self.client)

    def run(self, product_id: str, query: str = "", product_title: str = "") -> Dict[str, Any]:
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

        # 2. Trust Scoring (P1: Graph + XGBoost)
        try:
            # If collusion_graph injected, compute Louvain risk for current batch
            if self.collusion_graph is not None:
                try:
                    self.collusion_graph.graph.clear()
                    self.collusion_graph.build_graph(reviews)
                    risk_map = self.collusion_graph.detect_collusion_clusters()
                except Exception:
                    risk_map = {}
            else:
                risk_map = {}
            df = self.feature_extractor.extract_features(reviews)
            if 'graph_collusion_risk' not in df.columns:
                df['graph_collusion_risk'] = df['reviewer_id'].map(risk_map).fillna(0.0)
            else:
                df['graph_collusion_risk'] = df['graph_collusion_risk'].fillna(df['reviewer_id'].map(risk_map).fillna(0.0))
            trust_scores = self.trust_classifier.predict_trust_score(df)
            for i, rev in enumerate(reviews):
                rev["trust_score"] = float(trust_scores[i])
        except Exception as e:
            print(f"[Orchestrator] Trust scoring fallback: {e}")
            for rev in reviews:
                rev["trust_score"] = 1.0
                
        # 3. Aspect & Opinion (Optimized into one call)
        print(f"[{self.client.mode.upper()}] Extracting aspects and scoring opinions...")
        aspect_scores = self.aspect_opinion_agent.extract_and_score(reviews)
        
        # 4. Summarization
        print(f"[{self.client.mode.upper()}] Generating summary...")
        final_summary = self.summary_agent.generate_summary(reviews, aspect_scores, product_title=product_title)
        
        elapsed = time.time() - start_time
        final_summary["latency_seconds"] = round(elapsed, 2)
        print(f"Pipeline completed in {elapsed:.2f}s")
        
        return final_summary
