import pytest
import os
from src.retrieval.hybrid_search import HybridRetriever
from src.trust.features import FeatureExtractor
from src.trust.classifier import TrustClassifier
from src.agents.llm_client import LLMClient
from src.agents.orchestrator import PipelineOrchestrator

def test_full_pipeline_local_mode():
    os.environ["OLLAMA_MODE"] = "local"
    
    # 1. Setup mock data
    retriever = HybridRetriever(model_name="intfloat/multilingual-e5-small")
    extractor = FeatureExtractor()
    classifier = TrustClassifier()
    
    raw_reviews = [
        {"review_id": "r1", "reviewer_id": "u1", "product_id": "test_p1", "rating": 5, "verified_purchase": True, "timestamp": "2023-10-01T10:00:00", "text": "The battery life is amazing!"},
        {"review_id": "r2", "reviewer_id": "u2", "product_id": "test_p1", "rating": 1, "verified_purchase": False, "timestamp": "2023-10-01T10:05:00", "text": "Terrible battery. Fake!"},
        {"review_id": "r3", "reviewer_id": "u3", "product_id": "test_p1", "rating": 4, "verified_purchase": True, "timestamp": "2023-10-02T12:00:00", "text": "La duración de la batería es increíble."}
    ]
    
    # Train trust
    df = extractor.extract_features(raw_reviews)
    df['graph_collusion_risk'] = 0.0
    classifier.train(df)
    
    # Index
    all_texts = [r["text"] for r in raw_reviews]
    retriever.add_documents(all_texts, raw_reviews)
    
    client = LLMClient()
    
    # Since we can't reliably mock Ollama responses in this generic test environment without spinning up Ollama,
    # we just ensure the orchestrator instantiates correctly.
    # In a real CI, we'd mock the LLMClient's chat_structured method.
    
    # Mock LLM Client
    client.chat_structured = lambda msgs: {"summary_text": "Mock summary"} if "summary_text" in msgs[0]["content"] else {"battery": 0.8}

    
    orchestrator = PipelineOrchestrator(retriever, extractor, classifier, client)
    
    # Run pipeline
    result = orchestrator.run(product_id="test_p1", query="battery")
    
    assert result is not None
    assert "trust_adjusted_rating" in result
    assert "latency_seconds" in result
    assert "summary_text" in result
    assert result["summary_text"] == "Mock summary"
