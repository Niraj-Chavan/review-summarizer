import sys
import os
import json
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.hybrid_search import HybridRetriever
from src.trust.features import FeatureExtractor
from src.trust.classifier import TrustClassifier
from src.agents.llm_client import LLMClient
from src.agents.orchestrator import PipelineOrchestrator

def setup_mock_data():
    retriever = HybridRetriever(model_name="intfloat/multilingual-e5-small")
    
    raw_reviews = [
        {
            "review_id": "r1", "product_id": "p1", "rating": 5, "verified_purchase": True,
            "timestamp": "2023-10-01T10:00:00",
            "text": "The battery life is amazing! I charge it once a day. Highly recommend."
        },
        {
            "review_id": "r2", "product_id": "p1", "rating": 1, "verified_purchase": False,
            "timestamp": "2023-10-01T10:05:00",
            "text": "Terrible. Waste of money. Do not buy."
        },
        {
            "review_id": "r3", "product_id": "p1", "rating": 4, "verified_purchase": True,
            "timestamp": "2023-10-02T12:00:00",
            "text": "La duración de la batería es increíble. Pero el envío fue muy lento."
        }
    ]
    
    # Needs to be trained for TrustClassifier to work
    extractor = FeatureExtractor()
    df = extractor.extract_features(raw_reviews)
    df['graph_collusion_risk'] = 0.0 # Mock graph
    
    classifier = TrustClassifier()
    classifier.train(df)
    
    all_texts = [r["text"] for r in raw_reviews]
    retriever.add_documents(all_texts, raw_reviews)
    
    return retriever, extractor, classifier

def run_comparison():
    parser = argparse.ArgumentParser(description="Compare Local vs Cloud Ollama Modes")
    parser.add_argument("--product_id", type=str, default="p1")
    parser.add_argument("--query", type=str, default="battery")
    args = parser.parse_args()
    
    print("Setting up mock data...")
    retriever, extractor, classifier = setup_mock_data()
    
    print("\n=============================================")
    print(" RUNNING LOCAL MODE")
    print("=============================================")
    os.environ["OLLAMA_MODE"] = "local"
    local_client = LLMClient()
    local_orchestrator = PipelineOrchestrator(retriever, extractor, classifier, local_client)
    local_result = local_orchestrator.run(args.product_id, args.query)
    
    print("\n=============================================")
    print(" RUNNING CLOUD MODE")
    print("=============================================")
    os.environ["OLLAMA_MODE"] = "cloud"
    cloud_client = LLMClient()
    cloud_orchestrator = PipelineOrchestrator(retriever, extractor, classifier, cloud_client)
    cloud_result = cloud_orchestrator.run(args.product_id, args.query)
    
    print("\n=============================================")
    print(" COMPARISON RESULTS ")
    print("=============================================")
    print(f"--- LOCAL MODEL ({os.environ.get('OLLAMA_MODEL_LOCAL', 'llama3.1:8b')}) ---")
    print(json.dumps(local_result, indent=2))
    
    print(f"\n--- CLOUD MODEL ({os.environ.get('OLLAMA_MODEL_CLOUD', 'llama3.1:70b-cloud')}) ---")
    print(json.dumps(cloud_result, indent=2))

if __name__ == "__main__":
    run_comparison()
