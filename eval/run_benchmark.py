import os
import json
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from typing import List, Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.hybrid_search import HybridRetriever
from src.trust.features import FeatureExtractor
from src.trust.classifier import TrustClassifier
from src.agents.llm_client import LLMClient
from src.agents.orchestrator import PipelineOrchestrator

# --- BASELINES ---
class PlainRAGOrchestrator(PipelineOrchestrator):
    def run(self, product_id: str, query: str = "") -> Dict[str, Any]:
        print(f"[{self.client.mode.upper()}] Running Plain RAG...")
        retrieved_docs = self.retriever.search(query=query, product_id=product_id, top_k=10)
        reviews = [{"text": doc["content"], **doc["metadata"]} for doc in retrieved_docs]
        if not reviews: return {"error": "no reviews"}
        
        for rev in reviews:
            rev["trust_score"] = 1.0 # Ignore trust weighting
            
        aspects = self.aspect_agent.extract_aspects(reviews)
        aspect_scores = self.opinion_agent.score_opinions(reviews, aspects)
        return self.summary_agent.generate_summary(reviews, aspect_scores)

class BM25Orchestrator(PipelineOrchestrator):
    def run(self, product_id: str, query: str = "") -> Dict[str, Any]:
        print(f"[{self.client.mode.upper()}] Running BM25 Only...")
        query_tokens = query.lower().split()
        bm25_scores = self.retriever.bm25.get_scores(query_tokens)
        
        candidates = []
        for idx, s in enumerate(bm25_scores):
            if s > 0 and self.retriever.metadata[idx].get("product_id") == product_id:
                candidates.append((idx, s))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_indices = [c[0] for c in candidates[:10]]
        
        reviews = [{"text": self.retriever.original_texts[idx], **self.retriever.metadata[idx]} for idx in top_indices]
        if not reviews: return {"error": "no reviews"}
        
        for rev in reviews:
            rev["trust_score"] = 1.0 # Ignore trust weighting
            
        aspects = self.aspect_agent.extract_aspects(reviews)
        aspect_scores = self.opinion_agent.score_opinions(reviews, aspects)
        return self.summary_agent.generate_summary(reviews, aspect_scores)

# --- BENCHMARK DATASET ---
EVAL_SET = [
    {
        "product_id": "p1",
        "query": "battery and delivery",
        "reference_summary": "The product has amazing battery life that lasts all day, but customers complained about very slow delivery."
    },
    {
        "product_id": "p2",
        "query": "build quality",
        "reference_summary": "The build quality is generally poor and feels cheap, though a few fake reviews claim otherwise."
    }
]

def load_checkpoint(filepath: str) -> Dict[str, Any]:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_checkpoint(filepath: str, data: Dict[str, Any]):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def compute_metrics(predictions: List[str], references: List[str]):
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    for p, r in zip(predictions, references):
        scores = scorer.score(r, p)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
        
    n = len(predictions)
    # BERTScore
    P, R, F1 = bert_score_fn(predictions, references, lang='en', verbose=False)
    
    return {
        "rouge1": rouge1 / n,
        "rouge2": rouge2 / n,
        "rougeL": rougeL / n,
        "bertscore_f1": F1.mean().item()
    }

def run_benchmark(mode: str):
    os.environ["OLLAMA_MODE"] = mode
    checkpoint_file = f"eval/benchmark_results_{mode}.json"
    results = load_checkpoint(checkpoint_file)
    
    # Mock data setup (reusing previous data structure logic)
    retriever = HybridRetriever(model_name="intfloat/multilingual-e5-small")
    # ... In a real scenario, you'd load thousands of documents here.
    raw_reviews = [
        {"review_id": "r1", "product_id": "p1", "rating": 5, "verified_purchase": True, "timestamp": "2023-10-01T10:00:00", "text": "The battery life is amazing! I charge it once a day. Highly recommend."},
        {"review_id": "r2", "product_id": "p1", "rating": 1, "verified_purchase": False, "timestamp": "2023-10-01T10:05:00", "text": "Terrible. Waste of money. Do not buy."},
        {"review_id": "r3", "product_id": "p1", "rating": 4, "verified_purchase": True, "timestamp": "2023-10-02T12:00:00", "text": "La duración de la batería es increíble. Pero el envío fue muy lento."},
        {"review_id": "r4", "product_id": "p2", "rating": 1, "verified_purchase": True, "timestamp": "2023-10-03T10:00:00", "text": "Build quality is terrible. Feels like cheap plastic."},
        {"review_id": "r5", "product_id": "p2", "rating": 5, "verified_purchase": False, "timestamp": "2023-10-03T10:05:00", "text": "great product highly recommend best build ever"} # Fake
    ]
    
    extractor = FeatureExtractor()
    df = extractor.extract_features(raw_reviews)
    df['graph_collusion_risk'] = 0.0
    classifier = TrustClassifier()
    classifier.train(df)
    
    all_texts = [r["text"] for r in raw_reviews]
    retriever.add_documents(all_texts, raw_reviews)
    
    client = LLMClient()
    orchestrators = {
        "Trust-Aware RAG": PipelineOrchestrator(retriever, extractor, classifier, client),
        "Plain RAG": PlainRAGOrchestrator(retriever, extractor, classifier, client),
        "BM25 Only": BM25Orchestrator(retriever, extractor, classifier, client)
    }

    for system_name, orch in orchestrators.items():
        if system_name not in results:
            results[system_name] = {"predictions": [], "references": []}
            
        for i, item in enumerate(EVAL_SET):
            # Checkpoint check
            if i < len(results[system_name]["predictions"]):
                continue
                
            print(f"[{mode}] Running {system_name} on {item['product_id']}...")
            try:
                out = orch.run(item['product_id'], item['query'])
                pred = out.get("summary_text", "")
                results[system_name]["predictions"].append(pred)
                results[system_name]["references"].append(item['reference_summary'])
                
                save_checkpoint(checkpoint_file, results)
                
                # Rate Limiter: backoff to avoid cloud quotas
                if mode == "cloud":
                    print("Sleeping 5 seconds to avoid rate limits...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Failed to process {item['product_id']}: {e}")
                # Save state and exit to allow resumption later
                save_checkpoint(checkpoint_file, results)
                return

def generate_report():
    modes = ["local", "cloud"]
    all_metrics = []
    
    for mode in modes:
        filepath = f"eval/benchmark_results_{mode}.json"
        if not os.path.exists(filepath):
            continue
            
        results = load_checkpoint(filepath)
        for sys_name, data in results.items():
            preds = data["predictions"]
            refs = data["references"]
            if len(preds) > 0:
                print(f"Computing metrics for {sys_name} ({mode})...")
                metrics = compute_metrics(preds, refs)
                metrics["System"] = sys_name
                metrics["Mode"] = mode
                all_metrics.append(metrics)
                
    if not all_metrics:
        print("No results found to report.")
        return
        
    df = pd.DataFrame(all_metrics)
    df.to_csv("eval/benchmark_metrics.csv", index=False)
    print("Metrics saved to eval/benchmark_metrics.csv")
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    # ROUGE-L comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="System", y="rougeL", hue="Mode")
    plt.title("Summary Quality: ROUGE-L Score by System and LLM Mode")
    plt.ylabel("ROUGE-L F1")
    plt.savefig("eval/chart_rougeL.png")
    
    # BERTScore comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="System", y="bertscore_f1", hue="Mode")
    plt.title("Summary Quality: BERTScore by System and LLM Mode")
    plt.ylabel("BERTScore F1")
    plt.savefig("eval/chart_bertscore.png")
    
    print("Charts generated: eval/chart_rougeL.png, eval/chart_bertscore.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the benchmarks")
    parser.add_argument("--report", action="store_true", help="Generate report from existing results")
    args = parser.parse_args()
    
    if args.run:
        print("--- RUNNING LOCAL BENCHMARK ---")
        run_benchmark("local")
        print("\n--- RUNNING CLOUD BENCHMARK ---")
        run_benchmark("cloud")
        
    if args.report:
        generate_report()
