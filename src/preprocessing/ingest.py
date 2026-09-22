"""
P2: Data Ingestion Pipeline — 4-person scale
Handles CSV/JSONL Amazon reviews, multilingual cleaning, chunking, bulk indexing.
Usage: python -m src.preprocessing.ingest --input data/amazon_reviews.csv --product-id B0FNJS4N2H
"""
import os, json, pandas as pd, argparse
from typing import List, Dict, Any
from .cleaner import TextCleaner
from .chunker import Chunker
from ..retrieval.hybrid_search import HybridRetriever
from ..trust.features import FeatureExtractor
from ..trust.graph import CollusionGraph
from ..trust.classifier import TrustClassifier

def load_reviews(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[Ingest] No file at {path}, using synthetic demo")
        return []
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    raise ValueError("Unsupported format")

def ingest(product_id: str = None, input_path: str = "data/amazon_reviews.csv"):
    cleaner = TextCleaner()
    chunker = Chunker()
    retriever = HybridRetriever()
    extractor = FeatureExtractor()
    graph = CollusionGraph()
    classifier = TrustClassifier()

    reviews = load_reviews(input_path)
    if product_id:
        reviews = [r for r in reviews if r.get("product_id") == product_id] or reviews

    if not reviews:
        print("[Ingest] 0 reviews — run with --demo synthetic or place data/amazon_reviews.csv")
        return retriever

    # Clean + chunk (multilingual)
    for r in reviews:
        r["text"] = cleaner.clean_text(r["text"])
        r["lang"] = cleaner.detect_language(r["text"])
    # Expand long reviews via chunker
    expanded = []
    for r in reviews:
        chunks = chunker.chunk(r["text"])
        for i, c in enumerate(chunks):
            nr = r.copy()
            nr["review_id"] = f"{r['review_id']}_c{i}"
            nr["text"] = c
            expanded.append(nr)
    print(f"[Ingest] {len(reviews)} → {len(expanded)} chunks (lang sample: {expanded[0].get('lang')})")

    # Graph collusion
    graph.build_graph(expanded)
    risk = graph.detect_collusion_clusters()
    df = extractor.extract_features(expanded)
    df['graph_collusion_risk'] = df['reviewer_id'].map(risk).fillna(0.0)
    classifier.train(df)
    print(f"[Ingest] Trained XGBoost on {len(df)} rows, collusion clusters {len(risk)}")

    # Index
    retriever.add_documents([r["text"] for r in expanded], expanded)
    print(f"[Ingest] Indexed {retriever.faiss_index.ntotal} vectors → {retriever.index_path}")
    return retriever

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/amazon_reviews.csv")
    p.add_argument("--product-id", default=None)
    args = p.parse_args()
    ingest(args.product_id, args.input)
