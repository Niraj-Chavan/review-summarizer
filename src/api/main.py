import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from src.retrieval.hybrid_search import HybridRetriever
from src.trust.features import FeatureExtractor
from src.trust.classifier import TrustClassifier
from src.trust.graph import CollusionGraph
from src.agents.orchestrator import PipelineOrchestrator

app = FastAPI(title="Trust-Aware Review Summarizer API")

# Allow CORS for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to extension ID or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    product_id: str
    platform: str = "amazon"
    product_title: str = ""

# In-memory cache to protect Ollama Cloud quota
CACHE: Dict[str, Dict[str, Any]] = {}

# Pipeline Initialization — 4-person scale (P1+P2)
retriever = HybridRetriever(model_name="intfloat/multilingual-e5-small")
extractor = FeatureExtractor()
classifier = TrustClassifier()
collusion_graph = CollusionGraph()

import hashlib
import random

# PRELOAD MOCK DATA FOR DEMONSTRATION
raw_reviews = [
    {"review_id": "r1", "reviewer_id": "u1", "product_id": "B08J5F3G18", "rating": 5, "verified_purchase": False, "timestamp": "2023-10-01T10:00:00", "text": "This product is amazing. I highly recommend it. Great product!"},
    {"review_id": "r2", "reviewer_id": "u2", "product_id": "B08J5F3G18", "rating": 5, "verified_purchase": False, "timestamp": "2023-10-01T10:05:00", "text": "Great product. Perfect. The best thing ever made. Buy it now."},
    {"review_id": "r3", "reviewer_id": "u3", "product_id": "B08J5F3G18", "rating": 1, "verified_purchase": True, "timestamp": "2023-10-02T12:00:00", "text": "The battery died after two days. Also the build quality is cheap plastic."},
    {"review_id": "r4", "reviewer_id": "u4", "product_id": "B08J5F3G18", "rating": 2, "verified_purchase": True, "timestamp": "2023-10-03T15:30:00", "text": "Not worth the money. Build quality is terrible. Battery life is decent though."}
]

# Category-aware synthetic templates — fixes AirTag vs Headphone mismatch
_synthetic_pool = {
    "headphone": [
        ("Battery life is amazing, lasts 80 hours as advertised! ANC is great.", 5, True),
        ("Sound quality is excellent, KEF-tuned audio is crystal clear.", 5, True),
        ("Comfortable for long hours, build is premium white finish.", 4, True),
        ("Spatial sound is immersive, fast charging works in 30 mins.", 4, True),
        ("Battery died after two days. Also the build quality is cheap plastic.", 1, True),
        ("The headband cracked after a month, poor durability.", 2, True),
        ("ANC is weak compared to Sony, overpriced for the quality.", 2, True),
    ],
    "airtag": [
        ("Invisible pin holds perfectly on my kid's clothes, very secure.", 5, True),
        ("Waterproof cover works great, survived washing machine!", 5, True),
        ("Lightweight and comfortable, my toddler doesn't notice it.", 4, True),
        ("Perfect for tracking backpack and shoes, pin is sturdy.", 4, True),
        ("Pin is flimsy, broke after a week, not durable.", 2, True),
        ("Too bulky for small clothes, fell off easily.", 2, True),
        ("Not truly waterproof, got damaged in rain.", 1, True),
    ],
    "generic": [
        ("Great value for money, does exactly what it promises.", 4, True),
        ("Quality is okay for the price, shipping was fast.", 4, True),
        ("Stopped working after a month, poor durability.", 2, True),
        ("Overpriced for what you get, build feels cheap.", 2, True),
    ],
    "fake": [
        ("great product highly recommend best build ever", 5, False),
        ("Great product. Perfect. The best thing ever made. Buy it now.", 5, False),
    ]
}

def _detect_category(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ["airtag", "tracker", "gps", "finder", "pin", "holder"]):
        return "airtag"
    if any(k in t for k in ["headphone", "earphone", "earbud", "anc", "kef", "nothing"]):
        return "headphone"
    return "generic"

def _generate_synthetic_for_product(pid: str, n=10, title: str = ""):
    """Deterministic synthetic reviews per product_id + title category to match actual product."""
    h = int(hashlib.md5((pid + title).encode()).hexdigest()[:8], 16)
    rng = random.Random(h)
    cat = _detect_category(title)
    pool = _synthetic_pool[cat] + _synthetic_pool["generic"]
    reviews = []
    for i in range(n):
        text, rating, verified = rng.choice(pool)
        reviews.append({
            "review_id": f"{pid}_r{i}",
            "reviewer_id": f"{pid}_u{i%6}",
            "product_id": pid,
            "rating": rating,
            "verified_purchase": verified if rng.random() > 0.15 else False,
            "timestamp": f"2023-10-{rng.randint(1,28):02d}T{rng.randint(10,22):02d}:{rng.randint(0,59):02d}:00",
            "text": text
        })
    # Inject 30% fake burst for trust delta visibility
    for j in range(3):
        text, rating, verified = rng.choice(_synthetic_pool["fake"])
        reviews[j]["text"] = text
        reviews[j]["rating"] = 5
        reviews[j]["verified_purchase"] = False
        reviews[j]["timestamp"] = "2023-10-01T10:0{}:00".format(j)
    return reviews

try:
    collusion_graph.build_graph(raw_reviews)
    risk = collusion_graph.detect_collusion_clusters()
    df = extractor.extract_features(raw_reviews)
    df['graph_collusion_risk'] = df['reviewer_id'].map(risk).fillna(0.0)
    classifier.train(df)
    all_texts = [r["text"] for r in raw_reviews]
    retriever.add_documents(all_texts, raw_reviews)
    print(f"[Init] Graph collusion risk {risk}, indexed {len(raw_reviews)}")
except Exception as e:
    print(f"Warning: Mock data failed to load: {e}")

orchestrator = PipelineOrchestrator(retriever, extractor, classifier, collusion_graph=collusion_graph)

def get_timeout() -> int:
    """Returns timeout in seconds based on OLLAMA_MODE."""
    mode = os.getenv("OLLAMA_MODE", "local").lower()
    return 180 if mode == "cloud" else 60

@app.post("/api/summarize")
async def summarize(request: SummarizeRequest):
    # Dynamic product handling — generate synthetic data if product not seen before
    pid = request.product_id
    cache_key = f"{request.platform}:{pid}"
    
    if cache_key in CACHE:
        print(f"Returning cached result for {cache_key}")
        return CACHE[cache_key]

    # If product not in retriever, generate synthetic reviews on-the-fly (title-aware)
    title = getattr(request, "product_title", "") or ""
    print(f"Request pid={pid} title='{title[:60]}'")
    existing_pids = set(m.get("product_id") for m in retriever.metadata)
    if pid not in existing_pids:
        print(f"Product {pid} not in cache — generating synthetic reviews for category '{_detect_category(title)}'")
        synth = _generate_synthetic_for_product(pid, n=10, title=title)
        try:
            # Build bipartite graph for new product
            tmp_graph = CollusionGraph()
            tmp_graph.build_graph(synth)
            synth_risk = tmp_graph.detect_collusion_clusters()
            df_syn = extractor.extract_features(synth)
            df_syn['graph_collusion_risk'] = df_syn['reviewer_id'].map(synth_risk).fillna(0.0)
            classifier.train(df_syn)
            retriever.add_documents([r["text"] for r in synth], synth)
            print(f"[Synthetic] Graph risk {synth_risk} for {pid}")
        except Exception as e:
            print(f"Synthetic generation failed for {pid}: {e}")
            # Fallback: add without trust training
            retriever.add_documents([r["text"] for r in synth], synth)

    timeout = get_timeout()
    print(f"Processing request for {cache_key} with timeout {timeout}s")
    
    # Run the orchestrator in a background thread to avoid blocking the event loop
    # We wrap it in asyncio.wait_for to enforce the timeout
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(orchestrator.run, pid, "", title),
            timeout=timeout
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        # Store in cache
        CACHE[cache_key] = result
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, 
            detail=f"Ollama request timed out after {timeout}s. (Mode: {os.getenv('OLLAMA_MODE', 'local')})"
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama service is unreachable. Please check if 'ollama serve' is running.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
