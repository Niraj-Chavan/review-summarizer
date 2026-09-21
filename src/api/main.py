import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from src.retrieval.hybrid_search import HybridRetriever
from src.trust.features import FeatureExtractor
from src.trust.classifier import TrustClassifier
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

# In-memory cache to protect Ollama Cloud quota
CACHE: Dict[str, Dict[str, Any]] = {}

# Mock Pipeline Initialization (In real app, this runs at startup)
retriever = HybridRetriever(model_name="intfloat/multilingual-e5-small")
extractor = FeatureExtractor()
classifier = TrustClassifier()
orchestrator = PipelineOrchestrator(retriever, extractor, classifier)

def get_timeout() -> int:
    """Returns timeout in seconds based on OLLAMA_MODE."""
    mode = os.getenv("OLLAMA_MODE", "local").lower()
    return 180 if mode == "cloud" else 60

@app.post("/api/summarize")
async def summarize(request: SummarizeRequest):
    cache_key = f"{request.platform}:{request.product_id}"
    
    if cache_key in CACHE:
        print(f"Returning cached result for {cache_key}")
        return CACHE[cache_key]

    timeout = get_timeout()
    print(f"Processing request for {cache_key} with timeout {timeout}s")
    
    # Run the orchestrator in a background thread to avoid blocking the event loop
    # We wrap it in asyncio.wait_for to enforce the timeout
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(orchestrator.run, request.product_id, ""),
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
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama service is unreachable. Please check if 'ollama serve' is running.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
