# Demo Script: Trust-Aware Review Summarizer

**Goal:** Demonstrate the trust-adjusted rating, explainable trust flags, and benchmark results in under 5 minutes without relying on live cloud calls.

## Preparation (Before Demo)
1. Start `ollama serve` locally with `llama3.1:8b` downloaded.
2. Start the FastAPI backend: `uvicorn src.api.main:app`.
3. Load the Browser Extension in Chrome (`chrome://extensions` -> Load Unpacked).
4. Have the benchmark charts (`chart_rougeL.png` and `chart_bertscore.png`) pre-generated and open in tabs.

## Sequence of Actions

### 1. The Problem Statement (0:00 - 1:00)
- **Action:** Open a standard Amazon product page (e.g., a cheap electronics product with a 4.5★ rating but suspicious reviews).
- **Talking Track:** "Traditional opinion summarizers and platform ratings treat all reviews equally. But look at these 5-star reviews—many are unverified, posted on the exact same day, and use the exact same 'great product' phrasing. Our system fixes this by treating trust as a first-class input."

### 2. The Live Demonstration (1:00 - 2:30)
- **Action:** Refresh the Amazon page. Point to the Extension Badge appearing in the bottom right corner.
- **Talking Track:** "Behind the scenes, the extension queries our local FastAPI backend (running local Ollama for zero latency/network risk). The backend uses a hybrid FAISS/BM25 retriever, passes the reviews through an XGBoost Trust Classifier, and feeds the trust-weighted reviews to the LLM."
- **Action:** Expand the badge to show the results.
- **Talking Track:** "Notice how the **Trust-Adjusted Rating** drops to 3.2★ compared to the raw 4.5★. The LLM summary explicitly mentions that while battery life is praised, the build quality is poor, and the system flagged 45 reviews as low-trust."

### 3. The Analytics Dashboard (2:30 - 3:30) (If Option B was built)
- **Action:** Open the local Streamlit dashboard.
- **Talking Track:** "For sellers, we provide a time-series view. You can see a spike in fake reviews detected last weekend, which automatically triggered this red alert banner."

### 4. Benchmark & Validation (3:30 - 5:00)
- **Action:** Show the pre-generated `chart_rougeL.png` and `chart_bertscore.png` on screen.
- **Talking Track:** "To prove this works, we benchmarked our Trust-Aware RAG against Plain RAG and BM25-only retrieval. We ran this evaluation using the larger Ollama Cloud model (`llama3.1:70b-cloud`). As you can see, our Trust-Aware system significantly outperforms Plain RAG in ROUGE-L and BERTScore because it successfully ignores the contradictory noise injected by fake review farms."
