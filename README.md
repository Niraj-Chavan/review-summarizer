# Trust-Aware Multilingual Opinion Summarizer

An advanced multi-agent system that summarizes e-commerce product reviews across languages with a **trust-aware layer**. It detects fake/manipulated reviews (behavioral + linguistic features + reviewer-product graph via Louvain) and down-weights them before LLM summarization — producing a **Trust-Adjusted Rating** vs raw average.

[![Ollama](https://img.shields.io/badge/Ollama-0.24.0-black)](https://ollama.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Architecture

```mermaid
graph TD
    A[Raw Multilingual Reviews] --> B(Preprocessing & Chunking)
    B --> C(Hybrid Retrieval: BM25 + FAISS]
    A --> D(Trust & Fraud Agent)
    D --> |Behavioral/Linguistic Features| E[XGBoost Classifier]
    D --> |Bipartite Graph| F[Louvain Community Detection]
    E --> G[Trust Score per Review]
    F --> G
    C --> |Top K Reviews| H(Aspect & Opinion LLM Agent)
    G --> |Trust Weights| H
    H --> I[Trust-Weighted Aspect Scores]
    I --> J(Summarization LLM Agent)
    J --> K[Final Output JSON & Rating]
```

**Key Components:**
- `src/preprocessing/` — `TextCleaner` (HTML/URL/emoji/slang), `langid` language detection
- `src/retrieval/hybrid_search.py` — `HybridRetriever` (multilingual-e5-small, FAISS + BM25 + RRF)
- `src/trust/features.py` + `classifier.py` + `graph.py` — `FeatureExtractor`, `TrustClassifier` (XGBoost), `CollusionGraph`
- `src/agents/` — `LLMClient` (4 modes), `AspectOpinionAgent`, `SummarizationAgent`, `PipelineOrchestrator`
- `src/api/main.py` — FastAPI with caching, timeout, synthetic per-product generation, CORS for extension
- `extension/` — Premium Chrome overlay (manifest v3, content script, glassmorphism UI)
- `eval/` — Benchmarks (Plain RAG vs BM25 vs Trust-Aware) + ROUGE/BERTScore

---

## Quick Start for Friends (5 mins)

### 1. Clone & Install
```bash
git clone https://github.com/Niraj-Chavan/review-summarizer.git
cd review-summarizer
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install & Configure Ollama
```bash
# Install (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh
ollama --version                  # 0.24.0

# Start daemon
ollama serve &                    # http://localhost:11434
# Pull local model (4.9GB, ~5 mins on good internet)
ollama pull llama3.1:8b           # verified: 46e0c10c039e 4.9GB
ollama list                       # check

# For Cloud Mode (free tier, no 40GB download)
ollama signin                     # opens https://ollama.com/connect → Authorize
# Test cloud (uses gpt-oss:20b-cloud free, llama3.1:70b-cloud is 404)
ollama run gpt-oss:20b-cloud "hello"
```

### 3. Configure Environment
```bash
cp .env.example .env
cat .env
# OLLAMA_MODE=cloud               # local | cloud | auto | hybrid
# OLLAMA_MODEL_LOCAL=llama3.1:8b
# OLLAMA_MODEL_CLOUD=gpt-oss:20b-cloud  # was llama3.1:70b-cloud (404)
# OLLAMA_HOST=http://localhost:11434

# Switch modes:
# Local (offline, recommended for demo DEMO.md:6):
echo "OLLAMA_MODE=local" > .env
# Cloud (benchmark quality, 15-30s latency):
echo "OLLAMA_MODE=cloud" > .env
# Auto (prefers cloud if signed in else local):
echo "OLLAMA_MODE=auto" > .env
```

### 4. Run Backend API
```bash
source venv/bin/activate
# Port 8000 may be occupied by other app (web.app) → use 8080 (extension supports 8000/8001/8080)
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
# Preloads B08J5F3G18 mocks, synthetic generator for any ASIN, XGBoost, FAISS
# Docs: http://localhost:8080/docs
```

Test it:
```bash
curl -X POST http://localhost:8080/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"product_id":"B0F66XDSLF","platform":"amazon","product_title":"for AirTag Holder for Kids with Invisible Pin"}' | python3 -m json.tool
# {"raw_rating":4.0,"trust_adjusted_rating":3.7,"down_weighted_count":5,"total_reviews":10,
#  "aspects":[{"pin":0.17},{"durability":-0.5},{"waterproof_cover":0.5}...],
#  "summary_text":"This AirTag holder for kids offers... keeps AirTags secure on backpacks..."}
# Headphone: B0F66XDSLF (Nothing) → battery/sound, AirTag: B0FNJS4N2H → pin/waterproof (title-aware)
# Any product works — synthetic per-product + title category (headphone vs airtag vs generic)
```

### 5. Load Chrome Extension
1. `chrome://extensions` → enable **Developer mode** (top-right)
2. **Load unpacked** → select `review-summarizer/extension` (contains `manifest.json`, `content.js`, `style.css`)
3. Pin the **Trust-Aware** icon
4. Open Amazon: `https://www.amazon.com/dp/B0F66XDSLF` (Nothing Headphone) or any `/dp/B0XXXXXXXX`
5. Hard-refresh `Ctrl+Shift+R` → bottom-right **Trust-Aware Summary** slides in:
   - **Trust-Adjusted ★4.29 vs Raw 4.1** with delta chip, 5-star row
   - **Trust Gauge** (50-100% trusted), **5 flagged / 10 total**
   - **AI Summary** 2-4 sentences, **Aspect Breakdown** (battery, build quality with -0.5 to +0.6 bars)
   - **Trust Breakdown** accordion (verified/burstiness, duplicate, Louvain) + footer latency/model

> **If badge shows `Connection failed`:** backend not on 8080 → `ss -tlnp | grep 8080` or check `ollama serve`. Extension auto-fallback tries 8000 → 8001 → 8080.

### 6. Run Tests & Benchmarks
```bash
venv/bin/pytest -q                          # 5 passed (84s) - preprocessing, retrieval, integration
venv/bin/python eval/run_benchmark.py --run   # Trust-Aware vs Plain RAG vs BM25
venv/bin/python eval/run_benchmark.py --report # → eval/chart_rougeL.png, chart_bertscore.png
venv/bin/python eval/eval_trust.py            # Precision/Recall/F1 on 100 mock fakes
```

---

## How It Works (For Your Friend to Understand)

**Is it scanning Amazon live?** No — current demo uses **synthetic per-product** generation (not live Amazon scraping). Real scraping would need Amazon SP-API / DOM review scraping (`data-hook="review"`), which is not yet implemented. The extension now is **title-aware** to fix the "AirTag shows headphones" bug.

1. **You visit Amazon:** Content script `extension/content.js:14` extracts ASIN via regex `/dp/([A-Z0-9]{10})` + `extension/content.js:25` `extractProductTitle()` from `#productTitle`
2. **Extension calls API:** `fetchWithFallback()` POST `http://localhost:8080/api/summarize` with `{product_id, product_title}` (supports 8000/8001/8080 fallback)
3. **Backend:** `src/api/main.py:103` checks `retriever.metadata` — if new ASIN, calls `_generate_synthetic_for_product(pid, title)` → `_detect_category(title)` picks `airtag|headphone|generic` pool (+ `fake` burst 30%), trains XGBoost, indexes via FAISS/BM25. Else cache. Title also passed to `orchestrator.run(pid, "", title)` → `summarization_agent.py:32` includes `Product: {title}` in LLM prompt to avoid hallucinating unrelated product.
4. **Pipeline:** `PipelineOrchestrator.run()` → Hybrid retrieval top-10 → Trust scoring → LLM aspect extraction + summarization (`gpt-oss:20b-cloud` or `llama3.1:8b`) → JSON with `total_reviews`
5. **UI Renders:** `content.js:159` calculates trusted/flagged, gauge `conic-gradient`, aspect bars, collapsible breakdown. Cached per `amazon:ASIN` for Ollama quota.

**Why trust-adjusted differs?** Fake 5★ templated reviews (`great product highly recommend`) get `trust_score <0.5` → down-weighted in `SummarizationAgent` weighted average, so genuine 2★ critiques pull rating down (or up if fakes were negative).

**Your screenshot bug explained:** `B0FNJS4N2H` AirTag page showed *“wireless headphones ... ANC ... spatial sound”* because old `_synthetic_templates` was headphone-only. Fixed `src/api/main.py:49` → `_synthetic_pool` with `headphone` vs `airtag` vs `generic` + title-aware.

## Project Structure
```
src/
  preprocessing/cleaner.py, chunker.py
  retrieval/hybrid_search.py
  trust/features.py, classifier.py, graph.py
  agents/llm_client.py (4 modes), aspect_opinion_agent.py, summarization_agent.py, orchestrator.py
  api/main.py
extension/
  manifest.json (mv3, host_permissions for amazon + localhost 8000/8001/8080)
  content.js (FAB, gauge, fallback)
  style.css (glassmorphism, Inter font)
eval/
  run_benchmark.py, eval_trust.py, mock_results.py
tests/
  test_preprocessing.py, test_retrieval.py, test_integration.py
DEMO.md — 5-min demo script
```

## Troubleshooting
| Issue | Fix |
|-------|-----|
| `8000 occupied` | Use `8080` or `kill -f "uvicorn web.app"` |
| `Failed to fetch` | `ollama serve` + `uvicorn` on 8080 running? Check `curl http://localhost:8080/docs` |
| `llama3.1:70b-cloud not found` | Use `gpt-oss:20b-cloud` (free) — updated in `.env.example` |
| Same output for all products | Update to latest `main` — old `src/api/main.py:63` hard-coded `B08J5F3G18`, now dynamic + title-aware |
| `emoji ModuleNotFound` | `source venv/bin/activate` before pytest |

## Known Limitations
- Linguistic sentiment: `textblob` fallback, true ABSA via LLM
- Graph scalability: in-memory `networkx` → production needs Neo4j
- Synthetic data for unknown ASINs (no live Amazon scraping yet) — now title-aware (fixed AirTag vs headphone) but still mock; true live scrape needs Amazon review DOM/API

## DEMO.md Flow
1. Problem (0:00) — show 5★ fakes on Amazon
2. Live demo (1:00) — refresh → badge 3.2★ vs 4.5★ raw
3. Dashboard (2:30) — Streamlit optional
4. Benchmark (3:30) — `chart_rougeL.png` Trust-Aware wins

---

**Push to GitHub:** Friend clones via `git clone https://github.com/Niraj-Chavan/review-summarizer.git` and follows steps 1-5 above. No secrets needed (`.env` gitignored, share `.env.example`).
