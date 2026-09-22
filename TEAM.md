# Team — 4-Person Big Project

| Person | Role | Ownership | Files |
|--------|------|-----------|-------|
| **P1 — Data & Trust** | Data Engineer + Trust Research | Ingestion, multilingual, XLM-R, graph | `src/preprocessing/*`, `src/trust/*`, `data/`, `eval/eval_trust.py` |
| **P2 — Retrieval & RAG** | RAG Engineer | Hybrid search, VectorDB, eval | `src/retrieval/hybrid_search.py`, `src/preprocessing/ingest.py`, `eval/run_benchmark.py` |
| **P3 — Backend & Infra** | Backend / MLOps | FastAPI, scaling, Docker, CI | `src/api/main.py`, `docker-compose.yml`, `Dockerfile`, `.github/workflows/ci.yml` |
| **P4 — Frontend & Dashboard** | Full-stack | Extension + Seller Dashboard | `extension/*`, `dashboard/app.py` |

**Stand-ups:** Weekly PR review, `eval/benchmark_metrics.csv` as single source of truth.
**4-person artefact:** Each PR must touch only own pillar; cross-review required.
