# Contributing — 4-Person Workflow
1. Branch per pillar: `p1/trust-xlmr`, `p2/vector-qdrant`, `p3/redis-cache`, `p4/dashboard`
2. `pytest -q` must pass (5 tests). `eval/run_benchmark.py --run` for RAG changes.
3. Add docs to `README.md` + `TEAM.md`.
4. Docker: `docker compose up --build` should bring api:8080 + dashboard:8501 + ollama:11434.
