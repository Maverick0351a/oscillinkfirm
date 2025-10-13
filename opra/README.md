# OPRA: Oscillink Private RAG Appliance

Local-only appliance to ingest private docs, build a deterministic index, chat with grounded answers, and export reports with Oscillink receipts. All services bind to 127.0.0.1 and keep data on-device.

## Layout

```
opra/
  docker/
    Dockerfile.api
    Dockerfile.ui
    Dockerfile.llm
  docker-compose.local.yml
  .env.example
  config/
    opra.yaml
    prompts.yaml
  data/
    docs/
    index/
    receipts/
    reports/
  api/
    app.py
    routers/
      __init__.py
      ingest.py
      query.py
      chat.py
      report.py
    core/
      __init__.py
      faiss_store.py
      oscillink_adapter.py
      determinism.py
      receipts.py
      llm_adapter.py
      watcher.py
    eval/
      __init__.py
      eval_cli.py
  ui/
    README.md
  scripts/
    make_index.sh
    warmup.sh
    recall_check.py
```

## Quickstart

1) Build

```powershell
docker compose -f docker-compose.local.yml build
```

2) Run (loopback only)

```powershell
docker compose -f docker-compose.local.yml up -d
```

3) Drop docs

- Copy PDFs/DOCX into `./data/docs/`. A watcher will be added to auto-ingest; for now, use the scripts or call the API.

4) Try the API

- Health: `GET http://127.0.0.1:8080/health`
- Query: `POST http://127.0.0.1:8080/v1/query` with body:

```json
{
  "index_path": "/data/index/demo_index.jsonl",
  "q": "What is Oscillink?",
  "backend": "jsonl",
  "k": 5,
  "embed_model": "bge-small-en-v1.5"
}
```

5) UI (placeholder)

- UI will run at http://127.0.0.1:3000 once implemented.

## Endpoints

- POST /v1/ingest: { path, embed_model?, index_out? } → builds JSONL index + receipt
- POST /v1/index: { policy: incremental|full, docs_dir?, embed_model? } → batch build
- POST /v1/warmup: { backend, embed_model, index_path?, meta_path? } → preload models/index
- POST /v1/query: { index_path, q, k?, backend?, embed_model?, meta_path?, epsilon?, tau?, e2e? }
- GET  /v1/watcher/status: background auto-indexer status
- POST /v1/watcher/scan: run one scan iteration on demand

## Auto-ingest watcher

A lightweight background watcher polls `opra/data/docs` and auto-builds indices under `opra/data/index`.

Environment toggles:

- OPRA_WATCHER=1 (default): enable on API startup; set to 0 to disable
- OPRA_DOCS_DIR: default /data/docs
- OPRA_INDEX_DIR: default /data/index
- OPRA_RECEIPTS_DIR: default /data/receipts
- OPRA_EMBED_MODEL: default bge-small-en-v1.5
- OPRA_WATCH_INTERVAL: default 2.0 seconds

A manifest `index_manifest.json` is persisted under receipts to track processed files.

## Determinism & Security Defaults

- OSC_DETERMINISTIC=1; BLAS threads pinned.
- Embed model dim and sha256 verified against index; mismatch guarded.
- All services bound to 127.0.0.1; no egress.
- Read-only containers with tmpfs scratch; volumes only for data/config.

## Worklog

- 2025-10-12
  - [x] Scaffold created with minimal /health and /v1/query (wired to oscillink query service).
  - [ ] Watcher, ingest/index routers, chat/report endpoints.
  - [ ] UI chat with receipts modal; report builder.
