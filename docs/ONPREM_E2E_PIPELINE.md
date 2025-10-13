# On‑prem E2E Pipeline and Deterministic Container Plan

This document lays out the end-to-end local pipeline and containerization plan, tied to concrete files and acceptance criteria in this repo.

High-level flow (local):
file → OCR/text extract → chunk → embed → ANN recall → Oscillink settle → bundle + chained receipts

Deterministic mode: fixed model weights, pinned threads, stable tie-breaks, signed receipts.

Containerization: CPU-first image; optional GPU image.

## Repo layout additions

New modules and assets to add under existing package structure. Existing relevant modules include `oscillink/core/*` (lattice, receipts, determinism) and `cloud/app/*` (FastAPI). New code will live primarily under `oscillink/ingest` and thin glue in `oscillink/api`.

- oscillink/
  - ingest/
    - __init__.py
    - cli.py                  — Typer CLI: `osc ingest`, `osc query`
    - extract.py              — Tika client + OCRmyPDF wrapper dispatcher
    - ocr.py                  — Tesseract; optional PaddleOCR/docTR backends
    - chunk.py                — Unstructured partition + chunk_by_title
    - embed.py                — sentence-transformers loader + deterministic wrapper
    - index_faiss.py          — FAISS index builder/reader; ANN recall
    - receipts.py             — IngestReceipt, chaining into SettleReceipt
    - determinism.py          — thread pins, env guards, stable sort
    - models_registry.py      — local model registry with hashes
  - adapters/
    - recall.py               — mutual-kNN+prune wrapper (deterministic)
  - api/
    - routes_ingest.py        — POST /v1/ingest, POST /v1/query (end-to-end)
  - assets/schemas/
    - ingest_receipt.schema.json
- docker/
  - Dockerfile.cpu
  - Dockerfile.gpu
  - docker-compose.onprem.yml
- scripts/
  - build_index.py
  - print_receipt.py
  - demo_corpus/
    - doc1.pdf  img_scan.png  contract.docx

Where possible, reuse existing `oscillink/core/lattice.py`, `oscillink/core/receipts.py`, and `cloud/app/main.py` patterns (metrics, headers) for consistency.

## Open-source models (deterministic, container-friendly)

Embeddings (Sentence-Transformers):
- bge-small-en-v1.5 (≈33M params, 384 dim)
- gte-base-en-v1.5 (768 dim)
- e5-small-v2 (384 dim)
- Optional multilingual: bge-m3 small or LaBSE

Pin exact versions; record SHA256 of weights and tokenizer. When GPU is used, set `torch.set_deterministic(True)`. Always force `eval()` and disable dropout. Seeds set by determinism switch.

OCR:
- Default: Tesseract via OCRmyPDF for image-only pages (Apache-2.0)
- Optional: PaddleOCR or docTR (Apache-2.0) via `--ocr-backend paddle|doctr`

Chunking:
- Unstructured (partition_auto, chunk_by_title) with deterministic rules

ANN index:
- FAISS FlatIP/HNSW first; IVF/PQ optional; fixed seeds for graph construction

## Determinism mode

Env var: `OSC_DETERMINISTIC=1` toggles deterministic runtime:
- Set OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1
- Stable sorts with (−score, id) keys
- Fix seeds: PYTHONHASHSEED=0, numpy.random.seed(0), torch.manual_seed(0)
- Record package versions + model weight hashes
- Persist candidate set, edge list hash, model hash, index hash in receipts

Implementation: `oscillink/ingest/determinism.py` applies this at import, and expose a `pin_threads()` context (reusing `oscillink/core/determinism.py` where possible).

## Receipts (chain of custody)

Define IngestReceipt (Pydantic), JSON schema under `oscillink/assets/schemas/ingest_receipt.schema.json`:

```
{
  "file_sha256": "...",
  "filesize": 123456,
  "mimetype": "application/pdf",
  "parse_route": "pdf_text|image_pdf_ocr|image_ocr|html|docx",
  "ocr": {"engine":"tesseract","version":"5.3.4","languages":"eng","pages_ocrd":4,"avg_conf":0.93},
  "chunker": {"engine":"unstructured","version":"0.13.2","ruleset_id":"default","chunk_count":182},
  "embedding": {
    "model":"bge-small-en-v1.5",
    "weights_sha256":"3af9…",
    "tokenizer_sha256":"77be…",
    "device":"cpu",
    "precision":"fp32",
    "count":182,
    "dim":384
  },
  "index": {"type":"faiss:IVF1024,PQ64","index_sha256":"b112…"},
  "ingest_sig": "sha256(file_sha256∥parse_route∥ocr∥chunker∥embedding∥index)"
}
```

Extend SettleReceipt (existing in `oscillink/core/receipts.py` and lattice.receipt()) with:
- parent_ingest_sig
- candidate_set_hash
- edge_hash
- params + cg_iters, final_residual
- latency_ms

API `/v1/query` returns `{ bundle, settle_receipt, ingest_receipt }`.

## CLI commands

Ingest:
- `osc ingest --input /data/docs --tika-url http://tika:9998 --ocr-backend ocrmypdf --ocr-langs "eng" --chunker unstructured --ruleset default --embed-model bge-small-en-v1.5 --index-out /data/osc_index --deterministic`

Query (end-to-end):
- `osc query --index /data/osc_index --psi "Summarize the indemnity clause" --kneighbors 6 --lamC 0.5 --lamQ 4.0 --tol 1e-3 --bundle-k 5 --deterministic --out receipts/`

`oscillink/ingest/cli.py` will implement these and be wired from the existing `oscillink/cli.py` entrypoint (new subcommands or a separate console script).

## Minimal API routes (FastAPI)

- `POST /v1/ingest` → returns `{\"ingest_receipt\": {...}}`
- `POST /v1/query` → returns `{\"bundle\":[...], \"settle_receipt\": {...}, \"ingest_receipt\": {...}}`

Implement a new router `oscillink/api/routes_ingest.py`, then include it from `cloud/app/factory.py` or add a separate lightweight `oscillink.api.app` for the on‑prem image. Reuse `/license/status` from `cloud/app/main.py` or gate endpoints via a license middleware.

## Dockerization

- docker/Dockerfile.cpu: base `python:3.11-slim`; install tesseract-ocr, ocrmypdf, FAISS CPU wheels, unstructured, sentence-transformers, fastapi, uvicorn; non-root; read-only FS; metrics protection via env; set `OSC_DETERMINISTIC=1` by default.
- docker/Dockerfile.gpu: base `nvidia/cuda:12.1.0-runtime-ubuntu22.04`; install PyTorch CUDA, FAISS-gpu; same libs; requires `--gpus all`.
- docker-compose.onprem.yml: brings up `tika` and `oscillink` services; mounts `/data` and `/models` volumes; optional `ingest-worker` for batch OCR.

## Model registry

`~/.oscillink/models.json`:

```
{
  "bge-small-en-v1.5": {"source": "hf://BAAI/bge-small-en-v1.5", "sha256": "…", "dim": 384},
  "gte-base-en-v1.5": {"source": "hf://Alibaba-NLP/gte-base-en-v1.5", "sha256": "…", "dim": 768},
  "e5-small-v2": {"source": "hf://intfloat/e5-small-v2", "sha256": "…", "dim": 384}
}
```

At startup, create if absent; download/verify hash; cache under `/models` in container.

## Success criteria

- Determinism: with `OSC_DETERMINISTIC=1`, two runs on same machine produce byte-identical `ingest_receipt` and `settle_receipt`.
- Throughput: ingest 1k mixed docs on CPU; non‑OCR PDFs ≥ 40 docs/min.
- Latency: E2E query on ~1.2k chunks < 40 ms on laptop CPU.
- Auditability: receipts include file_sha256, weights_sha256, index_sha256, edge_hash, state_sig, cg_iters, final_residual.

## Licensing & compliance

- Prefer permissive components (Apache/BSD/MIT). Include third‑party license texts in container under `/licenses` and update `NOTICE`.

## Work breakdown and mapping

- T1 Determinism plumbing → `oscillink/ingest/determinism.py`; integrate in CLI and API startup; reuse `oscillink/core/determinism.py` snapshot/pin helpers.
- T2 Extract & OCR → `oscillink/ingest/extract.py`, `oscillink/ingest/ocr.py`; tests under `tests/ingest/test_extract.py`.
- T3 Chunker → `oscillink/ingest/chunk.py`; tests under `tests/ingest/test_chunk.py`.
- T4 Embeddings → `oscillink/ingest/models_registry.py`, `oscillink/ingest/embed.py`; tests under `tests/ingest/test_embed.py`.
- T5 FAISS → `oscillink/ingest/index_faiss.py`; tests under `tests/ingest/test_index.py`.
- T6 Recall+settle wiring → `oscillink/adapters/recall.py`; uses existing `oscillink/core/lattice.py`.
- T7 Receipts and chaining → `oscillink/ingest/receipts.py`, schema file; extend `oscillink/core/receipts.py` if needed; tests under `tests/receipts/`.
- T8 CLI & API → `oscillink/ingest/cli.py`, `oscillink/api/routes_ingest.py`; wire in `cloud/app/factory.py` for cloud mode or provide `oscillink.api.app` for on‑prem image.
- T9 Golden tests & smoke → add under `tests/e2e/` and `scripts/demo_corpus`.
- T10 Docker → `docker/*` and `docker-compose.onprem.yml`.

## Notes on current repo integration

- FastAPI lives under `cloud/app/main.py` with `create_app()` in `cloud/app/factory.py`. We'll either:
  1) add a new `oscillink.api.app` app focused on ingest/query for the on‑prem image, or
  2) include `oscillink/api/routes_ingest.py` into `cloud/app/factory.py` for unified routing.
- Determinism helpers exist in `oscillink/core/determinism.py` and will be reused by the ingest layer.
- Existing receipts in `oscillink/core/lattice.py` already contain rich fields (state_sig, edge_hash, term_energies, component_sizes, determinism_env); we will extend them with `parent_ingest_sig` and `candidate_set_hash` for chaining.
