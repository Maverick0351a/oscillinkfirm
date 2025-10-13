# On‑Prem Container Quickstart

This guide shows how to run the ingest CLI in containers with deterministic defaults, OCR/PDF extraction, and a Tika sidecar.

## What you get
- App image with `oscillink` installed, plus: `ocrmypdf`, `tesseract-ocr-eng`, `pdfminer.six`, `unstructured`, and `faiss-cpu`.
- A Tika server sidecar for robust document extraction.
- Determinism defaults via `OSC_DETERMINISTIC=1`.

## Prerequisites
- Docker and Docker Compose
- A folder of documents to ingest in `deploy/data/` (bind-mounted into the container)

## Build and start services

- Build the app image and start the Tika sidecar:

  (From repo root)
  - Windows PowerShell:
    - `docker compose -f deploy/docker-compose.onprem.yml build`
    - `docker compose -f deploy/docker-compose.onprem.yml up -d tika`

This leaves the `app` service defined but idle (sleep). We run CLI commands with `docker compose run` for one-shot jobs.

## Ingest examples

- Deterministic JSONL index with auto extraction (prefers Tika, then pdfminer), paragraph chunking, and simple embedding fallback:

  Windows PowerShell:
  - `docker compose -f deploy/docker-compose.onprem.yml run --rm app python -m oscillink.ingest.cli ingest --input /data/sample.txt --index-out /data/index.jsonl --extract-parser auto --tika-url $Env:OSC_TIKA_URL --chunker paragraph`

- Building a FAISS IVF index deterministically (requires faiss-cpu in the image):

  Windows PowerShell:
  - `docker compose -f deploy/docker-compose.onprem.yml run --rm app python -m oscillink.ingest.cli ingest --input /data/sample.txt --index-out /data/index.faiss --index-backend faiss --faiss-variant ivf --faiss-nlist 64 --faiss-seed 0`

Notes:
- Replace `/data/sample.txt` with your document path inside the container. Place files under `deploy/data/` on the host.
- Set `--extract-parser tika` or `auto` with `--tika-url http://tika:9998` to use the Tika sidecar.
- OCR is attempted with `ocrmypdf` when extraction returns no pages.

## Determinism checklist
- `OSC_DETERMINISTIC=1` is set in the compose file; it pins threads and seeds PRNGs where applicable.
- Keep input order, FAISS params (`--faiss-*`), and embedding model selection stable to reproduce index hashes.

## Cleanup
- Stop services: `docker compose -f deploy/docker-compose.onprem.yml down`

## Troubleshooting
- Ensure Tika is healthy: `http://localhost:9998` should respond.
- If OCR fails, confirm `ocrmypdf` runs inside the container: `docker compose run --rm app ocrmypdf --version`.
- For verbose ingest logs, add `--verbose` to CLI or set `PYTHONVERBOSE=1` temporarily.
