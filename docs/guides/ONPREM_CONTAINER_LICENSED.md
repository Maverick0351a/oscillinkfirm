# Licensed On‑Prem Ingest Container

This container verifies a license at startup, then lets you run deterministic ingest jobs with OCR, Tika/pdfminer extraction, Unstructured chunking, FAISS, and Sentence‑Transformers embeddings entirely on your own servers.

## Prepare
- Place your license at `deploy/license/oscillink.lic` (Ed25519‑signed JWT)
- Put documents under `deploy/data/` (mounted to `/data`)

## Build and start Tika

Windows PowerShell (run from repo root):
```powershell
docker compose -f deploy/docker-compose.onprem.licensed.yml build
docker compose -f deploy/docker-compose.onprem.licensed.yml up -d tika
```

## Run an ingest job (deterministic JSONL index)
```powershell
docker compose -f deploy/docker-compose.onprem.licensed.yml run --rm app `
  python -m oscillink.ingest.cli ingest `
    --input /data/sample.txt `
    --index-out /data/index.jsonl `
    --extract-parser auto `
    --tika-url $Env:OSC_TIKA_URL `
    --chunker unstructured `
    --embed-model bge-small-en-v1.5
```

## Build a FAISS IVF index with fixed seed
```powershell
docker compose -f deploy/docker-compose.onprem.licensed.yml run --rm app `
  python -m oscillink.ingest.cli ingest `
    --input /data/sample.txt `
    --index-out /data/index.faiss `
    --index-backend faiss `
    --faiss-variant ivf `
    --faiss-nlist 64 `
    --faiss-seed 0
```

Notes
- Determinism: `OSC_DETERMINISTIC=1` is set; keep FAISS params and input order stable to reproduce hashes.
- License verification uses `tools/license_verify.py` and exports entitlements to `/run/oscillink_entitlements.json` and env. The app reads these for limits/tiers.
- Tika sidecar runs at `http://tika:9998` (exposed on host `:9998`).
- OCR via `ocrmypdf` runs inside the container when extraction yields zero pages.

Troubleshooting
- If the container exits immediately, check for missing license or JWKS URL.
- For network‑restricted environments, the verifier supports ETag caching and an offline grace window; see environment knobs in the compose file.

Security
- No embeddings or content leave your network; only the license verification calls the JWKS endpoint. You can mirror the JWKS endpoint internally if required and point `OSCILLINK_JWKS_URL` to it.
