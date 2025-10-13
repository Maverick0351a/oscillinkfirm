# On‑Prem Container Benchmark

This guide shows how to time the ingest pipeline inside the licensed on‑prem container.

## Prepare
- License at `deploy/license/oscillink.lic`
- Documents in `deploy/data/`
- Tika sidecar up

## Build and start
```powershell
docker compose -f deploy/docker-compose.onprem.licensed.yml build
docker compose -f deploy/docker-compose.onprem.licensed.yml up -d tika
```

## Benchmark JSONL path
```powershell
docker compose -f deploy/docker-compose.onprem.licensed.yml run --rm app `
  python tools/bench_ingest.py `
    --input /data/sample.txt `
    --out /data/index.jsonl `
    --backend jsonl `
    --extract-parser auto `
    --chunker unstructured
```

## Benchmark FAISS IVF
```powershell
docker compose -f deploy/docker-compose.onprem.licensed.yml run --rm app `
  python tools/bench_ingest.py `
    --input /data/sample.txt `
    --out /data/index.faiss `
    --backend faiss `
    --faiss-variant ivf `
    --nlist 64 `
    --seed 0
```

The script prints a single JSON line with timing and settings, e.g.:

```json
{"ok": true, "rc": 0, "backend": "jsonl", "faiss_variant": null, "elapsed_seconds": 0.4231, ...}
```

Notes
- Determinism: `OSC_DETERMINISTIC=1` is set; use consistent seeds/params and input order.
- Use larger files and repeat runs to characterize warm vs cold paths (Tika / OCR / disk cache).
- For batch comparisons, redirect output to a file and aggregate in your preferred tool.
