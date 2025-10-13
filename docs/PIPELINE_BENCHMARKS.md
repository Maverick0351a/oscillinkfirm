# Pipeline Benchmarks

This document captures benchmark methodology and sample results for the Oscillink ingest and query pipeline.

## Setup

- OS: Windows (PowerShell)
- Python: 3.11 (venv in .venv)
- Query server: examples/query_server.py containerized via deploy/docker-compose.query.onprem.yml
- Index mount: deploy/data -> /data inside container

## Warmup

The HTTP service exposes POST /v1/warmup to pre-load the embedding model and JSONL indices. The query service caches JSONL index contents in-process.

## Query benchmarks

Example commands:

```powershell
# HTTP, warmup + concurrency
.venv\Scripts\python.exe scripts\bench_query.py --mode http --server-warmup --url http://localhost:8080/v1/query --index /data/demo_index.jsonl --q "What is Oscillink?" --runs 200 --warmup 5 --concurrency 8 --out _bench\query_http_warm8.json --out-csv _bench\query_http_warm8.csv

# Local programmatic
.venv\Scripts\python.exe scripts\bench_query.py --mode local --index deploy\data\demo_index.jsonl --q "What is Oscillink?" --runs 200 --warmup 5 --out _bench\query_local.json --out-csv _bench\query_local.csv
```

Sample (small index, localhost):

- HTTP warm (4-way): p50 ~ 12.8 ms, p95 ~ 31.5 ms, avg ~ 14.9 ms
- Local: p50 ~ 16.5 ms, p95 ~ 21.5 ms, avg ~ 17.0 ms

CSV contains raw latencies for plotting (one value per row, ms).

## Ingest benchmarks

Single-run and multi-run timing of extract -> ocr_if_needed -> chunk -> embed -> index.

```powershell
# Single small text file, multi-run, stub embeddings for fast deterministic timings
.venv\Scripts\python.exe scripts\bench_ingest.py --input _tmp_demo\sample.txt --parser auto --out _bench\index.demo.jsonl --runs 10 --warmup 2 --stub --out-csv yes
```

Outputs:

- JSON summary at _bench/index.demo.jsonl.timing.json, including per-stage p50/p95/avg/min/max
- CSV at _bench/index.demo.jsonl.runs.csv with columns: extract_ms, ocr_ms, chunk_ms, embed_ms, index_ms, total_ms

## Synthetic index generation

To scale the JSONL size deterministically:

```powershell
.venv\Scripts\python.exe scripts\generate_demo_index.py --out deploy\data\synth_index.jsonl --n 10000 --dim 64 --seed 42
```

Use the generated path in query benchmarks (`--index /data/synth_index.jsonl`).

## Notes

- For FAISS comparisons, build CPU FAISS and add a FAISS index builder to compare (TBD).
- Ensure /v1/warmup is called for realistic steady-state numbers.
- For deterministic fast ingest runs, prefer --stub to avoid heavy model load time dominating measurements.
