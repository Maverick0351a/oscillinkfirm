# Pipeline Benchmarks (Repeatable)

This folder captures reproducible, no-fluff numbers for latency, determinism, and throughput.

- Latency (warm caches):
  - ANN→Oscillink p50 ≤ 100 ms; p95 ≤ 180 ms at K=60, D=384 on CPU.
  - /v1/chat extractive end-to-end p50 ≤ 250 ms (context-size dependent).
- Determinism: Two identical /v1/chat calls under OSC_DETERMINISTIC=1 yield identical bundle order, identical receipts, and identical report bundle_hash.
- Throughput: Watcher ≥ 40 non-OCR PDFs/min per container; OCR throughput bounded by Tesseract.

Artifacts to include after running scripts:
- CSVs: latency_chat.csv, latency_ann.csv
- JSON: determinism_check.json
- Logs: watcher_throughput.json
