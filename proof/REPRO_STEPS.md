# Repro Steps (Clean Machine)

Prereqs: Python 3.11, Node 20+, Docker optional for compose.

1) Install runtime deps (local run)

- Ensure environment variables for local paths if not using Docker:
  - OPRA_DOCS_DIR=<repo>/opra/data/docs
  - OPRA_INDEX_DIR=<repo>/opra/data/index
  - OPRA_RECEIPTS_DIR=<repo>/opra/data/receipts
  - OPRA_REPORTS_DIR=<repo>/opra/data/reports

2) Start API (local)

- Run uvicorn against `opra/api/app.py` or use Docker compose per opra/README.md

3) Warm & index

- POST /v1/watcher/scan to prime caches.

4) Benchmarks

- Use scripts in scripts/ (bench_http_query.py, bench_report.py, check_determinism.py). See README usage.

5) Collate results

- Copy CSV/JSON outputs into proof/ and commit.
