This folder is mounted into the on-prem query container at /data (read-only).

Place your built indices here and reference them from requests via absolute in-container paths, e.g.:

- JSONL index: /data/demo_index.jsonl
- FAISS index: /data/index.faiss and meta: /data/index.meta.jsonl

Note: Paths are relative to this deploy/ directory because docker-compose.query.onprem.yml lives here.