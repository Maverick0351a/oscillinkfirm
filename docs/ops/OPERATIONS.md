# Operations runbook (self-hosted/licensed)

This runbook summarizes the high-signal, low-noise operational knobs for Oscillink in production.

## Readiness and health
- Liveness: GET /health (fast path)
- Readiness: GET /license/status (reflects JWT validity; set `OSCILLINK_LICENSE_REQUIRED=1` to fail on expired/missing)

## Access and security
- Admin endpoints: require `OSCILLINK_ADMIN_SECRET`; pass in header `X-Admin-Secret`
- Metrics: set `OSCILLINK_METRICS_PROTECTED=1` to require admin header for GET /metrics
- NetworkPolicy: restrict ingress to service port and egress to JWKS/usage endpoints; allow DNS

## Logging and tracing
- JSON access logs: `OSCILLINK_JSON_LOGS=1`; sampling via `OSCILLINK_LOG_SAMPLE` in [0.0,1.0]
- Request IDs: included by default; propagate in logs and responses
- Avoid logging request bodies to prevent accidental PII retention

## Scaling and performance
- HPA: start 2–6 replicas; target 60–70% CPU; refine with real traffic
- Cache: `OSCILLINK_CACHE_ENABLE=1`, set `OSCILLINK_CACHE_TTL` and `OSCILLINK_CACHE_CAP` per-key
- Rate-limits: tune global/per-IP/per-endpoint limits via env; verify headers `X-RateLimit-*`/`X-IPLimit-*`

## Quotas and caps
- Per-key quota: `OSCILLINK_KEY_NODE_UNITS_LIMIT` and `OSCILLINK_KEY_NODE_UNITS_WINDOW`
- Monthly cap (licensed): `OSCILLINK_MONTHLY_CAP`; headers `X-Monthly-*` on responses
- Feature flags: `OSCILLINK_FEAT_*` export overlays (e.g., `OSCILLINK_FEAT_DIFFUSION_GATES=1`)

## Redis
- Enable with `OSCILLINK_STATE_BACKEND=redis` and `OSCILLINK_REDIS_URL`
- CLI sessions: `OSCILLINK_CLI_SESSIONS_BACKEND=redis`; TTL via `OSCILLINK_CLI_TTL`

## TLS
- Prefer cert-manager for automatic certs; sample manifests under `deploy/helm/oscillink/examples/cert-manager/`
- Set Ingress class and annotations for your controller (nginx/ALB). Use cluster-specific values files.

## Proxies and private clusters
- See `values-private.yaml` and `values-proxy.yaml` for strict egress and proxy envs
- If using a mesh/egress gateway, disable public ingress and expose internally only

## Supply chain
- Base image pinned to python:3.11.9-slim; consider digest pinning in Helm values
- Image signing with cosign: see `docs/ops/IMAGE_SIGNING.md`
- CI includes pip-audit, SBOM (Syft), and Trivy; make blocking in your fork as desired

## Troubleshooting quick checks
- 403: verify X-API-Key and key store mode (env/Firestore)
- 429: inspect `X-Quota-*`, `X-RateLimit-*`, `X-IPLimit-*`, `X-Monthly-*`
- JWKS failures: check `OSCILLINK_JWKS_URL`, `OSCILLINK_JWKS_TTL`, offline grace; confirm egress
- Admin introspection: GET /admin/introspect with `X-Admin-Secret`

## OCR quality governance

Signals and flags

- Ingest receipts may include `ocr_avg_confidence` in [0,1] and a boolean `ocr_low_confidence` flag.
- Query responses surface these at the top level and per-result (non‑e2e). See README’s “OCR quality signals and UI badge”.

Retrieval behavior

- JSONL: apply small fixed score penalty (0.08) per low‑OCR chunk before re‑ranking to top‑k.
- FAISS: when the index is flagged low‑OCR, apply a uniform 0.08 penalty to all result scores and re-sort.
- E2E: if coherence thresholds fail and low‑OCR is flagged, abstain with reason `low-quality OCR`.

Operator thresholds (tunable)

- The fixed penalty is intentionally small and deterministic. If you need to tighten/loosen, adjust in a fork and document the delta in your ops notes.
- Keep ε (deltaH threshold) and τ (top-score threshold) at defaults initially; raise only when false positives are observed and metrics confirm.

Remediation checklist for low OCR

1) Re-scan documents at ≥300 DPI, grayscale or B/W with proper contrast.
2) Prefer native PDFs over images of text when possible.
3) Configure Tesseract language packs for the document’s dominant language(s).
4) For forms/columns, try `--psm 6` (assumes a uniform block of text) at OCR ingest time.
5) Remove heavy compression or re-export with lossless settings before ingest.
6) Validate results on a small sample; re‑ingest affected documents.

Metrics

- `osc_query_abstain_total{reason="low_ocr"|"insufficient",endpoint="query|query-e2e"}`
- `osc_ocr_low_conf_total{endpoint="query|query-e2e"}`
