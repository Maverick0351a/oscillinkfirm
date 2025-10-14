# Oscillink Firm

Build coherence into retrieval and generation. Deterministic receipts for every decision. Latency that scales gracefully with corpus size.
<p align="center"><img alt="Oscillink" src="assets/oscillink_hero.png" width="640"/></p>

<p align="center"><b>A physics‑inspired, model‑free coherence layer that transforms candidate embeddings into an explainable working‑memory state via convex energy minimization. Deterministic receipts for audit. Conjugate‑gradient solve with SPD guarantees.</b></p>
<p align="center">
	<a href="docs/API.md">API</a> ·
	<a href="docs/foundations/MATH_OVERVIEW.md">Math</a> ·
	<a href="docs/reference/RECEIPTS.md">Receipts</a> ·
	<a href="benchmarks/">Benchmarks</a> ·
	<a href="notebooks/">Notebooks</a> ·
	<a href="https://github.com/Maverick0351a/oscillinkfirm">oscillinkfirm</a>
</p>

--- 

## Contents
- Overview
- Quickstart
- Oscillink Firm (Local LLM)
- Adapters & Compatibility
- Reproducibility
- Performance
- Method (Technical)
- Deployment Options
- Security, Privacy, Legal
- Troubleshooting
- Contributing & License
- Changelog
---
Install: `pip install oscillink` · Docs: [API](docs/API.md) · Math: [Overview](docs/foundations/MATH_OVERVIEW.md) · Receipts: [Schema](docs/RECEIPTS.md)

## Quickstart
System requirements: Python 3.10–3.12, NumPy ≥ 1.22 (1.x/2.x supported). CPU only.

## Oscillink Firm (Local LLM)

What it is

Oscillink Firm is a local retrieval + report system. It runs on a single machine or VM, indexes your firm’s documents (including scanned PDFs), and lets an LLM answer only from that private index. Nothing leaves your environment. Every answer ships a receipt (hashes, ΔH, CG residuals, abstain flags) so you can verify provenance.

What problem it solves

Most “RAG” tools phone home, hallucinate, and can’t prove where an answer came from. Oscillink Firm runs offline, declines to answer when the corpus can’t support it, and proves the source of every citation.

How we solve it (succinct)

- Ingest (local): OCR → chunk → embed → build vector index (JSONL/FAISS).
- Retrieve (local): ANN recall → Oscillink Coherence Engine settles candidates into a coherent working memory.
- Answer (local): Extractive synthesis by default; optional OpenAI‑compatible LLM running locally.
- Receipts (local): Each response/report includes ΔH, CG iters/residual, file/index/model SHA‑256, and abstain flags if coherence is low. We also add lightweight context hints (e.g., email thread/message IDs and HTML page titles) to aid provenance.

Data we index (no egress)

- Scanned documents & images → OCR (Tesseract via OCRmyPDF).
- PDFs (text) → pdfminer.six.
- Office docs → DOCX, PPTX (paragraphs, slides, speaker notes).
- Plain text & Markdown → .txt, .md (headings preserved).
- Email archives → MBOX/EML (PST via offline export → EML), plus attachments.
- Spreadsheets & CSV → XLSX/CSV (table, header-aware chunking with row‑level retrieval).
- HTML/Confluence exports → saved pages or static export (readability‑style cleanse).
- (Optional regulated formats) HL7 FHIR JSON / CDA excerpts, DICOM metadata (headers only; images omitted unless explicitly enabled).

Access & safety

- Bind to 127.0.0.1.
- No runtime downloads; models verified by SHA‑256.
- Role and matter filters on metadata to restrict retrieval scope.
- Abstain when coherence < ε or top score < τ—no guessing.

What to implement for maximum value (prioritized)

Tier 0 — already high ROI (lock these)

PDFs (text) + Scanned PDFs

Extractors: pdfminer.six; OCRmyPDF (Tesseract).

Deterministic chunking by headings/pages; include page numbers in metadata.

DOCX, TXT/MD, PPTX

Keep heading hierarchy; for PPTX include slide text + notes.

Chunk by section/slide; attach section_title, slide_index.

XLSX/CSV (table-aware)

Parse headers, normalize types, one row = one retrievable unit with a compact text rendering.

Include sheet, row_index, and key columns in metadata.

Why: These cover 80% of firm knowledge and make your demos obviously useful.

Tier 1 — unlock “hidden knowledge” (biggest next leap)

Email archives with attachments (offline)

Sources: MBOX/EML (and PST→EML via local export).

Extract: subject, from/to/date, body, attachments (then push attachments back through the same ingest).

Thread grouping: thread_id, message_index.

High value because institutional memory often lives in email.

HTML / Confluence static export

Strip boilerplate; preserve heading hierarchy and table content.

Tag with space, page_title, last_modified.

Why: Captures “tribal knowledge” that never made it into PDFs.

Tier 2 — regulated & specialized (offer as add‑ons)

Healthcare

FHIR JSON (e.g., DocumentReference, Observation, Condition) → index textual fields only; apply PHI tagging and policy filters.

CDA (clinical notes) → text body only; redact detected PHI tokens as configured.

DICOM → metadata headers only by default (PatientName/ID redacted or hashed).

Legal / eDiscovery

Load files (e.g., OPT/DAT) to reconstruct doc families; maintain family_id, bates ranges.

Connect to on‑prem DMS exports (iManage/NetDocuments via filesystem export, not live API at first).

Why: These win you specific industries without overhauling the core.

Implementation details that matter (so it “just works”)

Metadata schema (uniform across all connectors)

collection, doc_id, chunk_id, source_type, path, title, author,
created_at, modified_at, page_or_row, section, tags[],
model_name, model_sha256, dim, file_sha256

Add ACL tags: matter_id, client_id, department, role_required.

Index filters: year, company/client, matter, document type.

Deterministic ingest

Stable ordering and tie‑breaks.

Record exact parser/OCR versions and embed model hash in an IngestReceipt, chained into the SettleReceipt.

Email specifics

Normalize to UTF‑8, strip quoted reply blocks (keep one level optionally).

Store message‑level embeddings and also attachment embeddings as separate chunks linked back to the parent message. Query receipts may include `email.thread_id` and `email.message_id` under a compact context hint when present.

De‑dup via content hash to avoid storing identical attachments.

Spreadsheet specifics

Respect header row; compact each row into “key=value” text with a max token budget; attach top 3 salient columns as facets for filtering.

Add a table preview (first N rows) to the sidecar report for transparency.

HTML/Confluence specifics

Use readability‑style boilerplate removal; keep H1‑H3 hierarchy.

Preserve code blocks and tables as plain text plus a small JSON “shape” in metadata. Query receipts may include the HTML page title under a compact context hint.

Regulated data controls

PHI/PII detectors (local regex + small ML model if available) with three modes: tag, mask, block.

Receipt flag: contains_phi: true|false with a count (no raw values in receipts).

What not to do (now)

Live IMAP/Graph connectors that require continuous network access—start with offline exports.

Full DICOM pixel OCR—leave imaging to future modules; start with headers only.

Complex collaborative editing—keep the UI minimal; focus on solid retrieval + receipts.

“Most value” backlog (4‑week, aggressive)

Week 1:

Finalize PDFs (text+OCR), DOCX, TXT/MD, PPTX; uniform metadata; receipts chaining.

Table‑aware CSV/XLSX (row retrievability + facets).

Week 2:

Email (MBOX/EML + attachments).

HTML/Confluence static export.

Week 3:

ACL filters (matter/client/department) enforced at query time.

Proof pack polishing: determinism tests and report sidecars for each connector.

Week 4:

Healthcare/legal add‑ons behind flags (FHIR/CDA text fields; DICOM metadata; eDiscovery load‑file mapping).

Basic PHI/PII tag/mask/block policy.

Acceptance on each item =

deterministic ingest receipts,

reproducible recall→settle latency within target,

and abstain working when context is weak.

How to frame this in the README (crisp)

Local LLM on your firm’s data

Oscillink Firm indexes your scanned documents, PDFs, Office files, emails (with attachments), spreadsheets/CSV, and HTML/Confluence exports—all on your hardware. The LLM answers only from this private index.
Every answer includes a receipt with hashes and coherence metrics. If the corpus doesn’t support an answer, the system abstains.

---

See also: [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
## Adapters & Compatibility

---
## Deployment Options

### A. SDK (local)
### B. Licensed container (customer‑managed)

### C. Cloud API (beta)
Cloud feature flags, quotas, and Stripe onboarding are documented under `docs/`:

- Cloud architecture & ops: `docs/cloud/CLOUD_ARCH_GCP.md`, `docs/ops/REDIS_BACKEND.md`
- Billing: `docs/billing/STRIPE_INTEGRATION.md`, `docs/billing/PRICING.md`

Production readiness: see `docs/ops/PRODUCTION_CHECKLIST.md` for the deployment gate we use before shipping new versions or enabling external access.
For Kubernetes, see `docs/ops/K8S_DEPLOYMENT.md`.
## Security, Privacy, Legal

Policies: [Security](SECURITY.md) · [Privacy](docs/product/PRIVACY.md) · [Terms](docs/product/TERMS.md) · [License](LICENSE) · [Patent notice](PATENTS.md)
## Troubleshooting

---
## Contributing & License

---
## Changelog

---
## Appendix: Datasets and Notebooks

---
## Pricing (licensed container)

---


## Minimal on‑prem HTTP query service

For air‑gapped or on‑prem setups that only need retrieval over a prebuilt index, we include a tiny FastAPI app that wraps the programmatic query API:

- Server module: `examples/query_server.py`
- Endpoints:
  - POST /v1/query — top‑k vector search against a JSONL or FAISS index
  - POST /v1/query-e2e — JSONL only; runs recall→settle and returns a bundled context and receipt

Quickstart (install extras first):

1) Install cloud extras

```bash
pip install -e .[cloud]
```

2) Start the server

```bash
uvicorn examples.query_server:app --host 0.0.0.0 --port 8080
```

3) Query it

```bash
curl -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{"index_path":"/path/to/index.jsonl","q":"what is the topic?","backend":"jsonl","k":6}'
```

For FAISS, pass `"backend":"faiss"` and `"meta_path":"/path/to/index.meta.jsonl"`.

Filtering by metadata (non‑e2e JSONL only):

Include an optional `filters` object for equality matches on chunk metadata (e.g., `matter_id`, `client_id`, `department`). Example:

```
{
	"index_path": "/path/to/index.jsonl",
	"q": "billing policy",
	"backend": "jsonl",
	"k": 6,
	"filters": {"matter_id": "M-123", "department": "litigation"}
}
```

Docker (local build):

Use the included Dockerfile and compose profile to run the query service as non‑root with a read‑only filesystem.

1) Build and run

```
docker compose -f deploy/docker-compose.yml up --build oscillink_local
```

2) Health check

```
curl -fsS http://127.0.0.1:8080/health
```

Licensed mode (no egress):

The query server container now supports the same license gate as the ingest container. Mount your license token and an internal JWKS file, and the entrypoint will validate before serving:

1) Prepare secrets

```
# Place files under deploy/license/
#   - oscillink.lic    (your Ed25519-signed license JWT)
#   - jwks.json        (your mirrored JWKS)
```

2) Run with compose (uses file:// JWKS and fails readiness if unlicensed)

```
docker compose -f deploy/docker-compose.yml up --build oscillink_local
```

3) Check license status

```
curl -fsS http://127.0.0.1:8080/license/status
```

Set OSCILLINK_LICENSE_REQUIRED=1 to make /license/status return 503 when entitlements are missing. For fully air‑gapped environments, keep OSCILLINK_JWKS_URL pointing to a file:// path you mount in the container.

Metrics and observability:

- Enable header protection for /metrics by setting `OSCILLINK_METRICS_PROTECTED=1` and defining `OSCILLINK_ADMIN_SECRET`. Send the secret via the `X-Admin-Secret` header.
- On Kubernetes, enable the Helm `serviceMonitor.enabled=true` to let Prometheus Operator scrape `/metrics`.

### OCR quality signals and UI badge

When an index was built from low-quality scans, the server surfaces two fields on query responses (both e2e and non‑e2e):

- `ocr_low_confidence`: boolean, present at the top level and also attached to each result item in non‑e2e responses when applicable
- `ocr_avg_confidence`: number in [0,1] when available (best-effort; may be null for backends that don’t emit confidences)

Deterministic filters: When you pass a `filters` object (equality on chunk metadata), the pipeline deterministically prefilters the candidate set before settling. This only changes the `meta.candidate_set_hash` and the selection itself—core receipt fields (model dims/hashes, `epsilon`, `tau`, convergence stats format, and schema keys) remain stable. Early-abstain occurs if all candidates are filtered out, with reason `"no candidates after filter"`.

Behavioral guardrails (env → firm.yaml → built-ins):

- JSONL path applies a configurable score penalty (default 0.08) to chunks with `ocr_low_confidence=true` before top‑k re‑ranking
- FAISS path applies the same configurable uniform penalty to all result scores when the index is flagged low‑OCR, then re‑sorts for determinism
- E2E path abstains with reason `"low-quality OCR"` when coherence thresholds fail and the index is flagged low‑OCR

Prometheus metrics (optional):

- `osc_query_abstain_total{reason="low_ocr"|"insufficient",endpoint="query|query-e2e"}`
- `osc_ocr_low_conf_total{endpoint="query|query-e2e"}`
- `osc_ocr_avg_conf_gauge{endpoint="query|query-e2e"}`

Simple UI badge example (client-side):

```python
def enrich_citations(items):
	out = []
	for it in items:
		badge = "\u26A0 OCR" if it.get("ocr_low_confidence") else None
		out.append({
			**it,
			"badges": ([badge] if badge else []),
		})
	return out

# Usage (non-e2e):
# resp = requests.post("/v1/query", json=payload).json()
# citations = enrich_citations(resp.get("results", []))

# Usage (e2e):
# resp = requests.post("/v1/query-e2e", json=payload).json()
# show a single top-level badge if resp.get("ocr_low_confidence") is True
```

Example response (compact):

```json
{
	"answer": "Clause 9 limits indemnity to direct damages...",
	"abstain": false,
	"receipt": { "deltaH_total": 0.014, "epsilon": 0.001, "tau": 0.30 },
	"ocr_low_confidence": true,
	"ocr_avg_confidence": 0.58,
	"citations": [
		{ "title": "MSA_2019.pdf", "page": 7, "ocr_low_confidence": true, "ocr_avg_confidence": 0.58 },
		{ "title": "SOW_Addendum.pdf", "page": 2, "ocr_low_confidence": false }
	]
}
```

When all top‑K candidates are flagged and ΔH < ε, the system abstains with reason "low-quality OCR" and suggests rescanning in the report appendix.

Configuration

- Drop a sample `firm.yaml` in the repo root (or set `OSCILLINK_CONFIG` to a path). Load precedence: environment variables → `firm.yaml` → built‑ins.
- See `docs/ops/ALERTS.md` for Prometheus alerts that watch low‑quality OCR signals.

## Environment and config quick reference

You can tune behavior with environment variables or a minimal `firm.yaml` (env overrides YAML). Core knobs:

- OSCILLINK_CONFIG: Path to config YAML (default: `./firm.yaml`).
- OSCILLINK_OCR_SCORE_PENALTY: Penalty applied to scores for low‑OCR items (default: `0.08`).
- OSCILLINK_OCR_ABSTAIN_ON_ALL_LOW: When all top‑K are low‑OCR and coherence fails, abstain (default: `true`).
- OSCILLINK_METRICS_PROTECTED: Require admin secret on `/metrics` when set to 1/true/on.
- OSCILLINK_ADMIN_SECRET: Secret value expected in the `X-Admin-Secret` header.
- OSCILLINK_LICENSE_REQUIRED: Make `/license/status` return 503 when entitlements are missing (1/true/on).
- OSCILLINK_JWKS_URL: JWKS URL or `file://` path for the license public keys (air‑gapped: use `file:///.../jwks.json`).
- OSCILLINK_ENTITLEMENTS_PATH: Path where the container writes validated entitlements JSON (default: `/run/oscillink_entitlements.json`).
- OSCILLINK_LICENSE_PATH: Path to the signed license token (default used by Docker/Helm examples: `/run/secrets/oscillink.lic`).
- OSC_DETERMINISTIC: Set to 1 to apply stricter deterministic settings at import time.

YAML layout (selected keys under `quality.ocr`):

```yaml
quality:
	ocr:
		score_penalty: 0.08
		abstain_on_all_low: true
```

### Windows PowerShell examples

Set environment variables for the current session and run the local server:

```powershell
$env:OSCILLINK_CONFIG = "C:\\data\\firm.yaml"
$env:OSCILLINK_OCR_SCORE_PENALTY = "0.12"
$env:OSCILLINK_OCR_ABSTAIN_ON_ALL_LOW = "true"
python -m pip install -e .[cloud]
uvicorn examples.query_server:app --host 127.0.0.1 --port 8080
```

Protected metrics endpoint from PowerShell (assuming a secret):

```powershell
$env:OSCILLINK_METRICS_PROTECTED = "1"
$env:OSCILLINK_ADMIN_SECRET = "supersecret"
Invoke-RestMethod -Uri "http://127.0.0.1:8080/metrics" -Headers @{"X-Admin-Secret" = $env:OSCILLINK_ADMIN_SECRET}
```

License status check from PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/license/status" | ConvertTo-Json -Depth 5
```

<!-- Pruned legacy duplication below -->
