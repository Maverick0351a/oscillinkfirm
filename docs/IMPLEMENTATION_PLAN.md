# Oscillink Firm — Implementation Plan

Headline: Local LLM that answers only from your firm’s private, vectorized index. No egress. Receipts for every answer. First‑class OCR so scanned content isn’t invisible.

## Scope and priorities

- Tier 0 (Weeks 1):
  - Ingestors: PDFs (pdfminer.six), Scanned PDFs (OCRmyPDF/Tesseract), DOCX, TXT/MD, PPTX
  - Table‑aware CSV/XLSX (row retrievability + facets)
  - Uniform metadata schema and ingest receipts chained into settle receipts
  - Deterministic chunking (headings/pages/sections/slides)
  - UI: Minimal chat + report; receipts modal; proof sidecars; abstain wiring
  - Acceptance: deterministic ingest receipts; reproducible recall→settle latency; abstain when weak context

- Tier 1 (Week 2):
  - Email MBOX/EML + attachments (attachments routed to the same ingest)
  - HTML/Confluence static export with readability‑style cleanup

- Tier 2 (Weeks 3–4):
  - Healthcare add‑ons: FHIR JSON text fields, CDA narrative text; DICOM metadata only (PHI redaction/hashing)
  - Legal/eDiscovery: Load files (OPT/DAT) for doc families; DMS exports via filesystem
  - ACL filters (matter/client/department/role) enforced at query time
  - PHI/PII tag/mask/block policy

## Metadata schema

Fields (present where applicable):

- collection, doc_id, chunk_id, source_type, path, title, author
- created_at, modified_at, page_or_row, section, tags[]
- model_name, model_sha256, dim, file_sha256
- ACL: matter_id, client_id, department, role_required

Index filters: year, company/client, matter, document type.

## Deterministic ingest

- Stable ordering and tie‑breaks (e.g., path, page, section, row)
- Record parser/OCR versions, embed model name+SHA256 in IngestReceipt
- Chain IngestReceipt -> SettleReceipt and include index/model/file hashes

## Email specifics

- Normalize to UTF‑8, strip quoted reply blocks (configurable)
- Store message embeddings and attachment embeddings; link by parent_id
- De‑duplicate by content hash for attachments

## Spreadsheet specifics

- Header aware; compact row text as key=value pairs within token budget
- Add salient columns as facets; include sheet, row_index in metadata

## HTML/Confluence specifics

- Boilerplate removal (readability‑style), keep H1–H3 hierarchy
- Preserve code blocks and tables as text plus small JSON shape in metadata

## Regulated data controls

- PHI/PII detectors (local regex or on‑device model if available)
- Modes: tag, mask, block
- Receipt flag: contains_phi with count (no raw values in receipts)

## Engineering plan (4‑week, aggressive)

- Week 1
  - Finalize Tier 0 connectors; implement uniform metadata
  - Deterministic chunkers and receipts chaining
  - CSV/XLSX table‑aware ingestion
  - Proof sidecars in reports

- Week 2
  - Email (MBOX/EML + attachments) connector
  - HTML/Confluence static export connector

- Week 3
  - ACL filters at query time; enforce via metadata predicates
  - Determinism tests across connectors; receipts validation

- Week 4
  - Healthcare/legal add‑ons behind feature flags
  - PHI/PII tag/mask/block policy; receipts flags for PHI counts

## Success criteria

- All Tier 0/1 connectors produce deterministic ingest receipts and metadata
- End‑to‑end recall→settle latency within documented targets
- Abstain behavior triggers when coherence < ε or top score < τ
- Receipts shipped with every answer/report with hashes and CG metrics
