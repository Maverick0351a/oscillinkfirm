# On-Prem Ingest: Extraction, OCR, and Determinism

This guide covers optional integrations for on-premise ingestion with deterministic behavior.

## Extraction backends

The `osc ingest` command supports multiple extract parsers:

- `plain` (default for .txt): Reads UTF-8 text files deterministically.
- `tika`: Calls an Apache Tika server (`--tika-url`) to extract text.
- `pdfminer`: Uses pdfminer.six (if installed) to extract text from PDFs.
- `auto` (recommended): Tries `plain` → `tika` (if URL provided) → `pdfminer`, then falls back to OCR.

Examples:

```bash
osc ingest \
  --input /path/to/doc.pdf \
  --index-out /tmp/index.jsonl \
  --extract-parser auto \
  --tika-url http://localhost:9998
```

If Tika is unavailable or pdfminer.six is not installed, the pipeline falls back gracefully.

## OCR

When extraction yields zero pages, the CLI can attempt OCR via `ocrmypdf` if installed:

- `--ocr-backend ocrmypdf` (default)
- `--ocr-langs eng` for Tesseract language packs

The OCR step uses `ocrmypdf --sidecar` to produce deterministic text. If `ocrmypdf` is missing or fails, the pipeline continues without OCR.

## Determinism

Set `OSC_DETERMINISTIC=1` to:

- Pin thread counts (OMP/MKL/OPENBLAS/NUMEXPR) to 1 when not pre-set
- Fix seeds for Python `random`, NumPy, and PyTorch (if available)
- Enable deterministic Torch ops when supported

Receipts and sidecars record determinism environment keys for provenance.

## Receipts

Ingest receipts (written next to the index as `*.ingest.json`) include:

- `input_path` and `file_sha256`
- `index_path` and `index_sha256`
- Embedding metadata (model, dim, license, weights hash)
- Determinism settings

These are automatically chained into query/e2e outputs.

## Embedding model registry

You can provide model metadata in either `models_registry.json` at the repo root or
`~/.oscillink/models.json`. Supported fields per model:

- `path` (optional local path or name understood by sentence-transformers)
- `dim` (embedding dimension)
- `license` (string)
- `sha256_weights` (hash of model weights)
- `sha256_tokenizer` (hash of tokenizer files)

The CLI and API propagate `weights_sha256` and `tokenizer_sha256` into receipts/meta
when available for stronger provenance.

## Troubleshooting

- Ensure a Tika server is reachable (e.g., `docker run -p 9998:9998 apache/tika:2.9.0`)
- Install `pdfminer.six` for local PDF text extraction
- Install `ocrmypdf` and Tesseract language packs for OCR
- For deterministic runs, export `OSC_DETERMINISTIC=1` before invoking CLI

## Chunking options

Two chunkers are available:

- `unstructured` (default): Uses the Unstructured library to partition text and then maps
  elements back to deterministic byte offsets. If Unstructured is not installed or fails,
  the pipeline falls back to paragraph chunking.
- `paragraph`: A lightweight deterministic paragraph chunker based on blank-line separation.

Example forcing paragraph chunker:

```bash
osc ingest \
  --input /path/to/doc.txt \
  --index-out /tmp/index.jsonl \
  --chunker paragraph
```

## FAISS index variants

When using the FAISS backend (`--index-backend faiss`), you can choose among:

- `flat` (default): Deterministic `IndexFlatIP`.
- `ivf`: Deterministic `IndexIVFFlat` with seeded training; configure via `--faiss-nlist` and `--faiss-seed`.
- `hnsw`: Deterministic `IndexHNSWFlat` with fixed `M` and `efConstruction`.

Example IVF build with fixed seed:

```bash
osc ingest \
  --input /path/to/doc.txt \
  --index-out /tmp/index.faiss \
  --index-backend faiss \
  --faiss-variant ivf \
  --faiss-nlist 64 \
  --faiss-seed 0
```

Example HNSW build:

```bash
osc ingest \
  --input /path/to/doc.txt \
  --index-out /tmp/index.faiss \
  --index-backend faiss \
  --faiss-variant hnsw \
  --faiss-M 16 \
  --faiss-efConstruction 200
```

Determinism tips:

- Use `OSC_DETERMINISTIC=1` to pin threads and seeds process-wide.
- For IVF, keep the same `--faiss-seed`, `--faiss-nlist`, and input order to reproduce hashes.
- For HNSW, consistent params and input order produce stable graphs.
