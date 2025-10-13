# Receipts Schema (JSON)

Top‑level fields:
- `version` (str): package version.
- `deltaH_total` (float): exact energy gap (non‑negative).
- `term_energies` (object): energy terms evaluated at \(U^*\)
	- `anchor` (float): \(\lambda_G\,\|U^*-Y\|_F^2\)
	- `coherence` (float): \(\lambda_C\,\mathrm{tr}((U^*)^\top L_{\mathrm{sym}} U^*)\)
	- `query` (float): \(\lambda_Q\,\mathrm{tr}((U^*-\mathbf{1}\psi^\top)^\top B (U^*-\mathbf{1}\psi^\top))\)
	- `path` (float): optional path prior energy (0 if unset)
- `coh_drop_sum`, `anchor_pen_sum`, `query_term_sum` (floats): component aggregates.
- `cg_iters` (int): last CG iterations used (from `settle()` if called, else from `solve_Ustar`).
- `residual` (float): legacy key for CG residual.
- `final_residual` (float): same as `residual` (explicit name).
- `t_ms` (float): wall time for the last `settle()` call (0 if unset).
- `null_points` (array): edges with high z‑score residuals (may be empty/capped).
- `null_points_top` (array): first 16 elements of `null_points` for convenience.
- `meta` (object): metadata block (see below).

## Receipt detail modes

You can control which diagnostics are computed to trade detail for latency:

- full (default): includes per‑node diagnostics (coh_drop, anchor_pen, query_term) and the full list of `null_points`.
- light: computes only the core energy delta (`deltaH_total`) and metadata; skips per‑node diagnostics and `null_points` discovery. This reduces receipt time substantially on large N.

How to set (SDK):

```python
lat = Oscillink(Y, kneighbors=6)
lat.set_receipt_detail("light")
rec = lat.receipt()
```

The receipt `meta` block includes `receipt_detail` and a `null_points_summary` indicating whether a cap was applied.

## Meta block

The `meta` object provides observability and reproducibility context:

- `ustar_cached` (bool): whether `U*` was served from cache.
- `ustar_solves`, `ustar_cache_hits` (ints): counters.
- `ustar_converged` (bool), `ustar_res` (float), `ustar_iters` (int), `ustar_solve_ms` (float): last stationary solve stats.
- `graph_build_ms` (float): time to build the mutual‑kNN graph.
- `last_settle_ms` (float): last `settle()` time in ms.
- `avg_degree` (float), `edge_density` (float): adjacency statistics.
- `gates_min`, `gates_max`, `gates_mean`, `gates_uniform`: gating summary over `B_diag`.
- `state_sig` (str): deterministic state signature hash.
- `edge_hash` (str): canonical adjacency hash (lex‑sorted COO with rounded weights).
- `receipt_detail` ("full"|"light"): detail level used.
- `null_points_summary` (object): `{ total_null_points, returned_null_points, null_cap_applied }`.
- `component_sizes` (array[int]): sizes of connected components in `A`.
- `latency_ms` (object): `{ build, solve, receipt }` consolidated timings.
- `determinism_env` (object|null): snapshot of relevant BLAS/threads env vars when available.
- `params` (object): `{ k, lambdaG, lambdaC, lambdaQ, lambdaP, tol }`.
- `signature` (object, optional): HMAC block when signing is enabled
	- `algorithm` ("HMAC-SHA256"), `payload` (object), `signature` (hex string)
	- Minimal payload: `{ sig_v, mode: "minimal", state_sig, deltaH_total }`
	- Extended payload adds convergence stats, params, and graph settings.

## Ingest receipt (deterministic index provenance)

Emitted by the ingest route/CLI when building a deterministic JSONL index:

- `version` (int)
- `input_path` (str)
- `chunks` (int)
- `index_path` (str)
- `index_sha256` (str, 64 hex chars)
- `embed_model` (str)
- `deterministic` (bool): whether OSC_DETERMINISTIC was enabled and core knobs pinned
- `determinism_env` (object): snapshot of relevant thread/env knobs (OMP_NUM_THREADS, MKL_NUM_THREADS, etc.)
- `signature` (str): SHA‑256 over canonical JSON of core fields (`version,input_path,chunks,index_path,index_sha256,embed_model,deterministic`)

## Settle receipt provenance fields

Settle receipts include additional provenance to chain back to the ingest index and candidate identity:

- `meta.parent_ingest_sig` (str|null): SHA‑256 of the parent JSONL index, matching the ingest receipt’s `index_sha256`
- `meta.candidate_set_hash` (str): stable candidate identity hash over `(source_path,page,start,end)`

## Chain Receipt
- `verdict` (bool): pass iff all chain edges' z‑scores <= threshold and coherence gain >= 0.
- `weakest_link`: `{k, edge: [i,j], zscore}`.
- `coherence_gain` (float): chain coherence improvement vs anchors.
- `edges`: per‑edge `{k, edge, z_struct, z_path, r_struct, r_path}`.

## Validation via JSON Schema

We publish JSON Schemas for both ingest and settle receipts so you can validate them in CI or offline audits:

- Ingest receipt schema: `oscillink/assets/schemas/ingest_receipt.schema.json`
- Settle receipt schema: `oscillink/assets/schemas/settle_receipt.schema.json`

You can validate a receipt file using the helper script. Validation requires the optional `jsonschema` package:

```powershell
# Install validator (optional; only needed when you want schema checks)
python -m pip install jsonschema

# Validate an ingest receipt
python scripts/print_receipt.py path\to\ingest_receipt.json --schema oscillink/assets/schemas/ingest_receipt.schema.json

# Validate a settle receipt
python scripts/print_receipt.py path\to\settle_receipt.json --schema oscillink/assets/schemas/settle_receipt.schema.json
```

If `jsonschema` is not installed, the script still pretty‑prints the receipt and skips validation.

### Where receipts appear

- CLI: `oscillink ingest` emits an ingest receipt sidecar next to the JSONL index; `oscillink query` includes an `ingest_receipt` field in the response when found.
- API: `/v1/ingest` persists the ingest receipt; `/v1/query` returns `bundle`, `settle_receipt`, `meta.parent_ingest_sig`, and `ingest_receipt` (when available).

Use the provenance fields to chain the outputs:

- Ensure `ingest_receipt.index_sha256 == settle_receipt.meta.parent_ingest_sig`.
- Ensure `settle_receipt.meta.candidate_set_hash` is stable across repeated queries with the same candidate set.
