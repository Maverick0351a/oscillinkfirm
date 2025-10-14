from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Header

from ..ingest import (
    RecallParams,
    build_jsonl_index,
    chunk_paragraphs,
    determinism_env,
    extract_text,
    is_deterministic,
    load_embedding_model,
    recall_and_settle_jsonl,
)
from ..ingest.ocr import ocr_if_needed
from ..ingest.receipts import load_ingest_receipt, save_ingest_receipt
from .license import require_license

router = APIRouter(prefix="/v1", tags=["ingest"])


@router.post("/ingest")
def api_ingest(
    *,
    input_path: str = Body(..., embed=True),
    embed_model: str = Body("bge-small-en-v1.5", embed=True),
    index_out: str = Body(..., embed=True),
    x_license_key: Optional[str] = Header(default=None, alias="X-License-Key"),
) -> Dict[str, Any]:
    require_license(x_license_key)
    ex = extract_text([input_path])
    ocr = ocr_if_needed(ex)
    # Aggregate OCR quality
    ocr_avg_conf: float | None = None
    ocr_low_conf: bool | None = None
    try:
        total_chars = 0
        page_count = 0
        for r in ocr:
            # backend-provided confidence preferred
            if getattr(r, "stats", None) and r.stats.avg_confidence is not None:
                if ocr_avg_conf is None:
                    ocr_avg_conf = float(r.stats.avg_confidence)
                else:
                    ocr_avg_conf = (ocr_avg_conf + float(r.stats.avg_confidence)) / 2.0
            # accumulate for heuristic
            for p in r.pages:
                total_chars += len(getattr(p, "text", ""))
                page_count += 1
        # If backend didn’t report, derive simple heuristic
        if ocr_avg_conf is None:
            # conservative static defaults; can be made configurable later
            low_by_text = (total_chars < 32) or ((total_chars / max(1, page_count)) < 16.0)
            ocr_low_conf = low_by_text
        else:
            # Threshold default 0.70; tune in operations
            ocr_low_conf = bool(ocr_avg_conf < 0.70)
    except Exception:
        ocr_avg_conf = None
        ocr_low_conf = None
    pages = [p for r in ocr for p in r.pages]
    ch = chunk_paragraphs(pages)
    model = load_embedding_model(embed_model)
    vecs = model.embed([c.text for c in ch.chunks])
    idx = build_jsonl_index(ch.chunks, vecs, out_path=index_out)
    det_env = determinism_env()
    det = is_deterministic()
    ingest_receipt = {
        "version": 1,
        "input_path": input_path,
        "chunks": len(ch.chunks),
        "index_path": idx.index_path,
        "index_sha256": idx.index_sha256,
        "embed_model": embed_model,
        "embed_dim": getattr(getattr(model, "spec", None), "dim", None) or getattr(model, "dim", None),
        "embed_license": getattr(getattr(model, "spec", None), "license", None),
        "embed_weights_sha256": getattr(getattr(model, "spec", None), "sha256_weights", None),
        "deterministic": bool(det),
        "determinism_env": det_env,
        "extract_parser": "auto",
        "ocr_backend": "ocrmypdf",
        "ocr_langs": "eng",
        "ocr_avg_confidence": ocr_avg_conf,
        "ocr_low_confidence": ocr_low_conf,
    }
    # Attach a stable signature for provenance (JSON of core fields)
    try:
        import hashlib
        import json

        core = {
            k: ingest_receipt[k]
            for k in (
                "version",
                "input_path",
                "chunks",
                "index_path",
                "index_sha256",
                "embed_model",
                "deterministic",
            )
        }
        payload = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ingest_receipt["signature"] = hashlib.sha256(payload).hexdigest()
    except Exception:
        pass
    # Persist sidecar alongside index for retrieval in /v1/query
    try:
        save_ingest_receipt(idx.index_path, ingest_receipt)
    except Exception:
        pass
    # Emit a lightweight JSON log for OCR quality
    try:
        log = {
            "event": "ocr_quality",
            "file": input_path,
            "avg_conf": ocr_avg_conf,
            "low_conf": ocr_low_conf,
        }
        sys.stderr.write(str(log) + "\n")
    except Exception:
        pass
    return {"ingest_receipt": ingest_receipt, "ocr_low_confidence": ingest_receipt.get("ocr_low_confidence"), "ocr_avg_confidence": ingest_receipt.get("ocr_avg_confidence")}


@router.post("/query")
def api_query(
    *,
    index_path: str = Body(..., embed=True),
    q: str = Body(..., embed=True),
    kneighbors: int = Body(6, embed=True),
    lamC: float = Body(0.5, embed=True),
    lamQ: float = Body(4.0, embed=True),
    lamP: float = Body(0.0, embed=True),
    tol: float = Body(1e-3, embed=True),
    bundle_k: int = Body(5, embed=True),
    embed_model: str = Body("bge-small-en-v1.5", embed=True),
    x_license_key: Optional[str] = Header(default=None, alias="X-License-Key"),
) -> Dict[str, Any]:
    require_license(x_license_key)
    model = load_embedding_model(embed_model)
    qvec = model.embed([q])[0]
    params = RecallParams(
        kneighbors=kneighbors,
        lamC=lamC,
        lamQ=lamQ,
        lamP=lamP,
        tol=tol,
        bundle_k=bundle_k,
    )
    bundle, receipt = recall_and_settle_jsonl(index_path, qvec, params=params)
    # Enrich receipt with embedding meta when present
    try:
        if isinstance(receipt, dict):
            m = receipt.setdefault("meta", {}) if isinstance(receipt.get("meta"), dict) else {}
            emb_block = {
                "model": model.spec.name,
                "dim": model.spec.dim or 384,
                "framework": "sentence-transformers",
            }
            if model.spec.sha256_weights:
                emb_block["weights_sha256"] = model.spec.sha256_weights
            m["embedding"] = emb_block
            receipt["meta"] = m
    except Exception:
        pass
    # Propagate parent_ingest_sig at top-level for convenience if present
    parent_ingest_sig = None
    try:
        parent_ingest_sig = receipt.get("meta", {}).get("parent_ingest_sig") if isinstance(receipt, dict) else None
    except Exception:
        parent_ingest_sig = None
    # Attach ingest_receipt if available
    ingest_sidecar = load_ingest_receipt(index_path)
    out = {"bundle": bundle, "settle_receipt": receipt, "parent_ingest_sig": parent_ingest_sig}
    if ingest_sidecar is not None:
        out["ingest_receipt"] = ingest_sidecar
        # Surface OCR quality flags at top-level for UI
        out["ocr_low_confidence"] = ingest_sidecar.get("ocr_low_confidence")
        out["ocr_avg_confidence"] = ingest_sidecar.get("ocr_avg_confidence")
    return out
