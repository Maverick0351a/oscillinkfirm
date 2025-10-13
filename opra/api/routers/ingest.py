from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from oscillink.ingest.chunk import chunk_paragraphs
from oscillink.ingest.embed import load_embedding_model
from oscillink.ingest.extract import extract_text
from oscillink.ingest.index_simple import build_jsonl_index
from oscillink.ingest.ocr import ocr_if_needed
from oscillink.ingest.query_service import warmup_index
from oscillink.ingest.receipts import save_ingest_receipt

router = APIRouter()


class IngestRequest(BaseModel):
    path: str = Field(..., description="Path to a file under /data/docs or absolute path mounted inside container")
    embed_model: str = Field("bge-small-en-v1.5")
    index_out: Optional[str] = Field(None, description="Output index .jsonl path; defaults to /data/index/<stem>.jsonl")


class IngestResponse(BaseModel):
    ingest_receipt: Dict[str, Any]


@router.post("/ingest", response_model=IngestResponse)
def api_ingest(req: IngestRequest) -> Dict[str, Any]:
    src = Path(req.path)
    # Default output path in OPRA layout
    default_index_dir = Path(os.getenv("OPRA_INDEX_DIR", "/data/index"))
    out = (
        default_index_dir / (src.stem + ".jsonl")
        if req.index_out is None
        else Path(req.index_out)
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # Extract → OCR fallback → chunk → embed → index
    ex = extract_text([str(src)], parser="auto")
    ocr = ocr_if_needed(ex)
    pages = [p for r in ocr for p in r.pages]
    ch = chunk_paragraphs(pages)
    model = load_embedding_model(req.embed_model)
    vecs = model.embed([c.text for c in ch.chunks])
    idx = build_jsonl_index(ch.chunks, vecs, out_path=str(out))

    # Build ingest receipt
    rec: Dict[str, Any] = {
        "version": 1,
        "input_path": str(src),
        "chunks": len(ch.chunks),
        "index_path": idx.index_path,
        "index_sha256": idx.index_sha256,
        "embed_model": req.embed_model,
        "embed_dim": getattr(getattr(model, "spec", None), "dim", None) or getattr(model, "dim", None),
        "embed_weights_sha256": getattr(getattr(model, "spec", None), "sha256_weights", None),
        "deterministic": True,
    }
    # Stable signature for provenance
    try:
        import hashlib
        import json

        core = {
            "version": rec["version"],
            "input_path": rec["input_path"],
            "chunks": rec["chunks"],
            "index_path": rec["index_path"],
            "index_sha256": rec["index_sha256"],
            "embed_model": rec["embed_model"],
        }
        payload = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        rec["signature"] = hashlib.sha256(payload).hexdigest()
    except Exception:
        pass

    # Persist sidecar next to index
    try:
        save_ingest_receipt(idx.index_path, rec)
    except Exception:
        pass

    return {"ingest_receipt": rec}


class IndexRequest(BaseModel):
    policy: str = Field("incremental", pattern=r"^(incremental|full)$")
    docs_dir: Optional[str] = Field("/data/docs")
    embed_model: str = Field("bge-small-en-v1.5")


@router.post("/index")
def api_index(req: IndexRequest) -> Dict[str, Any]:
    """Minimal batch indexer: scans docs_dir and (re)builds JSONL indices per file.

    - incremental: build indices for files lacking /data/index/<stem>.jsonl
    - full: rebuild all
    """
    docs = Path(req.docs_dir or os.getenv("OPRA_DOCS_DIR", "/data/docs"))
    docs.mkdir(parents=True, exist_ok=True)
    built: list[Dict[str, Any]] = []
    for p in sorted(docs.glob("**/*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".pdf", ".docx"}:
            continue
        default_index_dir = Path(os.getenv("OPRA_INDEX_DIR", "/data/index"))
        out = default_index_dir / (p.stem + ".jsonl")
        if req.policy == "incremental" and out.exists():
            continue
        one = api_ingest(IngestRequest(path=str(p), embed_model=req.embed_model, index_out=str(out)))
        built.append(one["ingest_receipt"])  # type: ignore[index]
    return {"indexed": len(built), "receipts": built}


class WarmupRequest(BaseModel):
    backend: str = Field("jsonl", pattern=r"^(jsonl|faiss)$")
    embed_model: str = Field("bge-small-en-v1.5")
    index_path: Optional[str] = Field(None)
    meta_path: Optional[str] = Field(None)


@router.post("/warmup")
def api_warmup(req: WarmupRequest) -> Dict[str, Any]:
    return warmup_index(embed_model=req.embed_model, backend=req.backend, index_path=req.index_path, meta_path=req.meta_path)
