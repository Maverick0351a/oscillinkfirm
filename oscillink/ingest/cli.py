from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

from ..adapters.recall import RecallParams, recall_and_settle_jsonl
from .chunk import chunk_paragraphs, chunk_unstructured
from .determinism import _DETERMINISM_STATE  # noqa: F401
from .embed import load_embedding_model
from .extract import extract_text
from .index_faiss import (
    build_faiss_flat,
    build_faiss_hnsw,
    build_faiss_ivf,
    faiss_available,
    faiss_query_topk,
)
from .index_simple import build_jsonl_index
from .ocr import ocr_if_needed
from .query_simple import load_jsonl_index, query_topk
from .receipts import IngestReceipt, load_ingest_receipt, save_ingest_receipt


def _embedding_meta(model, fallback_name: str) -> dict[str, Any]:
    name = getattr(getattr(model, "spec", None), "name", None) or fallback_name
    dim = getattr(getattr(model, "spec", None), "dim", None) or getattr(model, "dim", 384)
    meta: dict[str, Any] = {"model": name, "dim": dim, "framework": "sentence-transformers"}
    spec = getattr(model, "spec", None)
    if spec is not None:
        w = getattr(spec, "sha256_weights", None)
        if w:
            meta["weights_sha256"] = w
        t = getattr(spec, "sha256_tokenizer", None)
        if t:
            meta["tokenizer_sha256"] = t
        lic = getattr(spec, "license", None)
        if lic:
            meta["license"] = lic
    return meta


def _attach_embedding_meta(receipt: dict[str, Any], model, fallback_name: str) -> None:
    try:
        if isinstance(receipt, dict):
            meta = receipt.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["embedding"] = _embedding_meta(model, fallback_name)
            receipt["meta"] = meta
    except Exception:
        pass


def _sha256_file(path: str) -> str | None:
    try:
        import hashlib
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def cmd_ingest(args: argparse.Namespace) -> int:
    # Minimal deterministic flow: extract -> ocr (no-op) -> chunk
    inputs = [args.input]
    ex_results = extract_text(inputs, parser=args.extract_parser, tika_url=args.tika_url)
    ocr_results = ocr_if_needed(ex_results, backend=args.ocr_backend, langs=args.ocr_langs)
    # Flatten pages from OCR results deterministically
    pages = []
    for r in ocr_results:
        pages.extend(r.pages)
    # Select chunker with deterministic fallback
    if args.chunker == "unstructured":
        ch = chunk_unstructured(pages, ruleset=args.ruleset)
    else:
        ch = chunk_paragraphs(pages, ruleset=args.ruleset)
    # Embed chunks
    model = load_embedding_model(args.embed_model)
    vectors = model.embed([c.text for c in ch.chunks])
    # Build index based on backend
    backend = args.index_backend
    if backend == "faiss":
        if not faiss_available():
            raise SystemExit("FAISS backend requested but faiss is not installed. Install faiss-cpu or use --index-backend jsonl.")
        # Choose FAISS variant
        variant = getattr(args, "faiss_variant", "flat")
        if variant == "flat":
            built_f = build_faiss_flat(ch.chunks, vectors, out_path=args.index_out)
        elif variant == "ivf":
            built_f = build_faiss_ivf(ch.chunks, vectors, out_path=args.index_out, nlist=getattr(args, "faiss_nlist", 64), seed=getattr(args, "faiss_seed", 0))
        elif variant == "hnsw":
            built_f = build_faiss_hnsw(ch.chunks, vectors, out_path=args.index_out, M=getattr(args, "faiss_M", 16), efConstruction=getattr(args, "faiss_efConstruction", 200))
        else:
            raise SystemExit(f"Unknown FAISS variant: {variant}")
        index_path = built_f.index_path
        index_sha = built_f.index_sha256
    else:
        built_j = build_jsonl_index(ch.chunks, vectors, out_path=args.index_out)
        index_path = built_j.index_path
        index_sha = built_j.index_sha256

    # Build receipt
    det_env = _DETERMINISM_STATE.get("env", {}) if isinstance(_DETERMINISM_STATE, dict) else {}
    rec = IngestReceipt(
        version=1,
        input_path=args.input,
        file_sha256=_sha256_file(args.input),
        chunks=len(ch.chunks),
        index_path=index_path,
        index_sha256=index_sha,
        embed_model=args.embed_model,
        embed_dim=getattr(getattr(model, "spec", None), "dim", None) or getattr(model, "dim", None),
        embed_license=getattr(getattr(model, "spec", None), "license", None),
        embed_weights_sha256=getattr(getattr(model, "spec", None), "sha256_weights", None),
        embed_tokenizer_sha256=getattr(getattr(model, "spec", None), "sha256_tokenizer", None),
        deterministic=bool(args.deterministic),
        determinism_env=det_env,
    )
    # Persist sidecar next to index for provenance chaining
    try:
        save_ingest_receipt(index_path, rec)
    except Exception:
        pass
    print(rec.to_json())
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    # Deterministic simple query over JSONL/FAISS index using same embedding stub
    model = load_embedding_model(args.embed_model)
    qvec = model.embed([args.q])[0]
    backend = args.index_backend
    # Optional end-to-end recall->settle path over JSONL index
    if args.e2e and backend != "jsonl":
        raise SystemExit("--e2e is only supported with jsonl backend currently")
    if args.e2e:
        params = RecallParams(
            kneighbors=args.kneighbors,
            lamC=args.lamC,
            lamQ=args.lamQ,
            lamP=args.lamP,
            tol=args.tol,
            bundle_k=args.bundle_k,
        )
        bundle, receipt = recall_and_settle_jsonl(args.index, qvec, params=params)
        _attach_embedding_meta(receipt, model, args.embed_model)
        # If requested, print just the embedding metadata and exit
        if getattr(args, "print_embedding_meta", False):
            emb = receipt.get("meta", {}).get("embedding", {}) if isinstance(receipt, dict) else {}
            print(json.dumps(emb))
            return 0
        ingest_sidecar = load_ingest_receipt(args.index)
        e2e_out: dict[str, Any] = {"bundle": bundle, "settle_receipt": receipt}
        if ingest_sidecar is not None:
            e2e_out["ingest_receipt"] = ingest_sidecar
        print(json.dumps(e2e_out))
        return 0
    if backend == "faiss":
        if not faiss_available():
            raise SystemExit("FAISS backend requested but faiss is not installed. Install faiss-cpu or use --index-backend jsonl.")
        if not args.meta:
            raise SystemExit("--meta path to .meta.jsonl is required for faiss backend")
        topk_pairs = faiss_query_topk(args.index, args.meta, qvec, k=args.k)
        res = [
            {
                "score": float(s),
                "source_path": m.source_path,
                "page_number": m.page_number,
                "start": m.start,
                "end": m.end,
            }
            for s, m in topk_pairs
        ]
    else:
        records = load_jsonl_index(args.index)
        topk = query_topk(records, qvec, k=args.k)
        res = [
            {
                "score": float(s),
                "source_path": r.source_path,
                "page_number": r.page_number,
                "start": r.start,
                "end": r.end,
            }
            for s, r in topk
        ]
    # Attach ingest receipt when available
    ingest_sidecar = load_ingest_receipt(args.index)
    out: dict[str, Any] = {"results": res}
    if ingest_sidecar is not None:
        out["ingest_receipt"] = ingest_sidecar
    # For non-e2e mode, allow printing just embedding metadata too
    if getattr(args, "print_embedding_meta", False):
        emb_block = _embedding_meta(model, args.embed_model)
        print(json.dumps(emb_block))
        return 0
    print(json.dumps(out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="osc", description="Oscillink Ingest CLI")
    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("ingest", help="Ingest files into a local index")
    pi.add_argument("--input", required=True)
    pi.add_argument("--extract-parser", choices=["auto", "plain", "tika", "pdfminer"], default="auto")
    pi.add_argument("--tika-url", default=None)
    pi.add_argument("--ocr-backend", default="ocrmypdf")
    pi.add_argument("--ocr-langs", default="eng")
    pi.add_argument("--chunker", choices=["paragraph", "unstructured"], default="unstructured")
    pi.add_argument("--ruleset", default="default")
    pi.add_argument("--embed-model", default="bge-small-en-v1.5")
    pi.add_argument("--index-out", required=True)
    pi.add_argument("--index-backend", choices=["jsonl", "faiss"], default="jsonl")
    pi.add_argument("--faiss-variant", choices=["flat", "ivf", "hnsw"], default="flat")
    pi.add_argument("--faiss-nlist", type=int, default=64)
    pi.add_argument("--faiss-seed", type=int, default=0)
    pi.add_argument("--faiss-M", type=int, default=16)
    pi.add_argument("--faiss-efConstruction", type=int, default=200)
    pi.add_argument("--deterministic", action="store_true")
    pi.set_defaults(func=cmd_ingest)

    pq = sub.add_parser("query", help="Query a local index end-to-end")
    pq.add_argument("--index", required=True)
    pq.add_argument("--q", required=True, help="Query text")
    pq.add_argument("--k", type=int, default=5)
    pq.add_argument("--embed-model", default="bge-small-en-v1.5")
    pq.add_argument("--index-backend", choices=["jsonl", "faiss"], default="jsonl")
    pq.add_argument("--meta", default=None, help="Path to .meta.jsonl (required for faiss backend)")
    pq.add_argument("--kneighbors", type=int, default=6)
    pq.add_argument("--lamC", type=float, default=0.5)
    pq.add_argument("--lamQ", type=float, default=4.0)
    pq.add_argument("--lamP", type=float, default=0.0)
    pq.add_argument("--tol", type=float, default=1e-3)
    pq.add_argument("--bundle-k", type=int, default=5)
    pq.add_argument("--e2e", action="store_true", help="Run end-to-end recall→settle and return bundle + receipt (jsonl only)")
    pq.add_argument(
        "--print-embedding-meta",
        action="store_true",
        help="Print only the embedding metadata (model, dim, weights hash, license) and exit",
    )
    pq.add_argument("--deterministic", action="store_true")
    pq.add_argument("--out", default=None)
    pq.set_defaults(func=cmd_query)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = build_parser()
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    try:
        return int(args.func(args))
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
