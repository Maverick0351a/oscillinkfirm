from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from oscillink.ingest.cli import build_parser as build_ingest_parser
from oscillink.ingest.cli import cmd_ingest


def _run_ingest(args: list[str]) -> tuple[int, float]:
    p = build_ingest_parser()
    ns = p.parse_args(["ingest", *args])
    t0 = time.time()
    rc = cmd_ingest(ns)
    return rc, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark ingest pipeline")
    ap.add_argument("--input", required=True, help="Path to input inside container (e.g., /data/sample.txt)")
    ap.add_argument("--out", required=True, help="Index output path (e.g., /data/index.jsonl)")
    ap.add_argument("--backend", choices=["jsonl", "faiss"], default="jsonl")
    ap.add_argument("--faiss-variant", choices=["flat", "ivf", "hnsw"], default="flat")
    ap.add_argument("--nlist", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--extract-parser", choices=["auto", "plain", "tika", "pdfminer"], default="auto")
    ap.add_argument("--tika-url", default=os.environ.get("OSC_TIKA_URL"))
    ap.add_argument("--chunker", choices=["paragraph", "unstructured"], default="unstructured")
    ap.add_argument("--embed-model", default="bge-small-en-v1.5")
    args = ap.parse_args()

    ingest_args: list[str] = [
        "--input", args.input,
        "--index-out", args.out,
        "--extract-parser", args.extract_parser,
        "--chunker", args.chunker,
        "--embed-model", args.embed_model,
        "--index-backend", args.backend,
    ]
    if args.tika_url:
        ingest_args += ["--tika-url", args.tika_url]
    if args.backend == "faiss":
        ingest_args += ["--faiss-variant", args.faiss_variant]
        if args.faiss_variant == "ivf":
            ingest_args += ["--faiss-nlist", str(args.nlist), "--faiss-seed", str(args.seed)]
        elif args.faiss_variant == "hnsw":
            ingest_args += ["--faiss-M", str(args.M), "--faiss-efConstruction", str(args.efc)]

    rc, elapsed = _run_ingest(ingest_args)
    ok = (rc == 0) and Path(args.out).exists()
    print(json.dumps({
        "ok": ok,
        "rc": rc,
        "backend": args.backend,
        "faiss_variant": args.faiss_variant if args.backend == "faiss" else None,
        "elapsed_seconds": round(elapsed, 4),
        "input": args.input,
        "index_out": args.out,
        "extract_parser": args.extract_parser,
        "chunker": args.chunker,
        "embed_model": args.embed_model,
        "tika_url": args.tika_url,
    }))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
