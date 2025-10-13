from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from oscillink.ingest.chunk import chunk_paragraphs
from oscillink.ingest.embed import EmbeddingModel, load_embedding_model
from oscillink.ingest.extract import extract_text
from oscillink.ingest.index_simple import build_jsonl_index
from oscillink.ingest.ocr import ocr_if_needed


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark ingest pipeline: extract->ocr->chunk->embed->index")
    ap.add_argument("--input", default="_tmp_demo/sample.txt", help="Input document (txt or pdf)")
    ap.add_argument("--parser", choices=["auto", "plain", "tika", "pdfminer"], default="auto")
    ap.add_argument("--langs", default="eng", help="OCR languages (ocrmypdf)")
    ap.add_argument("--embed-model", default="bge-small-en-v1.5")
    ap.add_argument("--stub", action="store_true", help="Force deterministic stub embeddings (skip heavy model load)")
    ap.add_argument("--out", default="_bench/index.demo.jsonl", help="Output index path (JSONL)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1, help="Repeat measurement runs and compute stats")
    ap.add_argument("--out-csv", default=None, help="Write per-run stage timings to CSV")
    args = ap.parse_args()

    inp = args.input
    outp = args.out
    Path(Path(outp).parent).mkdir(parents=True, exist_ok=True)

    # Warmup extract path only (cheap, avoids loading heavy models)
    for _ in range(args.warmup):
        _ = extract_text([inp], parser=args.parser)

    def run_once() -> Dict[str, float]:
        t0 = time.perf_counter()
        ext = extract_text([inp], parser=args.parser)
        t1 = time.perf_counter()
        ocr = ocr_if_needed(ext, backend="ocrmypdf", langs=args.langs)
        t2 = time.perf_counter()
        pages: List[Any] = []
        for r in ocr:
            pages.extend(r.pages)
        ch = chunk_paragraphs(pages, ruleset="paragraph")
        t3 = time.perf_counter()
        model: EmbeddingModel
        if args.stub:
            # Force stub by creating an EmbeddingModel with dim only (no ST load)
            model = EmbeddingModel(spec=load_embedding_model(args.embed_model).spec, dim=64)
            # Monkey-patch to ensure stub path: set private loader to None by calling embed directly (no ST use)
        else:
            model = load_embedding_model(args.embed_model)
        vecs = model.embed([c.text for c in ch.chunks])
        t4 = time.perf_counter()
        _ = build_jsonl_index(ch.chunks, vecs, out_path=outp)
        t5 = time.perf_counter()
        return {
            "extract": (t1 - t0) * 1000.0,
            "ocr_if_needed": (t2 - t1) * 1000.0,
            "chunk": (t3 - t2) * 1000.0,
            "embed": (t4 - t3) * 1000.0,
            "index": (t5 - t4) * 1000.0,
            "total": (t5 - t0) * 1000.0,
            "pages": float(len(pages)),
            "chunks": float(len(ch.chunks)),
        }

    runs: List[Dict[str, float]] = []
    for _ in range(args.runs):
        runs.append(run_once())

    # Aggregate stats
    def stat_of(key: str) -> Dict[str, float]:
        vals = [r[key] for r in runs]
        ls = sorted(vals)
        n = len(ls)
        def pct(p: float) -> float:
            idx = min(n - 1, max(0, int(round((p/100.0)* (n - 1)))))
            return ls[idx]
        return {
            "count": float(n),
            "p50_ms": pct(50),
            "p95_ms": pct(95),
            "avg_ms": statistics.fmean(ls),
            "min_ms": ls[0],
            "max_ms": ls[-1],
        }

    stats = {k: stat_of(k) for k in ["extract", "ocr_if_needed", "chunk", "embed", "index", "total"]}
    pages = int(runs[-1]["pages"]) if runs else 0
    chunks = int(runs[-1]["chunks"]) if runs else 0

    res: Dict[str, Any] = {
        "input": inp,
        "out": outp,
        "stub": bool(args.stub),
        "counts": {"pages": pages, "chunks": chunks},
        "stage_stats": stats,
    }
    js = json.dumps(res, indent=2)
    print(js)
    Path(outp + ".timing.json").write_text(js, encoding="utf-8")
    if args.out_csv:
        with open(Path(outp).with_suffix(Path(outp).suffix + ".runs.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["extract_ms","ocr_ms","chunk_ms","embed_ms","index_ms","total_ms"]) 
            for r in runs:
                w.writerow([f"{r['extract']:.6f}", f"{r['ocr_if_needed']:.6f}", f"{r['chunk']:.6f}", f"{r['embed']:.6f}", f"{r['index']:.6f}", f"{r['total']:.6f}"])


if __name__ == "__main__":
    main()
