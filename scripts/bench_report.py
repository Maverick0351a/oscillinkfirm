#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--fmt", choices=["docx", "txt"], default="docx")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--index", default=str(Path("opra/data/index").resolve() / "demo_index.jsonl"))
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/report")
    args = ap.parse_args()

    lat = []
    with httpx.Client(timeout=30.0) as client:
        for _ in range(args.runs):
            payload = {
                "title": "Benchmark Report",
                "index_path": args.index,
                "q": args.q,
                "backend": "jsonl",
                "k": 60,
                "embed_model": "bge-small-en-v1.5",
                "fmt": args.fmt,
                "epsilon": 1e-3,
                "tau": 0.30,
            }
            t0 = time.perf_counter()
            r = client.post(args.url, json=payload)
            r.raise_for_status()
            _ = r.json()
            lat.append((time.perf_counter() - t0) * 1000.0)

    p50 = stats.median(lat)
    p95 = sorted(lat)[int(0.95 * len(lat)) - 1]
    print(json.dumps({"p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "runs": len(lat)}, indent=2))


if __name__ == "__main__":
    main()
