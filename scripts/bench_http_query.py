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
    ap.add_argument("--qs", required=True, help="Path to JSONL with {q} per line")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    ap.add_argument("--index", default=str(Path("opra/data/index").resolve() / "demo_index.jsonl"))
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat")
    args = ap.parse_args()

    lat = []
    with httpx.Client(timeout=20.0) as client, open(args.qs, encoding="utf-8") as f:
        qs = [json.loads(line)["q"] for line in f if line.strip()]
        for i in range(args.runs):
            q = qs[i % len(qs)]
            payload = {
                "index_path": args.index,
                "q": q,
                "backend": "jsonl",
                "k": args.k,
                "embed_model": "bge-small-en-v1.5",
                "synth_mode": args.mode,
                "epsilon": 1e-3,
                "tau": 0.30,
            }
            t0 = time.perf_counter()
            r = client.post(args.url, json=payload)
            r.raise_for_status()
            _ = r.json()
            lat.append((time.perf_counter() - t0) * 1000.0)

    lat.sort()
    p50 = stats.median(lat)
    p95 = lat[int(0.95 * len(lat)) - 1]
    print(json.dumps({"p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "runs": len(lat)}, indent=2))


if __name__ == "__main__":
    main()
