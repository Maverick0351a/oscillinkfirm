#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    ap.add_argument("--out", default=None, help="Write JSON summary to this file")
    ap.add_argument("--csv", default=None, help="Write CSV summary with one row of aggregate metrics")
    args = ap.parse_args()

    lat = []
    last_json = None
    def _extract_q(line: str) -> str | None:
        s = line.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                # Prefer 'q', but fall back to 'text' for generic samples
                return obj.get("q") or obj.get("text")
        except Exception:
            # Not JSON; treat entire line as the query text
            return s
        return None

    with httpx.Client(timeout=20.0) as client, open(args.qs, encoding="utf-8") as f:
        qs = [q for q in (_extract_q(line) for line in f) if q]
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
            last_json = r.json()
            lat.append((time.perf_counter() - t0) * 1000.0)

    lat.sort()
    p50 = stats.median(lat)
    p95 = lat[int(0.95 * len(lat)) - 1]
    summary = {"endpoint": args.url, "mode": args.mode, "k": args.k, "runs": len(lat), "p50_ms": round(p50, 2), "p95_ms": round(p95, 2)}
    print(json.dumps({"p50_ms": summary["p50_ms"], "p95_ms": summary["p95_ms"], "runs": summary["runs"]}, indent=2))
    # Optional outputs
    if args.out:
        # Add simple env metadata plus optional digests from last response's receipt
        import platform
        import subprocess
        import sys
        commit = None
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            pass
        env = {"os": platform.platform(), "python": sys.version.split(" ")[0], "commit": commit}
        digests = {}
        try:
            if isinstance(last_json, dict):
                rec = last_json.get("receipt") or {}
                if isinstance(rec, dict):
                    digests = {
                        "index_sha256": rec.get("index_sha256"),
                        "model_sha256": rec.get("query_model_sha256") or rec.get("index_model_sha256"),
                    }
        except Exception:
            pass
        payload = {"summary": summary, "env": env, "digests": digests, "time": time.time()}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.csv:
        # Write one-row CSV with headers
        headers = ["endpoint", "mode", "k", "runs", "p50_ms", "p95_ms", "time"]
        row = [summary["endpoint"], summary["mode"], summary["k"], summary["runs"], summary["p50_ms"], summary["p95_ms"], f"{time.time():.0f}"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerow(row)


if __name__ == "__main__":
    main()
