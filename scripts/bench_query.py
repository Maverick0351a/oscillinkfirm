from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, cast

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional path
    requests = None  # type: ignore

from oscillink.ingest.query_service import query_index


@dataclass
class RunStats:
    latencies_ms: List[float]

    def summary(self) -> Dict[str, float]:
        if not self.latencies_ms:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        ls = sorted(self.latencies_ms)
        n = len(ls)
        def pct(p: float) -> float:
            idx = min(n - 1, max(0, int(round((p/100.0)* (n - 1)))))
            return ls[idx]
        return {
            "count": float(n),
            "p50_ms": pct(50),
            "p95_ms": pct(95),
            "p99_ms": pct(99),
            "avg_ms": statistics.fmean(ls),
            "min_ms": ls[0],
            "max_ms": ls[-1],
        }


def call_http(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if requests is None:  # defensive for type-checkers
        raise RuntimeError("requests not available")
    r = cast(Any, requests).post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def time_http_query(url: str, payload: Dict[str, Any], *, warmup: int, runs: int, concurrency: int = 1) -> RunStats:
    if requests is None:
        raise SystemExit("requests is not installed. Install it or use --mode local")
    lat: List[float] = []
    # Warmup sequentially
    for _ in range(warmup):
        _ = call_http(url, payload)
    # Timed runs with basic concurrency (thread pool) to simulate parallel load
    if concurrency <= 1:
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = call_http(url, payload)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            lat.append(dt_ms)
    else:
        import concurrent.futures as cf
        batches = runs // concurrency
        rem = runs % concurrency
        def one():
            t0 = time.perf_counter()
            _ = call_http(url, payload)
            return (time.perf_counter() - t0) * 1000.0
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            for _ in range(batches):
                lat.extend(ex.map(lambda _: one(), range(concurrency)))
            if rem:
                lat.extend(ex.map(lambda _: one(), range(rem)))
    return RunStats(latencies_ms=lat)


def time_local_query(payload: Dict[str, Any], *, warmup: int, runs: int) -> RunStats:
    lat: List[float] = []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        _ = query_index(
            index_path=payload["index_path"],
            backend=payload.get("backend", "jsonl"),
            q=payload["q"],
            k=int(payload.get("k", 5)),
            embed_model=payload.get("embed_model", "bge-small-en-v1.5"),
            meta_path=payload.get("meta_path"),
            e2e=bool(payload.get("e2e", False)),
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            lat.append(dt_ms)
    return RunStats(latencies_ms=lat)


def run_http_mode(*, url: str, args, payload: Dict[str, Any]) -> tuple[RunStats, Dict[str, Any]]:
    if args.server_warmup:
        if requests is None:
            raise SystemExit("requests not installed for server warmup")
        _ = call_http(url.replace("/v1/query", "/v1/warmup"), {
            "backend": args.backend,
            "embed_model": args.embed_model,
            "index_path": args.index,
            "meta_path": args.meta,
        })
    # route to e2e endpoint when requested
    eff_url = url
    if args.e2e:
        eff_url = eff_url.replace("/v1/query", "/v1/query-e2e")
    if args.duration_sec and args.duration_sec > 0:
        if requests is None:
            raise SystemExit("requests not installed")
        import concurrent.futures as cf
        end = time.perf_counter() + float(args.duration_sec)
        # Warmup sequentially
        for _ in range(args.warmup):
            _ = call_http(eff_url, payload)
        lat: List[float] = []
        def one() -> float:
            t0 = time.perf_counter()
            _ = call_http(eff_url, payload)
            return (time.perf_counter() - t0) * 1000.0
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            while time.perf_counter() < end:
                futs = [ex.submit(one) for _ in range(args.concurrency)]
                for f in futs:
                    lat.append(f.result())
        stats = RunStats(latencies_ms=lat)
        mode_desc = {"mode": "http", "url": eff_url, "duration_sec": float(args.duration_sec), "concurrency": int(args.concurrency)}
    else:
        stats = time_http_query(eff_url, payload, warmup=args.warmup, runs=args.runs, concurrency=args.concurrency)
        mode_desc = {"mode": "http", "url": eff_url, "concurrency": int(args.concurrency)}
    return stats, mode_desc


def run_local_mode(*, args, payload: Dict[str, Any]) -> tuple[RunStats, Dict[str, Any]]:
    # In local mode, if the index path looks like an in-container path, try to map to deploy/data/demo_index.jsonl
    if payload["index_path"].startswith("/data/"):
        idx = Path("deploy") / "data" / Path(payload["index_path"]).name
        payload["index_path"] = str(idx)
    stats = time_local_query(payload, warmup=args.warmup, runs=args.runs)
    mode_desc = {"mode": "local"}
    return stats, mode_desc


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark query latency for HTTP or local mode")
    ap.add_argument("--mode", choices=["http", "local"], default="http")
    ap.add_argument("--url", default="http://localhost:8080/v1/query", help="HTTP endpoint for --mode http")
    ap.add_argument("--index", default="/data/demo_index.jsonl", help="Index path (absolute in-container path for http; host path for local)")
    ap.add_argument("--q", default="What is Oscillink?", help="Query text")
    ap.add_argument("--backend", choices=["jsonl", "faiss"], default="jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embed-model", default="bge-small-en-v1.5")
    ap.add_argument("--meta", default=None, help="FAISS meta .meta.jsonl path")
    ap.add_argument("--e2e", action="store_true", help="Use e2e recall/settle (jsonl only)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--server-warmup", action="store_true", help="POST /v1/warmup before running (http mode)")
    ap.add_argument("--duration-sec", type=float, default=0.0, help="If >0, run for this duration instead of a fixed number of runs")
    ap.add_argument("--out", default=None, help="Write JSON summary to this file")
    ap.add_argument("--out-csv", default=None, help="Write raw latencies to CSV (columns: ms)")
    args = ap.parse_args()

    payload: Dict[str, Any] = {
        "index_path": args.index,
        "q": args.q,
        "backend": args.backend,
        "k": args.k,
        "embed_model": args.embed_model,
        "meta_path": args.meta,
        "e2e": args.e2e,
    }

    if args.mode == "http":
        stats, mode_desc = run_http_mode(url=args.url, args=args, payload=payload)
    else:
        stats, mode_desc = run_local_mode(args=args, payload=payload)

    summary = stats.summary()
    # Compute RPS when duration mode used
    rps = None
    if args.duration_sec and args.duration_sec > 0 and stats.latencies_ms:
        dur = float(args.duration_sec)
        rps = float(len(stats.latencies_ms)) / dur if dur > 0 else None

    out: Dict[str, Any] = {
        "config": {
            "backend": args.backend,
            "k": args.k,
            "embed_model": args.embed_model,
            "e2e": args.e2e,
            **mode_desc,
        },
        "results": summary,
        **({"rps": rps} if rps is not None else {}),
    }
    js = json.dumps(out, indent=2)
    print(js)
    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
    if args.out_csv:
        # Write raw latencies
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["latency_ms"])
            for v in stats.latencies_ms:
                w.writerow([f"{v:.6f}"])


if __name__ == "__main__":
    main()
