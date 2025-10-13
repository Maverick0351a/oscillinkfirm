#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def bundle_hash_from_report_sidecar(path: str) -> str | None:
    p = Path(path)
    side = p.with_suffix(p.suffix + ".json")
    if not side.exists():
        return None
    data = json.loads(side.read_text(encoding="utf-8"))
    return data.get("bundle_hash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--assert-identical-receipts", action="store_true")
    ap.add_argument("--assert-identical-bundle-hash", action="store_true")
    ap.add_argument("--index", default=str(Path("opra/data/index").resolve() / "demo_index.jsonl"))
    args = ap.parse_args()

    chat_results = []
    report_paths = []
    with httpx.Client(timeout=20.0) as client:
        for _ in range(args.repeat):
            payload = {
                "index_path": args.index,
                "q": args.q,
                "backend": "jsonl",
                "k": 60,
                "embed_model": "bge-small-en-v1.5",
                "synth_mode": "extractive",
                "epsilon": 1e-3,
                "tau": 0.30,
            }
            r = client.post("http://127.0.0.1:8080/v1/chat", json=payload)
            r.raise_for_status()
            chat_results.append(r.json())

            rp = client.post("http://127.0.0.1:8080/v1/report", json={
                "title": "Determinism Check",
                "index_path": args.index,
                "q": args.q,
                "backend": "jsonl",
                "k": 60,
                "embed_model": "bge-small-en-v1.5",
                "fmt": "txt",
                "epsilon": 1e-3,
                "tau": 0.30,
            })
            rp.raise_for_status()
            report_paths.append(rp.json()["path"])  # path to txt

    # Receipts identical?
    identical_receipts = True
    if chat_results:
        first = json.dumps(chat_results[0].get("receipt"), sort_keys=True)
        identical_receipts = all(json.dumps(c.get("receipt"), sort_keys=True) == first for c in chat_results[1:])

    # Bundle hash identical?
    hashes = [bundle_hash_from_report_sidecar(p) for p in report_paths]
    identical_hashes = all(h == hashes[0] for h in hashes if h is not None) if hashes else True

    result = {
        "identical_receipts": identical_receipts,
        "identical_bundle_hash": identical_hashes,
        "receipts": [c.get("receipt") for c in chat_results],
        "report_paths": report_paths,
        "bundle_hashes": hashes,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.assert_identical_receipts and not identical_receipts:
        raise SystemExit(1)
    if args.assert_identical_bundle_hash and not identical_hashes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
