from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import List

# Deterministic synthetic index generator

def make_vector(text: str, dim: int = 64) -> List[float]:
    d = hashlib.sha256(text.encode("utf-8")).digest()
    vals: List[float] = []
    i = 0
    while len(vals) < dim:
        b = d[i % len(d)]
        vals.append((b + 1) / 257.0)
        i += 1
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic JSONL index deterministically")
    ap.add_argument("--out", default="deploy/data/synth_index.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            source = f"synthetic://doc/{i//10}#page={1 + (i % 5)}"
            text = f"Record {i} - Oscillink synthetic text block {random.randint(0, 9999)}"
            rec = {
                "source_path": source,
                "page_number": 1 + (i % 5),
                "start": i * 10,
                "end": i * 10 + len(text),
                "text": text,
                "vector": make_vector(text, args.dim),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {args.n} records to {args.out}")


if __name__ == "__main__":
    main()
