"""
Quick recall harness for FAISS variants.

Usage:
  python scripts/recall_check.py --index hnsw --N 5000 --D 384 --k 50
"""
from __future__ import annotations

import argparse

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover - optional dep
    raise SystemExit("faiss is required for recall_check.py; install faiss-cpu") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--D", type=int, default=384)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--index", default="hnsw", choices=["flat", "hnsw", "ivfpq"])
    args = ap.parse_args()

    np.random.seed(0)
    X = np.random.randn(args.N, args.D).astype("float32")
    # Normalize for inner product to mimic cosine
    faiss.normalize_L2(X)
    q = np.random.randn(args.D).astype("float32")
    q /= np.linalg.norm(q) + 1e-9

    # brute ground truth via dot product
    sims = X @ q
    gt = np.argsort(-sims)[: args.k]

    # build faiss index
    if args.index == "flat":
        ix = faiss.IndexFlatIP(args.D)
    elif args.index == "hnsw":
        ix = faiss.IndexHNSWFlat(args.D, 32)
        ix.hnsw.efSearch = 64
    else:
        nlist = 4096
        pqm = 64
        quant = faiss.IndexFlatIP(args.D)
        ivf = faiss.IndexIVFPQ(quant, args.D, nlist, pqm, 8)
        ivf.train(X[: min(50000, len(X))])
        ix = ivf
        ix.nprobe = 16

    ix.add(X)
    D, idxs = ix.search(q[None, :], args.k)
    recall = len(set(gt).intersection(set(idxs[0]))) / float(args.k)
    print(f"index={args.index} recall@{args.k}={recall:.3f}")


if __name__ == "__main__":
    main()
