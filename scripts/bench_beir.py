"""
BEIR-style retrieval benchmark: cosine baseline vs Oscillink post-processing.

- Embeddings via sentence-transformers
- Datasets via ir_datasets (BEIR subsets)
- Metrics via ranx (NDCG@10, Recall@50)

Outputs a JSON summary per dataset to stdout, or writes to --out if provided.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import ir_datasets as irds  # type: ignore
    from ranx import Qrels, Run, evaluate  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover - optional bench deps
    raise SystemExit(
        f"Bench extras are required. Install with: pip install -e .[bench]\nImport error: {e}"
    ) from e

from oscillink import Oscillink


@dataclass
class Config:
    dataset: str
    model_name: str
    top_k: int
    oscillink_kneighbors: int
    ndcg_at: int
    recall_at: int
    max_queries: int | None
    alpha: float
    mode: str  # 'bundle' | 'align'


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)
    return embs


def cosine_scores(q: np.ndarray, D: np.ndarray) -> np.ndarray:
    # q: (D,), D: (N, D) – assume unit-normalized
    return D @ q


def run_dataset(cfg: Config) -> dict[str, Any]:
    ds = irds.load(cfg.dataset)
    corpus = list(ds.docs_iter())
    queries = list(ds.queries_iter())
    qrels_dict = {q.query_id: {} for q in queries}
    for qrel in ds.qrels_iter():
        qrels_dict.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

    # Prepare text arrays
    doc_ids = [d.doc_id for d in corpus]
    doc_texts = [getattr(d, "text", getattr(d, "title", "")) or "" for d in corpus]
    query_ids = [q.query_id for q in queries]
    query_texts = [q.text for q in queries]

    if cfg.max_queries:
        query_ids = query_ids[: cfg.max_queries]
        query_texts = query_texts[: cfg.max_queries]

    # Embeddings
    model = SentenceTransformer(cfg.model_name)
    D = embed_texts(model, doc_texts)  # (N, dim)
    Q = embed_texts(model, query_texts)  # (M, dim)

    # Baseline and Oscillink runs
    baseline_run: dict[str, dict[str, float]] = {}
    oscillink_run: dict[str, dict[str, float]] = {}

    for qid, qvec in zip(query_ids, Q):
        # cosine baseline
        scores = cosine_scores(qvec, D)
        top_idx = np.argpartition(-scores, cfg.top_k)[: cfg.top_k]
        top_pairs = sorted(((doc_ids[i], float(scores[i])) for i in top_idx), key=lambda x: -x[1])
        baseline_run[qid] = {doc_id: score for doc_id, score in top_pairs}

        # Oscillink refinement on top-K
        Y = D[top_idx]
        lat = Oscillink(Y, kneighbors=cfg.oscillink_kneighbors)
        lat.set_query(qvec)
        lat.settle()
        if cfg.mode == "bundle":
            bundle = lat.bundle(k=cfg.top_k, alpha=cfg.alpha)
            # Map back to doc_ids using the bundle's local indices into top_idx
            refined = []
            for b in bundle:
                idx = int(b.get("id", -1))
                if 0 <= idx < len(top_idx):
                    refined.append((doc_ids[top_idx[idx]], float(b.get("score", 0.0))))
        else:
            # Alignment mode: rank by cosine alignment of U* with query (no diversification)
            Ustar = lat.solve_Ustar()
            u_norm = np.linalg.norm(Ustar, axis=1, keepdims=True) + 1e-12
            psi_n = qvec / (np.linalg.norm(qvec) + 1e-12)
            align_scores = (Ustar / u_norm) @ psi_n
            # Select top-k by alignment within the top_idx subset
            local_scores = [(i, float(align_scores[i])) for i in range(len(top_idx))]
            local_scores.sort(key=lambda x: -x[1])
            refined = [(doc_ids[top_idx[i]], s) for i, s in local_scores[: cfg.top_k]]
        oscillink_run[qid] = {doc_id: score for doc_id, score in refined}

    # Metrics via ranx
    qrels = Qrels(qrels_dict)
    run_base = Run(baseline_run)
    run_osc = Run(oscillink_run)

    metrics = [f"ndcg@{cfg.ndcg_at}", f"recall@{cfg.recall_at}"]
    base_scores = evaluate(qrels, run_base, metrics=metrics, make_comparable=True)
    osc_scores = evaluate(qrels, run_osc, metrics=metrics, make_comparable=True)

    return {
        "dataset": cfg.dataset,
        "model": cfg.model_name,
        "top_k": cfg.top_k,
        "kneighbors": cfg.oscillink_kneighbors,
        "metrics": {"cosine": base_scores, "oscillink": osc_scores},
        "counts": {"docs": len(doc_ids), "queries": len(query_ids)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="BEIR benchmark: cosine vs Oscillink")
    ap.add_argument("--dataset", default="beir/fiqa/dev")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--kneighbors", type=int, default=6)
    ap.add_argument("--ndcg-at", type=int, default=10)
    ap.add_argument("--recall-at", type=int, default=50)
    ap.add_argument("--max-queries", type=int)
    ap.add_argument("--alpha", type=float, default=0.5, help="Trade-off: 0=alignment, 1=coherence (bundle mode)")
    ap.add_argument("--mode", type=str, default="bundle", choices=["bundle", "align"], help="Oscillink ranking mode")
    ap.add_argument("--out", type=str)
    args = ap.parse_args()

    cfg = Config(
        dataset=args.dataset,
        model_name=args.model,
        top_k=args.top_k,
        oscillink_kneighbors=args.kneighbors,
        ndcg_at=args.ndcg_at,
        recall_at=args.recall_at,
        max_queries=args.max_queries,
        alpha=max(0.0, min(1.0, args.alpha)),
        mode=args.mode,
    )

    result = run_dataset(cfg)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
