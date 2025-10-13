from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

from .chunk import Chunk

faiss: Any
if find_spec("faiss") is not None:  # pragma: no cover - optional dependency
    try:
        import faiss as _faiss
        faiss = _faiss
    except Exception:
        faiss = None
else:
    faiss = None


def faiss_available() -> bool:
    return faiss is not None


@dataclass(frozen=True)
class BuiltFaissIndex:
    index_path: str
    meta_path: str
    index_sha256: str
    meta_sha256: str
    dim: int
    count: int


@dataclass(frozen=True)
class FaissMeta:
    source_path: str
    page_number: int
    start: int
    end: int


def _ensure_paths(out_path: str) -> tuple[Path, Path]:
    p = Path(out_path)
    if p.suffix.lower() != ".faiss":
        index_path = p.with_suffix(p.suffix + ".faiss") if p.suffix else p.with_suffix(".faiss")
    else:
        index_path = p
    meta_path = index_path.with_suffix(index_path.suffix + ".meta.jsonl")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    return index_path, meta_path


def build_faiss_flat(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, out_path: str) -> BuiltFaissIndex:
    if not faiss_available():
        raise RuntimeError("faiss is not available; install faiss-cpu to use this backend")
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors length mismatch")
    n = len(vectors)
    d = len(vectors[0]) if n else 0
    xb = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(d)  # deterministic flat inner-product index
    index.add(xb)

    index_path, meta_path = _ensure_paths(out_path)
    faiss.write_index(index, str(index_path))
    # write meta JSONL deterministically
    with meta_path.open("w", encoding="utf-8", newline="\n") as f:
        for ch in chunks:
            rec = {
                "source_path": ch.source_path,
                "page_number": ch.page_number,
                "start": ch.start,
                "end": ch.end,
            }
            f.write(__import__("json").dumps(rec, separators=(",", ":")) + "\n")

    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    return BuiltFaissIndex(
        index_path=str(index_path),
        meta_path=str(meta_path),
        index_sha256=index_sha,
        meta_sha256=meta_sha,
        dim=d,
        count=n,
    )


def build_faiss_ivf(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, out_path: str, nlist: int = 64, seed: int = 0) -> BuiltFaissIndex:
    """Build a deterministic FAISS IVF Flat index.

    Determinism notes:
    - We fix the training seed via faiss.randu if available and numpy seed.
    - Training order is the order of input vectors which we assume is already deterministic.
    - We use IndexIVFFlat with inner-product metric.
    """
    if not faiss_available():
        raise RuntimeError("faiss is not available; install faiss-cpu to use this backend")
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors length mismatch")
    n = len(vectors)
    d = len(vectors[0]) if n else 0
    xb = np.asarray(vectors, dtype="float32")
    # Seed numpy and faiss RNG when available
    np.random.seed(seed)
    if hasattr(faiss, "faiss_seed"):
        try:
            faiss.faiss_seed(seed)  # pragma: no cover - depends on faiss build
        except Exception:
            pass
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, max(1, nlist), faiss.METRIC_INNER_PRODUCT)
    index.train(xb)
    index.add(xb)
    index_path, meta_path = _ensure_paths(out_path)
    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8", newline="\n") as f:
        for ch in chunks:
            rec = {
                "source_path": ch.source_path,
                "page_number": ch.page_number,
                "start": ch.start,
                "end": ch.end,
            }
            f.write(__import__("json").dumps(rec, separators=(",", ":")) + "\n")
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    return BuiltFaissIndex(
        index_path=str(index_path),
        meta_path=str(meta_path),
        index_sha256=index_sha,
        meta_sha256=meta_sha,
        dim=d,
        count=n,
    )


def build_faiss_hnsw(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, out_path: str, M: int = 16, efConstruction: int = 200) -> BuiltFaissIndex:
    """Build a deterministic FAISS HNSW index.

    Determinism notes:
    - HNSW in FAISS is deterministic given fixed construction params and input order.
    """
    if not faiss_available():
        raise RuntimeError("faiss is not available; install faiss-cpu to use this backend")
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors length mismatch")
    n = len(vectors)
    d = len(vectors[0]) if n else 0
    xb = np.asarray(vectors, dtype="float32")
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efConstruction
    index.add(xb)
    index_path, meta_path = _ensure_paths(out_path)
    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8", newline="\n") as f:
        for ch in chunks:
            rec = {
                "source_path": ch.source_path,
                "page_number": ch.page_number,
                "start": ch.start,
                "end": ch.end,
            }
            f.write(__import__("json").dumps(rec, separators=(",", ":")) + "\n")
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    return BuiltFaissIndex(
        index_path=str(index_path),
        meta_path=str(meta_path),
        index_sha256=index_sha,
        meta_sha256=meta_sha,
        dim=d,
        count=n,
    )


def faiss_query_topk(index_path: str, meta_path: str, qvec: Sequence[float], *, k: int = 5) -> List[Tuple[float, FaissMeta]]:
    if not faiss_available():
        raise RuntimeError("faiss is not available; install faiss-cpu to use this backend")
    index = faiss.read_index(index_path)
    xq = np.asarray([qvec], dtype="float32")
    D, idxs = index.search(xq, k)
    # load metadata rows into list
    metas: List[FaissMeta] = []
    with Path(meta_path).open("r", encoding="utf-8") as f:
        import json

        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            metas.append(
                FaissMeta(
                    source_path=obj["source_path"],
                    page_number=int(obj["page_number"]),
                    start=int(obj["start"]),
                    end=int(obj["end"]),
                )
            )
    # Build (score, meta) pairs; faiss returns -1 for invalid when k>n
    pairs: List[Tuple[float, FaissMeta]] = []
    for score, idx in zip(D[0].tolist(), idxs[0].tolist()):
        if idx < 0:
            continue
        pairs.append((float(score), metas[idx]))
    # Deterministic tie-breaks
    pairs.sort(key=lambda t: (-t[0], t[1].source_path, t[1].page_number, t[1].start, t[1].end))
    return pairs[:k]
