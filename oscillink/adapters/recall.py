from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..core.lattice import OscillinkLattice
from ..ingest.query_simple import IndexRecord, load_jsonl_index


@dataclass(frozen=True)
class RecallParams:
    kneighbors: int = 6
    lamG: float = 1.0
    lamC: float = 0.5
    lamQ: float = 4.0
    lamP: float = 0.0
    tol: float = 1e-3
    bundle_k: int = 5


def _candidate_set_hash(records: Sequence[IndexRecord]) -> str:
    """Stable hash of candidate identity only (paths and spans)."""
    h = hashlib.sha256()
    # sort deterministically by the same key used elsewhere
    rows = sorted(records, key=lambda r: (r.source_path, r.page_number, r.start, r.end))
    for r in rows:
        h.update(r.source_path.encode("utf-8"))
        h.update(str(r.page_number).encode("utf-8"))
        h.update(str(r.start).encode("utf-8"))
        h.update(str(r.end).encode("utf-8"))
    return h.hexdigest()


def _records_to_matrix(records: Sequence[IndexRecord]) -> np.ndarray:
    if not records:
        return np.zeros((0, 0), dtype=np.float32)
    d = len(records[0].vector)
    Y = np.zeros((len(records), d), dtype=np.float32)
    for i, r in enumerate(records):
        Y[i] = np.asarray(r.vector, dtype=np.float32)
    return Y


def recall_and_settle_records(
    records: Sequence[IndexRecord],
    qvec: Sequence[float],
    *,
    params: RecallParams | None = None,
) -> tuple[List[dict], dict]:
    """Run deterministic recall→settle over provided records and return bundle + receipt.

    - records: full candidate set (each contains vector and metadata)
    - qvec: query embedding vector
    - params: lattice parameters and bundle size
    returns: (bundle_list, settle_receipt_dict)
    """
    cfg = params or RecallParams()
    Y = _records_to_matrix(records)
    lat = OscillinkLattice(
        Y,
        kneighbors=min(cfg.kneighbors, max(1, max(0, Y.shape[0] - 1))),
        lamG=cfg.lamG,
        lamC=cfg.lamC,
        lamQ=cfg.lamQ,
        deterministic_k=True,
    )
    lat.set_query(np.asarray(qvec, dtype=np.float32))
    # settle once to populate stats; then compute bundle
    lat.settle(tol=cfg.tol)
    bundle = lat.bundle(k=cfg.bundle_k)
    rec = lat.receipt()
    # Enrich with candidate set hash for chain-of-custody
    if isinstance(rec, dict) and "meta" in rec and isinstance(rec["meta"], dict):
        rec["meta"]["candidate_set_hash"] = _candidate_set_hash(records)
    return bundle, rec


def recall_and_settle_jsonl(
    index_path: str,
    qvec: Sequence[float],
    *,
    params: RecallParams | None = None,
) -> tuple[List[dict], dict]:
    records = load_jsonl_index(index_path)
    bundle, rec = recall_and_settle_records(records, qvec, params=params)
    # Add parent_ingest_sig to receipt meta for provenance
    try:
        parent_sig = records[0].index_sha256 if records else None
        if isinstance(rec, dict):
            rec_meta = rec.setdefault("meta", {}) if isinstance(rec.get("meta"), dict) else {}
            rec_meta["parent_ingest_sig"] = parent_sig
            rec["meta"] = rec_meta
    except Exception:
        pass
    return bundle, rec
