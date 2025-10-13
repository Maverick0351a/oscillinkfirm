from __future__ import annotations
# ruff: noqa: I001

from typing import Any, Optional

from importlib.util import find_spec

from ..adapters.recall import RecallParams, recall_and_settle_jsonl
from .embed import load_embedding_model
from .index_faiss import faiss_available, faiss_query_topk
from .index_simple import build_jsonl_index  # noqa: F401 (public API parity)
from .query_simple import load_jsonl_index, query_topk
from .receipts import load_ingest_receipt
from .models_registry import get_model_spec


_MODEL_CACHE: dict[str, Any] = {}
_INDEX_CACHE: dict[str, Any] = {}

# Optional Prometheus metrics
_PROM: Any
if find_spec("prometheus_client") is not None:  # pragma: no cover - optional dependency
    try:
        from prometheus_client import Counter, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

        _PROM = {
            "registry": CollectorRegistry(auto_describe=True),
        }
        _PROM["abstain_counter"] = Counter(
            "osc_query_abstain_total",
            "Total number of abstained queries",
            labelnames=("reason", "endpoint"),
            registry=_PROM["registry"],
        )
    except Exception:
        _PROM = None
else:
    _PROM = None


def _get_model(name: str):
    m = _MODEL_CACHE.get(name)
    if m is None:
        m = load_embedding_model(name)
        _MODEL_CACHE[name] = m
    return m


def _get_jsonl_records(index_path: str):
    recs = _INDEX_CACHE.get(index_path)
    if recs is None:
        recs = load_jsonl_index(index_path)
        _INDEX_CACHE[index_path] = recs
    return recs


def _get_index_model_sha(index_receipt: Optional[dict]) -> Optional[str]:
    if not isinstance(index_receipt, dict):
        return None
    return (
        index_receipt.get("embed_weights_sha256")
        or index_receipt.get("embed_model_sha256")
        or index_receipt.get("embed_sha256")
    )


def _guard_dims_and_hash(query_dim: int, idx_dim: int, query_sha: Optional[str], index_sha: Optional[str]) -> None:
    if query_dim != idx_dim:
        raise ValueError(f"embedding dim mismatch: query={query_dim} index={idx_dim}")
    if query_sha and index_sha and query_sha != index_sha:
        raise ValueError("embedding model sha256 mismatch between query and index")


def _safe_float(val: object) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)  # type: ignore[arg-type]
    except Exception:
        return None


def _build_query_receipt(ingest_sidecar: Optional[dict], query_model_sha: Optional[str], index_model_sha: Optional[str], dim: int, epsilon: float, tau: float) -> dict[str, Any]:
    return {
        "query_model_sha256": query_model_sha,
        "index_model_sha256": index_model_sha,
        "index_sha256": ingest_sidecar.get("index_sha256") if isinstance(ingest_sidecar, dict) else None,
        "dim": int(dim),
        "epsilon": float(epsilon),
        "tau": float(tau),
    }


def _run_e2e_jsonl(index_path: str, qvec: list[float], *, epsilon: float, tau: float, query_model_sha: Optional[str], index_model_sha: Optional[str], index_receipt: Optional[dict]) -> dict[str, Any]:
    params = RecallParams(kneighbors=6, lamC=0.5, lamQ=4.0, lamP=0.0, tol=1e-3, bundle_k=5)
    bundle, settle_receipt = recall_and_settle_jsonl(index_path, qvec, params=params)
    records = _get_jsonl_records(index_path)
    idx_dim = len(records[0].vector) if records else len(qvec)
    _guard_dims_and_hash(len(qvec), idx_dim, query_model_sha, index_model_sha)
    dH = _safe_float(settle_receipt.get("deltaH_total")) if isinstance(settle_receipt, dict) else None
    top_score = _safe_float(settle_receipt.get("coherence")) if isinstance(settle_receipt, dict) else None
    ingest_sidecar = index_receipt
    query_receipt = _build_query_receipt(ingest_sidecar, query_model_sha, index_model_sha, idx_dim, epsilon, tau)
    if (dH is not None and dH < float(epsilon)) or (top_score is not None and top_score < float(tau)):
        if _PROM is not None:
            try:
                _PROM["abstain_counter"].labels(reason="insufficient", endpoint="query-e2e").inc()
            except Exception:
                pass
        return {"abstain": True, "reason": "insufficient coherence", "receipt": query_receipt, "settle_receipt": settle_receipt, **({"ingest_receipt": ingest_sidecar} if ingest_sidecar is not None else {})}
    out_e2e: dict[str, Any] = {"bundle": bundle, "settle_receipt": settle_receipt, "receipt": query_receipt}
    if ingest_sidecar is not None:
        out_e2e["ingest_receipt"] = ingest_sidecar
    return out_e2e


def _non_e2e_faiss(index_path: str, meta_path: str, qvec: list[float], *, k: int, query_model_sha: Optional[str], index_model_sha: Optional[str]) -> list[dict[str, Any]]:
    if not faiss_available():
        raise RuntimeError("FAISS backend requested but faiss is not installed. Install faiss-cpu or use backend=jsonl.")
    # Avoid importing faiss directly here to keep it an optional dependency; rely on query function.
    idx_dim = len(qvec)
    _guard_dims_and_hash(len(qvec), idx_dim, query_model_sha, index_model_sha)
    topk_pairs = faiss_query_topk(index_path, meta_path, qvec, k=k)
    return [
        {"score": float(s), "source_path": m.source_path, "page_number": m.page_number, "start": m.start, "end": m.end}
        for s, m in topk_pairs
    ]


def _non_e2e_jsonl(index_path: str, qvec: list[float], *, k: int, query_model_sha: Optional[str], index_model_sha: Optional[str]) -> list[dict[str, Any]]:
    records = _INDEX_CACHE.get(index_path) or load_jsonl_index(index_path)
    _INDEX_CACHE[index_path] = records
    idx_dim = len(records[0].vector) if records else len(qvec)
    _guard_dims_and_hash(len(qvec), idx_dim, query_model_sha, index_model_sha)
    topk = query_topk(records, qvec, k=k)
    return [
        {"score": float(s), "source_path": r.source_path, "page_number": r.page_number, "start": r.start, "end": r.end}
        for s, r in topk
    ]


def _run_non_e2e(index_path: str, backend: str, meta_path: Optional[str], qvec: list[float], *, k: int, query_model_sha: Optional[str], index_model_sha: Optional[str], index_receipt: Optional[dict], epsilon: float, tau: float) -> dict[str, Any]:
    if backend == "faiss":
        if not meta_path:
            raise ValueError("meta_path (.meta.jsonl) is required for faiss backend")
        res = _non_e2e_faiss(index_path, meta_path, qvec, k=k, query_model_sha=query_model_sha, index_model_sha=index_model_sha)
    else:
        res = _non_e2e_jsonl(index_path, qvec, k=k, query_model_sha=query_model_sha, index_model_sha=index_model_sha)
    ingest_sidecar = index_receipt
    reason: Optional[str] = None
    if res:
        top_score = _safe_float(res[0].get("score")) if isinstance(res[0], dict) else None
        if top_score is not None and top_score < float(tau):
            reason = "top_score_below_tau"
    query_receipt = _build_query_receipt(ingest_sidecar, query_model_sha, index_model_sha, len(qvec), epsilon, tau)
    if reason is not None:
        if _PROM is not None:
            try:
                _PROM["abstain_counter"].labels(reason=reason, endpoint="query").inc()
            except Exception:
                pass
        return {"abstain": True, "reason": "insufficient coherence", "receipt": query_receipt, **({"ingest_receipt": ingest_sidecar} if ingest_sidecar is not None else {})}
    out_payload: dict[str, Any] = {"results": res, "receipt": query_receipt}
    if ingest_sidecar is not None:
        out_payload["ingest_receipt"] = ingest_sidecar
    return out_payload


def warmup_index(*, embed_model: str = "bge-small-en-v1.5", backend: str = "jsonl", index_path: Optional[str] = None, meta_path: Optional[str] = None) -> dict[str, Any]:
    """Warm up embedding model and optionally pre-load index into cache.

    - backend: "jsonl" or "faiss" (faiss warmup is a no-op except model init)
    - index_path: for jsonl, pre-loads into memory cache
    """
    _get_model(embed_model)
    loaded = False
    if backend == "jsonl" and index_path:
        _ = _get_jsonl_records(index_path)
        loaded = True
    elif backend == "faiss" and index_path:
        # FAISS warmup is a no-op here; ensure dependency present
        loaded = bool(faiss_available())
    return {"model": embed_model, "backend": backend, "index_cached": loaded}


def query_index(
    *,
    index_path: str,
    backend: str = "jsonl",
    q: str,
    k: int = 5,
    embed_model: str = "bge-small-en-v1.5",
    meta_path: Optional[str] = None,
    e2e: bool = False,
    epsilon: float = 1e-3,
    tau: float = 0.30,
) -> dict[str, Any]:
    """Programmatic query that mirrors CLI behavior and returns a JSON-serializable dict.

    - backend: "jsonl" or "faiss"
    - For faiss, meta_path is required (the .meta.jsonl produced at build time)
    - e2e: when True, runs recall→settle pipeline (jsonl only) and returns bundle+receipt
    """
    model = _get_model(embed_model)
    qvec = model.embed([q])[0]

    # Determine index dim and model hash (if available via ingest receipt)
    index_receipt = load_ingest_receipt(index_path)
    index_model_sha = _get_index_model_sha(index_receipt)
    # Current query model sha from registry (may be None)
    spec = get_model_spec(embed_model)
    query_model_sha = getattr(spec, "sha256_weights", None)

    if e2e:
        if backend != "jsonl":
            raise ValueError("--e2e is only supported with jsonl backend currently")
        return _run_e2e_jsonl(index_path, qvec, epsilon=epsilon, tau=tau, query_model_sha=query_model_sha, index_model_sha=index_model_sha, index_receipt=index_receipt)

    return _run_non_e2e(index_path, backend, meta_path, qvec, k=k, query_model_sha=query_model_sha, index_model_sha=index_model_sha, index_receipt=index_receipt, epsilon=epsilon, tau=tau)


def metrics_exposition() -> tuple[str, bytes]:
    """Return (content_type, payload) for Prometheus metrics if available; else empty."""
    if _PROM is None:
        return ("text/plain; charset=utf-8", b"")
    try:  # pragma: no cover - optional dep
        return (CONTENT_TYPE_LATEST, generate_latest(_PROM["registry"]))
    except Exception:
        return ("text/plain; charset=utf-8", b"")
