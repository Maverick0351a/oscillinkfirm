from __future__ import annotations
# ruff: noqa: I001

from typing import Any, Optional, Dict, List
import os

from importlib.util import find_spec

from ..adapters.recall import RecallParams, recall_and_settle_jsonl, recall_and_settle_records
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
        from prometheus_client import Counter, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

        _PROM = {
            "registry": CollectorRegistry(auto_describe=True),
        }
        _PROM["abstain_counter"] = Counter(
            "osc_query_abstain_total",
            "Total number of abstained queries",
            labelnames=("reason", "endpoint"),
            registry=_PROM["registry"],
        )
        _PROM["ocr_low_conf_total"] = Counter(
            "osc_ocr_low_conf_total",
            "Total number of queries referencing low-confidence OCR indexes",
            labelnames=("endpoint",),
            registry=_PROM["registry"],
        )
        _PROM["ocr_avg_conf_gauge"] = Gauge(
            "osc_ocr_avg_conf_gauge",
            "Average OCR confidence (0..1) observed on last query",
            labelnames=("endpoint",),
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


def _build_query_receipt(
    ingest_sidecar: Optional[dict],
    query_model_sha: Optional[str],
    index_model_sha: Optional[str],
    dim: int,
    epsilon: float,
    tau: float,
    *,
    context_hints: Optional[Dict[str, List[str]]] = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "query_model_sha256": query_model_sha,
        "index_model_sha256": index_model_sha,
        "index_sha256": ingest_sidecar.get("index_sha256") if isinstance(ingest_sidecar, dict) else None,
        "dim": int(dim),
        "epsilon": float(epsilon),
        "tau": float(tau),
    }
    # Attach optional context hints under a meta block to avoid breaking existing consumers
    if context_hints:
        rec["meta"] = {"context": context_hints}
    return rec


def _context_hints_from_records(selected: List[Any]) -> Dict[str, List[str]]:
    """Collect lightweight context hints from selected IndexRecord objects.

    Returns keys only when present to keep payload compact.
    """
    threads: set[str] = set()
    msg_ids: set[str] = set()
    html_titles: set[str] = set()
    for r in selected:
        meta = getattr(r, "meta", None)
        if not isinstance(meta, dict):
            continue
        tid = meta.get("email.thread_id")
        mid = meta.get("email.message_id") or meta.get("message_id")
        if isinstance(tid, str) and tid:
            threads.add(tid)
        if isinstance(mid, str) and mid:
            msg_ids.add(mid)
        try:
            st = meta.get("source_type")
            title = meta.get("title") or meta.get("page_title")
            if st == "html" and isinstance(title, str) and title:
                html_titles.add(title)
        except Exception:
            # defensive: never fail hints on odd metadata
            pass
    out: Dict[str, List[str]] = {}
    if threads:
        out["email_thread_ids"] = sorted(threads)
    if msg_ids:
        out["email_message_ids"] = sorted(msg_ids)
    if html_titles:
        out["html_page_titles"] = sorted(html_titles)
    return out


def _load_quality_ocr_config() -> Dict[str, Any]:
    """Read OCR quality config from env or optional YAML (OSCILLINK_CONFIG or ./firm.yaml).

    Returns keys:
      - score_penalty: float (default 0.08)
      - abstain_on_all_low: bool (default True)
    """
    cfg: Dict[str, Any] = {
        "score_penalty": float(os.getenv("OSCILLINK_OCR_SCORE_PENALTY", "0.08")),
        "abstain_on_all_low": os.getenv("OSCILLINK_OCR_ABSTAIN_ON_ALL_LOW", "true").lower() in {"1", "true", "on", "yes"},
    }
    path = os.getenv("OSCILLINK_CONFIG", "firm.yaml")
    if os.path.isfile(path):  # pragma: no cover - optional dependency and file
        try:
            import yaml  # type: ignore

            with open(path, encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            o = (((y or {}).get("quality") or {}).get("ocr") or {})
            if isinstance(o, dict):
                if "score_penalty" in o:
                    try:
                        cfg["score_penalty"] = float(o["score_penalty"])  # type: ignore[arg-type]
                    except Exception:
                        pass
                if "abstain_on_all_low" in o:
                    try:
                        v = o["abstain_on_all_low"]
                        if isinstance(v, bool):
                            cfg["abstain_on_all_low"] = v
                        else:
                            cfg["abstain_on_all_low"] = str(v).lower() in {"1", "true", "on", "yes"}
                    except Exception:
                        pass
        except Exception:
            # YAML is optional; ignore if not present or unparseable
            pass
    return cfg


def _extract_bundle_ids(bundle: List[dict]) -> List[int]:
    ids: List[int] = []
    for b in bundle:
        if not isinstance(b, dict):
            continue
        val = b.get("id")
        try:
            if val is None:
                continue
            i = int(val)
            ids.append(i)
        except (ValueError, TypeError):
            continue
    return ids


def _jsonl_hints_for_results(index_path: str, res: List[dict]) -> Dict[str, List[str]]:
    try:
        records_all = _get_jsonl_records(index_path)
        key_map: Dict[tuple, Any] = {}
        for r in records_all:
            key_map[(r.source_path, r.page_number, r.start, r.end)] = r
        matched: List[Any] = []
        for item in res:
            key = (
                item.get("source_path"),
                item.get("page_number"),
                item.get("start"),
                item.get("end"),
            )
            rec = key_map.get(key)
            if rec is not None:
                matched.append(rec)
        return _context_hints_from_records(matched)
    except Exception:
        return {}


def _inc_low_ocr_metric(endpoint: str) -> None:
    if _PROM is None:
        return
    try:  # pragma: no cover - optional dep
        _PROM["ocr_low_conf_total"].labels(endpoint=endpoint).inc()
    except Exception:
        pass


def _set_ocr_avg_metric(endpoint: str, avg: Optional[float]) -> None:
    if _PROM is None or avg is None:
        return
    try:  # pragma: no cover - optional dep
        _PROM["ocr_avg_conf_gauge"].labels(endpoint=endpoint).set(float(avg))
    except Exception:
        pass


def _faiss_apply_penalty_and_sort(items: list[dict[str, Any]], dec: float) -> None:
    for it in items:
        if "score" in it:
            try:
                it["score"] = float(it["score"]) - dec
            except Exception:
                pass
    try:
        items.sort(
            key=lambda d: (
                -float(d.get("score", 0.0)),
                d.get("source_path"),
                d.get("page_number", 0),
                d.get("start", 0),
                d.get("end", 0),
            )
        )
    except Exception:
        pass


def _settle_and_select(index_path: str, qvec: list[float], filters: Optional[Dict[str, Any]] = None) -> tuple[list[dict], dict, List[Any], int, bool]:
    """Run recall→settle and map bundle ids to in-memory records.

    Returns: (bundle, settle_receipt, selected_records, index_dim, no_candidates_after_filter)
    """
    params = RecallParams(kneighbors=6, lamC=0.5, lamQ=4.0, lamP=0.0, tol=1e-3, bundle_k=5)
    def _map_selected(ids: List[int], records: List[Any]) -> List[Any]:
        sel: List[Any] = []
        for i in ids:
            try:
                if 0 <= i < len(records):
                    sel.append(records[i])
            except Exception:
                continue
        return sel

    if filters:
        all_recs = _get_jsonl_records(index_path)
        cand = _apply_meta_filters(all_recs, filters)
        if not cand:
            dim0 = len(all_recs[0].vector) if all_recs else len(qvec)
            return [], {}, [], dim0, True
        bundle_f, receipt_f = recall_and_settle_records(cand, qvec, params=params)
        try:
            if isinstance(receipt_f, dict):
                m = receipt_f.setdefault("meta", {}) if isinstance(receipt_f.get("meta"), dict) else {}
                parent_sig = getattr(cand[0], "index_sha256", None)
                if parent_sig:
                    m["parent_ingest_sig"] = parent_sig
                receipt_f["meta"] = m
        except Exception:
            pass
        dim_f = len(cand[0].vector) if cand else len(qvec)
        ids_f = _extract_bundle_ids(bundle_f)
        sel_f = _map_selected(ids_f, cand)
        return bundle_f, receipt_f, sel_f, dim_f, False

    # Default path: settle over full index
    bundle, receipt = recall_and_settle_jsonl(index_path, qvec, params=params)
    recs = _get_jsonl_records(index_path)
    dim = len(recs[0].vector) if recs else len(qvec)
    ids = _extract_bundle_ids(bundle)
    sel = _map_selected(ids, recs)
    return bundle, receipt, sel, dim, False


def _all_low_ocr_records(records: List[Any]) -> bool:
    try:
        return bool(records) and all(bool(getattr(r, "meta", {}) and getattr(r, "meta", {}).get("ocr_low_confidence")) for r in records)
    except Exception:
        return False


def _abstain_reason_e2e(
    *,
    delta_h: Optional[float],
    top_score: Optional[float],
    epsilon: float,
    tau: float,
    selected: List[Any],
    index_receipt: Optional[dict],
    abstain_on_all_low: bool,
) -> Optional[str]:
    insufficient = (delta_h is not None and delta_h < float(epsilon)) or (top_score is not None and top_score < float(tau))
    if not insufficient:
        return None
    low_ocr = bool(index_receipt.get("ocr_low_confidence")) if isinstance(index_receipt, dict) else False
    if abstain_on_all_low and (delta_h is not None and delta_h < float(epsilon)) and _all_low_ocr_records(selected):
        low_ocr = True
    return "low-quality OCR" if low_ocr else "insufficient coherence"


def _build_e2e_payload(
    *,
    bundle: list[dict],
    settle_receipt: dict,
    query_receipt: dict,
    ingest_sidecar: Optional[dict],
) -> dict[str, Any]:
    out: dict[str, Any] = {"bundle": bundle, "settle_receipt": settle_receipt, "receipt": query_receipt}
    if ingest_sidecar is not None:
        out["ingest_receipt"] = ingest_sidecar
        out["ocr_low_confidence"] = ingest_sidecar.get("ocr_low_confidence")
        out["ocr_avg_confidence"] = ingest_sidecar.get("ocr_avg_confidence")
        if bool(ingest_sidecar.get("ocr_low_confidence")):
            _inc_low_ocr_metric("query-e2e")
        try:
            avg_val = ingest_sidecar.get("ocr_avg_confidence")
            avg_float = float(avg_val) if avg_val is not None else None
        except Exception:
            avg_float = None
        _set_ocr_avg_metric("query-e2e", avg_float)
    return out


def _run_e2e_jsonl(index_path: str, qvec: list[float], *, epsilon: float, tau: float, query_model_sha: Optional[str], index_model_sha: Optional[str], index_receipt: Optional[dict], filters: Optional[Dict[str, Any]] = None) -> dict[str, Any]:
    bundle, settle_receipt, selected, idx_dim, no_cands = _settle_and_select(index_path, qvec, filters)
    _guard_dims_and_hash(len(qvec), idx_dim, query_model_sha, index_model_sha)
    if no_cands:
        ingest_sidecar = index_receipt
        query_receipt = _build_query_receipt(ingest_sidecar, query_model_sha, index_model_sha, idx_dim, epsilon, tau, context_hints=None)
        return {"abstain": True, "reason": "no candidates after filter", "receipt": query_receipt, **({"ingest_receipt": ingest_sidecar} if ingest_sidecar is not None else {})}
    dH = _safe_float(settle_receipt.get("deltaH_total")) if isinstance(settle_receipt, dict) else None
    top_score = _safe_float(settle_receipt.get("coherence")) if isinstance(settle_receipt, dict) else None
    ingest_sidecar = index_receipt
    hints = _context_hints_from_records(selected) if selected else {}
    query_receipt = _build_query_receipt(ingest_sidecar, query_model_sha, index_model_sha, idx_dim, epsilon, tau, context_hints=hints or None)
    cfg = _load_quality_ocr_config()
    abstain_reason = _abstain_reason_e2e(
        delta_h=dH,
        top_score=top_score,
        epsilon=epsilon,
        tau=tau,
        selected=selected,
        index_receipt=index_receipt,
        abstain_on_all_low=cfg.get("abstain_on_all_low", True),
    )
    if abstain_reason is not None:
        if _PROM is not None:
            try:
                _PROM["abstain_counter"].labels(reason=("low_ocr" if abstain_reason == "low-quality OCR" else "insufficient"), endpoint="query-e2e").inc()
            except Exception:
                pass
        return {"abstain": True, "reason": abstain_reason, "receipt": query_receipt, "settle_receipt": settle_receipt, **({"ingest_receipt": ingest_sidecar} if ingest_sidecar is not None else {})}
    return _build_e2e_payload(bundle=bundle, settle_receipt=settle_receipt, query_receipt=query_receipt, ingest_sidecar=ingest_sidecar)


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


def _get_non_e2e_results(
    backend: str,
    index_path: str,
    meta_path: Optional[str],
    qvec: list[float],
    *,
    k: int,
    query_model_sha: Optional[str],
    index_model_sha: Optional[str],
    filters: Optional[Dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if backend == "faiss":
        if not meta_path:
            raise ValueError("meta_path (.meta.jsonl) is required for faiss backend")
        return _non_e2e_faiss(index_path, meta_path, qvec, k=k, query_model_sha=query_model_sha, index_model_sha=index_model_sha)
    return _non_e2e_jsonl(index_path, qvec, k=k, query_model_sha=query_model_sha, index_model_sha=index_model_sha, filters=filters)


def _evaluate_abstain_and_hints(
    res: list[dict[str, Any]],
    *,
    tau: float,
    backend: str,
    index_path: str,
) -> tuple[Optional[str], Dict[str, List[str]]]:
    reason: Optional[str] = None
    hints: Dict[str, List[str]] = {}
    if res:
        top_item = res[0] if isinstance(res[0], dict) else None
        top_score = _safe_float(top_item.get("score")) if isinstance(top_item, dict) else None
        if top_score is not None and top_score < float(tau):
            reason = "top_score_below_tau"
        if backend == "jsonl":
            hints = _jsonl_hints_for_results(index_path, res)
    return reason, hints


def _apply_low_ocr_annotations(
    *,
    backend: str,
    res: list[dict[str, Any]],
    ingest_sidecar: Optional[dict],
) -> None:
    if not (ingest_sidecar and ingest_sidecar.get("ocr_low_confidence")):
        return
    _inc_low_ocr_metric("query")
    _set_ocr_avg_metric("query", ingest_sidecar.get("ocr_avg_confidence") if ingest_sidecar else None)
    penalty = float(_load_quality_ocr_config().get("score_penalty", 0.08))
    for item in res:
        if isinstance(item, dict):
            item["ocr_low_confidence"] = True
            item["ocr_avg_confidence"] = ingest_sidecar.get("ocr_avg_confidence") if ingest_sidecar else None
    if backend == "faiss":
        _faiss_apply_penalty_and_sort(res, penalty)


def _apply_meta_filters(records: list[Any], filters: Optional[Dict[str, Any]]) -> list[Any]:
    if not filters:
        return records
    out: list[Any] = []
    for r in records:
        meta = getattr(r, "meta", None)
        if not isinstance(meta, dict):
            continue
        ok = True
        for k, v in filters.items():
            if k not in meta:
                ok = False
                break
            if meta.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _non_e2e_jsonl(index_path: str, qvec: list[float], *, k: int, query_model_sha: Optional[str], index_model_sha: Optional[str], filters: Optional[Dict[str, Any]] = None) -> list[dict[str, Any]]:
    records = _INDEX_CACHE.get(index_path) or load_jsonl_index(index_path)
    _INDEX_CACHE[index_path] = records
    # Apply optional metadata filters (equality match on provided fields)
    records = _apply_meta_filters(records, filters)
    idx_dim = len(records[0].vector) if records else len(qvec)
    _guard_dims_and_hash(len(qvec), idx_dim, query_model_sha, index_model_sha)
    # Fetch a slightly larger candidate set, apply OCR-aware penalty, then re-rank deterministically
    raw = query_topk(records, qvec, k=max(k, 10))
    cfg = _load_quality_ocr_config()
    penalty = float(cfg.get("score_penalty", 0.08))
    def _penalty(meta: Optional[Dict[str, Any]]) -> float:
        try:
            if isinstance(meta, dict) and meta.get("ocr_low_confidence"):
                return penalty
        except Exception:
            pass
        return 0.0
    adjusted: List[tuple[float, Any]] = []
    for s, r in raw:
        try:
            meta = getattr(r, "meta", None)
        except Exception:
            meta = None
        adjusted.append((float(s) - _penalty(meta), r))
    adjusted.sort(key=lambda t: (-t[0], t[1].source_path, t[1].page_number, t[1].start, t[1].end))
    return [
        {"score": float(s), "source_path": r.source_path, "page_number": r.page_number, "start": r.start, "end": r.end}
        for s, r in adjusted[:k]
    ]


def _run_non_e2e(index_path: str, backend: str, meta_path: Optional[str], qvec: list[float], *, k: int, query_model_sha: Optional[str], index_model_sha: Optional[str], index_receipt: Optional[dict], epsilon: float, tau: float, filters: Optional[Dict[str, Any]] = None) -> dict[str, Any]:
    res = _get_non_e2e_results(
        backend,
        index_path,
        meta_path,
        qvec,
        k=k,
        query_model_sha=query_model_sha,
        index_model_sha=index_model_sha,
        filters=filters,
    )
    ingest_sidecar = index_receipt
    reason, hints = _evaluate_abstain_and_hints(res, tau=tau, backend=backend, index_path=index_path)
    query_receipt = _build_query_receipt(
        ingest_sidecar, query_model_sha, index_model_sha, len(qvec), epsilon, tau, context_hints=(hints or None)
    )
    # Note: For non-e2e queries, always return results. If top score is below tau,
    # we include abstain flags alongside results rather than omitting them.
    if reason is not None and _PROM is not None:
        try:
            _PROM["abstain_counter"].labels(reason=reason, endpoint="query").inc()
        except Exception:
            pass
        if ingest_sidecar is not None:
            try:
                avg = ingest_sidecar.get("ocr_avg_confidence")
                if avg is not None:
                    _PROM["ocr_avg_conf_gauge"].labels(endpoint="query").set(float(avg))
            except Exception:
                pass
    _apply_low_ocr_annotations(backend=backend, res=res, ingest_sidecar=ingest_sidecar)
    out_payload: dict[str, Any] = {"results": res, "receipt": query_receipt}
    if ingest_sidecar is not None:
        out_payload["ingest_receipt"] = ingest_sidecar
        out_payload["ocr_low_confidence"] = ingest_sidecar.get("ocr_low_confidence")
        out_payload["ocr_avg_confidence"] = ingest_sidecar.get("ocr_avg_confidence")
    if reason is not None:
        out_payload["abstain"] = True
        out_payload["reason"] = "insufficient coherence"
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
    filters: Optional[Dict[str, Any]] = None,
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
        return _run_e2e_jsonl(index_path, qvec, epsilon=epsilon, tau=tau, query_model_sha=query_model_sha, index_model_sha=index_model_sha, index_receipt=index_receipt, filters=filters)

    return _run_non_e2e(index_path, backend, meta_path, qvec, k=k, query_model_sha=query_model_sha, index_model_sha=index_model_sha, index_receipt=index_receipt, epsilon=epsilon, tau=tau, filters=filters)


def metrics_exposition() -> tuple[str, bytes]:
    """Return (content_type, payload) for Prometheus metrics if available; else empty."""
    if _PROM is None:
        return ("text/plain; charset=utf-8", b"")
    try:  # pragma: no cover - optional dep
        return (CONTENT_TYPE_LATEST, generate_latest(_PROM["registry"]))
    except Exception:
        return ("text/plain; charset=utf-8", b"")
