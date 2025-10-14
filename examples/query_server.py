from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from oscillink.ingest.query_service import metrics_exposition, query_index, warmup_index

# Optional Prometheus metrics (import after standard libs)
try:  # pragma: no cover - optional dep
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - optional dep
    Counter = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]

_REQ_COUNTER = None
_LAT_HIST = None
if Counter is not None and Histogram is not None:  # type: ignore[truthy-function]
    try:
        _REQ_COUNTER = Counter("osc_http_requests_total", "HTTP requests", labelnames=("endpoint",))
        _LAT_HIST = Histogram("osc_http_request_duration_ms", "Request duration (ms)", labelnames=("endpoint",))
    except Exception:
        _REQ_COUNTER = None
        _LAT_HIST = None

app = FastAPI(title="Oscillink On-Prem Query Service", version="0.1")


class QueryRequest(BaseModel):
    index_path: str = Field(..., description="Path to index (.jsonl for JSONL or .faiss for FAISS)")
    q: str = Field(..., description="Query text")
    backend: str = Field("jsonl", pattern="^(jsonl|faiss)$")
    k: int = Field(6, ge=1, le=100)
    embed_model: str = Field("bge-small-en-v1.5")
    meta_path: Optional[str] = Field(None, description="Required for FAISS: path to .meta.jsonl")
    epsilon: float = Field(1e-3, ge=0.0)
    tau: float = Field(0.30, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional equality filters on metadata (e.g., matter_id, client_id)")


class QueryResult(BaseModel):
    score: float
    source_path: str
    page_number: int
    start: int
    end: int


class QueryResponse(BaseModel):
    results: Optional[list[QueryResult]] = None
    bundle: Optional[list[dict]] = None
    settle_receipt: Optional[dict] = None
    ingest_receipt: Optional[dict] = None
    receipt: Optional[dict] = None
    abstain: Optional[bool] = None
    reason: Optional[str] = None


class WarmupRequest(BaseModel):
    backend: str = Field("jsonl", pattern="^(jsonl|faiss)$")
    embed_model: str = Field("bge-small-en-v1.5")
    index_path: Optional[str] = Field(None)
    meta_path: Optional[str] = Field(None)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/license/status")
def license_status() -> Response:
    """Report licensed-container status based on exported entitlements.

    Behavior:
      - If /run/oscillink_entitlements.json exists and has a valid status, return it.
      - When OSCILLINK_LICENSE_REQUIRED=1|true|on, return 503 if entitlements are missing.
    """
    require = os.getenv("OSCILLINK_LICENSE_REQUIRED", "0").lower() in {"1", "true", "on"}
    ent_path = "/run/oscillink_entitlements.json"
    if not os.path.exists(ent_path):
        status = {"status": "unlicensed"}
        if require:
            return Response(content=json.dumps(status), media_type="application/json", status_code=503)
        return Response(content=json.dumps(status), media_type="application/json", status_code=200)
    try:
        with open(ent_path, encoding="utf-8") as f:
            data = json.load(f)
        payload = {
            "status": "ok",
            "tier": data.get("tier"),
            "license_id": data.get("license_id") or data.get("sub"),
            "expires_at": data.get("exp"),
        }
        return Response(content=json.dumps(payload), media_type="application/json", status_code=200)
    except Exception:
        status = {"status": "unlicensed"}
        if require:
            return Response(content=json.dumps(status), media_type="application/json", status_code=503)
        return Response(content=json.dumps(status), media_type="application/json", status_code=200)

@app.post("/v1/query", response_model=QueryResponse)
def http_query(req: QueryRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out = query_index(
        index_path=req.index_path,
        backend=req.backend,
        q=req.q,
        k=req.k,
        embed_model=req.embed_model,
        meta_path=req.meta_path,
        e2e=False,
        epsilon=req.epsilon,
        tau=req.tau,
        filters=req.filters,
    )
    dt = (time.perf_counter() - t0) * 1000.0
    # metrics
    if _REQ_COUNTER is not None:
        try:
            _REQ_COUNTER.labels(endpoint="query").inc()
        except Exception:
            pass
    if _LAT_HIST is not None:
        try:
            _LAT_HIST.labels(endpoint="query").observe(dt)
        except Exception:
            pass
    try:
        log = {
            "event": "query",
            "backend": req.backend,
            "k": req.k,
            "epsilon": req.epsilon,
            "tau": req.tau,
            "latency_ms": round(dt, 2),
            "abstain": bool(out.get("abstain", False)),
        }
        sys.stderr.write(json.dumps(log) + "\n")
    except Exception:
        pass
    return out

@app.post("/v1/query-e2e", response_model=QueryResponse)
def http_query_e2e(req: QueryRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out = query_index(
        index_path=req.index_path,
        backend="jsonl",
        q=req.q,
        k=req.k,
        embed_model=req.embed_model,
        meta_path=None,
        e2e=True,
        epsilon=req.epsilon,
        tau=req.tau,
        filters=req.filters,
    )
    dt = (time.perf_counter() - t0) * 1000.0
    if _REQ_COUNTER is not None:
        try:
            _REQ_COUNTER.labels(endpoint="query_e2e").inc()
        except Exception:
            pass
    if _LAT_HIST is not None:
        try:
            _LAT_HIST.labels(endpoint="query_e2e").observe(dt)
        except Exception:
            pass
    try:
        log = {
            "event": "query_e2e",
            "k": req.k,
            "epsilon": req.epsilon,
            "tau": req.tau,
            "latency_ms": round(dt, 2),
            "abstain": bool(out.get("abstain", False)),
        }
        sys.stderr.write(json.dumps(log) + "\n")
    except Exception:
        pass
    return out


@app.post("/v1/warmup")
def http_warmup(req: WarmupRequest) -> Dict[str, Any]:
    return warmup_index(embed_model=req.embed_model, backend=req.backend, index_path=req.index_path, meta_path=req.meta_path)


@app.get("/metrics")
def http_metrics(x_admin_secret: Optional[str] = Header(default=None, alias="X-Admin-Secret")) -> Response:
    # Optional protection: require admin secret only when enabled AND a secret is set
    if os.getenv("OSCILLINK_METRICS_PROTECTED", "0").lower() in {"1", "true", "on"}:
        configured = os.getenv("OSCILLINK_ADMIN_SECRET")
        if not configured:
            raise HTTPException(status_code=503, detail="admin secret not configured")
        if x_admin_secret != configured:
            raise HTTPException(status_code=401, detail="invalid admin secret")
    ctype, payload = metrics_exposition()
    return Response(content=payload, media_type=ctype)
