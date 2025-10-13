from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from oscillink.ingest.query_service import metrics_exposition, query_index, warmup_index

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

@app.post("/v1/query", response_model=QueryResponse)
def http_query(req: QueryRequest) -> Dict[str, Any]:
    return query_index(
        index_path=req.index_path,
        backend=req.backend,
        q=req.q,
        k=req.k,
        embed_model=req.embed_model,
        meta_path=req.meta_path,
        e2e=False,
        epsilon=req.epsilon,
        tau=req.tau,
    )

@app.post("/v1/query-e2e", response_model=QueryResponse)
def http_query_e2e(req: QueryRequest) -> Dict[str, Any]:
    return query_index(
        index_path=req.index_path,
        backend="jsonl",
        q=req.q,
        k=req.k,
        embed_model=req.embed_model,
        meta_path=None,
        e2e=True,
        epsilon=req.epsilon,
        tau=req.tau,
    )


@app.post("/v1/warmup")
def http_warmup(req: WarmupRequest) -> Dict[str, Any]:
    return warmup_index(embed_model=req.embed_model, backend=req.backend, index_path=req.index_path, meta_path=req.meta_path)


@app.get("/metrics")
def http_metrics() -> Response:
    ctype, payload = metrics_exposition()
    return Response(content=payload, media_type=ctype)
