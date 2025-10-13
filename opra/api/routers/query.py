from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

# Reuse existing query service from oscillink
from oscillink.ingest.query_service import query_index

router = APIRouter()

class QueryRequest(BaseModel):
    index_path: str = Field(...)
    q: str = Field(...)
    backend: str = Field("jsonl", pattern=r"^(jsonl|faiss)$")
    k: int = Field(5, ge=1, le=100)
    embed_model: str = Field("bge-small-en-v1.5")
    meta_path: Optional[str] = None
    epsilon: float = Field(1e-3, ge=0.0)
    tau: float = Field(0.30, ge=0.0, le=1.0)
    e2e: bool = False

class QueryResponse(BaseModel):
    results: Optional[list[dict]] = None
    bundle: Optional[list[dict]] = None
    ingest_receipt: Optional[dict] = None
    settle_receipt: Optional[dict] = None
    receipt: Optional[dict] = None
    abstain: Optional[bool] = None
    reason: Optional[str] = None

@router.post("/query", response_model=QueryResponse)
def post_query(req: QueryRequest) -> Dict[str, Any]:
    return query_index(
        index_path=req.index_path,
        backend=req.backend,
        q=req.q,
        k=req.k,
        embed_model=req.embed_model,
        meta_path=req.meta_path,
        e2e=req.e2e,
        epsilon=req.epsilon,
        tau=req.tau,
    )
