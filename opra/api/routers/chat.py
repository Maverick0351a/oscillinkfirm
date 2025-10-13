from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from oscillink.ingest.query_service import query_index

router = APIRouter()


class ChatRequest(BaseModel):
    index_path: str = Field(...)
    q: str = Field(..., description="User prompt/question")
    backend: str = Field("jsonl")
    k: int = Field(5, ge=1, le=20)
    embed_model: str = Field("bge-small-en-v1.5")
    meta_path: Optional[str] = None
    epsilon: float = Field(1e-3, ge=0.0)
    tau: float = Field(0.30, ge=0.0, le=1.0)
    e2e: bool = False
    synth_mode: str = Field("extractive", pattern=r"^(extractive|llm)$")
    llm_model: Optional[str] = Field(default=None, description="OpenAI-compatible model name")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=32, le=4096)


class Citation(BaseModel):
    score: Optional[float] = None
    source_path: Optional[str] = None
    page_number: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None


class ChatResponse(BaseModel):
    answer: Optional[str] = None
    citations: Optional[List[Citation]] = None
    receipt: Optional[dict] = None
    ingest_receipt: Optional[dict] = None
    settle_receipt: Optional[dict] = None
    abstain: Optional[bool] = None
    reason: Optional[str] = None
    mode: str = "extractive"


def _format_extractive_answer(q: str, items: List[dict]) -> str:
    lines = [f"Q: {q}", "", "Based on the top retrieved passages:"]
    for i, it in enumerate(items[:5], 1):
        src = it.get("source_path")
        page = it.get("page_number")
        start = it.get("start")
        end = it.get("end")
        sc = it.get("score")
        lines.append(f"{i}. {os.path.basename(str(src)) if src else 'unknown'} (page {page}, lines {start}-{end}) — score={sc}")
    lines.append("")
    lines.append("Answer: The information above contains the most relevant citations. Refer to the listed sources and passages.")
    return "\n".join(lines)


def _call_llm(prompt: str, *, model: Optional[str], temperature: float, max_tokens: int) -> Optional[str]:
    base = os.getenv("OPRA_LLM_BASE_URL")
    api_key = os.getenv("OPRA_LLM_API_KEY")
    if not base:
        return None
    try:
        import json

        import httpx

        url = base.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {
            "model": model or os.getenv("OPRA_LLM_MODEL", "gpt-3.5-turbo"),
            "messages": [
                {"role": "system", "content": "You are a precise assistant. Only answer from the provided context. If unsure, say you don't know."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        with httpx.Client(timeout=20.0) as client:
            r = client.post(url, headers=headers, content=json.dumps(body))
        r.raise_for_status()
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return msg.get("content")
    except Exception:
        return None


def _retrieve(req: ChatRequest) -> Dict[str, Any]:
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


def _citations_from_qres(qres: Dict[str, Any], *, use_bundle: bool) -> List[dict]:
    items: List[dict] = []
    if use_bundle and isinstance(qres.get("bundle"), list):
        for b in qres["bundle"]:
            if isinstance(b, dict):
                items.append({
                    "score": b.get("score"),
                    "source_path": b.get("source_path"),
                    "page_number": b.get("page_number"),
                    "start": b.get("start"),
                    "end": b.get("end"),
                })
    elif isinstance(qres.get("results"), list):
        items = list(qres["results"])  # shallow copy
    return items


def _to_citations(items: List[dict]) -> List[dict]:
    def _maybe_float(x: object) -> Optional[float]:
        try:
            if isinstance(x, (int, float)):
                return float(x)
            return None
        except Exception:
            return None

    return [
        {
            "score": _maybe_float(it.get("score")),
            "source_path": it.get("source_path"),
            "page_number": it.get("page_number"),
            "start": it.get("start"),
            "end": it.get("end"),
        }
        for it in items
    ]


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> Dict[str, Any]:
    # Normalize backend flag to support variants like "faiss:hnsw"
    backend_flag = req.backend.split(":", 1)[0]
    req.backend = backend_flag if backend_flag in {"jsonl", "faiss"} else "jsonl"
    qres = _retrieve(req)
    if qres.get("abstain"):
        return {
            "abstain": True,
            "reason": qres.get("reason") or "insufficient coherence",
            "receipt": qres.get("receipt"),
            "ingest_receipt": qres.get("ingest_receipt"),
            "settle_receipt": qres.get("settle_receipt"),
            "mode": req.synth_mode,
        }

    items = _citations_from_qres(qres, use_bundle=req.e2e)
    mode = req.synth_mode
    answer_text: Optional[str] = None
    if mode == "llm":
        preface = _format_extractive_answer(req.q, items)
        answer_text = _call_llm(
            preface + "\n\nPlease answer succinctly with citations by index (1..n).",
            model=req.llm_model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        if answer_text is None:
            mode = "extractive"

    if mode == "extractive" or answer_text is None:
        answer_text = _format_extractive_answer(req.q, items)

    return {
        "answer": answer_text,
        "citations": _to_citations(items),
        "receipt": qres.get("receipt"),
        "ingest_receipt": qres.get("ingest_receipt"),
        "settle_receipt": qres.get("settle_receipt"),
        "mode": mode,
    }
