"""Ingest pipeline package (extract -> OCR -> chunk -> embed -> index -> recall).

Modules are intentionally small and composable. Determinism plumbing is applied
at import time when OSC_DETERMINISTIC=1.
"""

from __future__ import annotations

from ..adapters.recall import RecallParams, recall_and_settle_jsonl, recall_and_settle_records
from .chunk import Chunk, ChunkResult, chunk_paragraphs
from .determinism import _DETERMINISM_STATE as _INGEST_DET_STATE
from .embed import EmbeddingModel, load_embedding_model
from .extract import ExtractedPage, ExtractResult, extract_text
from .index_faiss import (
	BuiltFaissIndex,
	FaissMeta,
	build_faiss_flat,
	faiss_available,
	faiss_query_topk,
)
from .index_simple import BuiltIndex, build_jsonl_index
from .models_registry import ModelSpec, get_model_spec, load_registry
from .ocr import OcrResult, OcrStats, ocr_if_needed
from .query_simple import IndexRecord, load_jsonl_index, query_topk
from .receipts import IngestReceipt

__all__ = [
	"ExtractedPage",
	"ExtractResult",
	"extract_text",
	"Chunk",
	"ChunkResult",
	"chunk_paragraphs",
	"EmbeddingModel",
	"load_embedding_model",
	"BuiltIndex",
	"build_jsonl_index",
	"BuiltFaissIndex",
	"FaissMeta",
	"build_faiss_flat",
	"faiss_available",
	"faiss_query_topk",
	"ModelSpec",
	"get_model_spec",
	"load_registry",
	"IngestReceipt",
	"OcrStats",
	"OcrResult",
	"ocr_if_needed",
	"IndexRecord",
	"load_jsonl_index",
	"query_topk",
	"recall_and_settle_records",
	"recall_and_settle_jsonl",
	"RecallParams",
]

# Convenience exports for API surfaces
def is_deterministic() -> bool:
	return bool(_INGEST_DET_STATE.get("enabled", False))


def determinism_env() -> dict[str, str]:
	env = _INGEST_DET_STATE.get("env", {})
	if isinstance(env, dict):
		# Narrow to str->str
		return {str(k): str(v) for k, v in env.items()}
	return {}
