from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class IngestReceipt:
    version: int
    input_path: str
    file_sha256: str | None
    chunks: int
    index_path: str
    index_sha256: str
    embed_model: str
    deterministic: bool
    determinism_env: Dict[str, str]
    # Optional embedding metadata for provenance
    embed_dim: int | None = None
    embed_license: str | None = None
    embed_weights_sha256: str | None = None
    embed_tokenizer_sha256: str | None = None
    # Optional extraction/OCR provenance
    extract_parser: str | None = None
    ocr_backend: str | None = None
    ocr_langs: str | None = None
    # Optional OCR quality indicators
    ocr_avg_confidence: float | None = None
    ocr_low_confidence: bool | None = None

    def signature(self) -> str:
        # Canonical JSON of the core fields (sorted keys, compact separators)
        payload = {
            "version": self.version,
            "input_path": self.input_path,
            "file_sha256": self.file_sha256,
            "chunks": self.chunks,
            "index_path": self.index_path,
            "index_sha256": self.index_sha256,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "embed_license": self.embed_license,
            "embed_weights_sha256": self.embed_weights_sha256,
            "embed_tokenizer_sha256": self.embed_tokenizer_sha256,
            "deterministic": self.deterministic,
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return _sha256_bytes(data)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


# --- Sidecar persistence helpers ---
def _ingest_sidecar_path(index_path: str | Path) -> Path:
    p = Path(index_path)
    # Avoid duplicating suffix: write alongside index as <name>.<ext>.ingest.json
    return p.with_suffix(p.suffix + ".ingest.json")


def save_ingest_receipt(index_path: str | Path, receipt: IngestReceipt | dict) -> Path:
    """Persist the ingest receipt as a JSON sidecar next to the index file.

    Returns the path written.
    """
    sidecar = _ingest_sidecar_path(index_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(receipt.to_json()) if isinstance(receipt, IngestReceipt) else receipt
    sidecar.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return sidecar


def load_ingest_receipt(index_path: str | Path) -> Optional[dict]:
    """Load the ingest receipt sidecar if present; return dict or None if missing."""
    sidecar = _ingest_sidecar_path(index_path)
    try:
        if sidecar.exists():
            return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None
