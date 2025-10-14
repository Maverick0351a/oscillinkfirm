from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .chunk import Chunk


@dataclass(frozen=True)
class BuiltIndex:
    index_path: str
    index_sha256: str
    dim: int
    count: int


def build_jsonl_index(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, out_path: str) -> BuiltIndex:
    assert len(chunks) == len(vectors)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic write: iterate in order and write one JSON per line
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for ch, vec in zip(chunks, vectors):
            rec = {
                "source_path": ch.source_path,
                "page_number": ch.page_number,
                "start": ch.start,
                "end": ch.end,
                "text": ch.text,
                "meta": ch.meta,
                "vector": vec,
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    # Compute SHA256 over file bytes
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    dim = len(vectors[0]) if vectors else 0
    return BuiltIndex(index_path=str(path), index_sha256=digest, dim=dim, count=len(vectors))
