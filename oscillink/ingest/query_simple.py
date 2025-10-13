from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class IndexRecord:
    source_path: str
    page_number: int
    start: int
    end: int
    text: str
    vector: List[float]
    index_sha256: str | None = None


def load_jsonl_index(path: str | Path) -> List[IndexRecord]:
    p = Path(path)
    rows: List[IndexRecord] = []
    # Pre-compute file SHA256 for provenance and attach to each record
    file_sha: str | None
    try:
        file_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        file_sha = None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                IndexRecord(
                    source_path=obj["source_path"],
                    page_number=int(obj["page_number"]),
                    start=int(obj["start"]),
                    end=int(obj["end"]),
                    text=obj["text"],
                    vector=[float(x) for x in obj["vector"]],
                    index_sha256=file_sha,
                )
            )
    return rows


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def query_topk(records: Sequence[IndexRecord], qvec: Sequence[float], *, k: int = 5) -> List[Tuple[float, IndexRecord]]:
    scored: List[Tuple[float, IndexRecord]] = []
    for r in records:
        s = _dot(qvec, r.vector)
        scored.append((s, r))
    # Deterministic sort: score desc, then path, page, start, end
    scored.sort(key=lambda t: (-t[0], t[1].source_path, t[1].page_number, t[1].start, t[1].end))
    return scored[:k]
