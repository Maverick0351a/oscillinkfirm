from __future__ import annotations
# ruff: noqa: I001

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import subprocess
import tempfile

from .extract import ExtractedPage, ExtractResult


@dataclass(frozen=True)
class OcrStats:
    backend: str
    langs: str
    avg_confidence: float | None


@dataclass(frozen=True)
class OcrResult:
    pages: List[ExtractedPage]
    stats: OcrStats


def _run_ocrmypdf(input_path: str, langs: str) -> List[ExtractedPage] | None:
    """Run ocrmypdf in a temp dir and read sidecar text.

    We use --sidecar to get extracted text deterministically and return a single page.
    Returns None on any failure.
    """
    try:
        p = Path(input_path)
        if not p.exists():
            return None
        with tempfile.TemporaryDirectory() as td:
            sidecar = Path(td) / "sidecar.txt"
            # We don't overwrite original PDF; output to temp file
            out_pdf = Path(td) / "out.pdf"
            cmd = [
                "ocrmypdf",
                "--force-ocr",
                "--optimize", "0",
                "--sidecar", str(sidecar),
                "-l", langs,
                str(p),
                str(out_pdf),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            if sidecar.exists():
                text = sidecar.read_text(encoding="utf-8", errors="replace")
                return [ExtractedPage(page_number=1, text=text, source_path=str(p))]
    except Exception:
        return None
    return None


def ocr_if_needed(results: Sequence[ExtractResult], *, backend: str = "ocrmypdf", langs: str = "eng") -> List[OcrResult]:
    """Deterministic OCR wrapper with graceful fallback.

    - If an ExtractResult has zero pages and backend is ocrmypdf, attempt OCR.
    - Otherwise, pass-through pages and attach placeholder stats.
    """
    out: List[OcrResult] = []
    for r in results:
        if len(r.pages) == 0 and backend == "ocrmypdf":
            # Attempt OCR only when we know the source path
            pages = _run_ocrmypdf(r.source_path, langs) or list(r.pages)
            out.append(OcrResult(pages=pages, stats=OcrStats(backend=backend, langs=langs, avg_confidence=None)))
        else:
            out.append(OcrResult(pages=list(r.pages), stats=OcrStats(backend="none", langs=langs, avg_confidence=None)))
    return out
