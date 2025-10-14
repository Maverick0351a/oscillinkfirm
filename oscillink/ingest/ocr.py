from __future__ import annotations
# ruff: noqa: I001

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import subprocess
import tempfile
from shutil import which

from .extract import ExtractedPage, ExtractResult
from .extract import _file_meta, _guess_mimetype


@dataclass(frozen=True)
class OcrStats:
    backend: str
    langs: str
    avg_confidence: float | None


@dataclass(frozen=True)
class OcrResult:
    pages: List[ExtractedPage]
    stats: OcrStats

def _tsv_avg_conf(tsv_text: str) -> float | None:
    vals: List[float] = []
    try:
        lines = tsv_text.splitlines()
        if not lines:
            return None
        # Skip header
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) > 10:
                try:
                    c = int(parts[10])
                except Exception:
                    continue
                if c >= 0:
                    vals.append(c / 100.0)
    except Exception:
        return None
    if not vals:
        return None
    return float(sum(vals) / len(vals))

def _run_tesseract_tsv(img_path: Path, *, oem: int = 1, psm: int = 6, langs: str = "eng") -> float | None:
    if which("tesseract") is None:
        return None
    try:
        cmd = ["tesseract", str(img_path), "stdout", "--oem", str(oem), "--psm", str(psm), "-l", str(langs), "tsv"]
        proc = subprocess.run(cmd, check=True, capture_output=True)
        tsv_text = proc.stdout.decode("utf-8", errors="replace")
        return _tsv_avg_conf(tsv_text)
    except Exception:
        return None


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
                # Split sidecar on form-feeds into per-page text (pdfminer convention)
                parts = text.split("\f") if "\f" in text else [text]
                base = _file_meta(p, source_type="pdf", mimetype=_guess_mimetype(p) or "application/pdf")
                pages: List[ExtractedPage] = []
                # Optional per-page OCR confidence via Tesseract TSV over the rendered page images
                page_conf: List[float | None] = []
                # Attempt to render single-page images using ocrmypdf temp output if available is out of scope;
                # fall back to running TSV directly on sidecar text is not possible, so leave None by default.
                for i, t in enumerate(parts, start=1):
                    t2 = t.strip()
                    if not t2:
                        continue
                    meta = {**base, "page": i, "page_or_row": f"page:{i}"}
                    # Best-effort: if we can locate a per-page image (p.png) in the temp dir, compute TSV conf
                    img_guess = Path(td) / f"page-{i}.png"
                    conf = _run_tesseract_tsv(img_guess, langs=langs) if img_guess.exists() else None
                    if conf is not None:
                        meta["ocr_page_confidence"] = float(conf)
                    page_conf.append(conf)
                    pages.append(ExtractedPage(page_number=i, text=t2, source_path=str(p), meta=meta))
                return pages or [ExtractedPage(page_number=1, text=text, source_path=str(p), meta={**base, "page": 1, "page_or_row": "page:1"})]
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
            # Compute document-level avg confidence from per-page values if present
            confs: List[float] = []
            for p in pages:
                try:
                    c = p.meta.get("ocr_page_confidence") if isinstance(p.meta, dict) else None
                    if c is not None:
                        confs.append(float(c))
                except Exception:
                    pass
            avg_conf = (sum(confs) / len(confs)) if confs else None
            out.append(OcrResult(pages=pages, stats=OcrStats(backend=backend, langs=langs, avg_confidence=avg_conf)))
        else:
            out.append(OcrResult(pages=list(r.pages), stats=OcrStats(backend="none", langs=langs, avg_confidence=None)))
    return out
