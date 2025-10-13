from __future__ import annotations
# ruff: noqa: I001

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import urllib.error
import urllib.parse
import urllib.request


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    source_path: str


@dataclass(frozen=True)
class ExtractResult:
    pages: List[ExtractedPage]
    parser: str  # e.g., "tika", "pdfminer", "plain"
    source_path: str


def _extract_with_tika(path: Path, tika_url: str) -> Optional[List[ExtractedPage]]:
    """Call a Tika server to extract plain text. Returns pages or None on failure.

    We issue a PUT to the Tika endpoint with Accept: text/plain and treat the
    entire response as a single page. Uses stdlib urllib to avoid hard deps.
    """
    try:
        url = tika_url.rstrip("/")
        # Common Tika endpoints: "/tika" (auto-detect). We'll accept either a base or a full path.
        if not url.endswith("/tika"):
            url = url + "/tika"
        req = urllib.request.Request(
            url,
            data=path.read_bytes(),
            headers={"Accept": "text/plain"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # nosec - user-provided URL, documented feature
            data = resp.read()
        text = data.decode("utf-8", errors="replace")
        pages = [ExtractedPage(page_number=1, text=text, source_path=str(path))]
        return pages
    except Exception:
        return None


def _extract_with_pdfminer(path: Path) -> Optional[List[ExtractedPage]]:
    """Use pdfminer.six when available to extract text. Returns pages or None on failure.

    We attempt to split pages using form-feed characters if present.
    """
    try:
        from pdfminer.high_level import extract_text  # type: ignore[import-not-found]

        text = extract_text(str(path))
        if text is None:
            return None
        # pdfminer often separates pages with \f
        parts = text.split("\f")
        pages: List[ExtractedPage] = []
        for i, t in enumerate(parts, start=1):
            t2 = t.strip()
            if not t2:
                continue
            pages.append(ExtractedPage(page_number=i, text=t2, source_path=str(path)))
        if not pages and text:
            pages = [ExtractedPage(page_number=1, text=text, source_path=str(path))]
        return pages
    except Exception:
        return None


def extract_text(paths: Iterable[str | Path], *, parser: str = "plain", tika_url: Optional[str] = None) -> List[ExtractResult]:
    """Deterministic text extraction stub.

    For now, provides a trivial, deterministic implementation:
    - If a path is a .txt file, read as a single page.
    - If parser is 'tika' and a tika_url is provided, call Tika server.
    - If parser is 'pdfminer' and pdfminer.six is installed, use it.
    - If parser is 'auto', try txt -> tika -> pdfminer, then fallback empty.
    - Otherwise, return empty result; OCR step may handle it.

    This keeps the repo green until heavy deps (Tika, pdfminer) are added.
    """
    results: List[ExtractResult] = []
    for p in paths:
        sp = str(p)
        path = Path(p)
        used_parser = parser
        pages: List[ExtractedPage] | None = None

        # Always handle simple txt deterministically
        if path.suffix.lower() == ".txt" and path.exists():
            txt = path.read_text(encoding="utf-8", errors="replace")
            pages = [ExtractedPage(page_number=1, text=txt, source_path=sp)]
            used_parser = "plain"
        else:
            # Auto-detect flow prefers lightweight/local first
            if parser == "auto":
                # Try Tika when URL provided
                if tika_url:
                    pages = _extract_with_tika(path, tika_url)
                    used_parser = "tika" if pages is not None else used_parser
                if pages is None:
                    # Try pdfminer next
                    pages = _extract_with_pdfminer(path)
                    used_parser = "pdfminer" if pages is not None else used_parser
            elif parser == "tika" and tika_url:
                pages = _extract_with_tika(path, tika_url)
                used_parser = "tika"
            elif parser == "pdfminer":
                pages = _extract_with_pdfminer(path)
                used_parser = "pdfminer"

        if pages is None:
            pages = []
        results.append(ExtractResult(pages=pages, parser=used_parser, source_path=sp))
    return results
