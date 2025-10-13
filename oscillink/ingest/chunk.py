from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import List, Sequence

from .extract import ExtractedPage


@dataclass(frozen=True)
class Chunk:
    source_path: str
    page_number: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ChunkResult:
    chunks: List[Chunk]
    ruleset: str


def _paragraph_spans(text: str) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        # skip leading newlines
        while i < n and text[i] in "\r\n":
            i += 1
        if i >= n:
            break
        # start of paragraph
        start = i
        # consume until blank line
        while i < n:
            if text[i] in "\r\n":
                # check if next non-newline is also newline => blank line
                j = i
                # consume one line break
                if text[j] == "\r" and j + 1 < n and text[j + 1] == "\n":
                    j += 2
                else:
                    j += 1
                # peek next linebreaks
                k = j
                while k < n and text[k] in "\r\n":
                    k += 1
                if k > j:  # there was at least one more newline => blank
                    end = i
                    spans.append((start, end))
                    i = k
                    break
                else:
                    i = j
                    continue
            else:
                i += 1
        else:
            # reached EOF
            spans.append((start, n))
            break
    if not spans and text:
        spans.append((0, n))
    return spans


def chunk_paragraphs(pages: Sequence[ExtractedPage], *, ruleset: str = "paragraph") -> ChunkResult:
    chunks: List[Chunk] = []
    for page in pages:
        for start, end in _paragraph_spans(page.text):
            if end > start:
                chunks.append(
                    Chunk(
                        source_path=page.source_path,
                        page_number=page.page_number,
                        start=start,
                        end=end,
                        text=page.text[start:end],
                    )
                )
    # Deterministic ordering
    chunks.sort(key=lambda c: (c.source_path, c.page_number, c.start, c.end))
    return ChunkResult(chunks=chunks, ruleset=ruleset)


def chunk_unstructured(pages: Sequence[ExtractedPage], *, ruleset: str = "unstructured") -> ChunkResult:
    """Chunk using unstructured if available; fallback to paragraph chunking.

    Determinism: we preserve source order and assign chunks in stable order by
    (source_path, page_number, start, end). If unstructured is missing, we fall
    back to paragraph spans.
    """
    try:  # pragma: no cover - optional dependency path
        if find_spec("unstructured") is None:
            raise RuntimeError("unstructured not installed")
        # Defer import until needed
        from unstructured.partition.text import partition_text  # type: ignore[import-not-found]

        chunks: List[Chunk] = []
        for page in pages:
            if not page.text.strip():
                continue
            # Partition plain text page; we avoid I/O to keep things simple
            elements = partition_text(text=page.text)
            # Construct deterministic slices by locating element text in the page text
            cursor = 0
            for el in elements:
                t = str(getattr(el, "text", "")).strip()
                if not t:
                    continue
                # Find next occurrence deterministically from cursor
                pos = page.text.find(t, cursor)
                if pos == -1:
                    # fallback: simple append at end if not located
                    pos = max(cursor, 0)
                start, end = pos, pos + len(t)
                cursor = end
                chunks.append(
                    Chunk(
                        source_path=page.source_path,
                        page_number=page.page_number,
                        start=start,
                        end=end,
                        text=page.text[start:end],
                    )
                )
        chunks.sort(key=lambda c: (c.source_path, c.page_number, c.start, c.end))
        return ChunkResult(chunks=chunks, ruleset=ruleset)
    except Exception:
        # Fallback deterministically to paragraph chunking
        return chunk_paragraphs(pages, ruleset="paragraph")
