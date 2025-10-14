from __future__ import annotations
# ruff: noqa: I001

from pathlib import Path

from oscillink.ingest.extract import ExtractResult, ExtractedPage
from oscillink.ingest.ocr import ocr_if_needed


def test_ocr_sidecar_splits_pages(monkeypatch, tmp_path: Path) -> None:
    # Create a dummy PDF file path
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%...mock...")

    # Prepare an ExtractResult with zero pages to trigger OCR
    er = ExtractResult(pages=[], parser="plain", source_path=str(pdf_path))

    # Monkeypatch the internal _run_ocrmypdf to simulate sidecar with two pages
    from oscillink.ingest import ocr as ocr_mod

    def fake_run_ocrmypdf(input_path: str, langs: str):
        # Return two pages as if sidecar had two form-feed separated pages
        base_meta = {"created_at": 0, "modified_at": 0, "source_type": "pdf", "mimetype": "application/pdf"}
        p1 = ExtractedPage(page_number=1, text="Page One", source_path=input_path, meta={**base_meta, "page": 1, "page_or_row": 1})
        p2 = ExtractedPage(page_number=2, text="Page Two", source_path=input_path, meta={**base_meta, "page": 2, "page_or_row": 2})
        return [p1, p2]

    monkeypatch.setattr(ocr_mod, "_run_ocrmypdf", fake_run_ocrmypdf)

    out = ocr_if_needed([er], backend="ocrmypdf", langs="eng")
    assert out and out[0].pages
    pages = out[0].pages
    assert len(pages) == 2
    assert pages[0].page_number == 1 and pages[0].text == "Page One"
    assert pages[1].page_number == 2 and pages[1].text == "Page Two"
    # Ensure metadata includes page and page_or_row
    assert pages[0].meta and pages[0].meta.get("page") == 1 and pages[0].meta.get("page_or_row") == 1
    assert pages[1].meta and pages[1].meta.get("page") == 2 and pages[1].meta.get("page_or_row") == 2
