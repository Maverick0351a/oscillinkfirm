from __future__ import annotations

from pathlib import Path

from oscillink.ingest.extract import extract_text


def _assert_common_meta(meta: dict, *, source_type: str) -> None:
    assert isinstance(meta.get("created_at"), int)
    assert isinstance(meta.get("modified_at"), int)
    assert meta.get("source_type") == source_type
    # page_or_row is present and is a positive int
    por = meta.get("page_or_row")
    # Accept int (legacy) or normalized string markers like 'section:1','row:2','msg:3'
    if isinstance(por, int):
        assert por >= 1
    else:
        assert isinstance(por, str) and any(por.startswith(prefix) for prefix in ("section:", "page:", "row:", "sheet:", "msg:"))


def test_txt_extraction_meta(tmp_path: Path) -> None:
    p = tmp_path / "sample.txt"
    p.write_text("Hello world", encoding="utf-8")

    res = extract_text([str(p)], parser="auto")
    assert res and res[0].pages
    pages = res[0].pages
    assert len(pages) == 1
    pg = pages[0]
    assert pg.page_number == 1
    assert pg.text == "Hello world"
    assert pg.meta is not None
    _assert_common_meta(pg.meta, source_type="txt")
    # mimetype should be text/plain for .txt
    assert pg.meta.get("mimetype") == "text/plain"


def test_csv_extraction_meta(tmp_path: Path) -> None:
    p = tmp_path / "rows.csv"
    p.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")

    res = extract_text([str(p)], parser="auto")
    assert res and res[0].pages
    pages = res[0].pages
    # Two data rows -> two pages
    assert len(pages) == 2
    # Row 1
    pg1 = pages[0]
    assert pg1.meta is not None
    _assert_common_meta(pg1.meta, source_type="csv")
    assert pg1.meta.get("mimetype") == "text/csv"
    assert pg1.meta.get("row_index") == 1
    assert pg1.page_number == 1
    # Row 2
    pg2 = pages[1]
    assert pg2.meta is not None
    _assert_common_meta(pg2.meta, source_type="csv")
    assert pg2.meta.get("row_index") == 2
    assert pg2.page_number == 2
