from __future__ import annotations

from pathlib import Path

from oscillink.ingest.extract import extract_text


def test_html_basic_headings_and_title(tmp_path: Path) -> None:
    html = """
    <html>
      <head><title>Sample Page</title></head>
      <body>
        <h1>Welcome</h1>
        <p>Intro paragraph.</p>
        <h2>Details</h2>
        <p>More info.</p>
      </body>
    </html>
    """
    p = tmp_path / "a.html"
    p.write_text(html, encoding="utf-8")
    res = extract_text([str(p)], parser="auto")
    assert res and res[0].pages
    page = res[0].pages[0]
    assert "# Welcome" in page.text
    assert "## Details" in page.text
    assert "Intro paragraph." in page.text
    assert page.meta is not None
    assert page.meta.get("title") == "Sample Page"
    assert page.meta.get("mimetype") == "text/html"
    assert page.meta.get("source_type") == "html"


def test_eml_headers_and_attachments(tmp_path: Path) -> None:
    # Construct a simple EML with plain text and one attachment (name only)
    eml = (
        "From: Alice <alice@example.com>\n"
        "To: Bob <bob@example.com>\n"
        "Subject: Greetings\n"
        "Date: Wed, 01 Jan 2025 00:00:00 +0000\n"
        "MIME-Version: 1.0\n"
        "Content-Type: multipart/mixed; boundary=BOUND\n\n"
        "--BOUND\n"
        "Content-Type: text/plain; charset=utf-8\n\n"
        "Hello Bob.\n\n"
        "--BOUND\n"
        "Content-Type: application/octet-stream\n"
        "Content-Disposition: attachment; filename=report.pdf\n\n"
        "<bytes>\n"
        "--BOUND--\n"
    )
    p = tmp_path / "m.eml"
    p.write_text(eml, encoding="utf-8")
    res = extract_text([str(p)], parser="auto")
    assert res and res[0].pages
    page = res[0].pages[0]
    assert "Hello Bob." in page.text
    assert page.meta is not None
    assert page.meta.get("subject") == "Greetings"
    assert "Alice" in page.meta.get("from", "")
    assert "Bob" in page.meta.get("to", "")
    # Attachment filename captured
    at = page.meta.get("attachments")
    assert isinstance(at, list) and "report.pdf" in at
    assert page.meta.get("source_type") == "eml"
