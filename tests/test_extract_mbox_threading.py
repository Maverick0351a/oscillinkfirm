from __future__ import annotations

from pathlib import Path

from oscillink.ingest.extract import extract_text


def _eml(subject: str, msg_id: str, refs: str | None, body: str, attach_name: str | None = None) -> str:
    boundary = "BOUND"
    parts = [
        f"Subject: {subject}",
        "From: Alice <alice@example.com>",
        "To: Bob <bob@example.com>",
        f"Message-ID: {msg_id}",
        (f"References: {refs}" if refs else None),
        "MIME-Version: 1.0",
        f"Content-Type: multipart/mixed; boundary={boundary}",
        "",
        f"--{boundary}",
        "Content-Type: text/plain; charset=utf-8",
        "",
        body,
    ]
    if attach_name:
        parts += [
            f"--{boundary}",
            "Content-Type: application/octet-stream",
            f"Content-Disposition: attachment; filename={attach_name}",
            "",
            "<bytes>",
        ]
    parts += [f"--{boundary}--", ""]
    return "\n".join([p for p in parts if p is not None])


def test_mbox_two_messages_threading(tmp_path: Path) -> None:
    # Build a minimal mbox file with two related messages
    mbox_path = tmp_path / "test.mbox"
    m1 = _eml("Subject A", "<id-1@x>", None, "Hello")
    m2 = _eml("Re: Subject A", "<id-2@x>", "<id-1@x>", "Reply", attach_name="notes.pdf")
    # mbox format expects messages separated with 'From ' lines; use simple separators
    mbox_content = "".join([
        "From alice@example.com Sat Jan  1 00:00:00 2022\n",
        m1,
        "From alice@example.com Sat Jan  1 00:00:01 2022\n",
        m2,
    ])
    mbox_path.write_text(mbox_content, encoding="utf-8")

    res = extract_text([str(mbox_path)], parser="auto")
    assert res and res[0].pages
    pages = res[0].pages
    assert len(pages) == 2
    # Natural mbox order must be preserved
    assert pages[0].meta and pages[0].meta.get("message_index") == 1 and pages[0].meta.get("page_or_row") == "msg:1"
    assert pages[1].meta and pages[1].meta.get("message_index") == 2 and pages[1].meta.get("page_or_row") == "msg:2"
    # Same thread_id
    tid1 = pages[0].meta.get("email.thread_id")  # type: ignore[assignment]
    tid2 = pages[1].meta.get("email.thread_id")  # type: ignore[assignment]
    assert tid1 and tid2 and tid1 == tid2
    # email.seq should be ascending within thread
    assert pages[0].meta.get("email.seq") == 1  # type: ignore[assignment]
    assert pages[1].meta.get("email.seq") == 2  # type: ignore[assignment]
    # Attachment captured on second message
    at = pages[1].meta.get("attachments")  # type: ignore[assignment]
    assert isinstance(at, list) and "notes.pdf" in at
