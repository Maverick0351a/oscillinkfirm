from __future__ import annotations

import hashlib
from pathlib import Path

from oscillink.ingest.embed import _verify_model_hash_dir


def _dir_sha256(files: list[tuple[str, bytes]]) -> str:
    h = hashlib.sha256()
    for _name, content in sorted(files):
        h.update(content)
    return h.hexdigest()


def test_verify_model_hash_dir_positive(tmp_path: Path):
    # Create a fake model directory with files we hash (*.bin/*.json are included)
    files = [
        ("a.bin", b"AAA"),
        ("b.safetensors", b"BBB"),
        ("config.json", b"{\n}\n"),
        ("notes.txt", b"ignored"),
    ]
    for name, content in files:
        (tmp_path / name).write_bytes(content)
    # Our helper walks os.walk and updates in sorted order; mirror that for expected hash
    expected = _dir_sha256([(n, c) for n, c in files if n.endswith((".bin", ".safetensors", ".json"))])
    # Use full hash; helper checks prefix case-insensitively
    assert _verify_model_hash_dir(str(tmp_path), expected)
    # Also validate short prefix matching (12 chars)
    assert _verify_model_hash_dir(str(tmp_path), expected[:12])


def test_verify_model_hash_dir_negative(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"AAA")
    wrong = hashlib.sha256(b"DIFFERENT").hexdigest()
    assert _verify_model_hash_dir(str(tmp_path), wrong) is False
