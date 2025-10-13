from __future__ import annotations

import json
from pathlib import Path

from oscillink.ingest.cli import build_parser, cmd_ingest, cmd_query


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    tmp_dir = repo / "_tmp_demo"
    tmp_dir.mkdir(exist_ok=True)

    # Create a tiny input file
    sample = tmp_dir / "sample.txt"
    sample.write_text("""Oscillink is a coherent memory engine.\nEmbedding models provide anchors.\nThis is a simple sample document.""", encoding="utf-8")

    index_path = str(tmp_dir / "demo_index.jsonl")

    # Ingest
    ingest_args = build_parser().parse_args([
        "ingest",
        "--input", str(sample),
        "--index-out", index_path,
        "--embed-model", "bge-small-en-v1.5",
        "--index-backend", "jsonl",
    ])
    rc = cmd_ingest(ingest_args)
    if rc != 0:
        print("Ingest failed", rc)
        return rc

    # Query end-to-end (recall -> settle) so we get a settle receipt
    query_args = build_parser().parse_args([
        "query",
        "--index", index_path,
        "--q", "What is Oscillink?",
        "--embed-model", "bge-small-en-v1.5",
        "--index-backend", "jsonl",
        "--e2e",
    ])
    # Capture printed JSON result
    import io
    import sys
    buf = io.StringIO()
    _stdout = sys.stdout
    try:
        sys.stdout = buf
        rc2 = cmd_query(query_args)
    finally:
        sys.stdout = _stdout
    if rc2 != 0:
        print("Query failed", rc2)
        return rc2

    out = json.loads(buf.getvalue())
    receipt = out.get("settle_receipt", {})
    emb = receipt.get("meta", {}).get("embedding", {})
    print("Embedding receipt block:", json.dumps(emb, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
