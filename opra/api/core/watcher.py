from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Optional

# Reuse existing ingest function to avoid duplication
from ..routers.ingest import IngestRequest, api_ingest


class Watcher:
    """Polling-based filesystem watcher to auto-ingest docs into indices.

    It scans a docs directory for supported files and builds/updates
    JSONL indices under an index directory. A manifest is persisted to
    track last processed modification times.
    """

    SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx"}

    def __init__(
        self,
        docs_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        receipts_dir: Optional[str] = None,
        embed_model: str = "bge-small-en-v1.5",
        interval_sec: float = 2.0,
    ) -> None:
        self.docs_dir = Path(docs_dir or os.getenv("OPRA_DOCS_DIR", "/data/docs")).resolve()
        self.index_dir = Path(index_dir or os.getenv("OPRA_INDEX_DIR", "/data/index")).resolve()
        self.receipts_dir = Path(receipts_dir or os.getenv("OPRA_RECEIPTS_DIR", "/data/receipts")).resolve()
        self.embed_model = embed_model
        self.interval_sec = max(0.5, float(interval_sec))

        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.receipts_dir / "index_manifest.json"
        self._manifest: Dict[str, float] = self._load_manifest()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _load_manifest(self) -> Dict[str, float]:
        try:
            if self.manifest_path.exists():
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # Coerce values to float
                    return {k: float(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_manifest(self) -> None:
        try:
            tmp = self.manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._manifest, sort_keys=True), encoding="utf-8")
            tmp.replace(self.manifest_path)
        except Exception:
            # Best-effort persistence
            pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="opra-watcher", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 3.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception:
                # Keep running even if one iteration fails
                pass
            finally:
                self._stop.wait(self.interval_sec)

    def scan_once(self) -> int:
        """Scan docs_dir, ingest changed/new files. Returns count processed."""
        processed = 0
        for p in sorted(self.docs_dir.glob("**/*")):
            if not p.is_file() or p.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            key = str(p)
            prev = self._manifest.get(key, 0.0)
            if mtime <= prev:
                continue
            out = self.index_dir / (p.stem + ".jsonl")
            # Perform ingest
            api_ingest(IngestRequest(path=str(p), embed_model=self.embed_model, index_out=str(out)))
            self._manifest[key] = mtime
            processed += 1
        if processed:
            self._save_manifest()
        return processed

    @property
    def status(self) -> Dict[str, object]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "docs_dir": str(self.docs_dir),
            "index_dir": str(self.index_dir),
            "receipts_dir": str(self.receipts_dir),
            "tracked_files": len(self._manifest),
            "interval_sec": self.interval_sec,
            "embed_model": self.embed_model,
        }
# TODO: watchfiles/watchdog to watch /data/docs and enqueue ingest tasks
