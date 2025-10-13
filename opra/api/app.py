from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.watcher import Watcher
from .routers import chat as chat_router
from .routers import ingest as ingest_router
from .routers import query as query_router
from .routers import report as report_router

app = FastAPI(title="OPRA API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

# Mount minimal query router (wired into oscillink query service)
app.include_router(query_router.router, prefix="/v1")
app.include_router(ingest_router.router, prefix="/v1")
app.include_router(chat_router.router, prefix="/v1")
app.include_router(report_router.router, prefix="/v1")

# CORS: allow local UI on 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Background watcher: start if enabled via env
watcher: Watcher | None = None

@app.on_event("startup")
def _on_startup() -> None:
    global watcher
    if os.getenv("OPRA_WATCHER", "1") not in {"0", "false", "False"}:
        watcher = Watcher(
            docs_dir=os.getenv("OPRA_DOCS_DIR"),
            index_dir=os.getenv("OPRA_INDEX_DIR"),
            receipts_dir=os.getenv("OPRA_RECEIPTS_DIR"),
            embed_model=os.getenv("OPRA_EMBED_MODEL", "bge-small-en-v1.5"),
            interval_sec=float(os.getenv("OPRA_WATCH_INTERVAL", "2.0")),
        )
        watcher.start()


@app.on_event("shutdown")
def _on_shutdown() -> None:
    global watcher
    if watcher:
        watcher.stop()


@app.get("/v1/watcher/status")
def watcher_status() -> dict:
    if not watcher:
        return {"running": False}
    return watcher.status


@app.post("/v1/watcher/scan")
def watcher_scan() -> dict:
    if not watcher:
        # Run one-off scan with transient watcher
        w = Watcher(
            docs_dir=os.getenv("OPRA_DOCS_DIR"),
            index_dir=os.getenv("OPRA_INDEX_DIR"),
            receipts_dir=os.getenv("OPRA_RECEIPTS_DIR"),
            embed_model=os.getenv("OPRA_EMBED_MODEL", "bge-small-en-v1.5"),
            interval_sec=1.0,
        )
        n = w.scan_once()
        return {"processed": n, "running": False}
    return {"processed": watcher.scan_once(), "running": True}
