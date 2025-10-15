import os
import shutil
import tempfile
from pathlib import Path

import pytest

from oscillink.ingest.embed import load_embedding_model
from oscillink.ingest.models_registry import get_model_spec


@pytest.mark.skipif(os.getenv("OSCILLINK_EMBED_STRICT_HASH") != "1", reason="Enable OSCILLINK_EMBED_STRICT_HASH=1 to exercise negative path")
def test_embed_strict_hash_mismatch(monkeypatch):
    spec = get_model_spec("bge-small-en-v1.5")
    if not spec.path:
        pytest.skip("No local model path available in registry to test strict hashing")
    src = Path(spec.path)
    if not src.exists():
        pytest.skip("Model directory not present on this machine")
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / src.name
        shutil.copytree(src, dst)
        # corrupt one eligible file deterministically
        mutated = False
        for p in sorted(dst.rglob("*")):
            if p.suffix in (".bin", ".safetensors", ".json") and p.is_file():
                data = bytearray(p.read_bytes())
                if data:
                    data[0] = (data[0] + 1) % 256
                    p.write_bytes(bytes(data))
                    mutated = True
                    break
        if not mutated:
            pytest.skip("No suitable file to mutate in model dir")

        # Monkeypatch registry to point to mutated copy
        from oscillink.ingest import models_registry as mr

        orig_get = mr.get_model_spec

        def _fake(name: str):
            sp = orig_get(name)
            return type(sp)(
                name=sp.name,
                revision=sp.revision,
                sha256_weights=sp.sha256_weights,
                sha256_tokenizer=sp.sha256_tokenizer,
                license=sp.license,
                dim=sp.dim,
                path=str(dst),
            )

        monkeypatch.setattr(mr, "get_model_spec", _fake)
        monkeypatch.setenv("OSCILLINK_EMBED_STRICT_HASH", "1")
        with pytest.raises((RuntimeError, SystemExit)) as e:
            _ = load_embedding_model(name="bge-small-en-v1.5")
        msg = str(e.value).lower()
        assert "hash" in msg and "mismatch" in msg
