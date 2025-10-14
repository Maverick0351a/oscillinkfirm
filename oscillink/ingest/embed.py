from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
import hashlib
import os
from typing import List, Sequence

from .models_registry import ModelSpec, get_model_spec

def _verify_model_hash_dir(model_dir: str, expected_sha256: str) -> bool:
    """Verify a directory of model files matches expected sha prefix.

    We hash all *.bin/*.safetensors/*.json files in sorted order.
    """
    try:
        import hashlib
        import os
        from pathlib import Path

        base = Path(model_dir)
        h = hashlib.sha256()
        for root, _dirs, files in os.walk(base):
            for f in sorted(files):
                if f.endswith((".bin", ".safetensors", ".json")):
                    fp = Path(root) / f
                    try:
                        with open(fp, "rb") as fh:
                            while True:
                                chunk = fh.read(1 << 20)
                                if not chunk:
                                    break
                                h.update(chunk)
                    except Exception:
                        continue
        digest = h.hexdigest().lower()
        return digest.startswith(str(expected_sha256).lower()[:12])
    except Exception:
        return False


@dataclass(frozen=True)
class EmbeddingModel:
    spec: ModelSpec
    dim: int = 64

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Compute embeddings for input texts.

        Uses sentence-transformers when available; falls back to a deterministic
        hash-based embedding to preserve testability/offline determinism.
        """
        # Try sentence-transformers first (optional dependency)
        st_name_or_path = self.spec.path or self.spec.name
        st_model = _load_st_model(st_name_or_path)
        if st_model is not None:  # pragma: no cover - exercised in environments with ST installed
            try:
                vecs = st_model.encode(
                    list(texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                # Convert to plain lists for JSON friendliness downstream
                return [[float(x) for x in row] for row in vecs.tolist()]
            except Exception:
                # Fall back to deterministic stub on any runtime error
                pass

        # Deterministic toy embedding: SHA256 digest expanded to dim floats in [0,1]
        out: List[List[float]] = []
        for t in texts:
            d = hashlib.sha256(t.encode("utf-8", errors="replace")).digest()
            needed = self.dim
            vals: List[float] = []
            i = 0
            while len(vals) < needed:
                b = d[i % len(d)]
                vals.append((b + 1) / 257.0)
                i += 1
            out.append(vals)
        return out


def load_embedding_model(name: str) -> EmbeddingModel:
    spec = get_model_spec(name)
    # Hash enforcement is opt-in to keep local/dev/test flows lightweight and offline-friendly.
    # Enable strict verification by setting OSCILLINK_EMBED_STRICT_HASH=1/true/on/yes
    if spec.path and spec.sha256_weights:
        strict = os.getenv("OSCILLINK_EMBED_STRICT_HASH", "0").lower() in {"1", "true", "on", "yes"}
        if strict:
            ok = _verify_model_hash_dir(spec.path, spec.sha256_weights)
            if not ok:
                # Raise a clear error to prevent silent drift when strict mode is enabled
                raise RuntimeError(
                    f"Embedding weights hash mismatch for {spec.name} at {spec.path} (expected prefix {spec.sha256_weights[:12]})"
                )
    dim = spec.dim if isinstance(spec.dim, int) and spec.dim > 0 else 64
    return EmbeddingModel(spec=spec, dim=dim)


@lru_cache(maxsize=2)
def _load_st_model(model_name: str):
    """Lazily load a sentence-transformers model if the package is present.

    Returns the model instance or None when unavailable. Kept private to avoid
    importing heavy deps unless actually requested at runtime.
    """
    try:  # pragma: no cover - optional dependency path
        if find_spec("sentence_transformers") is None:
            return None
        st_mod = import_module("sentence_transformers")
        SentenceTransformer = getattr(st_mod, "SentenceTransformer", None)
        if SentenceTransformer is None:
            return None
        # Allow local path override via registry by passing through the name which may be a path
        return SentenceTransformer(model_name)
    except Exception:
        return None
