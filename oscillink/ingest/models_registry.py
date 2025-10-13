from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    name: str
    revision: str | None
    sha256_weights: str | None
    sha256_tokenizer: str | None
    license: str | None
    dim: int | None = None
    path: str | None = None  # local path, if provided


DEFAULT_MODELS: Dict[str, ModelSpec] = {
    "bge-small-en-v1.5": ModelSpec(
        name="bge-small-en-v1.5",
        revision=None,
        sha256_weights=None,
        sha256_tokenizer=None,
        license="MIT",
    ),
}


def _registry_path() -> Path:
    home = Path(os.path.expanduser("~"))
    return home / ".oscillink" / "models.json"


def _repo_registry_path() -> Path:
    # Look for a repo-root file named models_registry.json
    return Path.cwd() / "models_registry.json"


def load_registry(path: Optional[str | Path] = None) -> Dict[str, ModelSpec]:
    # precedence: explicit path > repo-root models_registry.json > user home registry
    if path is not None:
        p = Path(path)
    else:
        repo_p = _repo_registry_path()
        p = repo_p if repo_p.exists() else _registry_path()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out: Dict[str, ModelSpec] = {}
            for name, meta in data.items():
                # Support both simple format (path/dim/sha256/license) and detailed keys
                dim = meta.get("dim")
                sha = meta.get("sha256") or meta.get("sha256_weights")
                out[name] = ModelSpec(
                    name=name,
                    revision=meta.get("revision"),
                    sha256_weights=sha,
                    sha256_tokenizer=meta.get("sha256_tokenizer"),
                    license=meta.get("license") or meta.get("licence"),
                    dim=int(dim) if isinstance(dim, int) else None,
                    path=meta.get("path"),
                )
            return out
        except Exception:
            # Fall back to defaults on parse errors
            return dict(DEFAULT_MODELS)
    return dict(DEFAULT_MODELS)


def get_model_spec(name: str, *, registry_path: Optional[str | Path] = None) -> ModelSpec:
    reg = load_registry(registry_path)
    if name in reg:
        return reg[name]
    # Unknown model: return a generic placeholder with unknown license
    return ModelSpec(name=name, revision=None, sha256_weights=None, sha256_tokenizer=None, license=None)
