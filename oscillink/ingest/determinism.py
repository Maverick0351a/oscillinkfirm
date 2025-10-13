from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional in minimal envs
    torch = None  # type: ignore[assignment]

# Reuse core determinism helpers for thread pinning/snapshot
try:
    from oscillink.core.determinism import pin_threads as _pin_threads
    from oscillink.core.determinism import snapshot_env as _snapshot_env
except Exception:  # pragma: no cover - fallback if core changes
    _pin_threads = None  # type: ignore[assignment]
    def _snapshot_env() -> Dict[str, str]:
        keys = [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]
        return {k: os.getenv(k, "") for k in keys}


def _truthy(val: Optional[str]) -> bool:
    return val in {"1", "true", "TRUE", "on", "On", "yes", "YES"}


def _set_env_if_unset(key: str, value: str) -> None:
    if os.getenv(key) is None:
        os.environ[key] = value


def apply_global_determinism() -> Dict[str, Any]:
    """Apply deterministic settings when OSC_DETERMINISTIC=1.

    - Set thread env vars to 1 (unless already set)
    - Fix seeds: PYTHONHASHSEED, random, numpy, torch
    - Enable deterministic torch ops where available
    Returns a snapshot of determinism-relevant environment variables.
    """
    if not _truthy(os.getenv("OSC_DETERMINISTIC")):
        return {"enabled": False, "env": _snapshot_env()}

    # Threads: set only if not already pinned by the parent/control plane
    _set_env_if_unset("OMP_NUM_THREADS", "1")
    _set_env_if_unset("MKL_NUM_THREADS", "1")
    _set_env_if_unset("OPENBLAS_NUM_THREADS", "1")
    _set_env_if_unset("NUMEXPR_NUM_THREADS", "1")

    # Seeds and hash randomization
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    if torch is not None:
        try:
            torch.manual_seed(0)
            # Some builds may not have this attribute; guard via hasattr
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)
            # cudnn flags (safe even on CPU-only build)
            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                # Access attributes conditionally to avoid attribute errors on CPU-only builds
                if hasattr(torch.backends.cudnn, "deterministic"):
                    torch.backends.cudnn.deterministic = True
                if hasattr(torch.backends.cudnn, "benchmark"):
                    torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    return {"enabled": True, "env": _snapshot_env()}


class PinnedThreads:
    """Context manager to pin threads using core helper, if present."""

    def __init__(self, n: int = 1) -> None:
        self.n = n
        self._ctx = _pin_threads(self.n) if _pin_threads is not None else None

    def __enter__(self):
        if self._ctx is not None:
            return self._ctx.__enter__()
        return None

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False

# Back-compat alias (if any code references the snake_case name during refactor)
pinned_threads = PinnedThreads


# Apply determinism eagerly if requested, so importers don't have to remember.
_DETERMINISM_STATE = apply_global_determinism()
