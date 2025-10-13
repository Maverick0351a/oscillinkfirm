from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict

DETERMINISM_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def snapshot_env() -> Dict[str, str]:
    """Return a snapshot of determinism-relevant environment variables."""
    return {k: os.getenv(k, "") for k in DETERMINISM_ENV_VARS}


@contextmanager
def pin_threads(n: int = 1):
    """Context manager to pin BLAS/OMP threads for more reproducible runs."""
    n_str = str(max(1, int(n)))
    prev = {k: os.getenv(k) for k in DETERMINISM_ENV_VARS}
    try:
        for k in DETERMINISM_ENV_VARS:
            os.environ[k] = n_str
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
