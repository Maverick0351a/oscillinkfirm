from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import pytest

from oscillink.api.license import require_license


@contextmanager
def env_var(key: str, value: str) -> Iterator[None]:
    prev = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
def test_require_license_enforced_and_optional() -> None:
    # Enforced missing token should raise SystemExit
    with env_var("OSCILLINK_LICENSE_OPTIONAL", "0"), pytest.raises(SystemExit):
        require_license(None)
    # Optional missing token should not raise
    with env_var("OSCILLINK_LICENSE_OPTIONAL", "1"):
        require_license(None)


def test_require_license_time_bounds() -> None:
    # Create a minimal JSON token with nbf/exp in window
    import time

    now = int(time.time())
    token = {"nbf": now - 1, "exp": now + 60}
    with env_var("OSCILLINK_LICENSE_OPTIONAL", "0"):
        # Should not raise
        require_license(str(token).replace("'", '"'))
