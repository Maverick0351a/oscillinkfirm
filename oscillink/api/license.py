from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Header


def _truthy(val: Optional[str]) -> bool:
    return val in {"1", "true", "TRUE", "on", "On", "yes", "YES"}


def _b64url_decode(data: str) -> bytes:
    padding = '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _parse_token(token: str) -> Tuple[Dict[str, Any] | None, str]:
    """Parse a token that may be a JSON string or a JWT-like payload.

    Returns (claims, reason_if_invalid). Signature verification is not implemented here.
    """
    token = token.strip()
    # JSON token fallback (used by dev license_svc)
    if token.startswith("{") and token.endswith("}"):
        try:
            return json.loads(token), ""
        except Exception as e:  # pragma: no cover
            return None, f"invalid json token: {e}"
    # JWT-like (header.payload.signature) without verification
    parts = token.split(".")
    if len(parts) == 3:
        try:
            payload_raw = _b64url_decode(parts[1])
            return json.loads(payload_raw.decode("utf-8")), ""
        except Exception as e:  # pragma: no cover
            return None, f"invalid jwt token: {e}"
    return None, "unknown token format"


class LicenseEnforcer:
    def __init__(self, *, optional: bool = False) -> None:
        self.optional = bool(optional)

    def check(self, token: Optional[str]) -> Tuple[bool, Dict[str, Any] | None, str]:
        if not token:
            if self.optional:
                return True, None, "optional: no token provided"
            return False, None, "missing token"
        claims, reason = _parse_token(token)
        if claims is None:
            if self.optional:
                return True, None, f"optional: {reason}"
            return False, None, reason
        # Basic time validation if present
        now = int(time.time())
        nbf = int(claims.get("nbf", now))
        exp = int(claims.get("exp", now + 1))
        if now < nbf:
            return False, claims, "not yet valid"
        if now >= exp:
            return False, claims, "expired"
        return True, claims, "ok"


router = APIRouter(prefix="/license", tags=["license"])


@router.get("/status")
def license_status(x_license_key: Optional[str] = Header(default=None, alias="X-License-Key")) -> Dict[str, Any]:
    optional = _truthy(os.getenv("OSCILLINK_LICENSE_OPTIONAL", "1"))
    enforcer = LicenseEnforcer(optional=optional)
    ok, claims, reason = enforcer.check(x_license_key)
    return {"ok": bool(ok), "reason": reason, "claims": claims or {}}


def require_license(x_license_key: Optional[str]) -> None:
    optional = _truthy(os.getenv("OSCILLINK_LICENSE_OPTIONAL", "1"))
    enforcer = LicenseEnforcer(optional=optional)
    ok, _claims, reason = enforcer.check(x_license_key)
    if not ok:
        # In route functions, raise SystemExit to preserve CLI-like error codes; FastAPI will convert to 500 if uncaught.
        # For now, we simply raise a SystemExit with message; you may replace with HTTPException(401, ...) later.
        raise SystemExit(f"license denied: {reason}")
