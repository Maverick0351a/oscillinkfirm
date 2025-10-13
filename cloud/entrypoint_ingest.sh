#!/usr/bin/env bash
set -euo pipefail

LICENSE_PATH="${OSCILLINK_LICENSE_PATH:-}"
JWKS_URL="${OSCILLINK_JWKS_URL:-}"

if [[ -z "${LICENSE_PATH:-}" ]] || [[ ! -f "${LICENSE_PATH}" ]]; then
  echo "ERROR: License file not found at ${LICENSE_PATH:-<unset>}" >&2
  exit 90
fi
if [[ -z "${JWKS_URL:-}" ]]; then
  echo "ERROR: JWKS URL is not set (OSCILLINK_JWKS_URL)" >&2
  exit 91
fi

# Verify license; writes entitlements to /run/oscillink_entitlements.json and env to /run/oscillink_entitlements.env
python /app/tools/license_verify.py --license "${LICENSE_PATH}" --jwks "${JWKS_URL}" || exit 92

# Export entitlements into env for downstream tools (limits, tiers, etc.)
if [[ -f /run/oscillink_entitlements.env ]]; then
  set -a
  source /run/oscillink_entitlements.env
  set +a
fi

# Now exec whatever command the user provided (e.g., python -m oscillink.ingest.cli ingest ...)
exec "$@"
