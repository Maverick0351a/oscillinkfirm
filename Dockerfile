# Minimal, hardened image for Oscillink on-prem query server
# Pinned digest recommended in CI; keeping tag here for readability
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# System deps (curl for healthcheck/debug), then drop unnecessary caches
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files needed to run the query server
COPY pyproject.toml README.md ./
COPY oscillink ./oscillink
COPY examples/query_server.py ./examples/query_server.py
COPY tools ./tools
COPY cloud/entrypoint_ingest.sh ./entrypoint.sh

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

# Install with cloud extras for FastAPI/uvicorn, without dev/test deps
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[cloud]" \
    && python -m pip install .

# Expose port and set non-root
EXPOSE 8080
USER 10001:10001

# Set read-only FS at runtime; use a writable tmpfs for uvicorn workers
VOLUME ["/tmp"]

# Default command binds to 0.0.0.0; consider binding to 127.0.0.1 behind a reverse proxy
# Licensed mode: entrypoint verifies license before launching the server.
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "examples.query_server:app", "--host", "0.0.0.0", "--port", "8080"]
