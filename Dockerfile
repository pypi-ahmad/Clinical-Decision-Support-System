# MediScan CDSS — production container image.
#
# Multi-stage build driven by uv and the uv_build backend declared in
# pyproject.toml. The runtime image carries only the resolved dependency
# set (no compilers, no git, no build tools).

# ---- builder stage --------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1

# System build deps for wheels that compile C extensions
# (Pillow, numpy, pdf2image bindings, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 1. Resolve the runtime layer (no dev deps) --------------------------
# Copy ONLY the manifest + lock first so Docker can cache the resolution
# layer independently of source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---- 2. Install the project itself (editable, so `import backend` works) -
COPY backend ./backend
COPY frontend ./frontend
COPY README.md ./
RUN uv sync --frozen --no-dev

# ---- runtime stage -------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    MEDISCAN_HOST=0.0.0.0 \
    MEDISCAN_PORT=8000

# Runtime system deps only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mediscan

# Copy the resolved site-packages + CLI scripts from the builder.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/backend /app/backend
COPY --from=builder /app/frontend /app/frontend
COPY --from=builder /app/README.md /app/README.md

# Make the venv's binaries first on PATH so `uvicorn` resolves correctly.
ENV PATH="/app/.venv/bin:${PATH}"

# Workspace directories.
RUN mkdir -p /data/uploads && chown -R mediscan:mediscan /app /data
USER mediscan

EXPOSE 8000

# Liveness probe: a 200 means the process is up.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${MEDISCAN_PORT}/health" || exit 1

# Stable, well-trodden production pattern: gunicorn + uvicorn worker. One
# worker per CPU is the typical default; operators can override via
# MEDISCAN_WORKERS.
CMD ["sh", "-c", "python -m uvicorn backend.main:app \
  --host ${MEDISCAN_HOST} --port ${MEDISCAN_PORT} \
  --proxy-headers --forwarded-allow-ip '*' \
  --workers ${MEDISCAN_WORKERS:-2}"]
