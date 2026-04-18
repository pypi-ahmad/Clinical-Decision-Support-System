# MediScan CDSS — production-ish container image.
# Multi-stage: builder installs deps, final stage is slim.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    poppler-utils \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Reproducible build: prefer the hashed lockfile when present, fall back to
# the range-pinned requirements.txt for first-time generation only.
COPY requirements.txt requirements.lock.txt* ./
RUN pip install --upgrade pip \
    && if [ -f requirements.lock.txt ]; then \
         pip install --require-hashes --no-deps -r requirements.lock.txt; \
       else \
         pip install -r requirements.txt; \
       fi


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEDISCAN_HOST=0.0.0.0 \
    MEDISCAN_PORT=8000

# Runtime system deps: poppler for pdf2image, libmagic for python-magic.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mediscan

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY backend ./backend
COPY frontend ./frontend
COPY pytest.ini pyproject.toml README.md ./

RUN mkdir -p /app/backend/uploads && chown -R mediscan:mediscan /app
USER mediscan

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${MEDISCAN_PORT}/health || exit 1

# gunicorn+uvicorn-worker is the stable production pattern; one worker per core.
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host ${MEDISCAN_HOST} --port ${MEDISCAN_PORT} --proxy-headers --forwarded-allow-ips '*' --workers ${MEDISCAN_WORKERS:-2}"]
