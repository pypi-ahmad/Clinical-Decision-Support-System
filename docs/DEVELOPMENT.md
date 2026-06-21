# Development

> **Purpose.** Conventions, tests, and CI/CD for contributors.

## Repository layout

```
backend/         FastAPI service + LangGraph + retrieval + observability
frontend/        Streamlit operator UI
tests/           Pytest suite (unit / integration / eval)
docs/            Handbook + focused reference docs
Dockerfile       Production image build
docker-compose.yml Local stack (backend + frontend + Qdrant + Ollama)
requirements*.txt  Pinned dependency manifests
pyproject.toml   Project metadata, ruff + mypy + pytest config
.pre-commit-config.yaml  Hooks: ruff, mypy
.github/workflows/ci.yml Lint + type-check + test + pip-audit
```

## Local setup

```bash
git clone https://github.com/pypi-ahmad/Clinical-Decision-Support-System.git
cd Clinical-Decision-Support-System

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt

export MEDISCAN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MRN_HMAC_PEPPER="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MEDISCAN_ALLOW_ANONYMOUS=1   # local dev only
```

The test suite (`tests/conftest.py`) sets these env vars automatically.

## Linting and formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for lint and format,
[MyPy](https://mypy-lang.org/) for types, and [pre-commit](https://pre-commit.com/)
to run them on every commit.

```bash
ruff check .                        # lint
ruff format --check .               # format check (CI runs this)
mypy backend/security.py backend/database.py backend/retrieval  # strict subset
pre-commit run --all-files          # full hook set
```

Configuration lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`,
`[tool.pytest.ini_options]`) and `.pre-commit-config.yaml`.

### Ignored rules (per-file)

| Rule | Why |
|---|---|
| `E501` (line length) | Handled by the formatter |
| `B008` (FastAPI defaults) | `Depends()` / `Form()` are by design |
| `N812` (lowercase import aliasing) | e.g. `import ollama as ollama` |
| `S110` (try/except/pass) | Documented audit-safe places |
| `S101` (assert) | Allowed in tests |
| `N802` / `N803` (naming) | Allowed in tests |
| All `S*` and `N*` | Allowed in tests |

## Tests

### Tier model

| Tier | Path | Skip rule | When to use |
|---|---|---|---|
| Unit | `tests/unit/` | Never | Pure logic, no I/O |
| Integration | `tests/integration/test_api_workflows.py` | Never | FastAPI `TestClient` against mocked services |
| Live | `tests/integration/test_live_pipeline.py` | Skips when Ollama / Qdrant / PaddleOCR are unreachable | Real end-to-end on running services |
| Eval | `tests/eval/test_extraction_quality.py` | Skips when `tests/eval/gold/*.json` is empty | Quality scoring on gold fixtures |

The test totals reported on the README badge are produced by:

```bash
python -m pytest -m "not live and not eval"  # 241 pass, 8 skip
```

### Running

```bash
python -m pytest                          # everything
python -m pytest tests/unit/              # unit only
python -m pytest tests/integration/       # integration (mocked + live)
python -m pytest -m live                  # live only
python -m pytest -m eval                  # quality harness
python -m pytest -k extraction_graph -v   # filter
python -m pytest -m "not live and not eval" --cov=backend --cov-report=term-missing
```

### What the integration suite covers

- Happy-path `/analyze` for direct, granular, and agentic modes
- Error paths: invalid `medical_json`, non-dict `/confirm` payload, oversized upload, wrong API key, missing `X-API-Key`
- Multi-page PDF rendering and OCR result aggregation
- Confidence-gate routing to the review queue
- Audit-event write-through

### What live tests cover

- Real Ollama OCR + structuring + reasoning on a sample PDF
- Real Qdrant index + search round-trip
- Multi-page document round-trip

## CI/CD

`.github/workflows/ci.yml` runs on every push to `main` / `master` and
on every pull request:

1. **lint** &mdash; `ruff check .` and `ruff format --check .`
2. **type-check** &mdash; `mypy` on the strict subset (security, database, retrieval)
3. **test** &mdash; `pytest -q -m "not live and not eval"` with the full dep set (excluding torch for the CPU runner)
4. **audit** &mdash; `pip-audit --strict -r requirements.txt` (non-blocking, `|| true`)

Concurrency: the latest push cancels any in-flight run for the same ref.

## Adding a new OCR backend

1. Implement `BaseOCRBackend.run` in a new file under
   `backend/ocr_backends/`. Return an `OCRResult` populated with at least
   `backend`, `model`, `ocr_mode`, `per_page_results`, and (when
   available) `bounding_boxes` (`backend/ocr_backends/base.py:38-50`).
2. Register the backend in `backend/ocr_backends/__init__.py:6-11`
   (extend `OCRBackendConfig.normalized_backend` if you add a new
   shortname &mdash; `backend/ocr_backends/base.py:64-72`).
3. If the backend is selectable from the Streamlit UI, extend the
   `selectbox` in `frontend/app.py:139-148` and the model mapping at
   `frontend/app.py:164-178`.
4. Add unit tests in `tests/unit/test_ocr_backends.py` covering the
   happy path, an empty-pages path, and any error handling.
5. Add a live test in `tests/integration/test_live_pipeline.py` that
   skips when the backend isn't reachable.

## Adding a new extraction node

1. Decide where the node belongs in the graph
   (`backend/workflows/extraction_graph.py:99-110`). Most new logic
   should be a pre- or post-processing step around an existing node.
2. Add a typed field to `ExtractionGraphState`
   (`backend/workflows/extraction_graph.py:49-82`).
3. Register the node in `build_extraction_graph` and add edges
   (`backend/workflows/extraction_graph.py:99-132`).
4. Add unit tests in `tests/unit/test_extraction_graph.py` covering
   the new field, the routing, and any error path.

## Adding a new LLM provider

1. Add a module-level client cache to `backend/ai_wrapper.py` (model
   after `_get_openai_client` at `backend/ai_wrapper.py:168`).
2. Add a `@_retry`-decorated provider function (model after
   `_call_openai` at `backend/ai_wrapper.py:201`).
3. Add a branch in `get_ai_response` (`backend/ai_wrapper.py:291-368`).
4. Update the `PROVIDER_MODELS` map in `frontend/app.py:32-45`.
5. Add unit tests in `tests/unit/test_ai_wrapper.py`.

## Local services

```bash
# Ollama (one-time)
ollama serve &
ollama pull glm-ocr
ollama pull glm-4.7-flash

# Qdrant (docker)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.12.4

# Full stack
docker compose up -d backend frontend ollama qdrant
```

## Release process

1. Bump `version` in `pyproject.toml` and add a `CHANGELOG.md` entry.
2. Update `requirements.lock.txt` if any range-pin changed:
   ```bash
   uv pip compile --python-version 3.11 --index-strategy unsafe-best-match \
     --generate-hashes --output-file requirements.lock.txt requirements.txt
   ```
3. Run the full pytest suite locally:
   ```bash
   python -m pytest -m "not live and not eval"
   ```
4. Open a PR; CI must be green before merge.
5. Tag the merge commit (`git tag vX.Y.Z && git push --tags`). The
   Docker image can be built with `docker build -t mediscan/cdss:X.Y.Z .`
   using the supplied `Dockerfile`.

## Where to read next

- [Handbook](HANDBOOK.md) &mdash; zero-to-hero walkthrough
- [ARCHITECTURE.md](ARCHITECTURE.md) &mdash; module map
- [SECURITY.md](SECURITY.md) &mdash; threat model
- [USAGE.md](../USAGE.md) &mdash; operator reference
- [TEST_REPORT.md](../TEST_REPORT.md) &mdash; current suite status
