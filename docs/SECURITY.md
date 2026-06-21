# Security

> **Purpose.** Document the threat model and the code that defends it.
> Every claim below is grounded in a `file:line` reference in
> `backend/`.

## Threat model

MediScan OCR processes Protected Health Information (PHI). The
deployment is assumed to be:

- Multi-user on a trusted network, or single-tenant.
- Fronted by HTTPS at the edge.
- Configured with a real `MEDISCAN_API_KEY` and `MRN_HMAC_PEPPER`.

Adversaries considered:

| Adversary | Goal | Primary defense |
|---|---|---|
| Unauthenticated internet attacker | Read or alter clinical records | API-key auth on every data route |
| Authenticated low-privilege user | Read another tenant's data | API key is single-tenant; cross-tenant isolation is the operator's responsibility |
| Malicious upload | RCE, storage exhaustion, SSRF pivot | Magic-byte check, byte cap, sanitized filename, SSRF guard |
| SSRF via crafted outbound URL | Pivot to internal services | `validate_outbound_url` blocks private/loopback/link-local IPs |
| Prompt-injection via uploaded document | Hijack the structuring or reasoning LLM | Boundary-nonce delimiter + firewall clause + retrieval-time scrub |
| Provider leaking API key in error | Credential theft | `AIProviderError` scrubs `sk-` / `sk-ant-` substrings (`backend/ai_wrapper.py:66`) |
| Operator with shell access | Read residual PHI from freed DB pages | `PRAGMA secure_delete=ON` (`backend/database.py:63`) |
| Rainbow-table attack on MRN | Re-identify patients | HMAC-SHA256 with server-held pepper; no weak-hash fallback |

## Defense in depth

```mermaid
flowchart LR
    A[Request] --> B[API Key]
    B --> C[Upload Validation]
    C --> D[SSRF Guard]
    D --> E[OCR Backend]
    E --> F[Prompt Firewall]
    F --> G[LLM Provider]
    G --> H[Retrieval Firewall]
    H --> I[Persist + Audit]
    I --> J[Structured JSON]
    J --> K[PHI Redactor in Logs]
```

## Layer 1 &mdash; API-key auth

| Concern | Code |
|---|---|
| Read env var | `backend/security.py:23-25` |
| Enforce or 503 (server misconfigured) | `backend/security.py:35-42` |
| Constant-time compare | `backend/security.py:43-47` |
| Opt-in anonymous mode | `backend/security.py:36-38` |
| Refuse to start with placeholder | `backend/main.py:83-88` |

The Streamlit client reads `MEDISCAN_API_KEY` and attaches it to every
outbound call (`frontend/app.py:30`).

## Layer 2 &mdash; Upload validation

| Concern | Code |
|---|---|
| Filename sanitization (no path separators, control chars, Windows reserved names) | `backend/security.py:72-83` |
| Suffix allow-list (`.pdf`, `.png`, `.jpg`, `.jpeg`, optionally `.txt` for policies) | `backend/security.py:62, 86-104` |
| Magic-byte check on first 4 KiB | `backend/security.py:107-121` |
| Streaming write with byte cap (default 50 MiB) | `backend/security.py:124-172` |
| Pillow decompression-bomb guard | `backend/artifacts.py:21` |

## Layer 3 &mdash; SSRF guard

`backend/security.py:198-247` resolves every outbound URL through
`socket.getaddrinfo` and rejects:

- Multicast (`ip.is_multicast`)
- Reserved (`ip.is_reserved`)
- Loopback (unless `allow_loopback=True`)
- Private (`ip.is_private`, unless `allow_loopback=True`)
- Link-local

Used by:

- `backend/ocr_backends/service_client.py:66` &mdash; PaddleOCR-VL
  service client validates the URL eagerly.

## Layer 4 &mdash; Prompt-injection firewall

| Concern | Code |
|---|---|
| Random nonce generator | `backend/security.py:255-257` |
| Untrusted-content wrapper (`<<<UNTRUSTED_DOCUMENT_<nonce>_BEGIN>>>` / `_END>>>`) | `backend/security.py:273-285` |
| Firewall clause embedded in the system prompt | `backend/security.py:288-293` |
| Used by structuring (direct pipeline) | `backend/extract.py:36-43, 95-103` |
| Used by structuring (granular graph) | `backend/workflows/extraction_graph.py:305-319` |
| Used by reasoning | `backend/logic.py:101, 118` |
| Used by insurance | `backend/logic.py:147, 164` |
| Defense-in-depth scrub of retrieved text (strips `ignore previous instructions`, role prefixes, inner delimiter markers) | `backend/retrieval/chunking.py:33-38` |

A nonce mismatch in the inner delimiters should not be trusted &mdash;
the firewall clause explicitly says so.

## Layer 5 &mdash; LLM provider hardening

| Concern | Code |
|---|---|
| Tenacity retries, transient-only | `backend/ai_wrapper.py:79-121` |
| Per-provider timeouts (OpenAI, Anthropic, Gemini, Ollama) | `backend/ai_wrapper.py:168-191, 311` |
| Provider clients cached at module scope | `backend/ai_wrapper.py:168-198` |
| Secret scrub in errors (`sk-`, `sk-ant-`) | `backend/ai_wrapper.py:66-71` |
| Resilient JSON parser (no greedy regex) | `backend/ai_wrapper.py:387-492` |
| Force native JSON modes where supported (Ollama, OpenAI, Gemini) | `backend/ai_wrapper.py:142, 220, 244` |

## Layer 6 &mdash; PHI redaction in logs

| Concern | Code |
|---|---|
| Sensitive-key deny list (`mrn`, `dob`, `full_name`, `api_key`, `token`, ...) | `backend/logging_config.py:24-38` |
| Token scrub (Bearer, OpenAI, Anthropic) | `backend/logging_config.py:47-57` |
| Structlog processor | `backend/logging_config.py:65-72` |

> The redactor is pattern-based on `mrn` as a key name, not as a value
> pattern. A 6-12 alphanumeric run is too ambiguous (it matches model
> names, correlation IDs, and git SHAs) to scrub blindly. Callers must
> never log raw MRN values in free-text messages.

## Layer 7 &mdash; PHI at rest

| Concern | Code |
|---|---|
| WAL + `synchronous=NORMAL` | `backend/database.py:48-49` |
| `busy_timeout=5000` | `backend/database.py:53` |
| `foreign_keys=ON` | `backend/database.py:55` |
| `trusted_schema=OFF` (CVE-2020-9327 / 13434 mitigation) | `backend/database.py:58` |
| `secure_delete=ON` (zero freed pages) | `backend/database.py:63` |
| `temp_store=MEMORY` (no on-disk temp files for PHI) | `backend/database.py:66` |
| 256 KiB payload cap on `save_record` | `backend/database.py:27, 152-154` |
| MRN hashing (HMAC-SHA256 with `MRN_HMAC_PEPPER`) | `backend/retrieval/__init__.py:27-47` |
| Optional PII scrub before non-Ollama calls | `backend/pii_scrub.py:49-71` |

## Layer 8 &mdash; Process and operator hardening

| Concern | Code |
|---|---|
| Bind to loopback by default | `backend/main.py:834` |
| CORS allow-list (no wildcards) | `backend/main.py:164-178` |
| Rate limit (slowapi, opt-in) | `backend/main.py:139-161` |
| `/docs` and `/openapi.json` hidden by default | `backend/main.py:113-119` |
| Artifact path canonicalization (no traversal) | `backend/security.py:180-190` |
| PaddleOCR service URL is server-side only (`PADDLE_SERVICE_URL`) | `backend/main.py:241-253` |
| Per-request correlation ID (no log injection) | `backend/observability.py:146-202` |

## Audit trail

Every meaningful operation writes a row to `audit_events`:

- `analyze_complete` &mdash; `backend/main.py:458-465`
- `insurance_check` &mdash; `backend/main.py:575-582`
- `record_confirmed` &mdash; `backend/main.py:611-618`
- `human_review_required` &mdash; `backend/workflows/extraction_graph.py:493-502`

MRNs are stored as HMAC hashes; correlation IDs are stored verbatim so
log lines can be cross-referenced. Audit writes are best-effort and
never raise into the call path (`backend/database.py:213-215`).

## Operator checklist

Before promoting a build to production:

- [ ] `MEDISCAN_API_KEY` is a 32+ byte random secret (`python -c 'import secrets; print(secrets.token_urlsafe(32))'`).
- [ ] `MRN_HMAC_PEPPER` is a 32+ byte random secret, **different** from `MEDISCAN_API_KEY`.
- [ ] `MEDISCAN_ALLOW_ANONYMOUS` is **not** set.
- [ ] `MEDISCAN_ALLOW_USER_API_KEYS` is intentionally set or intentionally unset (and documented).
- [ ] `MEDISCAN_ENABLE_DOCS=0` unless you have a public docs site.
- [ ] `MEDISCAN_PII_SCRUB=1` for any non-Ollama provider.
- [ ] `MEDISCAN_ALLOWED_ORIGINS` is the exact frontend origin, no wildcards.
- [ ] `MEDISCAN_DB_PATH` points to a path on an encrypted filesystem.
- [ ] The SQLite file is **not** in a directory served by a static file server.
- [ ] `MEDISCAN_UPLOAD_ROOT` is on a path with an enforced disk quota.
- [ ] Log shipping is configured and PHI-redacting (`LOG_LEVEL=INFO` or stricter).
- [ ] `pip-audit` is clean in CI.
- [ ] `MEDISCAN_PROMETHEUS=1` only behind a private network.

## Where to read next

- [Handbook &sect;9 &mdash; Security](HANDBOOK.md#9-security)
- [ARCHITECTURE.md](ARCHITECTURE.md) &mdash; module dependency map
- [USAGE.md](../USAGE.md#environment-variables) &mdash; full env-var reference
