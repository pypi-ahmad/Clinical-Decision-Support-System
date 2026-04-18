"""Clinical reasoning and insurance logic."""

from __future__ import annotations

import json
from typing import Any

from backend.ai_wrapper import AIProviderError, get_ai_response, parse_model_json
from backend.logging_config import get_logger
from backend.retrieval.chunking import sanitize_retrieved_text
from backend.security import firewall_clause, generate_boundary_nonce, wrap_untrusted


_logger = get_logger(__name__)


def _sanitize_chunks(chunks: list[dict] | None) -> list[dict]:
    """Scrub injection phrases from the ``text`` field of each retrieved chunk.

    Defense-in-depth on top of the callers in ``backend/main.py`` — guarantees
    that no unsanitized retrieved text reaches the prompt, even if a future
    caller skips the main-level scrub.
    """
    if not chunks:
        return []
    cleaned: list[dict] = []
    for chunk in chunks:
        if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
            cleaned.append({**chunk, "text": sanitize_retrieved_text(chunk["text"])})
        else:
            cleaned.append(chunk)
    return cleaned


_REASONING_SYSTEM_TEMPLATE = """You are a Clinical Decision Support System.
Compare the CURRENT_DATA against PAST_DATA and RETRIEVED_CONTEXT.
Task 1: TRENDS. Compare vitals (BP, Weight, HR) — mark them Increasing, Decreasing, or Stable.
Task 2: CONSISTENCY. Check whether prescribed medications match the diagnoses.

Output JSON with this exact schema:
{
  "alerts": ["High Priority Alert", "Medium Priority Warning"],
  "trends": [{"metric": "BP", "status": "Worsening", "details": "120/80 -> 140/90"}],
  "summary": "Brief clinical summary of changes."
}
Return ONLY the JSON object.
"""


_INSURANCE_SYSTEM_TEMPLATE = """You are an Insurance Claims Adjuster.
Review MEDICAL_DATA and INSURANCE_POLICY_TEXT and decide whether the patient's
condition is likely covered.

Rules:
1. Match diagnoses against policy inclusions/exclusions.
2. Check for waiting periods or pre-existing-condition clauses.

Output JSON with this exact schema:
{
  "eligible": true,
  "confidence": "High",
  "reasoning": "Explanation of why it is covered or rejected.",
  "missing_info": ["List of documents or details needed to confirm"]
}
Return ONLY the JSON object.
"""


def _failure_response(kind: str, reason: str = "provider_error") -> dict[str, Any]:
    """Return a stable fallback schema annotated with a ``_status`` block so
    the client can distinguish ``ok`` responses from silent failures.
    """
    if kind == "analysis":
        return {
            "summary": "Analysis failed",
            "alerts": [],
            "trends": [],
            "_status": {"ok": False, "reason": reason},
        }
    if kind == "insurance":
        return {
            "eligible": False,
            "confidence": "Low",
            "reasoning": "Analysis failed",
            "missing_info": [],
            "_status": {"ok": False, "reason": reason},
        }
    return {"_status": {"ok": False, "reason": reason}}


def analyze_medical_logic(
    current_data: dict,
    past_data: dict | None,
    provider: str = "Ollama",
    model: str = "lfm2.5-thinking",
    api_key: str | None = None,
    *,
    retrieved_context: list[dict] | None = None,
) -> dict:
    """Run the clinical reasoning LLM call."""
    nonce = generate_boundary_nonce()
    safe_context = _sanitize_chunks(retrieved_context)
    current_section, _ = wrap_untrusted(json.dumps(current_data, ensure_ascii=False), nonce)
    past_section, _ = wrap_untrusted(
        json.dumps(past_data, ensure_ascii=False) if past_data else "None (New Patient)",
        nonce,
    )
    context_section, _ = wrap_untrusted(
        json.dumps(safe_context, ensure_ascii=False),
        nonce,
    )

    user_text = (
        f"CURRENT_DATA:\n{current_section}\n\n"
        f"PAST_DATA:\n{past_section}\n\n"
        f"RETRIEVED_CONTEXT:\n{context_section}"
    )
    system_prompt = _REASONING_SYSTEM_TEMPLATE + "\n" + firewall_clause(nonce)

    _logger.info("reasoning_start", provider=provider, model=model)
    try:
        response = get_ai_response(provider, model, api_key, system_prompt, user_text, force_json=True)
    except AIProviderError as exc:
        _logger.warning("reasoning_provider_failed", reason=exc.detail)
        return _failure_response("analysis", reason="provider_error")
    try:
        parsed = parse_model_json(response)
    except ValueError as exc:
        _logger.warning("reasoning_parse_failed", reason=str(exc))
        return _failure_response("analysis", reason="parse_error")
    if isinstance(parsed, dict):
        parsed.setdefault("_status", {"ok": True})
    return parsed


def check_insurance_coverage(
    medical_data: dict,
    policy_text: str,
    provider: str = "Ollama",
    model: str = "glm-4.7-flash",
    api_key: str | None = None,
    *,
    relevant_policy_chunks: list[dict] | None = None,
    policy_char_limit: int = 8000,
) -> dict:
    """Run the insurance-eligibility LLM call."""
    nonce = generate_boundary_nonce()
    # Defense-in-depth: scrub injection phrases from raw policy OCR text and
    # from any retrieved policy chunks before wrapping them as untrusted data.
    trimmed_policy = sanitize_retrieved_text((policy_text or "")[:policy_char_limit])
    safe_chunks = _sanitize_chunks(relevant_policy_chunks)
    medical_section, _ = wrap_untrusted(json.dumps(medical_data, ensure_ascii=False), nonce)
    policy_section, _ = wrap_untrusted(trimmed_policy, nonce)
    retrieved_section, _ = wrap_untrusted(
        json.dumps(safe_chunks, ensure_ascii=False),
        nonce,
    )

    user_text = (
        f"MEDICAL_DATA:\n{medical_section}\n\n"
        f"INSURANCE_POLICY_TEXT (truncated to {policy_char_limit} chars):\n{policy_section}\n\n"
        f"RETRIEVED_POLICY_CONTEXT:\n{retrieved_section}"
    )
    system_prompt = _INSURANCE_SYSTEM_TEMPLATE + "\n" + firewall_clause(nonce)

    _logger.info("insurance_start", provider=provider, model=model)
    try:
        response = get_ai_response(provider, model, api_key, system_prompt, user_text, force_json=True)
    except AIProviderError as exc:
        _logger.warning("insurance_provider_failed", reason=exc.detail)
        return _failure_response("insurance", reason="provider_error")
    try:
        parsed = parse_model_json(response)
    except ValueError as exc:
        _logger.warning("insurance_parse_failed", reason=str(exc))
        return _failure_response("insurance", reason="parse_error")
    if isinstance(parsed, dict):
        parsed.setdefault("_status", {"ok": True})
    return parsed
