"""Evaluation metrics for the CDSS quality harness.

These helpers are deliberately dependency-free so they can run in CI without
installing ML libraries. They are thin, unit-tested implementations of the
handful of scores the audit called for:

* :func:`char_error_rate` — Levenshtein-based CER for OCR quality.
* :func:`word_error_rate` — token-level WER (whitespace tokenizer).
* :func:`field_set_f1` — precision / recall / F1 over a set of string values
  (e.g. diagnosis lists, medication lists).
* :func:`exact_match` — normalized equality for scalar fields (e.g. MRN).

Metrics are returned as plain floats in ``[0.0, 1.0]``. Callers decide
pass/fail thresholds; this module never raises.
"""

from __future__ import annotations

from collections.abc import Iterable


def _levenshtein(ref: str, hyp: str) -> int:
    """Classic DP edit distance. O(len(ref) * len(hyp)) time, O(len(hyp)) space."""
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        curr = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + (rc != hc),  # substitution
            )
        prev = curr
    return prev[-1]


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character Error Rate (lower is better).

    Conventions:
      * ``CER("", "")``  -> 0.0 (trivially correct)
      * ``CER("abc", "")`` -> 1.0 (total deletion)
      * ``CER("", "abc")`` -> 1.0 (total insertion; denom = len(hyp))
    """
    r = reference or ""
    h = hypothesis or ""
    if not r and not h:
        return 0.0
    denom = max(len(r), 1) if r else len(h)
    return _levenshtein(r, h) / denom


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Whitespace-token WER (lower is better). Same edge cases as CER."""
    r_tokens = (reference or "").split()
    h_tokens = (hypothesis or "").split()
    if not r_tokens and not h_tokens:
        return 0.0
    # Use the same DP but over token sequences.
    if not r_tokens:
        return 1.0
    if not h_tokens:
        return 1.0
    prev = list(range(len(h_tokens) + 1))
    for i, rt in enumerate(r_tokens, 1):
        curr = [i] + [0] * len(h_tokens)
        for j, ht in enumerate(h_tokens, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (rt != ht),
            )
        prev = curr
    return prev[-1] / len(r_tokens)


def _normalize_set(values: Iterable | None) -> set[str]:
    if not values:
        return set()
    return {str(v).strip().lower() for v in values if str(v).strip()}


def field_set_f1(reference: Iterable | None, hypothesis: Iterable | None) -> dict[str, float]:
    """Set-level precision / recall / F1 over case-insensitive string values.

    Returns a dict with keys ``precision``, ``recall``, ``f1`` (all in
    ``[0.0, 1.0]``). Both-empty inputs are treated as trivially correct
    (all scores 1.0), matching the extraction-quality harness convention.
    """
    ref = _normalize_set(reference)
    hyp = _normalize_set(hypothesis)
    if not ref and not hyp:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(ref & hyp)
    precision = tp / len(hyp) if hyp else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match(reference: str | None, hypothesis: str | None) -> bool:
    """Case-insensitive, whitespace-trimmed exact match for scalar fields."""
    r = (reference or "").strip().lower()
    h = (hypothesis or "").strip().lower()
    return r == h
