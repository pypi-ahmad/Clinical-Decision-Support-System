"""Extraction-quality evaluation harness.

Gold fixtures live in ``tests/eval/gold/`` as JSON files with the shape::

    {
        "reference_text": "... full ground-truth OCR text ...",
        "hypothesis_text": "... OCR output to score ...",
        "reference_fields": {"patient": {...}, "diagnosis_list": [...], ...},
        "hypothesis_fields": {"patient": {...}, "diagnosis_list": [...], ...}
    }

When no gold data is present the tests skip — the harness is safe to commit
without fixtures. Run explicitly with::

    pytest -m eval
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eval.metrics import char_error_rate, exact_match, field_set_f1

pytestmark = pytest.mark.eval

GOLD_DIR = Path(__file__).parent / "gold"


def _load_gold() -> list[tuple[str, dict]]:
    if not GOLD_DIR.exists():
        return []
    out: list[tuple[str, dict]] = []
    for path in sorted(GOLD_DIR.glob("*.json")):
        try:
            out.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


_GOLD = _load_gold()

if not _GOLD:
    @pytest.mark.eval
    def test_eval_harness_requires_gold():
        pytest.skip("No gold fixtures in tests/eval/gold/ — add JSON fixtures to enable.")
else:
    @pytest.mark.parametrize("name,case", _GOLD, ids=[n for n, _ in _GOLD])
    def test_ocr_cer(name: str, case: dict) -> None:
        cer = char_error_rate(case.get("reference_text", ""), case.get("hypothesis_text", ""))
        # Soft threshold — tune per dataset; kept lenient to avoid flakes.
        assert cer <= float(case.get("max_cer", 0.35)), f"{name}: CER={cer:.3f}"

    @pytest.mark.parametrize("name,case", _GOLD, ids=[n for n, _ in _GOLD])
    def test_diagnosis_field_f1(name: str, case: dict) -> None:
        ref_fields = case.get("reference_fields") or {}
        hyp_fields = case.get("hypothesis_fields") or {}
        scores = field_set_f1(ref_fields.get("diagnosis_list"), hyp_fields.get("diagnosis_list"))
        assert scores["f1"] >= float(case.get("min_diagnosis_f1", 0.5)), f"{name}: {scores}"

    @pytest.mark.parametrize("name,case", _GOLD, ids=[n for n, _ in _GOLD])
    def test_patient_mrn_exact_match(name: str, case: dict) -> None:
        ref = ((case.get("reference_fields") or {}).get("patient") or {}).get("mrn")
        hyp = ((case.get("hypothesis_fields") or {}).get("patient") or {}).get("mrn")
        assert exact_match(ref, hyp), f"{name}: mrn {ref!r} vs {hyp!r}"
