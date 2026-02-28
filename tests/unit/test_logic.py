import backend.logic as logic


def test_analyze_medical_logic_success_builds_new_patient_context(monkeypatch):
    """Targets backend.logic.analyze_medical_logic in backend/logic.py."""
    captured = {"context": None}

    def fake_get_ai_response(provider, model, api_key, system_prompt, user_text):
        captured["context"] = user_text
        return '{"alerts": ["a"], "trends": [], "summary": "ok"}'

    monkeypatch.setattr(logic, "get_ai_response", fake_get_ai_response)
    result = logic.analyze_medical_logic({"patient": {"mrn": "1"}}, None, "Ollama", "m", None)

    assert result["summary"] == "ok"
    assert "PAST_DATA: None (New Patient)" in captured["context"]


def test_analyze_medical_logic_failure_returns_fallback(monkeypatch):
    """Targets backend.logic.analyze_medical_logic exception path in backend/logic.py."""

    def raise_error(*args, **kwargs):
        raise RuntimeError("failed")

    monkeypatch.setattr(logic, "get_ai_response", raise_error)
    result = logic.analyze_medical_logic({"x": 1}, {"y": 2})

    assert result == {"summary": "Analysis failed", "alerts": [], "trends": []}


def test_check_insurance_coverage_success_and_policy_truncation(monkeypatch):
    """Targets backend.logic.check_insurance_coverage in backend/logic.py."""
    captured = {"prompt": None}

    def fake_get_ai_response(provider, model, api_key, system_prompt, user_text):
        captured["prompt"] = user_text
        return '{"eligible": true, "confidence": "High", "reasoning": "ok", "missing_info": []}'

    monkeypatch.setattr(logic, "get_ai_response", fake_get_ai_response)
    long_policy = "P" * 4500
    result = logic.check_insurance_coverage({"dx": ["x"]}, long_policy)

    assert result["eligible"] is True
    assert "INSURANCE_POLICY_TEXT:" in captured["prompt"]
    assert "P" * 4000 in captured["prompt"]
    assert "P" * 4100 not in captured["prompt"]


def test_check_insurance_coverage_failure_returns_error_payload(monkeypatch):
    """Targets backend.logic.check_insurance_coverage exception path in backend/logic.py."""

    def raise_error(*args, **kwargs):
        raise RuntimeError("provider error")

    monkeypatch.setattr(logic, "get_ai_response", raise_error)
    result = logic.check_insurance_coverage({"dx": []}, "policy")

    assert result["eligible"] is False
    assert result["reasoning"].startswith("Error:")
    assert result["missing_info"] == []
