# Gold Fixtures

Place evaluation fixtures here as JSON files. Each file becomes a parametrised
test case. Schema:

```json
{
  "reference_text": "ground-truth OCR text",
  "hypothesis_text": "OCR output to score",
  "max_cer": 0.35,
  "reference_fields": {
    "patient": {"mrn": "MRN-12345"},
    "diagnosis_list": ["Hypertension", "Type 2 Diabetes"]
  },
  "hypothesis_fields": {
    "patient": {"mrn": "MRN-12345"},
    "diagnosis_list": ["Hypertension", "T2DM"]
  },
  "min_diagnosis_f1": 0.5
}
```

Run: `pytest -m eval`.
