import backend.models as models


def test_medical_record_model_accepts_defined_shape():
    """Targets backend.models.MedicalRecord schema in backend/models.py."""
    record = models.MedicalRecord(
        patient={"full_name": "Jane", "dob": "1990-01-01", "mrn": "M1"},
        encounter_date="2026-02-01",
        clinical={"diagnosis_list": ["Dx"], "medications": [{"name": "A"}], "vitals": {"bp": "120/80"}},
    )

    assert record.patient.mrn == "M1"
    assert record.clinical.diagnosis_list == ["Dx"]


def test_clinical_data_default_fields_exist():
    """Targets backend.models.ClinicalData defaults in backend/models.py."""
    data = models.ClinicalData()
    assert data.diagnosis_list == []
    assert data.medications == []
    assert data.vitals == {}
