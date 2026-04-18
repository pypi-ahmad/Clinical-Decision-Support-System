import backend.models as models


def test_medical_record_model_accepts_defined_shape():
    record = models.MedicalRecord(
        patient={"full_name": "Jane", "dob": "1990-01-01", "mrn": "M1"},
        encounter_date="2026-02-01",
        clinical={
            "diagnosis_list": ["Dx"],
            "medications": [{"name": "A"}],
            "vitals": {"bp": "120/80"},
        },
    )

    assert record.patient.mrn == "M1"
    assert record.clinical.diagnosis_list == ["Dx"]


def test_clinical_data_default_fields_exist():
    data = models.ClinicalData()
    assert data.diagnosis_list == []
    assert data.medications == []
    # Vitals is now a typed Vitals model with optional fields defaulting to None.
    assert data.vitals.bp is None
    assert data.vitals.hr is None


def test_encounter_rejects_invalid_date():
    enc = models.Encounter(date="not-a-date")
    assert enc.date is None


def test_patient_truncates_long_name():
    long_name = "Z" * 1000
    patient = models.Patient(full_name=long_name)
    assert patient.full_name is not None
    assert len(patient.full_name) <= 128
