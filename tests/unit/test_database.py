import json
import sqlite3

import backend.database as database


def test_init_db_creates_records_table(tmp_path, monkeypatch):
    """Targets backend.database.init_db in backend/database.py."""
    db_file = tmp_path / "records.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))

    database.init_db()

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='records'")
    row = cur.fetchone()
    conn.close()
    assert row[0] == "records"


def test_save_record_inserts_expected_payload(tmp_path, monkeypatch):
    """Targets backend.database.save_record in backend/database.py."""
    db_file = tmp_path / "records.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    database.init_db()

    payload = {
        "patient": {"mrn": "MRN-1"},
        "encounter": {"date": "2026-01-15"},
        "clinical": {"diagnosis_list": []},
    }
    database.save_record(payload)

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT mrn, date, full_json FROM records")
    mrn, date, full_json = cur.fetchone()
    conn.close()

    assert mrn == "MRN-1"
    assert date == "2026-01-15"
    assert json.loads(full_json)["patient"]["mrn"] == "MRN-1"


def test_save_record_defaults_unknown_when_fields_missing(tmp_path, monkeypatch):
    """Targets backend.database.save_record fallback values in backend/database.py."""
    db_file = tmp_path / "records.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    database.init_db()

    database.save_record({})

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT mrn, date FROM records")
    mrn, date = cur.fetchone()
    conn.close()

    assert mrn == "UNKNOWN"
    assert date == "UNKNOWN"


def test_get_patient_history_returns_latest_by_date_desc(tmp_path, monkeypatch):
    """Targets backend.database.get_patient_history in backend/database.py."""
    db_file = tmp_path / "records.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    database.init_db()

    older = {"patient": {"mrn": "MRN-2"}, "encounter": {"date": "2025-12-01"}, "v": 1}
    newer = {"patient": {"mrn": "MRN-2"}, "encounter": {"date": "2026-02-01"}, "v": 2}
    database.save_record(older)
    database.save_record(newer)

    result = database.get_patient_history("MRN-2")
    assert result["v"] == 2


def test_get_patient_history_returns_none_when_missing(tmp_path, monkeypatch):
    """Targets backend.database.get_patient_history missing-record path in backend/database.py."""
    db_file = tmp_path / "records.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    database.init_db()

    assert database.get_patient_history("NOPE") is None
