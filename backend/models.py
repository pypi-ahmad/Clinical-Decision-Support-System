"""Data Models Module.

Pydantic v2 schemas for the clinical domain. Validation is intentionally
forgiving for *extracted* records (OCR + LLM output is noisy) but strict for
*confirmed* records that are persisted (see ``MedicalRecord.strict``).
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.logging_config import get_logger

_logger = get_logger(__name__)


# Maximum character counts for each free-text field. Protects the DB, logs,
# and the LLM prompt budget from pathological inputs.
_MAX_FIELD_LEN = 256
_MAX_LIST_ITEMS = 200


def _truncate(value: str | None, limit: int = _MAX_FIELD_LEN) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > limit:
        _logger.warning(
            "field_truncated",
            original_len=len(value),
            new_len=limit,
        )
        return value[:limit]
    return value


class Medication(BaseModel):
    """Single medication entry. Used inside :class:`ClinicalData`."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    dosage: str | None = None
    frequency: str | None = None

    @field_validator("name", "dosage", "frequency", mode="before")
    @classmethod
    def _clean(cls, value):
        if value is None:
            return None
        return _truncate(str(value))


class Vitals(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bp: str | None = None
    hr: str | None = None
    temp: str | None = None
    weight: str | None = None

    @field_validator("bp", "hr", "temp", "weight", mode="before")
    @classmethod
    def _clean(cls, value):
        if value is None:
            return None
        return _truncate(str(value), limit=64)


class Encounter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str | None = None
    provider: str | None = None
    facility: str | None = None

    @field_validator("provider", "facility", mode="before")
    @classmethod
    def _clean(cls, value):
        if value is None:
            return None
        return _truncate(str(value))

    @field_validator("date", mode="before")
    @classmethod
    def _validate_date(cls, value):
        if value in (None, ""):
            return None
        text = str(value).strip()
        # Coerce common YYYY-MM-DD; reject anything we can't parse into a date.
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", text):
            return None
        try:
            date.fromisoformat(text)
        except ValueError:
            return None
        return text


class Patient(BaseModel):
    model_config = ConfigDict(extra="ignore")

    full_name: str | None = None
    dob: str | None = None
    mrn: str | None = None

    @field_validator("full_name", "mrn", mode="before")
    @classmethod
    def _clean(cls, value):
        if value is None:
            return None
        return _truncate(str(value), limit=128)

    @field_validator("dob", mode="before")
    @classmethod
    def _validate_dob(cls, value):
        if value in (None, ""):
            return None
        text = str(value).strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", text):
            return None
        try:
            date.fromisoformat(text)
        except ValueError:
            return None
        return text


class ClinicalData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    diagnosis_list: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    vitals: Vitals = Field(default_factory=Vitals)

    @field_validator("diagnosis_list", mode="before")
    @classmethod
    def _clean_list(cls, value):
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value[:_MAX_LIST_ITEMS]:
            if isinstance(item, str):
                trimmed = _truncate(item)
                if trimmed:
                    cleaned.append(trimmed)
        return cleaned

    @field_validator("medications", mode="before")
    @classmethod
    def _clean_medications(cls, value):
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value[:_MAX_LIST_ITEMS]:
            if isinstance(item, dict):
                cleaned.append(item)
            # Drop any non-dict gracefully (e.g. stray strings).
        return cleaned

    @field_validator("vitals", mode="before")
    @classmethod
    def _coerce_vitals(cls, value):
        if isinstance(value, dict):
            return value
        return {}


class MedicalRecord(BaseModel):
    """Root schema representing a complete Medical Record."""

    model_config = ConfigDict(extra="ignore")

    patient: Patient = Field(default_factory=Patient)
    encounter: Encounter | None = None
    encounter_date: str | None = None
    clinical: ClinicalData = Field(default_factory=ClinicalData)

    @field_validator("encounter_date", mode="before")
    @classmethod
    def _validate_encounter_date(cls, value):
        return Encounter._validate_date(value)


# The ``strict`` variant should be used for /confirm (persistence boundary).
# It forbids extra keys and demands at least one of mrn / full_name so the
# row has a meaningful anchor.
class MedicalRecordStrict(MedicalRecord):
    model_config = ConfigDict(extra="forbid")
