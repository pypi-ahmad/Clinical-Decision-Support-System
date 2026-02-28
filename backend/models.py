"""
Data Models Module
------------------
This module defines the Pydantic schemas used to enforce data structure consistency 
across the application. These models ensure that the AI outputs match the expected format.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class Encounter(BaseModel):
    """Schema for encounter metadata."""
    date: Optional[str] = None
    provider: Optional[str] = None
    facility: Optional[str] = None


class Patient(BaseModel):
    """Schema for Patient demographics."""
    full_name: Optional[str] = None
    dob: Optional[str] = None
    mrn: Optional[str] = None

class ClinicalData(BaseModel):
    """Schema for clinical findings."""
    diagnosis_list: List[str] = Field(default_factory=list)
    medications: List[dict] = Field(default_factory=list)
    vitals: dict = Field(default_factory=dict)

class MedicalRecord(BaseModel):
    """
    Root Schema representing a complete Medical Record.
    Combines Patient demographics and Clinical Data.
    """
    patient: Patient
    encounter: Optional[Encounter] = None
    encounter_date: Optional[str] = None
    clinical: ClinicalData = Field(default_factory=ClinicalData)
