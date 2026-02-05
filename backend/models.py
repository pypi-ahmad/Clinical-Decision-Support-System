"""
Data Models Module
------------------
This module defines the Pydantic schemas used to enforce data structure consistency 
across the application. These models ensure that the AI outputs match the expected format.
"""

from pydantic import BaseModel
from typing import List, Optional

class Patient(BaseModel):
    """Schema for Patient demographics."""
    full_name: Optional[str] = None
    dob: Optional[str] = None
    mrn: Optional[str] = None

class ClinicalData(BaseModel):
    """Schema for clinical findings."""
    diagnosis_list: List[str] = []
    medications: List[dict] = []
    vitals: dict = {}

class MedicalRecord(BaseModel):
    """
    Root Schema representing a complete Medical Record.
    Combines Patient demographics and Clinical Data.
    """
    patient: Patient
    encounter_date: Optional[str] = None
    clinical: ClinicalData
