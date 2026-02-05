"""
Database Module (The Memory)
----------------------------
This module handles SQLite database interactions. It provides a simple persistent layer
to store patient medical records, allowing the application to retrieve past history
for trend analysis (Comparing "This Visit" vs "Last Visit").
"""

import sqlite3
import json
import os

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), 'medical_records.db')

def init_db():
    """
    Initialize the database table if it doesn't exist.
    
    Creates a 'records' table with columns:
    - id: Auto-incrementing primary key
    - mrn: Medical Record Number (Patient ID)
    - date: Date of the encounter
    - full_json: The complete structured data as a JSON string
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Store critical fields + raw JSON blob for flexibility
    c.execute('''CREATE TABLE IF NOT EXISTS records 
                 (id INTEGER PRIMARY KEY, mrn TEXT, date TEXT, full_json TEXT)''')
    conn.commit()
    conn.close()

def save_record(data: dict):
    """
    Save a new medical record to the database.

    Args:
        data (dict): The dictionary containing patient and clinical data.
                     Expected to have 'patient.mrn' and 'encounter.date'.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Extract key fields for querying
    mrn = data.get('patient', {}).get('mrn', 'UNKNOWN')
    date = data.get('encounter', {}).get('date', 'UNKNOWN')
    
    # Insert the record
    c.execute("INSERT INTO records (mrn, date, full_json) VALUES (?, ?, ?)",
              (mrn, date, json.dumps(data)))
    conn.commit()
    conn.close()

def get_patient_history(mrn: str):
    """
    Fetch the most recent PREVIOUS visit for a specific patient.

    Args:
        mrn (str): Medical Record Number to search for.

    Returns:
        dict: The full JSON data of the most recent previous visit.
        None: If no history is found.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    c = conn.cursor()
    
    # Get the latest record for this MRN
    c.execute("SELECT full_json FROM records WHERE mrn = ? ORDER BY date DESC LIMIT 1", (mrn,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return json.loads(row['full_json'])
    return None
