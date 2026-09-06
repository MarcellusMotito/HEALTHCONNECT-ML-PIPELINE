import pandas as pd
import numpy as np
import pytest
import sys, os

# Ensure project root is on sys.path so src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import clean_missing_values, encode_categorical, scale_features


def test_clean_missing_values():
    df = pd.DataFrame({
        "distance_to_clinic_km": [1.0, None, 3.0],
        "waiting_time_minutes": [10, None, 30]
    })
    df_clean = clean_missing_values(df.copy())
    assert df_clean["distance_to_clinic_km"].isnull().sum() == 0
    assert df_clean["waiting_time_minutes"].isnull().sum() == 0


def test_encode_categorical():
    # Include all categorical columns expected by preprocessing.py
    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "appointment_type": ["Checkup", "Emergency"],
        "reminder_channel": ["SMS", "Email"],
        "appointment_time": ["Morning", "Afternoon"],
        "appointment_day": ["Monday", "Tuesday"]
    })
    df_encoded = encode_categorical(df.copy())

    # Check that dummy columns were created
    assert any(col.startswith("gender_") for col in df_encoded.columns)
    assert any(col.startswith("appointment_type_") for col in df_encoded.columns)
    assert any(col.startswith("reminder_channel_") for col in df_encoded.columns)
    assert any(col.startswith("appointment_time_") for col in df_encoded.columns)
    assert any(col.startswith("appointment_day_") for col in df_encoded.columns)


def test_scale_features():
    df = pd.DataFrame({
        "age": [20, 40, 60],
        "distance_to_clinic_km": [1, 5, 10],
        "waiting_time_minutes": [5, 15, 25],
        "booking_lead_days": [2, 4, 6]
    })
    df_scaled = scale_features(df.copy())

    # mean should be ~0 after scaling
    assert np.isclose(df_scaled["age"].mean(), 0, atol=1e-6)
    assert np.isclose(df_scaled["distance_to_clinic_km"].mean(), 0, atol=1e-6)
    assert np.isclose(df_scaled["waiting_time_minutes"].mean(), 0, atol=1e-6)
    assert np.isclose(df_scaled["booking_lead_days"].mean(), 0, atol=1e-6)
