import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_missing_values(df):
    df["distance_to_clinic_km"].fillna(df["distance_to_clinic_km"].median(), inplace=True)
    df["waiting_time_minutes"].fillna(df["waiting_time_minutes"].median(), inplace=True)
    return df

def encode_categorical(df):
    categorical_cols = ["gender","appointment_type","reminder_channel","appointment_time","appointment_day"]
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def scale_features(df):
    numeric_cols = ["age","distance_to_clinic_km","waiting_time_minutes","booking_lead_days"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
