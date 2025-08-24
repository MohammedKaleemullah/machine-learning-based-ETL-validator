# validator.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import joblib
import os

MODEL_PATH = "./uber_iforest.pkl"

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in km between two lat/lon points"""
    R = 6371  # Earth radius
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def infer_schema(df: pd.DataFrame):
    schema = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            schema[col] = "numeric"
        else:
            schema[col] = "categorical"
    return schema

def train_isolation_forest(df: pd.DataFrame, columns_to_avoid=None):
    if columns_to_avoid is None:
        columns_to_avoid = []

    df = df.copy()
    df["trip_distance_km"] = haversine(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"]
    )
    df["fare_per_km"] = df["fare_amount"] / df["trip_distance_km"].replace(0, 1)

    train_df = df.drop(columns=columns_to_avoid, errors="ignore")

    num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    model = IsolationForest(contamination=0.05, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("isolation_forest", model)
    ])

    pipeline.fit(train_df)

    joblib.dump({
        "pipeline": pipeline,
        "columns_to_avoid": columns_to_avoid
    }, MODEL_PATH)

    print(f"Isolation Forest trained on features: {num_cols + cat_cols} (excluding {columns_to_avoid})")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Isolation Forest model not trained ")
    return joblib.load(MODEL_PATH)

def validate_new_rows(new_rows: pd.DataFrame, schema: dict):
    valid_rows, invalid_rows = [], []
    errors = []

    for idx, row in new_rows.iterrows():
        row_errors = []

        for col, expected_type in schema.items():
            if expected_type == "numeric":
                try:
                    float(row[col])
                except ValueError:
                    row_errors.append(f"Column {col} should be numeric, got {row[col]}")

        if row_errors:
            errors.extend(row_errors)
            invalid_rows.append(row)
        else:
            valid_rows.append(row)

    valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame()
    invalid_df = pd.DataFrame(invalid_rows) if invalid_rows else pd.DataFrame()

    # ML model anomaly detection is applied only on valid rows
    if not valid_df.empty:
        saved_obj = load_model()
        pipeline = saved_obj["pipeline"]
        columns_to_avoid = saved_obj.get("columns_to_avoid", [])

        valid_df = valid_df.copy()
        valid_df["trip_distance_km"] = haversine(
            valid_df["pickup_longitude"], valid_df["pickup_latitude"],
            valid_df["dropoff_longitude"], valid_df["dropoff_latitude"]
        )
        valid_df["fare_per_km"] = valid_df["fare_amount"] / valid_df["trip_distance_km"].replace(0, 1)

        predict_df = valid_df.drop(columns=columns_to_avoid, errors="ignore")

        preds = pipeline.predict(predict_df)  # 1=normal, -1=anomaly
        scores = pipeline.decision_function(predict_df)

        valid_df["anomaly"] = preds
        valid_df["anomaly_score"] = scores

        anomalies = valid_df[valid_df["anomaly"] == -1]
        if not anomalies.empty:
            errors.append(f"ML flagged {len(anomalies)} anomalies")
            invalid_df = pd.concat([invalid_df, anomalies])
            valid_df = valid_df[valid_df["anomaly"] == 1]
            print("\nML detected anomalies:")
            print(anomalies[["uid","fare_amount","passenger_count","trip_distance_km","fare_per_km","anomaly_score"]].to_string(index=False))

    return {
        "valid": valid_df.drop(columns=["anomaly","anomaly_score"], errors="ignore"),
        "invalid": invalid_df,
        "errors": errors
    }
