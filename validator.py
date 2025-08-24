# validator.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import joblib
import os

MODEL_PATH = "./uber_iforest.pkl"

def infer_schema(df: pd.DataFrame):
    """Infer schema dynamically from existing dataframe"""
    schema = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            schema[col] = "numeric"
        else:
            schema[col] = "categorical"
    return schema


def train_isolation_forest(df: pd.DataFrame, columns_to_avoid=None):
    """Train preprocessing pipeline + Isolation Forest model"""
    if columns_to_avoid is None:
        columns_to_avoid = []

    # Keep only training features
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

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("isolation_forest", model)
    ])

    pipeline.fit(train_df)

    joblib.dump({
        "pipeline": pipeline,
        "columns_to_avoid": columns_to_avoid
    }, MODEL_PATH)

    print(f"✅ Isolation Forest trained (excluding {columns_to_avoid}) and saved to {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Isolation Forest model not trained yet! Please run training.")
    return joblib.load(MODEL_PATH)


def validate_new_rows(new_rows: pd.DataFrame, schema: dict):
    """Validate schema + detect anomalies"""
    valid_rows, invalid_rows = [], []
    errors = []

    # Schema validation
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

    # ML anomaly detection
    if not valid_df.empty:
        saved_obj = load_model()
        pipeline = saved_obj["pipeline"]
        columns_to_avoid = saved_obj.get("columns_to_avoid", [])

        # Drop avoid columns before prediction
        predict_df = valid_df.drop(columns=columns_to_avoid, errors="ignore")

        preds = pipeline.predict(predict_df)   # 1 = normal, -1 = anomaly
        scores = pipeline.decision_function(predict_df)

        valid_df["anomaly"] = preds
        valid_df["anomaly_score"] = scores

        anomalies = valid_df[valid_df["anomaly"] == -1]
        if not anomalies.empty:
            errors.append(f"ML flagged {len(anomalies)} anomalies")
            invalid_df = pd.concat([invalid_df, anomalies])
            valid_df = valid_df[valid_df["anomaly"] == 1]
            print("\n🚨 ML detected anomalies:")
            print(anomalies[["uid", "fare_amount", "passenger_count", "anomaly_score"]].to_string(index=False))

    return {
        "valid": valid_df.drop(columns=["anomaly", "anomaly_score"], errors="ignore"),
        "invalid": invalid_df,
        "errors": errors
    }
