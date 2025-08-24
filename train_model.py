# train_model.py
import pandas as pd
from validator import train_isolation_forest, load_model, haversine
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import random


df = pd.read_csv("./watched_dir/uber.csv")
print(f"Training on {df.shape[0]} rows, {df.shape[1]} columns...")

train_isolation_forest(df, columns_to_avoid=["uid", "key", "pickup_datetime"])
print("Model training complete!")


saved_obj = load_model()
pipeline = saved_obj["pipeline"]
columns_to_avoid = saved_obj.get("columns_to_avoid", [])


df_eval = df.copy()
df_eval["trip_distance_km"] = haversine(
    df_eval["pickup_longitude"], df_eval["pickup_latitude"],
    df_eval["dropoff_longitude"], df_eval["dropoff_latitude"]
)
df_eval["fare_per_km"] = df_eval["fare_amount"] / df_eval["trip_distance_km"].replace(0, 1)

X = df_eval.drop(columns=columns_to_avoid, errors="ignore")

base_lat = 40.758
coord = random.uniform(-180, 180)

# Create 50 synthetic anomalies
synthetic_anomalies = df_eval.sample(50, replace=True).copy()
synthetic_anomalies["fare_amount"] = [round(random.uniform(0, 100), 2) for _ in range(len(synthetic_anomalies))]
synthetic_anomalies["pickup_longitude"] = [random.uniform(-180, 180) for _ in range(len(synthetic_anomalies))]
synthetic_anomalies["pickup_latitude"] = [base_lat for _ in range(len(synthetic_anomalies))]
synthetic_anomalies["trip_distance_km"] *= np.random.randint(50, 200, size=len(synthetic_anomalies))

# Labelling data: 0=normal, 1=anomaly
X_all = pd.concat([X[:50], synthetic_anomalies.drop(columns=columns_to_avoid, errors="ignore")])
y_true = np.array([0]*50 + [1]*len(synthetic_anomalies))

# Predict
y_pred = pipeline.predict(X_all)
y_pred = np.where(y_pred == -1, 1, 0)

# Metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n=== Model Evaluation with Synthetic Anomalies ===")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
