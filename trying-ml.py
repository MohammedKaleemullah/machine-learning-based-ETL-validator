# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import IsolationForest
# from sklearn.model_selection import train_test_split

# # 1. Load dataset
# print("Loading Uber dataset...")
# df = pd.read_csv("./watched_dir/uber.csv")
# print(f"Dataset shape: {df.shape}")
# # 2. Feature Engineering
# # Convert datetime features to useful columns
# if "pickup_datetime" in df.columns:
#     df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
#     df["pickup_hour"] = df["pickup_datetime"].dt.hour
#     df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
#     df.drop(columns=["pickup_datetime"], inplace=True)

# if "dropoff_datetime" in df.columns:
#     df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
#     df["trip_duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
#     df.drop(columns=["dropoff_datetime"], inplace=True, errors="ignore")

# # 3. Identify column types
# numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# print("Numeric cols:", numeric_cols)
# print("Categorical cols:", categorical_cols)

# # 4. Preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numeric_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
#     ]
# )

# print("Preprocessing pipeline created.")

# # 5. Full pipeline: preprocessing + model
# pipeline = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("model", IsolationForest(
#         n_estimators=100,
#         contamination=0.01,   # expected anomaly % (tuneable)
#         random_state=42
#     ))
# ])
# print("Full pipeline created.")
# # 6. Train (fit) the model
# pipeline.fit(df)

# print("✅ Isolation Forest model trained on Uber data")

# # 7. Example: Predict anomalies
# df["anomaly_score"] = pipeline.decision_function(df)
# df["anomaly"] = pipeline.predict(df)   # -1 = anomaly, 1 = normal

# print(df[["anomaly", "anomaly_score"]].head())
