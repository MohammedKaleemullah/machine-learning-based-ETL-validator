import pandas as pd
from validator import load_model

df = pd.read_csv("./watched_dir/uber.csv")
model = load_model()

preds = model.predict(df)
scores = model.decision_function(df)

df["anomaly"] = preds
df["anomaly_score"] = scores

print(df[["uid", "fare_amount", "passenger_count", "anomaly", "anomaly_score"]].head(20000))
