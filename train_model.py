# train_model.py
import pandas as pd
from validator import train_isolation_forest

df = pd.read_csv("./watched_dir/uber.csv")
print(f"Training on {df.shape[0]} rows, {df.shape[1]} columns...")

train_isolation_forest(df, columns_to_avoid=["uid", "key", "pickup_datetime"])

print("Model training complete!")
