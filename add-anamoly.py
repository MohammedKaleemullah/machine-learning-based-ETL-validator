# add-anomaly.py
import pandas as pd
from datetime import datetime
import random

csv_file = "./watched_dir/uber.csv"

def generate_anomaly():
    print(f"🚨 Injecting anomaly")
    
    base_lat = 40.758  # Manhattan-ish coords

    coord = random.uniform(-180, 180)
    return {
        "uid": random.randint(10000000, 99999999),
        "key": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "fare_amount": round(random.uniform(-50, -1), 2),  # 🚨 negative fare
        "pickup_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "pickup_longitude": coord,  # 🚨 way off NYC
        "pickup_latitude": base_lat,
        "dropoff_longitude": coord,  # 🚨 same as pickup
        "dropoff_latitude": base_lat + 0.005,
        "passenger_count": 0,  # 🚨 invalid passenger count
        }




# Generate & append anomaly
row = generate_anomaly()
df = pd.DataFrame([row])
df.to_csv(csv_file, mode="a", header=False, index=False)

print(f"✅ Anomaly row added to {csv_file}: {row}")
