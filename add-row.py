#add-row.py
import pandas as pd
from datetime import datetime
import random

csv_file = "./watched_dir/uber.csv"

# Define a new row (dummy data)
new_row = {
    "uid": random.randint(10000000, 99999999),   # random uid
    "key": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), 
    "fare_amount": round(random.uniform(5, 30), 2),  # random fare
    "pickup_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
    "pickup_longitude": -73.985 + random.uniform(-0.01, 0.01),
    "pickup_latitude": 40.758 + random.uniform(-0.01, 0.01),
    "dropoff_longitude": -73.975 + random.uniform(-0.01, 0.01),
    "dropoff_latitude": 40.768 + random.uniform(-0.01, 0.01),
    "passenger_count": random.randint(1, 4)
}

# Append the row without re-writing the header
df = pd.DataFrame([new_row])
df.to_csv(csv_file, mode="a", header=False, index=False)

print(f"✅ Row added to {csv_file}: {new_row}")
