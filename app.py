#app.py
import time
import os
import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from IPython.display import display
from validator import validate_new_rows, infer_schema

CDC_LOG = "./watched_dir/cdc_events.csv"   # log file path

ANAMOLY_LOG = "./watched_dir/anomalies.csv"  # anomaly log file path

class CSVWatcher(FileSystemEventHandler):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.last_df = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()
        print(f"Initial file loaded with {len(self.last_df)} rows.")

        # infer schema only once at start
        self.schema = infer_schema(self.last_df)
        print("\n📑 Inferred Schema:\n")
        for col in self.schema.items():
            print(f"{col}")

    def on_modified(self, event):
        if event.src_path.endswith(".csv"):
            try:
                new_df = pd.read_csv(self.file_path)

                # Find new rows compared to previous snapshot
                if len(new_df) > len(self.last_df):
                    diff = new_df.iloc[len(self.last_df):]

                    # ✅ Validate new rows
                    validation_result = validate_new_rows(diff, self.schema)

                    if not validation_result["errors"]:   # no errors
                        display(f"[CDC EVENT] ✅ Valid rows:\n{validation_result['valid'].to_string(index=False)}\n")
                        
                        # Append valid rows to CDC log file
                        if not os.path.exists(CDC_LOG):
                            validation_result["valid"].to_csv(CDC_LOG, index=False, mode="w")
                        else:
                            validation_result["valid"].to_csv(CDC_LOG, index=False, mode="a", header=False)
                    else:
                        display(f"[CDC EVENT] ⚠️ Validation errors:\n{validation_result['errors']}\n")
                        if not validation_result["invalid"].empty:
                            display(f"❌ Invalid rows skipped:\n{validation_result['invalid'].to_string(index=False)}\n")
                        # Append invalid rows to anomaly log file
                        if not os.path.exists(ANAMOLY_LOG):
                            validation_result["invalid"].to_csv(ANAMOLY_LOG, index=False, mode="w")
                        else:
                            validation_result["invalid"].to_csv(ANAMOLY_LOG, index=False, mode="a", header=False)

                self.last_df = new_df
            except Exception as e:
                print(f"Error reading CSV: {e}")


def watch_csv(file_path: str):
    event_handler = CSVWatcher(file_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(file_path) or ".", recursive=False)
    observer.start()

    print(f"👀 Watching CSV file for new rows: {file_path}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    csv_file = "./watched_dir/uber.csv"

    if not os.path.exists(csv_file):
        print("File does not exist!!")
        exit(1)

    watch_csv(csv_file)
