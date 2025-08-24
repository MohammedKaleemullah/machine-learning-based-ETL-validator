#app.py
import time
import os
import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from IPython.display import display
from validator import validate_new_rows, infer_schema

CDC_LOG = "./watched_dir/cdc_events.csv"

ANAMOLY_LOG = "./watched_dir/anomalies.csv"

class CSVWatcher(FileSystemEventHandler):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        if file_path.endswith(".csv"):
            self.last_df = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()
        elif file_path.endswith(".parquet"):
            self.last_df = pd.read_parquet(file_path) if os.path.exists(file_path) else pd.DataFrame()
        else:
            raise ValueError("Only CSV and Parquet files are supported.")
        
        print(f"File has {len(self.last_df)} rows.")

        self.schema = infer_schema(self.last_df)
        print("\nSchema:\n")
        for col in self.schema.items():
            print(f"{col}")

    def on_modified(self, event):
        if event.src_path.endswith(".csv") or event.src_path.endswith(".parquet"):
            try:
                if event.src_path.endswith(".csv"):
                    new_df = pd.read_csv(self.file_path)
                elif event.src_path.endswith(".parquet"):
                    new_df = pd.read_parquet(self.file_path)
                else:
                    return
                
                if len(new_df) > len(self.last_df):
                    diff = new_df.iloc[len(self.last_df):]

                    validation_result = validate_new_rows(diff, self.schema)

                    if not validation_result["errors"]:
                        display(f"[CDC EVENT] Valid rows:\n{validation_result['valid'].to_string(index=False)}\n")

                        if not os.path.exists(CDC_LOG):
                            validation_result["valid"].to_csv(CDC_LOG, index=False, mode="w")
                        else:
                            validation_result["valid"].to_csv(CDC_LOG, index=False, mode="a", header=False)
                    else:
                        display(f"[CDC EVENT] Validation errors:\n{validation_result['errors']}\n")
                        if not validation_result["invalid"].empty:
                            display(f"Invalid rows skipped:\n{validation_result['invalid'].to_string(index=False)}\n")
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

    print(f"watchdog watching : {file_path}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":

    #For local testing, csv file path is hardcoded
    csv_file = "./watched_dir/uber.csv"

    if not os.path.exists(csv_file):
        print("File does not exist!!")
        exit(1)

    watch_csv(csv_file)
