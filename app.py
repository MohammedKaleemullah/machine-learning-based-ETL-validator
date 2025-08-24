# #app.py
# import time
# import os
# import pandas as pd
# from watchdog.events import FileSystemEventHandler
# from watchdog.observers import Observer
# from IPython.display import display

# class CSVWatcher(FileSystemEventHandler):
#     def __init__(self, file_path: str):
#         super().__init__()
#         self.file_path = file_path
#         self.last_df = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()
#         print(f"Initial file loaded with {len(self.last_df)} rows.")

#     def on_modified(self, event):
#         if event.src_path.endswith(".csv"):
#             try:
#                 new_df = pd.read_csv(self.file_path)

#                 # Find new rows compared to previous snapshot
#                 if len(new_df) > len(self.last_df):
#                     diff = new_df.iloc[len(self.last_df):]
#                     display(f"[CDC EVENT] New rows detected:\n{diff.to_string(index=False)}\n")

#                 self.last_df = new_df
#             except Exception as e:
#                 print(f"Error reading CSV: {e}")


# def watch_csv(file_path: str):
#     event_handler = CSVWatcher(file_path)
#     observer = Observer()
#     observer.schedule(event_handler, path=os.path.dirname(file_path) or ".", recursive=False)
#     observer.start()

#     print(f"Watching CSV file for new rows: {file_path}")
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()


# if __name__ == "__main__":
#     csv_file = "./watched_dir/uber.csv"

#     if not os.path.exists(csv_file):
#         print("File does not exist!!")
#         exit(1)

#     watch_csv(csv_file)

import time
import os
import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from IPython.display import display

CDC_LOG = "./watched_dir/cdc_events.csv"   # log file path

class CSVWatcher(FileSystemEventHandler):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.last_df = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()
        print(f"Initial file loaded with {len(self.last_df)} rows.")

    def on_modified(self, event):
        if event.src_path.endswith(".csv"):
            try:
                new_df = pd.read_csv(self.file_path)

                # Find new rows compared to previous snapshot
                if len(new_df) > len(self.last_df):
                    diff = new_df.iloc[len(self.last_df):]

                    # Display in console (for feedback)
                    display(f"[CDC EVENT] New rows detected:\n{diff.to_string(index=False)}\n")

                    # ✅ Append to CDC log file
                    if not os.path.exists(CDC_LOG):
                        diff.to_csv(CDC_LOG, index=False, mode="w")   # create new log with header
                    else:
                        diff.to_csv(CDC_LOG, index=False, mode="a", header=False)  # append rows only

                self.last_df = new_df
            except Exception as e:
                print(f"Error reading CSV: {e}")


def watch_csv(file_path: str):
    event_handler = CSVWatcher(file_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(file_path) or ".", recursive=False)
    observer.start()

    print(f"Watching CSV file for new rows: {file_path}")
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
