# 🚖 Uber CDC Validator with ML Anomaly Detection

This project implements a **Change Data Capture (CDC) validator** for Uber trip data.  
It continuously watches a data file (`uber.csv` or `.parquet`) and validates any **newly appended rows** against a schema and an **Isolation Forest machine learning model** to detect anomalies.

## Why this Project ???

When building ETL pipelines for **real-time processes**, one of the biggest challenges is handling unexpected or invalid data. New rows may arrive with **inconsistent schemas, missing values, or anomalies** that can either distort downstream analytics (e.g., visualizations) or, in the worst case, **completely break the pipeline**. This project was designed to simulate such **real-time data validation**.

Using uber.csv as a sample dataset, new data is continuously ingested and validated in **two stages**: first, through **rule-based** schema checks to catch structural issues, and second, through **Isolation Forest (_unsupervised ML_)** to detect anomalies. All anomalies are logged and stored separately so users can identify what might have broken or corrupted the ETL flow.

While this is a **_simulation_**, the approach can be extended and optimized for production-grade, real-time ETL pipelines.

## What this project does

- Watches a CSV/Parquet file for new rows (using `watchdog`).
- Infers a schema automatically (numeric vs categorical).
- Validates new rows against schema rules:
  - Numeric type checks
  - Negative fares
  - Invalid passenger counts
- Engineers extra features:
  - **Trip distance** (via haversine formula)
  - **Fare per km**
- Runs an **Isolation Forest anomaly detection model**:
  - Flags rows that don’t fit the learned data distribution
  - Helps catch suspicious trips (e.g., impossible coordinates, abnormal fares, wrong passenger counts)
- Writes:
  - ✅ Valid rows → `cdc_events.csv`
  - ❌ Invalid/anomalous rows → `anomalies.csv`

---

## 🤔 Why Isolation Forest?

### What is Isolation Forest?

Isolation Forest is an **unsupervised anomaly detection algorithm**.  
Instead of modeling "normal" explicitly, it works by **randomly partitioning the dataset**:

- Outliers (anomalies) are easier to "isolate" since they lie far from the majority.
- Normal points require more partitions to separate.

### Why chosen here?

- Works well on **high-dimensional, mixed-type data** (numeric + categorical).
- Handles **imbalanced data** where anomalies are rare.
- Requires **no labels** — fits naturally for Uber trips where anomalies are not pre-tagged.
- Efficient on large datasets (linear time complexity).

This makes it a great fit for **Uber trip anomaly detection**, where we expect mostly normal rides but want to automatically flag unusual ones.

---

## ▶️ Pipeline Architecture

```mermaid
flowchart
    watched_dir → schema validator → feature engineering
        → ML anomaly detector → cdc_events.csv / anomalies.csv
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your data

- Place your CSV file(s) inside the `./watched_dir/` folder.
- Example file: `uber.csv`

### 3. Train the model

Run the training script on your dataset:

```bash
python train_model.py
```

This will:

- Preprocess numeric and categorical columns.
- Train an **Isolation Forest** anomaly detection model.
- Save the trained model as `isolation_forest.pkl`.

You can optionally exclude columns from training:

```python
train_isolation_forest(df, columns_to_avoid=["uid", "key", "pickup_datetime"])
```

---

## ▶️ Running Continuous Validation (CDC-style)

We use a **watchdog** observer to monitor a directory (`./watched_dir/`) for new incoming CSV files.
When a new file is detected:

- It is validated by the trained ML model.
- Anomalies are flagged.
- Results are split into ✅ valid rows and ❌ invalid rows.
- Logs are printed with `[CDC EVENT]`.

Run:

```bash
python app.py
```

Now drop new CSV files into `./watched_dir/` — anomalies will be detected in real time.

---

## ▶️ Testing with Sample Scripts

You can simulate adding new rows:

- **Add valid rows:**

```bash
python add-row.py
```

- **Add anomalies:**

```bash
python add-anamoly.py
```

---

## ▶️ Logs and Output Files

- Check the console logs from `app.py` to see real-time validation events.
- **cdc_events.csv** → Contains all valid rows that passed validation.
- **anomalies.csv** → Contains all invalid rows flagged as anomalies and skipped.

---

## ▶️ Example Output

```bash
ML detected anomalies:
     uid  fare_amount  passenger_count  anomaly_score
56186116         7.07                1       -0.15955
[CDC EVENT] Validation errors: ['ML flagged 1 anomalies']

Invalid rows skipped:
     uid  fare_amount  passenger_count  anomaly_score
56186116         7.07                1       -0.15955
```

---

# Model Evaluation with Synthetic Anomalies

---

## Metrics

| Metric    | Value |
| --------- | ----- |
| Precision | 0.933 |
| Recall    | 0.840 |
| F1 Score  | 0.884 |

---

## Confusion Matrix

| Actual \ Predicted | Normal (0) | Anomaly (1) |
| ------------------ | ---------- | ----------- |
| Normal (0)         | 47         | 3           |
| Anomaly (1)        | 8          | 42          |

---
