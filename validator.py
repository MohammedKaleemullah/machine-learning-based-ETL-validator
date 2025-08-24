# validator.py
import pandas as pd
import hashlib
import json
from datetime import datetime

def infer_schema(df: pd.DataFrame):
    schema = {}
    for col, dtype in df.dtypes.items():
        col_rules = {"not_null": True}

        if "int" in str(dtype) or "float" in str(dtype):
            col_rules["min"] = 0 if "fare" in col or "count" in col else None
            if "longitude" in col:
                col_rules["range"] = [-75, -72]
            if "latitude" in col:
                col_rules["range"] = [40, 42]

        if "datetime" in col or "date" in col:
            col_rules["datetime"] = True

        if "count" in col:
            col_rules["allowed"] = list(range(1, 7))

        schema[col] = {"dtype": str(dtype), "rules": col_rules}
    return schema


def coerce_types(row, schema):
    errors = []
    coerced = {}
    for col, spec in schema.items():
        try:
            if spec["dtype"].startswith("int"):
                coerced[col] = int(row[col])
            elif spec["dtype"].startswith("float"):
                coerced[col] = float(row[col])
            elif spec["rules"].get("datetime"):
                coerced[col] = pd.to_datetime(row[col])
            else:
                coerced[col] = str(row[col])
        except Exception:
            errors.append(f"type_error:{col}")
            coerced[col] = None
    return coerced, errors


def run_rules(row, schema):
    errors = []
    for col, spec in schema.items():
        val = row.get(col)

        if spec["rules"].get("not_null") and (val is None or pd.isna(val)):
            errors.append(f"{col}:null")

        if "min" in spec["rules"] and spec["rules"]["min"] is not None:
            if val is not None and val < spec["rules"]["min"]:
                errors.append(f"{col}:below_min")

        if "range" in spec["rules"]:
            lo, hi = spec["rules"]["range"]
            if val is not None and not (lo <= val <= hi):
                errors.append(f"{col}:out_of_range")

        if "allowed" in spec["rules"]:
            if val not in spec["rules"]["allowed"]:
                errors.append(f"{col}:not_allowed")
    return errors


def row_hash(row):
    return hashlib.md5(json.dumps(row, default=str).encode()).hexdigest()
