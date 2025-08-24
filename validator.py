# validator.py
import pandas as pd
import json

def infer_schema(df: pd.DataFrame):
    """
    Dynamically infer schema (dtypes + simple rules) from the given dataframe.
    """
    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)

        # rules = {}
        # if pd.api.types.is_numeric_dtype(df[col]):
        #     rules["min"] = df[col].min(skipna=True)
        #     rules["max"] = df[col].max(skipna=True)
        # elif pd.api.types.is_datetime64_any_dtype(df[col]):
        #     rules["parseable"] = True
        # elif pd.api.types.is_string_dtype(df[col]):
        #     rules["allowed_values"] = df[col].dropna().unique().tolist() if df[col].nunique() < 20 else None

        schema[col] = {
            "dtype": dtype
            # "rules": rules
        }
    return schema


def validate_new_rows(new_df: pd.DataFrame, schema: dict):
    """
    Validate new rows dynamically based on inferred schema.
    Returns dict: {"valid": DataFrame, "invalid": DataFrame, "errors": list}
    """
    valid_rows = []
    invalid_rows = []
    errors = []

    for idx, row in new_df.iterrows():
        row_errors = []

        for col, ruleset in schema.items():
            expected_dtype = ruleset["dtype"]

            # dtype check
            try:
                if "int" in expected_dtype:
                    _ = int(row[col])
                elif "float" in expected_dtype:
                    _ = float(row[col])
                elif "datetime" in expected_dtype:
                    pd.to_datetime(row[col])
            except Exception:
                row_errors.append(f"{col}: type mismatch (expected {expected_dtype})")

            # # rules check
            # rules = ruleset["rules"]
            # if "min" in rules and row[col] < rules["min"]:
            #     row_errors.append(f"{col}: below min {rules['min']}")
            # if "max" in rules and row[col] > rules["max"]:
            #     row_errors.append(f"{col}: above max {rules['max']}")
            # if rules.get("allowed_values") and row[col] not in rules["allowed_values"]:
            #     row_errors.append(f"{col}: value not allowed")

        if row_errors:
            errors.append({ "row": idx, "violations": row_errors, "preview": row.to_dict() })
            invalid_rows.append(row)
        else:
            valid_rows.append(row)

    return {
        "valid": pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame(columns=new_df.columns),
        "invalid": pd.DataFrame(invalid_rows) if invalid_rows else pd.DataFrame(columns=new_df.columns),
        "errors": errors
    }
