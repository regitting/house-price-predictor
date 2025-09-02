from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_california(as_log_target=True):
    ds = fetch_california_housing(as_frame=True)
    X = ds.frame.drop(columns=["MedHouseVal"])
    y_usd = ds.frame["MedHouseVal"] * 100_000  # USD
    if as_log_target:
        return X, np.log1p(y_usd)
    return X, y_usd

def write_example_csv(out_path="data/california_example.csv", nrows=None):
    X, y_log = load_california(as_log_target=True)
    if nrows:
        X = X.iloc[:nrows].copy()
        y_log = y_log.iloc[:nrows].copy()
    df = X.copy()
    df["target_log_price"] = y_log
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Write California example CSV")
    ap.add_argument("--out", default="data/california_example.csv")
    ap.add_argument("--rows", type=int, default=None, help="Optional row limit")
    args = ap.parse_args()

    if args.demo:
        write_example_csv(args.out, args.rows)
    else:
        print(
            "Nothing to do. Use --demo to generate a sample CSV, or provide your own CSV to predict.py."
        )
